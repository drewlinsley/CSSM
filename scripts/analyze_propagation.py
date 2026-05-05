#!/usr/bin/env python3
"""
Analyze whether the CSSM performs contour propagation over recurrence steps.

Two analyses:
1. Train separate classification probes per timestep — does accuracy ramp up?
2. Track features at endpoint dot locations over time — do they converge
   (connected) or stay separate (disconnected)?

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analyze_propagation.py \
        --checkpoint checkpoints/gdn_d3_pf_no_spectral/epoch_125 \
        --cssm gdn_d2 --embed_dim 64 --kernel_size 11 --gate_type channel \
        --stem_layers 2 --norm_type temporal_layer --use_complex32 \
        --scan_mode pallas --short_conv_size 0 --no_spectral_l2_norm \
        --qkv_conv_size 1 --pos_embed spatiotemporal --pool_type max \
        --image_size 224 --difficulty 14 --seq_len 8 \
        --num_samples 20 \
        --output_dir visualizations/propagation
"""

import os
import sys
from pathlib import Path

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training import train_state
from PIL import Image as PILImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pathfinder_data import IMAGENET_MEAN, IMAGENET_STD


def denormalize(img):
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


# ─── Per-timestep classifier (mirrors actual readout) ───

class TimestepProbe(nn.Module):
    """Classification probe: LN → SiLU → max_pool → LN → Dense(2)."""
    num_classes: int = 2

    @nn.compact
    def __call__(self, x):
        # x: (B, H', W', C)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = x.max(axis=(1, 2))         # (B, C)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.num_classes)(x)
        return x


@jax.jit
def probe_train_step(state, features, labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, features)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, loss, acc


def train_per_timestep_probe(features_t, labels, C, lr=1e-3, epochs=100,
                              batch_size=64):
    """Train a classification probe on features from a single timestep."""
    probe = TimestepProbe(num_classes=2)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, features_t.shape[1], features_t.shape[2], C))
    variables = probe.init(rng, dummy)
    state = train_state.TrainState.create(
        apply_fn=probe.apply, params=variables['params'],
        tx=optax.adam(lr))

    N = len(features_t)
    for epoch in range(epochs):
        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            state, _, _ = probe_train_step(
                state, jnp.array(features_t[idx]),
                jnp.array(labels[idx], dtype=jnp.int32))

    # Evaluate
    all_logits = []
    for i in range(0, N, batch_size):
        logits = probe.apply({'params': state.params},
                             jnp.array(features_t[i:i+batch_size]))
        all_logits.append(np.array(logits))
    all_logits = np.concatenate(all_logits)
    preds = np.argmax(all_logits, axis=-1)
    return float(np.mean(preds == labels))


# ─── Dot detection ───

def find_dots(image_denorm, feat_h, feat_w):
    """Find the two endpoint dots in a Pathfinder image.

    Dots are the brightest small regions. Returns (row, col) in
    FEATURE-SPACE coordinates (H', W').
    """
    H, W = image_denorm.shape[:2]
    gray = np.mean(image_denorm, axis=-1)

    # Dots are much brighter than dashed contour segments
    # Use a high threshold to find them
    from scipy import ndimage

    # Threshold at bright pixels
    thresh = np.percentile(gray[gray > 0.1], 95)
    binary = gray > thresh

    # Label connected components
    labeled, n_components = ndimage.label(binary)

    if n_components < 2:
        # Fallback: just use top-2 brightest pixels
        flat = gray.flatten()
        top2 = np.argsort(flat)[-2:]
        coords = [np.unravel_index(idx, gray.shape) for idx in top2]
    else:
        # Find centroids of components, pick the 2 with highest mean brightness
        centroids = ndimage.center_of_mass(gray, labeled, range(1, n_components + 1))
        means = ndimage.mean(gray, labeled, range(1, n_components + 1))
        top2_idx = np.argsort(means)[-2:]
        coords = [centroids[i] for i in top2_idx]

    # Convert to feature-space coordinates
    scale_h = feat_h / H
    scale_w = feat_w / W
    feat_coords = []
    for (r, c) in coords:
        fr = min(int(r * scale_h), feat_h - 1)
        fc = min(int(c * scale_w), feat_w - 1)
        feat_coords.append((fr, fc))

    return feat_coords


def find_dots_simple(image_denorm, feat_h, feat_w):
    """Simpler dot detection: find the two brightest local maxima."""
    H, W = image_denorm.shape[:2]
    gray = np.mean(image_denorm, axis=-1)

    # Blur slightly to merge nearby bright pixels
    from scipy.ndimage import maximum_filter, gaussian_filter
    gray_smooth = gaussian_filter(gray, sigma=2)

    # Find local maxima
    local_max = maximum_filter(gray_smooth, size=15)
    is_max = (gray_smooth == local_max) & (gray_smooth > 0.3)

    # Get all local max positions, sorted by brightness
    max_positions = np.argwhere(is_max)
    if len(max_positions) < 2:
        # Fallback
        max_positions = np.argwhere(gray > np.percentile(gray, 99))

    brightnesses = [gray_smooth[r, c] for r, c in max_positions]
    sorted_idx = np.argsort(brightnesses)[::-1]

    # Pick top 2 that are far apart (dots are at opposite ends)
    dot1 = max_positions[sorted_idx[0]]
    dot2 = None
    min_dist = min(H, W) * 0.15  # at least 15% of image apart
    for idx in sorted_idx[1:]:
        candidate = max_positions[idx]
        dist = np.sqrt(np.sum((candidate - dot1) ** 2))
        if dist > min_dist:
            dot2 = candidate
            break

    if dot2 is None:
        dot2 = max_positions[sorted_idx[1]]

    # Convert to feature-space coordinates
    scale_h = feat_h / H
    scale_w = feat_w / W
    feat_coords = []
    for pos in [dot1, dot2]:
        fr = min(int(pos[0] * scale_h), feat_h - 1)
        fc = min(int(pos[1] * scale_w), feat_w - 1)
        feat_coords.append((fr, fc))

    return feat_coords, [tuple(dot1), tuple(dot2)]


# ─── Feature collection ───

def collect_features(model, variables, samples, batch_size=4):
    all_features, all_images, all_labels = [], [], []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        videos = np.stack([s[0] for s in batch])
        labels = np.array([s[1] for s in batch])
        x_5d = jnp.array(videos)
        feat = model.apply(variables, x_5d, training=False, return_features=True)
        all_features.append(np.array(feat))
        all_images.append(videos[:, 0])
        all_labels.append(labels)
        if (i // batch_size) % 20 == 0:
            print(f"  {min(i+batch_size, len(samples))}/{len(samples)}")
    return (np.concatenate(all_features),
            np.concatenate(all_images),
            np.concatenate(all_labels))


# ─── Main ───

def main():
    from scripts.visualize_saliency_video import (
        make_parser, _args_to_model_kwargs, load_checkpoint,
        build_model, load_pathfinder_samples)

    parser = make_parser()
    parser.add_argument('--num_train', type=int, default=500)
    parser.add_argument('--probe_epochs', type=int, default=100)
    args = parser.parse_args()

    output_dir = args.output_dir or 'visualizations/propagation'
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Loading checkpoint...")
    params, batch_stats, saved_args = load_checkpoint(args.checkpoint)
    if saved_args is not None:
        from scripts.visualize_saliency_video import build_model_config
        model_kwargs = build_model_config(saved_args, args, parser)
    elif args.cssm is not None:
        model_kwargs = _args_to_model_kwargs(vars(args))
    else:
        print("ERROR: Need --cssm for legacy checkpoint")
        return

    print("Building model...")
    model = build_model(model_kwargs)
    variables = {'params': params, 'batch_stats': batch_stats}
    seq_len = model_kwargs.get('seq_len', args.seq_len)

    # Collect features
    print(f"Loading {args.num_train} samples per class...")
    np.random.seed(123)
    samples = load_pathfinder_samples(
        args.data_dir, args.difficulty, args.image_size,
        args.num_train, seq_len)
    print(f"  Got {len(samples)} samples")

    print("Collecting features...")
    features, images, labels = collect_features(model, variables, samples, batch_size=4)
    N, T, Hp, Wp, C = features.shape
    print(f"  Features: {features.shape}")

    # ═══════════════════════════════════════════════════════
    # Analysis 1: Per-timestep classification probes
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Analysis 1: Per-timestep classification probes")
    print(f"{'='*60}")
    print(f"Training separate LN→SiLU→maxpool→LN→Dense(2) probes per timestep...")

    per_t_acc = []
    for t in range(T):
        acc = train_per_timestep_probe(
            features[:, t], labels, C,
            lr=1e-3, epochs=args.probe_epochs, batch_size=64)
        per_t_acc.append(acc)
        print(f"  t={t+1}: accuracy = {acc:.3f}")

    # ═══════════════════════════════════════════════════════
    # Analysis 2: Feature similarity at dot locations
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Analysis 2: Feature similarity at endpoint dots over time")
    print(f"{'='*60}")

    conn_sims = [[] for _ in range(T)]
    disc_sims = [[] for _ in range(T)]
    conn_dists = [[] for _ in range(T)]
    disc_dists = [[] for _ in range(T)]
    dot_detect_failures = 0

    for i in range(N):
        img_denorm = denormalize(images[i])
        try:
            feat_coords, img_coords = find_dots_simple(img_denorm, Hp, Wp)
        except Exception:
            dot_detect_failures += 1
            continue

        (r1, c1), (r2, c2) = feat_coords

        for t in range(T):
            f1 = features[i, t, r1, c1]  # (C,)
            f2 = features[i, t, r2, c2]  # (C,)

            # Cosine similarity
            cos_sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
            # L2 distance
            l2_dist = np.linalg.norm(f1 - f2)

            if labels[i] == 1:
                conn_sims[t].append(cos_sim)
                conn_dists[t].append(l2_dist)
            else:
                disc_sims[t].append(cos_sim)
                disc_dists[t].append(l2_dist)

    print(f"  Dot detection: {N - dot_detect_failures}/{N} successful")

    # Compute means
    conn_sim_means = [np.mean(s) for s in conn_sims]
    disc_sim_means = [np.mean(s) for s in disc_sims]
    conn_dist_means = [np.mean(d) for d in conn_dists]
    disc_dist_means = [np.mean(d) for d in disc_dists]
    conn_sim_stds = [np.std(s) for s in conn_sims]
    disc_sim_stds = [np.std(s) for s in disc_sims]
    conn_dist_stds = [np.std(d) for d in conn_dists]
    disc_dist_stds = [np.std(d) for d in disc_dists]

    print(f"\n  Per-timestep dot-pair feature similarity:")
    print(f"  {'t':>3} | {'conn_cos':>10} | {'disc_cos':>10} | {'diff':>10} | {'conn_L2':>10} | {'disc_L2':>10}")
    print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for t in range(T):
        diff = conn_sim_means[t] - disc_sim_means[t]
        print(f"  {t+1:>3} | {conn_sim_means[t]:>10.4f} | {disc_sim_means[t]:>10.4f} | {diff:>+10.4f} | {conn_dist_means[t]:>10.2f} | {disc_dist_means[t]:>10.2f}")

    # ═══════════════════════════════════════════════════════
    # Plot
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ts = np.arange(1, T + 1)

    # Panel 1: Per-timestep probe accuracy
    ax = axes[0]
    ax.plot(ts, per_t_acc, 'k-o', linewidth=2, markersize=8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
    ax.set_xlabel('timestep', fontsize=12)
    ax.set_ylabel('accuracy', fontsize=12)
    ax.set_title('Per-timestep probe accuracy\n(separate probe trained per t)', fontsize=11)
    ax.set_ylim(0.45, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Cosine similarity between dot features
    ax = axes[1]
    ax.plot(ts, conn_sim_means, 'r-o', label='connected', linewidth=2)
    ax.fill_between(ts,
                    [m-s for m, s in zip(conn_sim_means, conn_sim_stds)],
                    [m+s for m, s in zip(conn_sim_means, conn_sim_stds)],
                    color='red', alpha=0.15)
    ax.plot(ts, disc_sim_means, 'b-o', label='disconnected', linewidth=2)
    ax.fill_between(ts,
                    [m-s for m, s in zip(disc_sim_means, disc_sim_stds)],
                    [m+s for m, s in zip(disc_sim_means, disc_sim_stds)],
                    color='blue', alpha=0.15)
    ax.set_xlabel('timestep', fontsize=12)
    ax.set_ylabel('cosine similarity', fontsize=12)
    ax.set_title('Dot-pair feature similarity\n(higher = more similar features at 2 dots)', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: L2 distance between dot features
    ax = axes[2]
    ax.plot(ts, conn_dist_means, 'r-o', label='connected', linewidth=2)
    ax.fill_between(ts,
                    [m-s for m, s in zip(conn_dist_means, conn_dist_stds)],
                    [m+s for m, s in zip(conn_dist_means, conn_dist_stds)],
                    color='red', alpha=0.15)
    ax.plot(ts, disc_dist_means, 'b-o', label='disconnected', linewidth=2)
    ax.fill_between(ts,
                    [m-s for m, s in zip(disc_dist_means, disc_dist_stds)],
                    [m+s for m, s in zip(disc_dist_means, disc_dist_stds)],
                    color='blue', alpha=0.15)
    ax.set_xlabel('timestep', fontsize=12)
    ax.set_ylabel('L2 distance', fontsize=12)
    ax.set_title('Dot-pair feature distance\n(lower = more similar features at 2 dots)', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'propagation_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {path}")

    # ═══════════════════════════════════════════════════════
    # Visualize dot detection on a few samples
    # ═══════════════════════════════════════════════════════
    print(f"\nGenerating dot detection examples...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for idx in range(10):
        ax = axes[idx // 5, idx % 5]
        img = denormalize(images[idx])
        try:
            feat_coords, img_coords = find_dots_simple(img, Hp, Wp)
            ax.imshow(img)
            for (r, c) in img_coords:
                ax.plot(c, r, 'go', markersize=12, markeredgewidth=2,
                       markerfacecolor='none')
            lbl = 'conn' if labels[idx] == 1 else 'disc'
            ax.set_title(f'{lbl} (sample {idx})', fontsize=9)
        except Exception as e:
            ax.imshow(img)
            ax.set_title(f'detection failed', fontsize=9)
        ax.axis('off')

    plt.suptitle('Dot detection (green circles)', fontsize=12)
    plt.tight_layout()
    dot_path = os.path.join(output_dir, 'dot_detection.png')
    plt.savefig(dot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Dot detection examples: {dot_path}")

    print(f"\nOutput: {output_dir}/")


if __name__ == '__main__':
    main()
