#!/usr/bin/env python3
"""
Train a dual-head probe with shared bottleneck latent space.

Architecture:
    post-CSSM features (H', W', C=64)
        → 1x1 projection → latent (H', W', D)
        → Head 1: 1x1 conv → RGB reconstruction (H', W', 3)
        → Head 2: LayerNorm → SiLU → max_pool → LayerNorm → Dense(2) classification

The shared latent must serve both tasks. After training, we find the latent
dimensions most aligned with the classification decision and visualize them
as spatial heatmaps — showing WHERE the model encodes decision-relevant info.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_dual_probe.py \
        --checkpoint checkpoints/gdn_d3_pf_no_spectral/epoch_125 \
        --cssm gdn_d2 --embed_dim 64 --kernel_size 11 --gate_type channel \
        --stem_layers 2 --norm_type temporal_layer --use_complex32 \
        --scan_mode pallas --short_conv_size 0 --no_spectral_l2_norm \
        --qkv_conv_size 1 --pos_embed spatiotemporal --pool_type max \
        --image_size 224 --difficulty 14 --seq_len 8 \
        --num_samples 20 \
        --output_dir visualizations/dual_probe
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from functools import partial

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
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pathfinder_data import IMAGENET_MEAN, IMAGENET_STD


def denormalize(img):
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def downsample(img_hw3, target_h, target_w):
    out = np.zeros((target_h, target_w, 3), dtype=np.float32)
    for c in range(3):
        pil = PILImage.fromarray(img_hw3[:, :, c].astype(np.float32), mode='F')
        pil = pil.resize((target_w, target_h), PILImage.BILINEAR)
        out[:, :, c] = np.array(pil)
    return out


# ─── Dual-head probe model ───

class DualProbe(nn.Module):
    """Shared bottleneck with reconstruction + classification heads."""
    latent_dim: int = 16
    num_classes: int = 2

    @nn.compact
    def __call__(self, x, training=False):
        """
        Args:
            x: (B, H', W', C) features at one timestep
        Returns:
            dict with 'latent', 'recon', 'logits'
        """
        # Project to bottleneck latent
        z = nn.Dense(self.latent_dim, name='proj')(x)   # (B, H', W', D)

        # Head 1: per-pixel reconstruction
        recon = nn.Dense(3, name='recon_head')(z)        # (B, H', W', 3)

        # Head 2: classification (mirrors actual model readout)
        h = nn.LayerNorm(name='cls_norm_pre')(z)
        h = nn.silu(h)
        h = h.max(axis=(1, 2))                          # max pool → (B, D)
        h = nn.LayerNorm(name='cls_norm_post')(h)
        logits = nn.Dense(self.num_classes, name='cls_head')(h)  # (B, 2)

        return {'latent': z, 'recon': recon, 'logits': logits}


# ─── Training ───

def create_train_state(rng, model, learning_rate, input_shape):
    variables = model.init(rng, jnp.ones(input_shape))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=variables['params'], tx=tx)


@jax.jit
def train_step(state, features, targets, labels, recon_weight, cls_weight):
    """One training step.

    Args:
        features: (B, H', W', C)
        targets: (B, H', W', 3) downsampled images
        labels: (B,) integer class labels
    """
    def loss_fn(params):
        out = state.apply_fn({'params': params}, features)
        # Reconstruction loss (MSE)
        recon_loss = jnp.mean((out['recon'] - targets) ** 2)
        # Classification loss (cross-entropy)
        cls_loss = optax.softmax_cross_entropy_with_integer_labels(
            out['logits'], labels).mean()
        total = recon_weight * recon_loss + cls_weight * cls_loss
        return total, {'recon_loss': recon_loss, 'cls_loss': cls_loss,
                       'logits': out['logits']}

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(aux['logits'], axis=-1) == labels)
    return state, loss, aux['recon_loss'], aux['cls_loss'], acc


def collect_features(model, variables, samples, batch_size=4):
    """Run frozen model, collect features at each timestep."""
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


# ─── Visualization ───

def apply_probe_cls_pipeline(latent, probe_params):
    """Apply the probe's classification pipeline to latent features.

    Replicates: LayerNorm → SiLU → (before max_pool)
    Returns per-pixel post-activation features.

    Args:
        latent: (H', W', D) or (T, H', W', D)
        probe_params: probe model params dict
    Returns:
        activated: same shape, post-LN+SiLU features
    """
    # LayerNorm params
    scale = np.array(probe_params['cls_norm_pre']['scale'])  # (D,)
    bias = np.array(probe_params['cls_norm_pre']['bias'])    # (D,)

    # Apply LayerNorm over last axis
    mean = latent.mean(axis=-1, keepdims=True)
    var = latent.var(axis=-1, keepdims=True)
    normed = (latent - mean) / np.sqrt(var + 1e-5)
    normed = normed * scale + bias

    # SiLU activation
    activated = normed * (1.0 / (1.0 + np.exp(-normed)))
    return activated


def make_evidence_gif(image, latent_per_t, probe_params, w_decision,
                      label, pred, sample_idx, output_dir, frame_duration=600):
    """GIF showing max-pool winner locations and their decision contributions.

    The classification pipeline is: latent → LN → SiLU → max_pool → LN → head.
    After LN+SiLU, each channel has a spatial activation map. Max pool picks
    ONE winner location per channel. We show:
      [input]  [winner locations colored by decision weight]
      [top +conn channel]  [top -conn channel]
    """
    T, Hp, Wp, D = latent_per_t.shape
    H_img, W_img = image.shape[:2]

    correct = pred == label
    label_str = 'connected' if label == 1 else 'disconnected'
    pred_str = 'connected' if pred == 1 else 'disconnected'
    status = 'correct' if correct else 'wrong'

    # Apply LN + SiLU to get post-activation features at each timestep
    activated = apply_probe_cls_pipeline(latent_per_t, probe_params)  # (T, H', W', D)

    # Sort channels by decision weight
    top_pos = np.argsort(w_decision)[::-1]  # most +conn first
    top_neg = np.argsort(w_decision)        # most -conn first

    # For the summary, compute max-pooled vector at each t and project
    maxpool_per_t = activated.max(axis=(1, 2))  # (T, D)

    # Apply norm_post to max-pooled vectors
    scale_post = np.array(probe_params['cls_norm_post']['scale'])
    bias_post = np.array(probe_params['cls_norm_post']['bias'])
    mp_mean = maxpool_per_t.mean(axis=-1, keepdims=True)
    mp_var = maxpool_per_t.var(axis=-1, keepdims=True)
    mp_normed = (maxpool_per_t - mp_mean) / np.sqrt(mp_var + 1e-5)
    mp_normed = mp_normed * scale_post + bias_post

    # Project onto decision direction → scalar evidence per timestep
    W_cls = np.array(probe_params['cls_head']['kernel'])
    b_cls = np.array(probe_params['cls_head']['bias'])
    logits_per_t = mp_normed @ W_cls + b_cls  # (T, 2)

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(24, 5))
        gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1.2, 1, 1, 1.3], wspace=0.15)

        # Panel 1: Input
        ax = fig.add_subplot(gs[0])
        ax.imshow(image)
        ax.set_title('input', fontsize=10)
        ax.axis('off')

        # Panel 2: Max-pool winner locations
        # For each channel, find argmax location and mark it
        ax = fig.add_subplot(gs[1])
        ax.imshow(image, alpha=0.5)

        act_t = activated[t]  # (H', W', D)
        scale_h = H_img / Hp
        scale_w = W_img / Wp
        for d in range(D):
            ch_map = act_t[:, :, d]
            max_idx = np.unravel_index(np.argmax(ch_map), ch_map.shape)
            max_val = ch_map[max_idx]
            # Map to image coordinates
            y_img = (max_idx[0] + 0.5) * scale_h
            x_img = (max_idx[1] + 0.5) * scale_w
            # Color by decision weight: red = +conn, blue = -conn
            # Size by activation magnitude
            w = w_decision[d]
            color = 'red' if w > 0 else 'blue'
            size = 30 + 100 * (np.abs(w) / np.abs(w_decision).max())
            ax.scatter(x_img, y_img, c=color, s=size, alpha=0.7,
                      edgecolors='white', linewidth=0.5, zorder=5)

        logit_conn = logits_per_t[t, 1]
        logit_disc = logits_per_t[t, 0]
        ax.set_title(f't={t+1}/{T} max-pool winners\n'
                     f'logit: conn={logit_conn:.1f} disc={logit_disc:.1f}',
                     fontsize=9)
        ax.axis('off')

        # Panel 3: Top +conn channel spatial map
        d_pos = top_pos[0]
        ax = fig.add_subplot(gs[2])
        ch_map = act_t[:, :, d_pos]
        ch_up = np.array(PILImage.fromarray(ch_map.astype(np.float32), mode='F')
                         .resize((W_img, H_img), PILImage.BILINEAR))
        ax.imshow(ch_up, cmap='Reds')
        ax.imshow(image, alpha=0.3)
        max_idx = np.unravel_index(np.argmax(ch_map), ch_map.shape)
        ax.scatter((max_idx[1]+0.5)*scale_w, (max_idx[0]+0.5)*scale_h,
                  c='lime', s=80, marker='*', zorder=5)
        ax.set_title(f'dim {d_pos} (+conn, w={w_decision[d_pos]:.2f})\n'
                     f'max={ch_map.max():.2f}', fontsize=9)
        ax.axis('off')

        # Panel 4: Top -conn channel spatial map
        d_neg = top_neg[0]
        ax = fig.add_subplot(gs[3])
        ch_map = act_t[:, :, d_neg]
        ch_up = np.array(PILImage.fromarray(ch_map.astype(np.float32), mode='F')
                         .resize((W_img, H_img), PILImage.BILINEAR))
        ax.imshow(ch_up, cmap='Blues')
        ax.imshow(image, alpha=0.3)
        max_idx = np.unravel_index(np.argmax(ch_map), ch_map.shape)
        ax.scatter((max_idx[1]+0.5)*scale_w, (max_idx[0]+0.5)*scale_h,
                  c='lime', s=80, marker='*', zorder=5)
        ax.set_title(f'dim {d_neg} (-conn, w={w_decision[d_neg]:.2f})\n'
                     f'max={ch_map.max():.2f}', fontsize=9)
        ax.axis('off')

        # Panel 5: Logit timecourse
        ax = fig.add_subplot(gs[4])
        ts_arr = np.arange(1, T+1)
        ax.plot(ts_arr, logits_per_t[:, 1], 'r-o', label='connected', linewidth=2)
        ax.plot(ts_arr, logits_per_t[:, 0], 'b-o', label='disconnected', linewidth=2)
        ax.axvline(t+1, color='black', linestyle='--', alpha=0.5)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('timestep')
        ax.set_ylabel('logit')
        ax.set_title('probe classification', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f'GT: {label_str} | Pred: {pred_str} ({"OK" if correct else "WRONG"})  '
            f'[red dots=+conn channels, blue dots=-conn channels, star=argmax]',
            fontsize=10, fontweight='bold',
            color='green' if correct else 'red')

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    gif_path = os.path.join(output_dir, f'dual_{sample_idx:03d}_{label_str}_{status}.gif')
    pil_frames = [PILImage.fromarray(f) for f in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                        duration=frame_duration, loop=0)
    pil_frames[-1].save(gif_path.replace('.gif', '_final.png'))
    return gif_path, logits_per_t


def make_summary(all_logits, all_labels, output_dir):
    """Summary: probe logit timecourses for connected vs disconnected."""
    T = all_logits[0].shape[0]

    conn_logits = [lg for lg, lab in zip(all_logits, all_labels) if lab == 1]
    disc_logits = [lg for lg, lab in zip(all_logits, all_labels) if lab == 0]

    if not conn_logits or not disc_logits:
        return

    # logits shape: (T, 2) per sample
    # Compute logit difference (conn - disc) per timestep
    conn_diff = [np.mean([lg[t, 1] - lg[t, 0] for lg in conn_logits]) for t in range(T)]
    disc_diff = [np.mean([lg[t, 1] - lg[t, 0] for lg in disc_logits]) for t in range(T)]
    conn_std = [np.std([lg[t, 1] - lg[t, 0] for lg in conn_logits]) for t in range(T)]
    disc_std = [np.std([lg[t, 1] - lg[t, 0] for lg in disc_logits]) for t in range(T)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ts = np.arange(1, T + 1)

    # Panel 1: Logit difference (conn - disc class) over time
    ax = axes[0]
    ax.plot(ts, conn_diff, 'r-o', label='connected samples', linewidth=2)
    ax.fill_between(ts,
                    [m - s for m, s in zip(conn_diff, conn_std)],
                    [m + s for m, s in zip(conn_diff, conn_std)],
                    color='red', alpha=0.15)
    ax.plot(ts, disc_diff, 'b-o', label='disconnected samples', linewidth=2)
    ax.fill_between(ts,
                    [m - s for m, s in zip(disc_diff, disc_std)],
                    [m + s for m, s in zip(disc_diff, disc_std)],
                    color='blue', alpha=0.15)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('timestep')
    ax.set_ylabel('logit(conn) - logit(disc)')
    ax.set_title('Probe decision evidence over time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Classification accuracy per timestep
    ax = axes[1]
    accs = []
    for t in range(T):
        correct = 0
        total = 0
        for lg, lab in zip(all_logits, all_labels):
            pred_t = int(np.argmax(lg[t]))
            correct += int(pred_t == lab)
            total += 1
        accs.append(correct / total)
    ax.plot(ts, accs, 'k-o', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('timestep')
    ax.set_ylabel('accuracy')
    ax.set_title('Probe classification accuracy over time')
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'summary_dual_probe.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Summary saved: {path}")

    print(f"\n  Per-timestep probe logit difference (+ = votes connected):")
    print(f"  {'t':>3} | {'conn_samples':>12} | {'disc_samples':>12} | {'accuracy':>8}")
    print(f"  {'-'*3}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    for t in range(T):
        print(f"  {t+1:>3} | {conn_diff[t]:>+12.3f} | {disc_diff[t]:>+12.3f} | {accs[t]:>8.3f}")


# ─── Main ───

def main():
    from scripts.visualize_saliency_video import (
        make_parser, _args_to_model_kwargs, load_checkpoint,
        build_model, load_pathfinder_samples)

    parser = make_parser()
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--probe_lr', type=float, default=1e-3)
    parser.add_argument('--probe_epochs', type=int, default=100)
    parser.add_argument('--probe_batch_size', type=int, default=64)
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--cls_weight', type=float, default=1.0)
    parser.add_argument('--num_train', type=int, default=500,
                        help='Samples per class for probe training')
    args = parser.parse_args()

    output_dir = args.output_dir or 'visualizations/dual_probe'
    os.makedirs(output_dir, exist_ok=True)

    # ── Load frozen model ──
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

    # ── Collect training features ──
    print(f"Loading {args.num_train} training samples per class...")
    np.random.seed(123)
    train_samples = load_pathfinder_samples(
        args.data_dir, args.difficulty, args.image_size,
        args.num_train, seq_len)
    print(f"  Got {len(train_samples)} samples")

    print("Collecting features...")
    features, images, labels = collect_features(
        model, variables, train_samples, batch_size=4)
    print(f"  Features: {features.shape}")  # (N, T, H', W', C)

    N, T, Hp, Wp, C = features.shape

    # Downsample target images to feature resolution
    targets_ds = np.zeros((N, Hp, Wp, 3), dtype=np.float32)
    for i in range(N):
        targets_ds[i] = downsample(images[i], Hp, Wp)

    # Use T=8 features for training (last timestep, where readout happens)
    train_feats = features[:, -1]   # (N, H', W', C)
    train_labels = labels            # (N,)

    # ── Train dual probe ──
    print(f"\nTraining DualProbe (latent_dim={args.latent_dim}, "
          f"recon_weight={args.recon_weight}, cls_weight={args.cls_weight})...")

    probe = DualProbe(latent_dim=args.latent_dim, num_classes=2)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(
        rng, probe, args.probe_lr, (1, Hp, Wp, C))

    # Training loop
    n_train = len(train_feats)
    batch_size = min(args.probe_batch_size, n_train)

    for epoch in range(args.probe_epochs):
        perm = np.random.permutation(n_train)
        epoch_loss, epoch_recon, epoch_cls, epoch_acc = 0, 0, 0, 0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            batch_feat = jnp.array(train_feats[idx])
            batch_tgt = jnp.array(targets_ds[idx])
            batch_lab = jnp.array(train_labels[idx], dtype=jnp.int32)

            state, loss, rl, cl, acc = train_step(
                state, batch_feat, batch_tgt, batch_lab,
                args.recon_weight, args.cls_weight)

            epoch_loss += float(loss)
            epoch_recon += float(rl)
            epoch_cls += float(cl)
            epoch_acc += float(acc)
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={epoch_loss/n_batches:.4f} "
                  f"recon={epoch_recon/n_batches:.6f} "
                  f"cls={epoch_cls/n_batches:.4f} "
                  f"acc={epoch_acc/n_batches:.3f}")

    # ── Extract decision direction in latent space ──
    probe_params = state.params
    W_cls = np.array(probe_params['cls_head']['kernel'])   # (D, 2)
    b_cls = np.array(probe_params['cls_head']['bias'])     # (2,)

    # Decision direction in latent space (before norm/act/pool)
    # This is approximate — the actual decision goes through LN+SiLU+pool+LN
    # But the latent dims most aligned with this direction are the ones
    # that matter for classification
    w_decision = W_cls[:, 1] - W_cls[:, 0]  # (D,)
    print(f"\n  Decision direction in latent space:")
    print(f"  w_decision norm: {np.linalg.norm(w_decision):.3f}")
    for d in np.argsort(np.abs(w_decision))[::-1]:
        sign = '+conn' if w_decision[d] > 0 else '-conn'
        print(f"    dim {d:2d}: {w_decision[d]:+.4f} ({sign})")

    # ── Evaluate probe on all timesteps ──
    print(f"\n  Per-timestep probe classification accuracy:")
    for t in range(T):
        feat_t = jnp.array(features[:, t])  # (N, H', W', C)
        # Process in batches to avoid OOM
        all_logits = []
        for i in range(0, N, batch_size):
            out = probe.apply({'params': probe_params}, feat_t[i:i+batch_size])
            all_logits.append(np.array(out['logits']))
        all_logits = np.concatenate(all_logits)
        preds = np.argmax(all_logits, axis=-1)
        acc = np.mean(preds == labels)
        print(f"    t={t+1}: {acc:.3f}")

    # Save probe
    probe_path = os.path.join(output_dir, 'dual_probe.pkl')
    with open(probe_path, 'wb') as f:
        pickle.dump({
            'params': jax.tree.map(np.array, probe_params),
            'latent_dim': args.latent_dim,
            'w_decision': w_decision,
        }, f)
    print(f"  Saved probe: {probe_path}")

    # ── Visualization on separate samples ──
    print(f"\nGenerating decision evidence GIFs...")
    np.random.seed(42)
    vis_samples = load_pathfinder_samples(
        args.data_dir, args.difficulty, args.image_size,
        args.num_samples, seq_len)

    all_logits_vis = []
    all_vis_labels = []
    correct_count = 0

    # Convert probe params to numpy for the manual pipeline
    probe_params_np = jax.tree.map(np.array, probe_params)

    for i, (img_video, label) in enumerate(vis_samples):
        x_5d = jnp.array(img_video)[None]

        # Get features at all timesteps
        feat = np.array(model.apply(variables, x_5d, training=False,
                                     return_features=True)[0])  # (T, H', W', C)

        # Get model prediction
        logits = np.array(model.apply(variables, x_5d, training=False)[0])
        pred = int(np.argmax(logits))
        correct_count += int(pred == label)

        # Project through probe to get latent at each timestep
        latent_per_t = []
        for t in range(T):
            out = probe.apply({'params': probe_params}, feat[t:t+1])
            latent_per_t.append(np.array(out['latent'][0]))
        latent_per_t = np.stack(latent_per_t)  # (T, H', W', D)

        gif_path, logits_per_t = make_evidence_gif(
            denormalize(img_video[0]), latent_per_t, probe_params_np,
            w_decision, label, pred, i, output_dir,
            frame_duration=args.frame_duration)

        all_logits_vis.append(logits_per_t)
        all_vis_labels.append(label)

        status = "OK" if pred == label else "WRONG"
        lbl = "conn" if label == 1 else "disc"
        prd = "conn" if pred == 1 else "disc"
        diff_t8 = logits_per_t[-1, 1] - logits_per_t[-1, 0]
        print(f"  [{i+1}/{len(vis_samples)}] GT={lbl} Pred={prd} ({status}) "
              f"logit_diff@T8={diff_t8:+.2f} -> {gif_path}")

    print(f"\nModel accuracy: {correct_count}/{len(vis_samples)}")

    # Summary
    make_summary(all_logits_vis, all_vis_labels, output_dir)
    print(f"\nOutput: {output_dir}/")


if __name__ == '__main__':
    main()
