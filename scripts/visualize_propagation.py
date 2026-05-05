#!/usr/bin/env python3
"""
Visualize contour propagation: how features at the two endpoint dots
converge (connected) or diverge (disconnected) over recurrence timesteps.

For each sample, shows:
  [input with dots]  [similarity-to-dot1 heatmap]  [similarity-to-dot2 heatmap]
  [dot-pair similarity timecourse]

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_propagation.py \
        --checkpoint checkpoints/gdn_d3_pf_no_spectral/epoch_125 \
        --cssm gdn_d2 --embed_dim 64 --kernel_size 11 --gate_type channel \
        --stem_layers 2 --norm_type temporal_layer --use_complex32 \
        --scan_mode pallas --short_conv_size 0 --no_spectral_l2_norm \
        --qkv_conv_size 1 --pos_embed spatiotemporal --pool_type max \
        --image_size 224 --difficulty 14 --seq_len 8 \
        --num_samples 10 \
        --output_dir visualizations/propagation
"""

import os
import sys
from pathlib import Path

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter
from PIL import Image as PILImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pathfinder_data import IMAGENET_MEAN, IMAGENET_STD


def denormalize(img):
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def find_dots(image_denorm, feat_h, feat_w):
    """Find endpoint dots. Returns feat-space and image-space coords."""
    H, W = image_denorm.shape[:2]
    gray = np.mean(image_denorm, axis=-1)
    gray_smooth = gaussian_filter(gray, sigma=2)

    local_max = maximum_filter(gray_smooth, size=15)
    is_max = (gray_smooth == local_max) & (gray_smooth > 0.3)
    max_positions = np.argwhere(is_max)
    if len(max_positions) < 2:
        max_positions = np.argwhere(gray > np.percentile(gray, 99))

    brightnesses = [gray_smooth[r, c] for r, c in max_positions]
    sorted_idx = np.argsort(brightnesses)[::-1]

    dot1 = max_positions[sorted_idx[0]]
    dot2 = None
    min_dist = min(H, W) * 0.15
    for idx in sorted_idx[1:]:
        candidate = max_positions[idx]
        dist = np.sqrt(np.sum((candidate - dot1) ** 2))
        if dist > min_dist:
            dot2 = candidate
            break
    if dot2 is None:
        dot2 = max_positions[sorted_idx[1]]

    scale_h, scale_w = feat_h / H, feat_w / W
    feat_coords = []
    for pos in [dot1, dot2]:
        fr = min(int(pos[0] * scale_h), feat_h - 1)
        fc = min(int(pos[1] * scale_w), feat_w - 1)
        feat_coords.append((fr, fc))

    return feat_coords, [tuple(dot1), tuple(dot2)]


def cosine_sim_map(features_hw, ref_vec):
    """Cosine similarity between each spatial location and a reference vector.

    Args:
        features_hw: (H', W', C)
        ref_vec: (C,)
    Returns:
        sim_map: (H', W') in [-1, 1]
    """
    norms = np.linalg.norm(features_hw, axis=-1, keepdims=True)
    ref_norm = np.linalg.norm(ref_vec)
    if ref_norm < 1e-8:
        return np.zeros(features_hw.shape[:2])
    normed = features_hw / (norms + 1e-8)
    ref_normed = ref_vec / ref_norm
    return np.einsum('hwc,c->hw', normed, ref_normed)


def make_propagation_gif(image, features, feat_coords, img_coords,
                         label, pred, sample_idx, output_dir,
                         frame_duration=600):
    """GIF showing feature propagation from the two endpoint dots.

    Panels:
      [input + dots]  [sim to dot1]  [sim to dot2]  [dot similarity timecourse]
    """
    T, Hp, Wp, C = features.shape
    H_img, W_img = image.shape[:2]

    correct = pred == label
    label_str = 'connected' if label == 1 else 'disconnected'
    pred_str = 'connected' if pred == 1 else 'disconnected'
    status = 'correct' if correct else 'wrong'

    (fr1, fc1), (fr2, fc2) = feat_coords
    (ir1, ic1), (ir2, ic2) = img_coords

    # Precompute dot-pair similarity and sim maps for consistent scales
    dot_sims = []
    sim_maps_1 = []
    sim_maps_2 = []
    for t in range(T):
        f1 = features[t, fr1, fc1]
        f2 = features[t, fr2, fc2]
        cos = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
        dot_sims.append(cos)
        sim_maps_1.append(cosine_sim_map(features[t], f1))
        sim_maps_2.append(cosine_sim_map(features[t], f2))

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(22, 5.5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.12)

        # Panel 1: Input with dots marked
        ax = fig.add_subplot(gs[0])
        ax.imshow(image)
        ax.plot(ic1, ir1, 'o', color='lime', markersize=14, markeredgewidth=2.5,
                markerfacecolor='none', zorder=5)
        ax.plot(ic2, ir2, 's', color='cyan', markersize=14, markeredgewidth=2.5,
                markerfacecolor='none', zorder=5)
        ax.set_title('input\n(circle=dot1, square=dot2)', fontsize=10)
        ax.axis('off')

        # Panel 2: Similarity to dot1
        ax = fig.add_subplot(gs[1])
        sm1 = sim_maps_1[t]
        sm1_up = np.array(PILImage.fromarray(sm1.astype(np.float32), mode='F')
                          .resize((W_img, H_img), PILImage.BILINEAR))
        ax.imshow(sm1_up, cmap='hot', vmin=0, vmax=1)
        ax.imshow(image, alpha=0.3)
        ax.plot(ic1, ir1, 'o', color='lime', markersize=10, markeredgewidth=2,
                markerfacecolor='none')
        ax.plot(ic2, ir2, 's', color='cyan', markersize=10, markeredgewidth=2,
                markerfacecolor='none')
        # Show similarity value at dot2's location
        sim_at_dot2 = sm1[fr2, fc2]
        ax.set_title(f't={t+1}/{T}  sim to dot1\n'
                     f'(dot2 sim={sim_at_dot2:.3f})', fontsize=10)
        ax.axis('off')

        # Panel 3: Similarity to dot2
        ax = fig.add_subplot(gs[2])
        sm2 = sim_maps_2[t]
        sm2_up = np.array(PILImage.fromarray(sm2.astype(np.float32), mode='F')
                          .resize((W_img, H_img), PILImage.BILINEAR))
        ax.imshow(sm2_up, cmap='hot', vmin=0, vmax=1)
        ax.imshow(image, alpha=0.3)
        ax.plot(ic1, ir1, 'o', color='lime', markersize=10, markeredgewidth=2,
                markerfacecolor='none')
        ax.plot(ic2, ir2, 's', color='cyan', markersize=10, markeredgewidth=2,
                markerfacecolor='none')
        sim_at_dot1 = sm2[fr1, fc1]
        ax.set_title(f't={t+1}/{T}  sim to dot2\n'
                     f'(dot1 sim={sim_at_dot1:.3f})', fontsize=10)
        ax.axis('off')

        # Panel 4: Dot-pair similarity timecourse
        ax = fig.add_subplot(gs[3])
        ts_arr = np.arange(1, T + 1)
        ax.plot(ts_arr[:t+1], dot_sims[:t+1], 'k-o', linewidth=2.5,
                markersize=8, zorder=5)
        # Show future as faint
        if t < T - 1:
            ax.plot(ts_arr[t:], dot_sims[t:], 'k--', linewidth=1, alpha=0.3)
            ax.plot(ts_arr[t+1:], dot_sims[t+1:], 'ko', markersize=5, alpha=0.3)
        ax.set_xlabel('timestep', fontsize=11)
        ax.set_ylabel('cosine similarity\n(dot1 vs dot2)', fontsize=11)
        ax.set_title('dot-pair feature similarity', fontsize=10)
        ax.set_ylim(0.4, 1.0)
        ax.set_xticks(ts_arr)
        ax.grid(True, alpha=0.3)

        # Annotate current value
        ax.annotate(f'{dot_sims[t]:.3f}',
                    xy=(t+1, dot_sims[t]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

        fig.suptitle(
            f'GT: {label_str} | Pred: {pred_str} ({"OK" if correct else "WRONG"})  '
            f'— Feature propagation from endpoint dots',
            fontsize=11, fontweight='bold',
            color='green' if correct else 'red')

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    gif_path = os.path.join(output_dir,
                            f'prop_{sample_idx:03d}_{label_str}_{status}.gif')
    pil_frames = [PILImage.fromarray(f) for f in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                        duration=frame_duration, loop=0)
    pil_frames[-1].save(gif_path.replace('.gif', '_final.png'))
    return gif_path, dot_sims


def main():
    from scripts.visualize_saliency_video import (
        make_parser, _args_to_model_kwargs, load_checkpoint,
        build_model, load_pathfinder_samples)

    parser = make_parser()
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

    # Load samples
    print(f"Loading {args.num_samples} samples per class...")
    np.random.seed(42)
    samples = load_pathfinder_samples(
        args.data_dir, args.difficulty, args.image_size,
        args.num_samples, seq_len)
    print(f"  Got {len(samples)} samples")

    # Generate per-sample GIFs
    conn_sims_all = []
    disc_sims_all = []
    correct_count = 0

    for i, (img_video, label) in enumerate(samples):
        x_5d = jnp.array(img_video)[None]
        feat = np.array(model.apply(variables, x_5d, training=False,
                                     return_features=True)[0])  # (T, H', W', C)
        logits = np.array(model.apply(variables, x_5d, training=False)[0])
        pred = int(np.argmax(logits))
        correct_count += int(pred == label)

        T, Hp, Wp, C = feat.shape
        img_denorm = denormalize(img_video[0])

        try:
            feat_coords, img_coords = find_dots(img_denorm, Hp, Wp)
        except Exception:
            print(f"  [{i+1}] dot detection failed, skipping")
            continue

        gif_path, dot_sims = make_propagation_gif(
            img_denorm, feat, feat_coords, img_coords,
            label, pred, i, output_dir, frame_duration=args.frame_duration)

        if label == 1:
            conn_sims_all.append(dot_sims)
        else:
            disc_sims_all.append(dot_sims)

        status = "OK" if pred == label else "WRONG"
        lbl = "conn" if label == 1 else "disc"
        sim_t1 = dot_sims[0]
        sim_t8 = dot_sims[-1]
        delta = sim_t8 - sim_t1
        print(f"  [{i+1}/{len(samples)}] GT={lbl} ({status}) "
              f"sim: t1={sim_t1:.3f} t8={sim_t8:.3f} delta={delta:+.3f} -> {gif_path}")

    print(f"\nAccuracy: {correct_count}/{len(samples)}")

    # Summary plot: all individual timecourses + class means
    if conn_sims_all and disc_sims_all:
        T = len(conn_sims_all[0])
        ts = np.arange(1, T + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: individual traces
        ax = axes[0]
        for sims in conn_sims_all:
            ax.plot(ts, sims, 'r-', alpha=0.25, linewidth=1)
        for sims in disc_sims_all:
            ax.plot(ts, sims, 'b-', alpha=0.25, linewidth=1)
        # Means
        conn_mean = np.mean(conn_sims_all, axis=0)
        disc_mean = np.mean(disc_sims_all, axis=0)
        ax.plot(ts, conn_mean, 'r-o', linewidth=3, label='connected (mean)',
                markersize=8, zorder=5)
        ax.plot(ts, disc_mean, 'b-o', linewidth=3, label='disconnected (mean)',
                markersize=8, zorder=5)
        ax.set_xlabel('timestep', fontsize=12)
        ax.set_ylabel('dot-pair cosine similarity', fontsize=12)
        ax.set_title('Feature similarity between endpoints\nover recurrence', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ts)

        # Right: delta from t=1
        ax = axes[1]
        for sims in conn_sims_all:
            delta = [s - sims[0] for s in sims]
            ax.plot(ts, delta, 'r-', alpha=0.25, linewidth=1)
        for sims in disc_sims_all:
            delta = [s - sims[0] for s in sims]
            ax.plot(ts, delta, 'b-', alpha=0.25, linewidth=1)
        conn_delta_mean = [np.mean([s[t] - s[0] for s in conn_sims_all]) for t in range(T)]
        disc_delta_mean = [np.mean([s[t] - s[0] for s in disc_sims_all]) for t in range(T)]
        ax.plot(ts, conn_delta_mean, 'r-o', linewidth=3, label='connected',
                markersize=8, zorder=5)
        ax.plot(ts, disc_delta_mean, 'b-o', linewidth=3, label='disconnected',
                markersize=8, zorder=5)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('timestep', fontsize=12)
        ax.set_ylabel('change in similarity from t=1', fontsize=12)
        ax.set_title('Similarity change relative to t=1\n(+ = dots becoming more similar)',
                     fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ts)

        plt.tight_layout()
        path = os.path.join(output_dir, 'propagation_summary.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Summary: {path}")

    print(f"\nOutput: {output_dir}/")


if __name__ == '__main__':
    main()
