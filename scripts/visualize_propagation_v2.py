#!/usr/bin/env python3
"""
Clean visualization of contour propagation: paired connected vs disconnected
samples showing how endpoint dot features converge or diverge over time.

For each pair, GIF animates through timesteps showing:
  Row 1: Connected sample — input + which contour connects the dots
  Row 2: Disconnected sample — input + dots on separate contours
  Right: Overlaid timecourse of dot-pair cosine similarity for both

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_propagation_v2.py \
        --checkpoint checkpoints/gdn_d3_pf_no_spectral/epoch_125 \
        --cssm gdn_d2 --embed_dim 64 --kernel_size 11 --gate_type channel \
        --stem_layers 2 --norm_type temporal_layer --use_complex32 \
        --scan_mode pallas --short_conv_size 0 --no_spectral_l2_norm \
        --qkv_conv_size 1 --pos_embed spatiotemporal --pool_type max \
        --image_size 224 --difficulty 14 --seq_len 8 \
        --num_samples 10 \
        --output_dir visualizations/propagation_v2
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
        if np.sqrt(np.sum((candidate - dot1) ** 2)) > min_dist:
            dot2 = candidate
            break
    if dot2 is None:
        dot2 = max_positions[sorted_idx[1]]
    scale_h, scale_w = feat_h / H, feat_w / W
    feat_coords = []
    for pos in [dot1, dot2]:
        feat_coords.append((min(int(pos[0] * scale_h), feat_h - 1),
                            min(int(pos[1] * scale_w), feat_w - 1)))
    return feat_coords, [tuple(dot1), tuple(dot2)]


def compute_dot_sims(features, feat_coords):
    """Cosine similarity between the two dots at each timestep."""
    T = features.shape[0]
    (fr1, fc1), (fr2, fc2) = feat_coords
    sims = []
    for t in range(T):
        f1 = features[t, fr1, fc1]
        f2 = features[t, fr2, fc2]
        cos = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
        sims.append(cos)
    return sims


def make_paired_gif(conn_img, conn_sims, conn_img_dots,
                    disc_img, disc_sims, disc_img_dots,
                    pair_idx, output_dir, frame_duration=600):
    """GIF: connected vs disconnected side by side with timecourse."""
    T = len(conn_sims)

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.5], wspace=0.2, hspace=0.25)

        # Top-left: Connected input with dots
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(conn_img)
        (r1, c1), (r2, c2) = conn_img_dots
        ax.plot(c1, r1, 'o', color='lime', markersize=16, markeredgewidth=3,
                markerfacecolor='none')
        ax.plot(c2, r2, 'o', color='lime', markersize=16, markeredgewidth=3,
                markerfacecolor='none')
        ax.set_title('CONNECTED', fontsize=13, fontweight='bold', color='red')
        ax.axis('off')

        # Bottom-left: Disconnected input with dots
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(disc_img)
        (r1, c1), (r2, c2) = disc_img_dots
        ax.plot(c1, r1, 'o', color='cyan', markersize=16, markeredgewidth=3,
                markerfacecolor='none')
        ax.plot(c2, r2, 'o', color='cyan', markersize=16, markeredgewidth=3,
                markerfacecolor='none')
        ax.set_title('DISCONNECTED', fontsize=13, fontweight='bold', color='blue')
        ax.axis('off')

        # Right: Timecourse (spans both rows)
        ax = fig.add_subplot(gs[:, 1])
        ts = np.arange(1, T + 1)

        # Plot full traces faintly
        ax.plot(ts, conn_sims, 'r-', linewidth=1, alpha=0.3)
        ax.plot(ts, disc_sims, 'b-', linewidth=1, alpha=0.3)

        # Plot up to current timestep bold
        ax.plot(ts[:t+1], conn_sims[:t+1], 'r-o', linewidth=3,
                markersize=10, label='connected', zorder=5)
        ax.plot(ts[:t+1], disc_sims[:t+1], 'b-s', linewidth=3,
                markersize=10, label='disconnected', zorder=5)

        # Annotate current values
        ax.annotate(f'{conn_sims[t]:.3f}',
                    xy=(t+1, conn_sims[t]), xytext=(12, 8),
                    textcoords='offset points', fontsize=13, fontweight='bold',
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                              edgecolor='red', alpha=0.9))
        ax.annotate(f'{disc_sims[t]:.3f}',
                    xy=(t+1, disc_sims[t]), xytext=(12, -18),
                    textcoords='offset points', fontsize=13, fontweight='bold',
                    color='blue',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                              edgecolor='blue', alpha=0.9))

        ax.set_xlabel('recurrence timestep', fontsize=13)
        ax.set_ylabel('cosine similarity between\nendpoint dot features', fontsize=13)
        ax.set_title(f't = {t+1} / {T}', fontsize=14, fontweight='bold')
        ax.set_ylim(0.1, 1.05)
        ax.set_xticks(ts)
        ax.set_xticklabels([str(i) for i in ts], fontsize=11)
        ax.legend(fontsize=12, loc='lower left')
        ax.grid(True, alpha=0.3)

        # Shade background to show which class "wins"
        if t == T - 1:
            if conn_sims[t] > disc_sims[t]:
                ax.text(0.98, 0.02, 'dots converge → connected',
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=11, color='green', fontweight='bold')
            else:
                ax.text(0.98, 0.02, 'dots diverge → disconnected?',
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=11, color='orange', fontweight='bold')

        fig.suptitle('Feature propagation: do endpoint dots converge?',
                     fontsize=14, fontweight='bold')

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    gif_path = os.path.join(output_dir, f'pair_{pair_idx:02d}.gif')
    pil_frames = [PILImage.fromarray(f) for f in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                        duration=frame_duration, loop=0)
    pil_frames[-1].save(gif_path.replace('.gif', '_final.png'))
    return gif_path


def main():
    from scripts.visualize_saliency_video import (
        make_parser, _args_to_model_kwargs, load_checkpoint,
        build_model, load_pathfinder_samples)

    parser = make_parser()
    args = parser.parse_args()

    output_dir = args.output_dir or 'visualizations/propagation_v2'
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

    # Process all samples
    conn_data = []  # (image, sims, img_dots)
    disc_data = []

    for i, (img_video, label) in enumerate(samples):
        x_5d = jnp.array(img_video)[None]
        feat = np.array(model.apply(variables, x_5d, training=False,
                                     return_features=True)[0])
        T, Hp, Wp, C = feat.shape
        img_denorm = denormalize(img_video[0])

        try:
            feat_coords, img_coords = find_dots(img_denorm, Hp, Wp)
        except Exception:
            continue

        sims = compute_dot_sims(feat, feat_coords)
        entry = (img_denorm, sims, img_coords)

        if label == 1:
            conn_data.append(entry)
        else:
            disc_data.append(entry)

        lbl = "conn" if label == 1 else "disc"
        delta = sims[-1] - sims[0]
        print(f"  [{i+1}/{len(samples)}] {lbl}: sim t1={sims[0]:.3f} t8={sims[-1]:.3f} delta={delta:+.3f}")

    # Make paired GIFs
    n_pairs = min(len(conn_data), len(disc_data))
    print(f"\nGenerating {n_pairs} paired GIFs...")

    all_conn_sims = []
    all_disc_sims = []

    for p in range(n_pairs):
        c_img, c_sims, c_dots = conn_data[p]
        d_img, d_sims, d_dots = disc_data[p]

        all_conn_sims.append(c_sims)
        all_disc_sims.append(d_sims)

        gif_path = make_paired_gif(
            c_img, c_sims, c_dots,
            d_img, d_sims, d_dots,
            p, output_dir, frame_duration=args.frame_duration)
        print(f"  Pair {p}: {gif_path}")

    # Summary with all traces
    if all_conn_sims and all_disc_sims:
        T = len(all_conn_sims[0])
        ts = np.arange(1, T + 1)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Individual traces
        for sims in all_conn_sims:
            ax.plot(ts, sims, 'r-', alpha=0.2, linewidth=1)
        for sims in all_disc_sims:
            ax.plot(ts, sims, 'b-', alpha=0.2, linewidth=1)

        # Means
        conn_mean = np.mean(all_conn_sims, axis=0)
        disc_mean = np.mean(all_disc_sims, axis=0)
        conn_std = np.std(all_conn_sims, axis=0)
        disc_std = np.std(all_disc_sims, axis=0)

        ax.plot(ts, conn_mean, 'r-o', linewidth=3, markersize=10,
                label=f'connected (n={len(all_conn_sims)})', zorder=5)
        ax.fill_between(ts, conn_mean - conn_std, conn_mean + conn_std,
                        color='red', alpha=0.15)
        ax.plot(ts, disc_mean, 'b-s', linewidth=3, markersize=10,
                label=f'disconnected (n={len(all_disc_sims)})', zorder=5)
        ax.fill_between(ts, disc_mean - disc_std, disc_mean + disc_std,
                        color='blue', alpha=0.15)

        ax.set_xlabel('recurrence timestep', fontsize=14)
        ax.set_ylabel('cosine similarity between\nendpoint dot features', fontsize=14)
        ax.set_title('Contour propagation evidence:\nconnected dots converge, disconnected dots diverge',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.set_xticks(ts)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.2, 1.0)

        plt.tight_layout()
        path = os.path.join(output_dir, 'summary.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Summary: {path}")

    print(f"\nOutput: {output_dir}/")


if __name__ == '__main__':
    main()
