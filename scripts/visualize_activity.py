#!/usr/bin/env python3
"""
Visualize activity propagating along contours over recurrence timesteps.

Contour pixels are colored by cumulative feature change (||f[t] - f[0]||).
Connected vs disconnected samples shown side-by-side — you can SEE
activity building up along the contour path to the dots.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_activity.py \
        --checkpoint checkpoints/gdn_d3_pf_no_spectral/epoch_125 \
        --cssm gdn_d2 --embed_dim 64 --kernel_size 11 --gate_type channel \
        --stem_layers 2 --norm_type temporal_layer --use_complex32 \
        --scan_mode pallas --short_conv_size 0 --no_spectral_l2_norm \
        --qkv_conv_size 1 --pos_embed spatiotemporal --pool_type max \
        --image_size 224 --difficulty 14 --seq_len 8 \
        --num_samples 10 \
        --output_dir visualizations/activity
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
import matplotlib.colors as mcolors

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
    for idx in sorted_idx[1:]:
        candidate = max_positions[idx]
        if np.sqrt(np.sum((candidate - dot1) ** 2)) > min(H, W) * 0.15:
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


def get_contour_mask(image_denorm, sigma=1.5, threshold=0.12):
    """Extract contour pixel mask from the image. Contours are bright."""
    gray = np.mean(image_denorm, axis=-1)
    gray_smooth = gaussian_filter(gray, sigma=sigma)
    return gray_smooth > threshold


def upsample(arr_2d, H, W):
    """Upsample a 2D array to (H, W)."""
    pil = PILImage.fromarray(arr_2d.astype(np.float32), mode='F')
    return np.array(pil.resize((W, H), PILImage.BILINEAR))


def contour_activity_image(image_denorm, activity_map, contour_mask, vmax,
                           cmap_name='inferno'):
    """Color contour pixels by activity level on a dark background.

    Returns an RGB image (H, W, 3) float in [0, 1].
    """
    H, W = image_denorm.shape[:2]
    cmap = plt.get_cmap(cmap_name)

    # Normalize activity to [0, 1] using global vmax
    activity_norm = np.clip(activity_map / (vmax + 1e-8), 0, 1)

    # Start with very dark background (dim version of input)
    out = image_denorm * 0.15

    # Color contour pixels by activity
    contour_colors = cmap(activity_norm)[:, :, :3]  # (H, W, 3) RGB

    # Blend: contour pixels get colored by activity, with some input visible
    mask_3d = contour_mask[:, :, None].astype(np.float32)
    # Activity-weighted blending: low activity = dim contour, high activity = bright color
    activity_weight = np.clip(activity_norm, 0.15, 1.0)[:, :, None]
    contour_vis = contour_colors * activity_weight

    out = out * (1 - mask_3d) + contour_vis * mask_3d

    return np.clip(out, 0, 1)


def make_paired_gif(conn_img, conn_feat, conn_dots_img, conn_dots_feat,
                    disc_img, disc_feat, disc_dots_img, disc_dots_feat,
                    pair_idx, output_dir, frame_duration=600):
    """Side-by-side connected vs disconnected: contours glow with activity."""
    T, Hp, Wp, C = conn_feat.shape
    H_img, W_img = conn_img.shape[:2]

    # Contour masks
    conn_mask = get_contour_mask(conn_img)
    disc_mask = get_contour_mask(disc_img)

    # Compute cumulative feature change from t=0: ||f[t] - f[0]|| per pixel
    conn_delta = np.zeros((T, Hp, Wp))
    disc_delta = np.zeros((T, Hp, Wp))
    for t in range(T):
        conn_delta[t] = np.linalg.norm(conn_feat[t] - conn_feat[0], axis=-1)
        disc_delta[t] = np.linalg.norm(disc_feat[t] - disc_feat[0], axis=-1)

    # Upsample all activity maps
    conn_activity = np.array([upsample(conn_delta[t], H_img, W_img) for t in range(T)])
    disc_activity = np.array([upsample(disc_delta[t], H_img, W_img) for t in range(T)])

    # Global vmax for consistent colorscale (max over time, masked to contours)
    conn_on_contour = conn_activity[:, conn_mask]  # (T, num_contour_px)
    disc_on_contour = disc_activity[:, disc_mask]  # (T, num_contour_px)
    vmax = max(conn_on_contour.max() if conn_on_contour.size else 1,
               disc_on_contour.max() if disc_on_contour.size else 1) * 0.85

    # Track activity at dot locations (in feature space)
    (cr1, cc1), (cr2, cc2) = conn_dots_feat
    (dr1, dc1), (dr2, dc2) = disc_dots_feat
    conn_dot1_act = [conn_delta[t, cr1, cc1] for t in range(T)]
    conn_dot2_act = [conn_delta[t, cr2, cc2] for t in range(T)]
    disc_dot1_act = [disc_delta[t, dr1, dc1] for t in range(T)]
    disc_dot2_act = [disc_delta[t, dr2, dc2] for t in range(T)]

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(18, 9))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.2],
                               hspace=0.2, wspace=0.08)

        # ── Row 1: Connected ──
        # Input
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(conn_img)
        (ir1, ic1), (ir2, ic2) = conn_dots_img
        ax.plot(ic1, ir1, 'o', color='lime', markersize=14, markeredgewidth=3,
                markerfacecolor='none')
        ax.plot(ic2, ir2, 'o', color='lime', markersize=14, markeredgewidth=3,
                markerfacecolor='none')
        ax.set_title('CONNECTED (input)', fontsize=12, fontweight='bold',
                     color='red')
        ax.axis('off')

        # Contour activity
        ax = fig.add_subplot(gs[0, 1])
        conn_vis = contour_activity_image(conn_img, conn_activity[t],
                                          conn_mask, vmax)
        ax.imshow(conn_vis)
        ax.plot(ic1, ir1, 'o', color='lime', markersize=14, markeredgewidth=3,
                markerfacecolor='none')
        ax.plot(ic2, ir2, 'o', color='lime', markersize=14, markeredgewidth=3,
                markerfacecolor='none')
        ax.set_title(f'contour activity  t={t+1}/{T}', fontsize=11,
                     fontweight='bold')
        ax.axis('off')

        # ── Row 2: Disconnected ──
        # Input
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(disc_img)
        (ir1d, ic1d), (ir2d, ic2d) = disc_dots_img
        ax.plot(ic1d, ir1d, 'o', color='cyan', markersize=14, markeredgewidth=3,
                markerfacecolor='none')
        ax.plot(ic2d, ir2d, 'o', color='cyan', markersize=14, markeredgewidth=3,
                markerfacecolor='none')
        ax.set_title('DISCONNECTED (input)', fontsize=12, fontweight='bold',
                     color='blue')
        ax.axis('off')

        # Contour activity
        ax = fig.add_subplot(gs[1, 1])
        disc_vis = contour_activity_image(disc_img, disc_activity[t],
                                          disc_mask, vmax)
        ax.imshow(disc_vis)
        ax.plot(ic1d, ir1d, 'o', color='cyan', markersize=14, markeredgewidth=3,
                markerfacecolor='none')
        ax.plot(ic2d, ir2d, 'o', color='cyan', markersize=14, markeredgewidth=3,
                markerfacecolor='none')
        ax.set_title(f'contour activity  t={t+1}/{T}', fontsize=11,
                     fontweight='bold')
        ax.axis('off')

        # ── Right: Activity at dots over time (spans both rows) ──
        ax = fig.add_subplot(gs[:, 2])
        ts = np.arange(1, T + 1)

        ax.plot(ts[:t+1], conn_dot1_act[:t+1], 'r-o', linewidth=2.5,
                markersize=8, label='conn dot 1')
        ax.plot(ts[:t+1], conn_dot2_act[:t+1], 'r--s', linewidth=2.5,
                markersize=8, label='conn dot 2')
        ax.plot(ts[:t+1], disc_dot1_act[:t+1], 'b-o', linewidth=2.5,
                markersize=8, label='disc dot 1')
        ax.plot(ts[:t+1], disc_dot2_act[:t+1], 'b--s', linewidth=2.5,
                markersize=8, label='disc dot 2')

        if t < T - 1:
            ax.plot(ts[t:], conn_dot1_act[t:], 'r-', alpha=0.15, linewidth=1)
            ax.plot(ts[t:], conn_dot2_act[t:], 'r--', alpha=0.15, linewidth=1)
            ax.plot(ts[t:], disc_dot1_act[t:], 'b-', alpha=0.15, linewidth=1)
            ax.plot(ts[t:], disc_dot2_act[t:], 'b--', alpha=0.15, linewidth=1)

        ax.set_xlabel('recurrence timestep', fontsize=12)
        ax.set_ylabel('activity at dot\n(feature change from t=1, L2)', fontsize=11)
        ax.set_title('Activity at endpoint dots', fontsize=12,
                     fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.set_xticks(ts)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, T + 0.5)

        fig.suptitle(
            'Contour activity over recurrence — brighter = more '
            'feature change from t=1',
            fontsize=13, fontweight='bold')

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    gif_path = os.path.join(output_dir, f'activity_{pair_idx:02d}.gif')
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

    output_dir = args.output_dir or 'visualizations/activity'
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

    # Split into connected / disconnected
    conn_data = []
    disc_data = []

    for i, (img_video, label) in enumerate(samples):
        x_5d = jnp.array(img_video)[None]
        feat = np.array(model.apply(variables, x_5d, training=False,
                                     return_features=True)[0])  # (T, H', W', C)
        T, Hp, Wp, C = feat.shape
        img_denorm = denormalize(img_video[0])

        try:
            feat_coords, img_coords = find_dots(img_denorm, Hp, Wp)
        except Exception:
            print(f"  [{i+1}] dot detection failed")
            continue

        entry = (img_denorm, feat, img_coords, feat_coords)
        if label == 1:
            conn_data.append(entry)
        else:
            disc_data.append(entry)

        lbl = "conn" if label == 1 else "disc"
        print(f"  [{i+1}/{len(samples)}] {lbl}")

    # Make paired GIFs
    n_pairs = min(len(conn_data), len(disc_data))
    print(f"\nGenerating {n_pairs} paired activity GIFs...")

    for p in range(n_pairs):
        c_img, c_feat, c_dots_img, c_dots_feat = conn_data[p]
        d_img, d_feat, d_dots_img, d_dots_feat = disc_data[p]

        gif_path = make_paired_gif(
            c_img, c_feat, c_dots_img, c_dots_feat,
            d_img, d_feat, d_dots_img, d_dots_feat,
            p, output_dir, frame_duration=args.frame_duration)
        print(f"  Pair {p}: {gif_path}")

    print(f"\nOutput: {output_dir}/")


if __name__ == '__main__':
    main()
