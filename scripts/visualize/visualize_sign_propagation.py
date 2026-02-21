#!/usr/bin/env python3
"""
Trace sign propagation along contours.

Key hypothesis: The two dots get opposite signs, and these signs
propagate along the contour via the kernel convolution. If connected,
the signs meet; if disconnected, they don't.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, label, distance_transform_edt
from scipy.ndimage import maximum_filter

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp

sys.path.insert(0, '/media/data_cifs/projects/prj_video_imagenet/CSSM')
from visualize_qka_states import apply_stem, extract_qka_states


def find_dots(img_display):
    """Find the two marker dots in the image."""
    gray = img_display.mean(axis=-1)
    # Dots are brightest points
    local_max = maximum_filter(gray, size=7) == gray
    candidates = gray * local_max

    # Find top 2 brightest
    flat_idx = np.argsort(candidates.flatten())[::-1]
    dots = []
    for idx in flat_idx:
        y, x = np.unravel_index(idx, gray.shape)
        if gray[y, x] > 0.5:  # Must be bright
            # Check not too close to existing dots
            too_close = False
            for dy, dx in dots:
                if abs(y - dy) < 5 and abs(x - dx) < 5:
                    too_close = True
                    break
            if not too_close:
                dots.append((y, x))
        if len(dots) >= 2:
            break
    return dots


def main():
    print("Loading checkpoint...")
    checkpoint_path = 'checkpoints/KQV_64/epoch_20/checkpoint.pkl'
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']
    cssm_params = params['cssm_0']
    embed_dim = cssm_params['kernel'].shape[0]

    print("\nLoading Pathfinder images...")
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    TFRECORD_DIR = '/home/dlinsley/pathfinder_tfrecord/difficulty_14/val'

    def parse_example(example):
        features = tf.io.parse_single_example(example, {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
        image = tf.io.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, [224, 224, 3])
        return image, features['label']

    val_files = sorted(tf.io.gfile.glob(f'{TFRECORD_DIR}/*.tfrecord'))
    ds = tf.data.TFRecordDataset(val_files[:5]).map(parse_example)

    pos_img, neg_img = None, None
    for img, lbl in ds:
        if lbl.numpy() == 1 and pos_img is None:
            pos_img = img.numpy()
        elif lbl.numpy() == 0 and neg_img is None:
            neg_img = img.numpy()
        if pos_img is not None and neg_img is not None:
            break

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    def denormalize(img):
        return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)

    os.makedirs('visualizations', exist_ok=True)

    # Get display images
    pos_display = denormalize(pos_img)
    neg_display = denormalize(neg_img)
    pos_display_small = zoom(pos_display, (56/224, 56/224, 1), order=1)
    neg_display_small = zoom(neg_display, (56/224, 56/224, 1), order=1)

    # Find dots
    pos_dots = find_dots(pos_display_small)
    neg_dots = find_dots(neg_display_small)
    print(f"Positive dots: {pos_dots}")
    print(f"Negative dots: {neg_dots}")

    # Get states
    seq_len = 8
    x_pos = jnp.array(pos_img)[None, None, ...]
    x_pos = jnp.repeat(x_pos, seq_len, axis=1)
    x_neg = jnp.array(neg_img)[None, None, ...]
    x_neg = jnp.repeat(x_neg, seq_len, axis=1)

    x_pos_embed = apply_stem(params, x_pos, embed_dim=embed_dim)
    x_neg_embed = apply_stem(params, x_neg, embed_dim=embed_dim)

    Q_pos, K_pos, A_pos = extract_qka_states(cssm_params, x_pos_embed, channels=embed_dim)
    Q_neg, K_neg, A_neg = extract_qka_states(cssm_params, x_neg_embed, channels=embed_dim)

    A_pos = np.array(A_pos[0])
    A_neg = np.array(A_neg[0])

    # =================================================================
    # ANALYSIS 1: Track A value at each dot over time
    # =================================================================
    print("\n=== Tracking A at dot locations ===")

    # Get A at dot locations (mean over small window)
    def get_A_at_location(A, y, x, window=2):
        y_lo, y_hi = max(0, y-window), min(A.shape[1], y+window+1)
        x_lo, x_hi = max(0, x-window), min(A.shape[2], x+window+1)
        return A[:, y_lo:y_hi, x_lo:x_hi, :].mean(axis=(1, 2))  # (T, C)

    if len(pos_dots) >= 2:
        dot1_A = get_A_at_location(A_pos, pos_dots[0][0], pos_dots[0][1])
        dot2_A = get_A_at_location(A_pos, pos_dots[1][0], pos_dots[1][1])

        # Mean over channels
        dot1_mean = dot1_A.mean(axis=1)
        dot2_mean = dot2_A.mean(axis=1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot time course
        axes[0].plot(range(8), dot1_mean, 'r-o', label='Dot 1', linewidth=2)
        axes[0].plot(range(8), dot2_mean, 'b-o', label='Dot 2', linewidth=2)
        axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Timestep')
        axes[0].set_ylabel('Mean A (over channels)')
        axes[0].set_title('A State at Dot Locations Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Show dots on image
        axes[1].imshow(pos_display_small)
        axes[1].scatter([pos_dots[0][1]], [pos_dots[0][0]], c='red', s=100, marker='o', label='Dot 1')
        axes[1].scatter([pos_dots[1][1]], [pos_dots[1][0]], c='blue', s=100, marker='o', label='Dot 2')
        axes[1].set_title('Dot Locations (Connected)')
        axes[1].legend()
        axes[1].axis('off')

        # Correlation between dot activations over channels
        corr_over_time = []
        for t in range(8):
            corr = np.corrcoef(dot1_A[t], dot2_A[t])[0, 1]
            corr_over_time.append(corr)

        axes[2].plot(range(8), corr_over_time, 'g-o', linewidth=2)
        axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Timestep')
        axes[2].set_ylabel('Correlation')
        axes[2].set_title('Correlation of A vectors at Dot 1 vs Dot 2')
        axes[2].set_ylim(-1, 1)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig('visualizations/dot_A_timecourse.png', dpi=150, bbox_inches='tight')
        print("Saved: visualizations/dot_A_timecourse.png")
        plt.close()

    # =================================================================
    # ANALYSIS 2: Compare pos vs neg - correlation at dots
    # =================================================================
    print("\n=== Comparing pos vs neg: dot correlation ===")

    if len(pos_dots) >= 2 and len(neg_dots) >= 2:
        # For positive (connected): dots should have correlated/similar A
        # For negative (disconnected): dots should have uncorrelated A

        pos_dot1_A = get_A_at_location(A_pos, pos_dots[0][0], pos_dots[0][1])
        pos_dot2_A = get_A_at_location(A_pos, pos_dots[1][0], pos_dots[1][1])

        neg_dot1_A = get_A_at_location(A_neg, neg_dots[0][0], neg_dots[0][1])
        neg_dot2_A = get_A_at_location(A_neg, neg_dots[1][0], neg_dots[1][1])

        pos_corr = []
        neg_corr = []
        for t in range(8):
            pos_corr.append(np.corrcoef(pos_dot1_A[t], pos_dot2_A[t])[0, 1])
            neg_corr.append(np.corrcoef(neg_dot1_A[t], neg_dot2_A[t])[0, 1])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(8), pos_corr, 'g-o', label='Connected', linewidth=2)
        ax.plot(range(8), neg_corr, 'r-o', label='Disconnected', linewidth=2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Correlation of A at Dot 1 vs Dot 2')
        ax.set_title('Do Connected Dots Become More Correlated?')
        ax.set_ylim(-1, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig('visualizations/pos_neg_dot_correlation.png', dpi=150, bbox_inches='tight')
        print("Saved: visualizations/pos_neg_dot_correlation.png")
        plt.close()

    # =================================================================
    # ANALYSIS 3: Sign propagation along contour
    # =================================================================
    print("\n=== Sign propagation analysis ===")

    # Extract contour mask
    gray = pos_display_small.mean(axis=-1)
    contour_mask = gray > 0.3

    # Get signed A along contour only
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))

    for t in range(8):
        A_signed = A_pos[t].mean(axis=-1)  # (H, W)
        A_on_contour = np.where(contour_mask, A_signed, np.nan)

        vmax = np.nanmax(np.abs(A_on_contour))
        if np.isnan(vmax) or vmax == 0:
            vmax = 1

        # Top row: signed A on contour only
        axes[0, t].imshow(pos_display_small, alpha=0.3)
        im = axes[0, t].imshow(A_on_contour, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        if len(pos_dots) >= 2:
            axes[0, t].scatter([pos_dots[0][1]], [pos_dots[0][0]], c='black', s=50, marker='x')
            axes[0, t].scatter([pos_dots[1][1]], [pos_dots[1][0]], c='black', s=50, marker='+')
        axes[0, t].set_title(f't={t}')
        axes[0, t].axis('off')

        # Bottom row: just the sign (positive vs negative on contour)
        A_sign = np.sign(A_signed)
        A_sign_contour = np.where(contour_mask, A_sign, np.nan)
        axes[1, t].imshow(pos_display_small, alpha=0.3)
        axes[1, t].imshow(A_sign_contour, cmap='RdBu_r', vmin=-1, vmax=1, alpha=0.8)
        axes[1, t].axis('off')

    axes[0, 0].set_ylabel('Signed A\n(on contour)', fontsize=11)
    axes[1, 0].set_ylabel('Sign only\n(+1 or -1)', fontsize=11)

    plt.suptitle('Sign Propagation Along Contour (Connected Example)\nDot 1: X, Dot 2: +', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/sign_propagation.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/sign_propagation.png")
    plt.close()

    # =================================================================
    # ANALYSIS 4: Distance from each dot - does sign correlate with distance?
    # =================================================================
    print("\n=== Sign vs distance from dots ===")

    if len(pos_dots) >= 2:
        # Create distance maps from each dot
        dot1_mask = np.zeros_like(contour_mask)
        dot2_mask = np.zeros_like(contour_mask)
        dot1_mask[pos_dots[0][0], pos_dots[0][1]] = 1
        dot2_mask[pos_dots[1][0], pos_dots[1][1]] = 1

        dist_from_dot1 = distance_transform_edt(1 - dot1_mask)
        dist_from_dot2 = distance_transform_edt(1 - dot2_mask)

        # For each contour pixel, compute: distance to dot1 - distance to dot2
        # Negative = closer to dot1, Positive = closer to dot2
        relative_dist = dist_from_dot1 - dist_from_dot2
        relative_dist_contour = relative_dist[contour_mask]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        for t_idx, t in enumerate([0, 2, 5, 7]):
            A_signed = A_pos[t].mean(axis=-1)
            A_contour = A_signed[contour_mask]

            # Scatter: relative distance vs A value
            axes[0, t_idx].scatter(relative_dist_contour, A_contour, alpha=0.3, s=10)
            axes[0, t_idx].axhline(0, color='gray', linestyle='--')
            axes[0, t_idx].axvline(0, color='gray', linestyle='--')
            axes[0, t_idx].set_xlabel('Dist(dot1) - Dist(dot2)')
            axes[0, t_idx].set_ylabel('A value')
            axes[0, t_idx].set_title(f't={t}')

            # Correlation
            corr = np.corrcoef(relative_dist_contour, A_contour)[0, 1]
            axes[0, t_idx].text(0.05, 0.95, f'r={corr:.3f}', transform=axes[0, t_idx].transAxes,
                               fontsize=12, verticalalignment='top')

            # Show the relative distance map
            rel_dist_vis = np.where(contour_mask, relative_dist, np.nan)
            axes[1, t_idx].imshow(pos_display_small, alpha=0.3)
            axes[1, t_idx].imshow(rel_dist_vis, cmap='RdBu_r', alpha=0.7)
            axes[1, t_idx].scatter([pos_dots[0][1]], [pos_dots[0][0]], c='blue', s=100, marker='o')
            axes[1, t_idx].scatter([pos_dots[1][1]], [pos_dots[1][0]], c='red', s=100, marker='o')
            axes[1, t_idx].axis('off')

        axes[1, 0].set_ylabel('Relative distance\nBlue=near dot1\nRed=near dot2', fontsize=10)

        plt.suptitle('Does A encode distance from dots?\nIf grouping works by spreading from dots, A should correlate with relative distance', fontsize=12)
        plt.tight_layout()
        fig.savefig('visualizations/A_vs_distance.png', dpi=150, bbox_inches='tight')
        print("Saved: visualizations/A_vs_distance.png")
        plt.close()

    # =================================================================
    # ANALYSIS 5: Look at specific "grouping" channels
    # =================================================================
    print("\n=== Looking for grouping channels ===")

    # Find channels where the two dots have most different values (potential label channels)
    if len(pos_dots) >= 2:
        dot1_vals = A_pos[-1, pos_dots[0][0], pos_dots[0][1], :]  # (C,)
        dot2_vals = A_pos[-1, pos_dots[1][0], pos_dots[1][1], :]  # (C,)

        dot_diff = dot1_vals - dot2_vals
        top_diff_channels = np.argsort(np.abs(dot_diff))[-4:]

        print(f"Channels with largest dot1-dot2 difference:")
        for ch in top_diff_channels[::-1]:
            print(f"  Ch {ch}: dot1={dot1_vals[ch]:.3f}, dot2={dot2_vals[ch]:.3f}, diff={dot_diff[ch]:.3f}")

        fig, axes = plt.subplots(4, 8, figsize=(24, 12))

        for row, ch in enumerate(top_diff_channels[::-1]):
            for t in range(8):
                ch_data = A_pos[t, :, :, ch]
                vmax = np.abs(ch_data).max()

                axes[row, t].imshow(pos_display_small, alpha=0.3)
                axes[row, t].imshow(ch_data, cmap='RdBu_r', alpha=0.7, vmin=-vmax, vmax=vmax)
                axes[row, t].scatter([pos_dots[0][1]], [pos_dots[0][0]], c='black', s=30, marker='x')
                axes[row, t].scatter([pos_dots[1][1]], [pos_dots[1][0]], c='black', s=30, marker='+')

                if t == 0:
                    axes[row, t].set_ylabel(f'Ch {ch}\ndiff={dot_diff[ch]:.2f}', fontsize=10)
                if row == 0:
                    axes[row, t].set_title(f't={t}')
                axes[row, t].axis('off')

        plt.suptitle('Channels with largest Dot1-Dot2 difference (potential grouping channels)', fontsize=14)
        plt.tight_layout()
        fig.savefig('visualizations/grouping_channels.png', dpi=150, bbox_inches='tight')
        print("Saved: visualizations/grouping_channels.png")
        plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
