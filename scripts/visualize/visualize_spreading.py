#!/usr/bin/env python3
"""
Visualize how attention (A state) spreads along the contour over time.

This script focuses on:
1. Consistent normalization across timesteps (to see growth)
2. A state evolution (the attention accumulator)
3. Difference between consecutive timesteps (what's being added)
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp

sys.path.insert(0, '/media/data_cifs/projects/prj_video_imagenet/CSSM')
from visualize_qka_states import apply_stem, extract_qka_states


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
    ds = tf.data.TFRecordDataset(val_files[:2]).map(parse_example)

    pos_img, neg_img = None, None
    for img, label in ds:
        if label.numpy() == 1 and pos_img is None:
            pos_img = img.numpy()
        elif label.numpy() == 0 and neg_img is None:
            neg_img = img.numpy()
        if pos_img is not None and neg_img is not None:
            break

    # Denormalize for display
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    def denormalize(img):
        return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)

    pos_display = denormalize(pos_img)
    neg_display = denormalize(neg_img)

    # Downsample display images to match state resolution (56x56)
    from scipy.ndimage import zoom
    pos_display_small = zoom(pos_display, (56/224, 56/224, 1), order=1)
    neg_display_small = zoom(neg_display, (56/224, 56/224, 1), order=1)

    print("\nExtracting Q, K, A states...")
    seq_len = 8
    x_pos = jnp.array(pos_img)[None, None, ...]
    x_pos = jnp.repeat(x_pos, seq_len, axis=1)
    x_neg = jnp.array(neg_img)[None, None, ...]
    x_neg = jnp.repeat(x_neg, seq_len, axis=1)

    x_pos_embed = apply_stem(params, x_pos, embed_dim=embed_dim)
    x_neg_embed = apply_stem(params, x_neg, embed_dim=embed_dim)

    Q_pos, K_pos, A_pos = extract_qka_states(cssm_params, x_pos_embed, channels=embed_dim)
    Q_neg, K_neg, A_neg = extract_qka_states(cssm_params, x_neg_embed, channels=embed_dim)

    # Convert to numpy, remove batch dim
    Q_pos, K_pos, A_pos = np.array(Q_pos[0]), np.array(K_pos[0]), np.array(A_pos[0])
    Q_neg, K_neg, A_neg = np.array(Q_neg[0]), np.array(K_neg[0]), np.array(A_neg[0])

    print(f"State shapes: {A_pos.shape}")

    os.makedirs('visualizations', exist_ok=True)

    # =================================================================
    # VISUALIZATION 1: A state evolution with GLOBAL normalization
    # =================================================================
    print("\nGenerating A state evolution (global normalization)...")

    A_pos_mag = np.abs(A_pos).mean(axis=-1)  # (T, H, W)
    A_neg_mag = np.abs(A_neg).mean(axis=-1)

    # Global min/max for consistent colorscale
    vmin = min(A_pos_mag.min(), A_neg_mag.min())
    vmax = max(A_pos_mag.max(), A_neg_mag.max())

    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    for t in range(8):
        axes[0, t].imshow(pos_display_small, alpha=0.4)
        im = axes[0, t].imshow(A_pos_mag[t], cmap='hot', alpha=0.7, vmin=vmin, vmax=vmax)
        axes[0, t].set_title(f't={t}')
        axes[0, t].axis('off')

        axes[1, t].imshow(neg_display_small, alpha=0.4)
        axes[1, t].imshow(A_neg_mag[t], cmap='hot', alpha=0.7, vmin=vmin, vmax=vmax)
        axes[1, t].axis('off')

    axes[0, 0].set_ylabel('CONNECTED', fontsize=12)
    axes[1, 0].set_ylabel('DISCONNECTED', fontsize=12)

    plt.suptitle('Attention (A) State Evolution - Global Normalization\n(Same colorscale across all frames)', fontsize=14)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='|A| magnitude')
    plt.tight_layout()
    fig.savefig('visualizations/A_state_global_norm.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/A_state_global_norm.png")
    plt.close()

    # =================================================================
    # VISUALIZATION 2: Temporal difference (what's added at each step)
    # =================================================================
    print("\nGenerating temporal difference (dA/dt)...")

    fig, axes = plt.subplots(2, 7, figsize=(21, 6))
    for t in range(1, 8):
        dA_pos = A_pos_mag[t] - A_pos_mag[t-1]
        dA_neg = A_neg_mag[t] - A_neg_mag[t-1]

        vmax_d = max(np.abs(dA_pos).max(), np.abs(dA_neg).max())

        axes[0, t-1].imshow(pos_display_small, alpha=0.3)
        im = axes[0, t-1].imshow(dA_pos, cmap='RdBu_r', alpha=0.7, vmin=-vmax_d, vmax=vmax_d)
        axes[0, t-1].set_title(f't={t-1}→{t}')
        axes[0, t-1].axis('off')

        axes[1, t-1].imshow(neg_display_small, alpha=0.3)
        axes[1, t-1].imshow(dA_neg, cmap='RdBu_r', alpha=0.7, vmin=-vmax_d, vmax=vmax_d)
        axes[1, t-1].axis('off')

    axes[0, 0].set_ylabel('CONNECTED', fontsize=12)
    axes[1, 0].set_ylabel('DISCONNECTED', fontsize=12)

    plt.suptitle('Attention Change Between Timesteps (A[t] - A[t-1])\nRed = increase, Blue = decrease', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/A_state_temporal_diff.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/A_state_temporal_diff.png")
    plt.close()

    # =================================================================
    # VISUALIZATION 3: Cumulative A growth from t=0
    # =================================================================
    print("\nGenerating cumulative growth from t=0...")

    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    for t in range(8):
        cumul_pos = A_pos_mag[t] - A_pos_mag[0]
        cumul_neg = A_neg_mag[t] - A_neg_mag[0]

        if t == 0:
            vmax_c = 1e-6  # Avoid zero
        else:
            vmax_c = max(np.abs(cumul_pos).max(), np.abs(cumul_neg).max(), 1e-6)

        axes[0, t].imshow(pos_display_small, alpha=0.3)
        axes[0, t].imshow(cumul_pos, cmap='hot', alpha=0.7, vmin=0, vmax=vmax_c if t > 0 else 1)
        axes[0, t].set_title(f't={t}')
        axes[0, t].axis('off')

        axes[1, t].imshow(neg_display_small, alpha=0.3)
        axes[1, t].imshow(cumul_neg, cmap='hot', alpha=0.7, vmin=0, vmax=vmax_c if t > 0 else 1)
        axes[1, t].axis('off')

    axes[0, 0].set_ylabel('CONNECTED', fontsize=12)
    axes[1, 0].set_ylabel('DISCONNECTED', fontsize=12)

    plt.suptitle('Cumulative Attention Growth: A[t] - A[0]\n(Shows what attention has accumulated since start)', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/A_state_cumulative.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/A_state_cumulative.png")
    plt.close()

    # =================================================================
    # VISUALIZATION 4: Q-K interaction (product) over time
    # =================================================================
    print("\nGenerating Q*K interaction...")

    Q_pos_mag = np.abs(Q_pos).mean(axis=-1)
    K_pos_mag = np.abs(K_pos).mean(axis=-1)
    Q_neg_mag = np.abs(Q_neg).mean(axis=-1)
    K_neg_mag = np.abs(K_neg).mean(axis=-1)

    QK_pos = Q_pos_mag * K_pos_mag
    QK_neg = Q_neg_mag * K_neg_mag

    vmax_qk = max(QK_pos.max(), QK_neg.max())

    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    for t in range(8):
        axes[0, t].imshow(pos_display_small, alpha=0.3)
        im = axes[0, t].imshow(QK_pos[t], cmap='plasma', alpha=0.7, vmin=0, vmax=vmax_qk)
        axes[0, t].set_title(f't={t}')
        axes[0, t].axis('off')

        axes[1, t].imshow(neg_display_small, alpha=0.3)
        axes[1, t].imshow(QK_neg[t], cmap='plasma', alpha=0.7, vmin=0, vmax=vmax_qk)
        axes[1, t].axis('off')

    axes[0, 0].set_ylabel('CONNECTED', fontsize=12)
    axes[1, 0].set_ylabel('DISCONNECTED', fontsize=12)

    plt.suptitle('Q*K Product (Attention Interaction) Over Time', fontsize=14)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='|Q|*|K|')
    plt.tight_layout()
    fig.savefig('visualizations/QK_product.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/QK_product.png")
    plt.close()

    # =================================================================
    # VISUALIZATION 5: Per-channel A state at select channels
    # =================================================================
    print("\nGenerating per-channel A visualization...")

    # Find channels with highest variance (most informative)
    A_pos_var = np.var(A_pos, axis=(0, 1, 2))  # Variance across T,H,W
    top_channels = np.argsort(A_pos_var)[-4:]  # Top 4 most varying channels

    fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    for row, ch in enumerate(top_channels):
        A_ch = A_pos[:, :, :, ch]
        vmin_ch, vmax_ch = A_ch.min(), A_ch.max()
        for t in range(8):
            axes[row, t].imshow(A_ch[t], cmap='viridis', vmin=vmin_ch, vmax=vmax_ch)
            if t == 0:
                axes[row, t].set_ylabel(f'Ch {ch}', fontsize=11)
            if row == 0:
                axes[row, t].set_title(f't={t}')
            axes[row, t].axis('off')

    plt.suptitle('Per-Channel A State (Positive Example)\nTop 4 most varying channels', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/A_per_channel.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/A_per_channel.png")
    plt.close()

    # =================================================================
    # VISUALIZATION 6: Thresholded A (above mean) to show where attention concentrates
    # =================================================================
    print("\nGenerating thresholded attention...")

    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    for t in range(8):
        # Threshold: show only above-mean attention
        thresh_pos = A_pos_mag[t] > A_pos_mag[t].mean() + 0.5 * A_pos_mag[t].std()
        thresh_neg = A_neg_mag[t] > A_neg_mag[t].mean() + 0.5 * A_neg_mag[t].std()

        axes[0, t].imshow(pos_display_small)
        axes[0, t].imshow(thresh_pos.astype(float), cmap='Reds', alpha=0.5 * thresh_pos.astype(float))
        axes[0, t].set_title(f't={t}')
        axes[0, t].axis('off')

        axes[1, t].imshow(neg_display_small)
        axes[1, t].imshow(thresh_neg.astype(float), cmap='Reds', alpha=0.5 * thresh_neg.astype(float))
        axes[1, t].axis('off')

    axes[0, 0].set_ylabel('CONNECTED', fontsize=12)
    axes[1, 0].set_ylabel('DISCONNECTED', fontsize=12)

    plt.suptitle('Thresholded Attention (> mean + 0.5*std)\nShows where attention concentrates', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/A_thresholded.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/A_thresholded.png")
    plt.close()

    print("\nDone! Check visualizations/ folder.")


if __name__ == '__main__':
    main()
