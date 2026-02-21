#!/usr/bin/env python3
"""
Visualize how attention correlates with contour structure over time.

Key insight: Instead of looking at raw magnitudes, we should look at
how attention concentrates specifically on contour pixels vs background.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp

sys.path.insert(0, '/media/data_cifs/projects/prj_video_imagenet/CSSM')
from visualize_qka_states import apply_stem, extract_qka_states


def extract_contour_mask(img_display, threshold=0.3):
    """
    Extract a binary mask of contour pixels from the image.
    Pathfinder images have white curves on black background.
    """
    gray = img_display.mean(axis=-1)
    mask = gray > threshold
    return mask.astype(float)


def compute_contour_attention_ratio(A_mag, contour_mask):
    """
    Compute ratio of attention on contour vs background.
    Higher ratio means attention is more concentrated on contours.
    """
    contour_attention = A_mag[contour_mask > 0.5].mean() if (contour_mask > 0.5).sum() > 0 else 0
    background_attention = A_mag[contour_mask <= 0.5].mean() if (contour_mask <= 0.5).sum() > 0 else 1
    return contour_attention / (background_attention + 1e-8)


def main():
    print("Loading checkpoint...")
    checkpoint_path = 'checkpoints/KQV_64/epoch_20/checkpoint.pkl'
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']
    cssm_params = params['cssm_0']
    embed_dim = cssm_params['kernel'].shape[0]

    print("\nLoading multiple Pathfinder images...")
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

    # Collect multiple examples
    pos_imgs, neg_imgs = [], []
    n_examples = 8
    for img, label in ds:
        if label.numpy() == 1 and len(pos_imgs) < n_examples:
            pos_imgs.append(img.numpy())
        elif label.numpy() == 0 and len(neg_imgs) < n_examples:
            neg_imgs.append(img.numpy())
        if len(pos_imgs) >= n_examples and len(neg_imgs) >= n_examples:
            break

    print(f"Loaded {len(pos_imgs)} positive and {len(neg_imgs)} negative examples")

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    def denormalize(img):
        return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)

    os.makedirs('visualizations', exist_ok=True)

    # =================================================================
    # ANALYSIS 1: Contour-to-background attention ratio over time
    # =================================================================
    print("\nComputing contour attention ratios over time...")

    pos_ratios = []
    neg_ratios = []

    for i, (pos_img, neg_img) in enumerate(zip(pos_imgs, neg_imgs)):
        # Denormalize for contour extraction
        pos_display = denormalize(pos_img)
        neg_display = denormalize(neg_img)

        # Downsample display and extract contour mask
        pos_display_small = zoom(pos_display, (56/224, 56/224, 1), order=1)
        neg_display_small = zoom(neg_display, (56/224, 56/224, 1), order=1)

        pos_mask = extract_contour_mask(pos_display_small, threshold=0.3)
        neg_mask = extract_contour_mask(neg_display_small, threshold=0.3)

        # Extract states
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

        # Compute ratios at each timestep
        pos_ratio_t = []
        neg_ratio_t = []
        for t in range(8):
            A_pos_mag = np.abs(A_pos[t]).mean(axis=-1)
            A_neg_mag = np.abs(A_neg[t]).mean(axis=-1)

            pos_ratio_t.append(compute_contour_attention_ratio(A_pos_mag, pos_mask))
            neg_ratio_t.append(compute_contour_attention_ratio(A_neg_mag, neg_mask))

        pos_ratios.append(pos_ratio_t)
        neg_ratios.append(neg_ratio_t)

        print(f"  Example {i+1}: pos ratio t=7: {pos_ratio_t[-1]:.2f}, neg ratio t=7: {neg_ratio_t[-1]:.2f}")

    pos_ratios = np.array(pos_ratios)
    neg_ratios = np.array(neg_ratios)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual traces with low alpha
    for i in range(len(pos_ratios)):
        ax.plot(range(8), pos_ratios[i], 'g-', alpha=0.2)
        ax.plot(range(8), neg_ratios[i], 'r-', alpha=0.2)

    # Plot means
    ax.plot(range(8), pos_ratios.mean(axis=0), 'g-', linewidth=3, label='Connected (mean)')
    ax.plot(range(8), neg_ratios.mean(axis=0), 'r-', linewidth=3, label='Disconnected (mean)')

    # Fill between for std
    ax.fill_between(range(8),
                    pos_ratios.mean(axis=0) - pos_ratios.std(axis=0),
                    pos_ratios.mean(axis=0) + pos_ratios.std(axis=0),
                    color='green', alpha=0.2)
    ax.fill_between(range(8),
                    neg_ratios.mean(axis=0) - neg_ratios.std(axis=0),
                    neg_ratios.mean(axis=0) + neg_ratios.std(axis=0),
                    color='red', alpha=0.2)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Uniform attention')

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Contour/Background Attention Ratio', fontsize=12)
    ax.set_title('Does Attention Concentrate on Contours Over Time?', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('visualizations/contour_attention_ratio.png', dpi=150, bbox_inches='tight')
    print("\nSaved: visualizations/contour_attention_ratio.png")
    plt.close()

    # =================================================================
    # ANALYSIS 2: Visualize attention on first example with contour overlay
    # =================================================================
    print("\nGenerating detailed attention visualization for first example...")

    pos_img = pos_imgs[0]
    neg_img = neg_imgs[0]

    pos_display = denormalize(pos_img)
    neg_display = denormalize(neg_img)
    pos_display_small = zoom(pos_display, (56/224, 56/224, 1), order=1)
    neg_display_small = zoom(neg_display, (56/224, 56/224, 1), order=1)

    pos_mask = extract_contour_mask(pos_display_small, threshold=0.3)
    neg_mask = extract_contour_mask(neg_display_small, threshold=0.3)

    # Get states
    seq_len = 8
    x_pos = jnp.array(pos_img)[None, None, ...]
    x_pos = jnp.repeat(x_pos, seq_len, axis=1)

    x_pos_embed = apply_stem(params, x_pos, embed_dim=embed_dim)
    Q_pos, K_pos, A_pos = extract_qka_states(cssm_params, x_pos_embed, channels=embed_dim)
    A_pos = np.array(A_pos[0])

    # Create visualization showing A masked to contours only
    fig, axes = plt.subplots(3, 8, figsize=(24, 9))

    for t in range(8):
        A_mag = np.abs(A_pos[t]).mean(axis=-1)

        # Row 1: Full attention
        axes[0, t].imshow(pos_display_small, alpha=0.3)
        im = axes[0, t].imshow(A_mag, cmap='hot', alpha=0.7)
        axes[0, t].set_title(f't={t}')
        axes[0, t].axis('off')

        # Row 2: Attention ON contours only
        A_on_contour = A_mag * pos_mask
        axes[1, t].imshow(pos_display_small, alpha=0.3)
        axes[1, t].imshow(A_on_contour, cmap='hot', alpha=0.8)
        axes[1, t].axis('off')

        # Row 3: Attention OFF contours (background)
        A_off_contour = A_mag * (1 - pos_mask)
        axes[2, t].imshow(pos_display_small, alpha=0.3)
        axes[2, t].imshow(A_off_contour, cmap='hot', alpha=0.8)
        axes[2, t].axis('off')

    axes[0, 0].set_ylabel('Full A', fontsize=12)
    axes[1, 0].set_ylabel('A on contours', fontsize=12)
    axes[2, 0].set_ylabel('A off contours', fontsize=12)

    plt.suptitle('Attention Decomposition: Contour vs Background (Connected Example)', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/attention_contour_decomposition.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/attention_contour_decomposition.png")
    plt.close()

    # =================================================================
    # ANALYSIS 3: Q-K similarity map
    # =================================================================
    print("\nGenerating Q-K similarity analysis...")

    x_pos = jnp.array(pos_img)[None, None, ...]
    x_pos = jnp.repeat(x_pos, seq_len, axis=1)
    x_pos_embed = apply_stem(params, x_pos, embed_dim=embed_dim)
    Q_pos, K_pos, A_pos = extract_qka_states(cssm_params, x_pos_embed, channels=embed_dim)

    Q_pos = np.array(Q_pos[0])
    K_pos = np.array(K_pos[0])

    # Compute Q-K cosine similarity at each position
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))

    for t in range(8):
        Q_t = Q_pos[t]  # (56, 56, 128)
        K_t = K_pos[t]

        # Normalize
        Q_norm = Q_t / (np.linalg.norm(Q_t, axis=-1, keepdims=True) + 1e-8)
        K_norm = K_t / (np.linalg.norm(K_t, axis=-1, keepdims=True) + 1e-8)

        # Cosine similarity at each position
        similarity = (Q_norm * K_norm).sum(axis=-1)

        # Also compute Q-K dot product magnitude
        qk_dot = (Q_t * K_t).sum(axis=-1)

        axes[0, t].imshow(pos_display_small, alpha=0.3)
        im1 = axes[0, t].imshow(similarity, cmap='RdYlBu', alpha=0.7, vmin=-1, vmax=1)
        axes[0, t].set_title(f't={t}')
        axes[0, t].axis('off')

        vmax_qk = np.abs(qk_dot).max()
        axes[1, t].imshow(pos_display_small, alpha=0.3)
        axes[1, t].imshow(qk_dot, cmap='RdBu_r', alpha=0.7, vmin=-vmax_qk, vmax=vmax_qk)
        axes[1, t].axis('off')

    axes[0, 0].set_ylabel('Q-K cosine sim', fontsize=11)
    axes[1, 0].set_ylabel('Q·K dot product', fontsize=11)

    plt.suptitle('Q-K Similarity Over Time (Connected Example)', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/QK_similarity.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/QK_similarity.png")
    plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
