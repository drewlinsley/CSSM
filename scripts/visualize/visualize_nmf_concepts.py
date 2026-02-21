#!/usr/bin/env python3
"""
Use NMF (Non-negative Matrix Factorization) to decompose CSSM activations
into interpretable "concepts" or components.

The idea: instead of averaging 128 channels, find a small number (e.g., 8)
of components that capture different aspects of the visual processing:
- Edge detection components
- Dot/marker components
- Curve-following components
- Background components
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.ndimage import zoom

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
    ds = tf.data.TFRecordDataset(val_files[:3]).map(parse_example)

    pos_imgs, neg_imgs = [], []
    n_examples = 4
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
    # Collect activations from multiple images
    # =================================================================
    print("\nExtracting activations from all images...")

    all_A = []
    all_Q = []
    all_displays = []

    for imgs in [pos_imgs, neg_imgs]:
        for img in imgs:
            display = denormalize(img)
            display_small = zoom(display, (56/224, 56/224, 1), order=1)
            all_displays.append(display_small)

            seq_len = 8
            x = jnp.array(img)[None, None, ...]
            x = jnp.repeat(x, seq_len, axis=1)
            x_embed = apply_stem(params, x, embed_dim=embed_dim)
            Q, K, A = extract_qka_states(cssm_params, x_embed, channels=embed_dim)

            all_A.append(np.array(A[0]))  # (T, H, W, C)
            all_Q.append(np.array(Q[0]))

    # Stack: (n_images, T, H, W, C)
    all_A = np.array(all_A)
    all_Q = np.array(all_Q)
    n_images, T, H, W, C = all_A.shape

    print(f"Activation shape: {all_A.shape}")

    # =================================================================
    # NMF on A state at final timestep
    # =================================================================
    print("\nFitting NMF on A state (final timestep)...")

    n_components = 8

    # Reshape for NMF: (n_samples, n_features) where samples = images*H*W
    A_final = all_A[:, -1, :, :, :]  # (n_images, H, W, C)

    # Take absolute value to make non-negative
    A_final_abs = np.abs(A_final)

    # Reshape to (n_images * H * W, C)
    A_flat = A_final_abs.reshape(-1, C)

    # Fit NMF
    nmf = NMF(n_components=n_components, init='nndsvd', max_iter=500, random_state=42)
    W = nmf.fit_transform(A_flat)  # (n_samples, n_components)
    H = nmf.components_  # (n_components, C)

    print(f"NMF reconstruction error: {nmf.reconstruction_err_:.4f}")
    print(f"W shape: {W.shape}, H shape: {H.shape}")

    # Reshape W back to spatial: (n_images, H, W, n_components)
    W_spatial = W.reshape(n_images, 56, 56, n_components)

    # =================================================================
    # Visualize NMF components
    # =================================================================
    print("\nVisualizing NMF components...")

    # Plot components for each image
    fig, axes = plt.subplots(n_images, n_components + 1, figsize=(3*(n_components+1), 3*n_images))

    for i in range(n_images):
        # Original image
        axes[i, 0].imshow(all_displays[i])
        label = 'Pos' if i < len(pos_imgs) else 'Neg'
        axes[i, 0].set_title(f'{label} #{i%len(pos_imgs)+1}' if i==0 or i==len(pos_imgs) else '')
        axes[i, 0].axis('off')

        # NMF components
        for c in range(n_components):
            comp = W_spatial[i, :, :, c]
            axes[i, c+1].imshow(all_displays[i], alpha=0.3)
            axes[i, c+1].imshow(comp, cmap='hot', alpha=0.7)
            if i == 0:
                axes[i, c+1].set_title(f'Comp {c}')
            axes[i, c+1].axis('off')

    plt.suptitle(f'NMF Decomposition of A State ({n_components} components, final timestep)', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/nmf_components.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/nmf_components.png")
    plt.close()

    # =================================================================
    # Show temporal evolution of NMF components
    # =================================================================
    print("\nVisualizing NMF component evolution over time...")

    # Apply learned NMF to all timesteps for first positive example
    img_idx = 0
    A_example = np.abs(all_A[img_idx])  # (T, H, W, C)

    fig, axes = plt.subplots(n_components, T, figsize=(2.5*T, 2.5*n_components))

    for t in range(T):
        A_t = A_example[t].reshape(-1, C)  # (H*W, C)
        W_t = nmf.transform(A_t)  # (H*W, n_components)
        W_t_spatial = W_t.reshape(56, 56, n_components)

        for c in range(n_components):
            comp = W_t_spatial[:, :, c]
            axes[c, t].imshow(all_displays[0], alpha=0.3)
            axes[c, t].imshow(comp, cmap='hot', alpha=0.7)
            if t == 0:
                axes[c, t].set_ylabel(f'Comp {c}', fontsize=10)
            if c == 0:
                axes[c, t].set_title(f't={t}')
            axes[c, t].axis('off')

    plt.suptitle('NMF Component Evolution Over Time (Positive Example #1)', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/nmf_temporal_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/nmf_temporal_evolution.png")
    plt.close()

    # =================================================================
    # Component channel weights visualization
    # =================================================================
    print("\nVisualizing what channels contribute to each component...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for c in range(n_components):
        ax = axes[c // 4, c % 4]
        ax.bar(range(C), H[c], width=1.0)
        ax.set_title(f'Component {c} channel weights')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Weight')

    plt.suptitle('NMF: Which channels contribute to each component?', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/nmf_channel_weights.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/nmf_channel_weights.png")
    plt.close()

    # =================================================================
    # Compare components between positive and negative examples
    # =================================================================
    print("\nComparing component activations between pos/neg...")

    # Compute mean component activation for each image
    component_means = np.zeros((n_images, n_components))
    for i in range(n_images):
        A_i = np.abs(all_A[i, -1]).reshape(-1, C)
        W_i = nmf.transform(A_i)
        component_means[i] = W_i.mean(axis=0)

    pos_means = component_means[:len(pos_imgs)].mean(axis=0)
    neg_means = component_means[len(pos_imgs):].mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_components)
    width = 0.35

    ax.bar(x - width/2, pos_means, width, label='Connected', color='green', alpha=0.7)
    ax.bar(x + width/2, neg_means, width, label='Disconnected', color='red', alpha=0.7)

    ax.set_xlabel('NMF Component')
    ax.set_ylabel('Mean Activation')
    ax.set_title('Component Activation: Connected vs Disconnected')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in range(n_components)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('visualizations/nmf_pos_vs_neg.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/nmf_pos_vs_neg.png")
    plt.close()

    # =================================================================
    # Also try on Q state
    # =================================================================
    print("\nFitting NMF on Q state...")

    Q_final = all_Q[:, -1, :, :, :]
    Q_final_abs = np.abs(Q_final)
    Q_flat = Q_final_abs.reshape(-1, C)

    nmf_q = NMF(n_components=n_components, init='nndsvd', max_iter=500, random_state=42)
    W_q = nmf_q.fit_transform(Q_flat)
    H_q = nmf_q.components_

    print(f"Q NMF reconstruction error: {nmf_q.reconstruction_err_:.4f}")

    W_q_spatial = W_q.reshape(n_images, 56, 56, n_components)

    fig, axes = plt.subplots(n_images, n_components + 1, figsize=(3*(n_components+1), 3*n_images))

    for i in range(n_images):
        axes[i, 0].imshow(all_displays[i])
        axes[i, 0].axis('off')

        for c in range(n_components):
            comp = W_q_spatial[i, :, :, c]
            axes[i, c+1].imshow(all_displays[i], alpha=0.3)
            axes[i, c+1].imshow(comp, cmap='viridis', alpha=0.7)
            if i == 0:
                axes[i, c+1].set_title(f'Q Comp {c}')
            axes[i, c+1].axis('off')

    plt.suptitle(f'NMF Decomposition of Q State ({n_components} components)', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/nmf_q_components.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/nmf_q_components.png")
    plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
