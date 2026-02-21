#!/usr/bin/env python3
"""
Analyze how TransformerCSSM performs incremental grouping.

The only way to solve Pathfinder is incremental grouping -
detecting that two dots belong to the same connected component.

Key questions:
1. What do the NMF components actually encode? (edges, orientations, grouping?)
2. Are there signed activation patterns that encode "same group" vs "different group"?
3. Does the CONNECTED curve have different activation than DISCONNECTED curves?
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, PCA
from scipy.ndimage import zoom, label, binary_dilation

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp

sys.path.insert(0, '/media/data_cifs/projects/prj_video_imagenet/CSSM')
from visualize_qka_states import apply_stem, extract_qka_states


def extract_contour_mask(img_display, threshold=0.3):
    """Extract binary mask of contours."""
    gray = img_display.mean(axis=-1)
    return (gray > threshold).astype(float)


def find_connected_components(mask):
    """Find connected components in the contour mask."""
    labeled, n_components = label(mask > 0.5)
    return labeled, n_components


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

    pos_imgs, neg_imgs = [], []
    n_examples = 4
    for img, label in ds:
        if label.numpy() == 1 and len(pos_imgs) < n_examples:
            pos_imgs.append(img.numpy())
        elif label.numpy() == 0 and len(neg_imgs) < n_examples:
            neg_imgs.append(img.numpy())
        if len(pos_imgs) >= n_examples and len(neg_imgs) >= n_examples:
            break

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    def denormalize(img):
        return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)

    os.makedirs('visualizations', exist_ok=True)

    # =================================================================
    # ANALYSIS 1: Look at SIGNED activations (not magnitudes)
    # =================================================================
    print("\n=== ANALYSIS 1: Signed activations ===")

    pos_img = pos_imgs[0]
    neg_img = neg_imgs[0]

    pos_display = denormalize(pos_img)
    neg_display = denormalize(neg_img)
    pos_display_small = zoom(pos_display, (56/224, 56/224, 1), order=1)
    neg_display_small = zoom(neg_display, (56/224, 56/224, 1), order=1)

    # Get states (with sign preserved)
    seq_len = 8
    x_pos = jnp.array(pos_img)[None, None, ...]
    x_pos = jnp.repeat(x_pos, seq_len, axis=1)
    x_neg = jnp.array(neg_img)[None, None, ...]
    x_neg = jnp.repeat(x_neg, seq_len, axis=1)

    x_pos_embed = apply_stem(params, x_pos, embed_dim=embed_dim)
    x_neg_embed = apply_stem(params, x_neg, embed_dim=embed_dim)

    Q_pos, K_pos, A_pos = extract_qka_states(cssm_params, x_pos_embed, channels=embed_dim)
    Q_neg, K_neg, A_neg = extract_qka_states(cssm_params, x_neg_embed, channels=embed_dim)

    A_pos = np.array(A_pos[0])  # (T, H, W, C) - keep sign!
    A_neg = np.array(A_neg[0])
    Q_pos = np.array(Q_pos[0])
    K_pos = np.array(K_pos[0])

    # Look at SIGNED mean across channels
    A_pos_signed = A_pos.mean(axis=-1)  # (T, H, W)
    A_neg_signed = A_neg.mean(axis=-1)

    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    for t in range(8):
        vmax = max(np.abs(A_pos_signed[t]).max(), np.abs(A_neg_signed[t]).max())

        axes[0, t].imshow(pos_display_small, alpha=0.3)
        axes[0, t].imshow(A_pos_signed[t], cmap='RdBu_r', alpha=0.7, vmin=-vmax, vmax=vmax)
        axes[0, t].set_title(f't={t}')
        axes[0, t].axis('off')

        axes[1, t].imshow(neg_display_small, alpha=0.3)
        axes[1, t].imshow(A_neg_signed[t], cmap='RdBu_r', alpha=0.7, vmin=-vmax, vmax=vmax)
        axes[1, t].axis('off')

    axes[0, 0].set_ylabel('CONNECTED', fontsize=12)
    axes[1, 0].set_ylabel('DISCONNECTED', fontsize=12)
    plt.suptitle('SIGNED A State (Red=positive, Blue=negative)\nLooking for grouping signals', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/A_signed.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/A_signed.png")
    plt.close()

    # =================================================================
    # ANALYSIS 2: Per-channel analysis - find channels that differ on contour
    # =================================================================
    print("\n=== ANALYSIS 2: Channels that encode contour structure ===")

    pos_mask = extract_contour_mask(pos_display_small, threshold=0.3)

    # For each channel, compute: mean on contour vs mean off contour
    A_final = A_pos[-1]  # (H, W, C) at final timestep

    contour_mean = A_final[pos_mask > 0.5].mean(axis=0)  # (C,)
    background_mean = A_final[pos_mask <= 0.5].mean(axis=0)  # (C,)

    # Channels where contour differs most from background
    contour_diff = contour_mean - background_mean
    top_contour_channels = np.argsort(np.abs(contour_diff))[-8:]

    print(f"Top 8 channels by contour/background difference:")
    for ch in top_contour_channels[::-1]:
        print(f"  Channel {ch}: contour={contour_mean[ch]:.3f}, bg={background_mean[ch]:.3f}, diff={contour_diff[ch]:.3f}")

    # Visualize these channels over time
    fig, axes = plt.subplots(8, 9, figsize=(27, 24))

    for row, ch in enumerate(top_contour_channels[::-1]):
        # First column: show the channel's difference pattern
        diff_sign = "+" if contour_diff[ch] > 0 else "-"
        axes[row, 0].text(0.5, 0.5, f'Ch {ch}\n{diff_sign}{abs(contour_diff[ch]):.2f}',
                         ha='center', va='center', fontsize=10)
        axes[row, 0].axis('off')

        # Remaining columns: show this channel over time
        for t in range(8):
            ch_data = A_pos[t, :, :, ch]
            vmax = np.abs(ch_data).max()

            axes[row, t+1].imshow(pos_display_small, alpha=0.3)
            axes[row, t+1].imshow(ch_data, cmap='RdBu_r', alpha=0.7, vmin=-vmax, vmax=vmax)
            if row == 0:
                axes[row, t+1].set_title(f't={t}')
            axes[row, t+1].axis('off')

    plt.suptitle('Top 8 Channels by Contour/Background Difference (CONNECTED example)\nRed=positive, Blue=negative', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/contour_channels.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/contour_channels.png")
    plt.close()

    # =================================================================
    # ANALYSIS 3: Do connected regions share similar activation patterns?
    # =================================================================
    print("\n=== ANALYSIS 3: Within-component similarity ===")

    # For the positive example, find connected components
    labeled_pos, n_comp_pos = find_connected_components(pos_mask)
    print(f"Found {n_comp_pos} connected components in positive example")

    # For each component, compute mean activation vector
    component_vectors = []
    component_sizes = []
    for comp_id in range(1, n_comp_pos + 1):
        comp_mask = labeled_pos == comp_id
        comp_size = comp_mask.sum()
        if comp_size > 10:  # Skip tiny components
            comp_activations = A_final[comp_mask].mean(axis=0)  # (C,)
            component_vectors.append(comp_activations)
            component_sizes.append(comp_size)
            print(f"  Component {comp_id}: {comp_size} pixels")

    if len(component_vectors) >= 2:
        component_vectors = np.array(component_vectors)

        # Compute pairwise cosine similarity
        norms = np.linalg.norm(component_vectors, axis=1, keepdims=True)
        normalized = component_vectors / (norms + 1e-8)
        similarity_matrix = normalized @ normalized.T

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Show components
        axes[0].imshow(pos_display_small)
        axes[0].imshow(labeled_pos, cmap='tab20', alpha=0.5)
        axes[0].set_title(f'Connected Components ({n_comp_pos} found)')
        axes[0].axis('off')

        # Similarity matrix
        im = axes[1].imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_title('Component Similarity (cosine)')
        axes[1].set_xlabel('Component')
        axes[1].set_ylabel('Component')
        plt.colorbar(im, ax=axes[1])

        # PCA of component vectors
        if len(component_vectors) >= 3:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(component_vectors)
            axes[2].scatter(coords[:, 0], coords[:, 1], c=range(len(coords)), cmap='tab10', s=100)
            for i, (x, y) in enumerate(coords):
                axes[2].annotate(f'C{i}', (x, y), fontsize=10)
            axes[2].set_title('PCA of Component Activations')
            axes[2].set_xlabel('PC1')
            axes[2].set_ylabel('PC2')
        else:
            axes[2].text(0.5, 0.5, 'Need 3+ components\nfor PCA', ha='center', va='center')
            axes[2].axis('off')

        plt.suptitle('Do connected regions have similar activation patterns?', fontsize=14)
        plt.tight_layout()
        fig.savefig('visualizations/component_similarity.png', dpi=150, bbox_inches='tight')
        print("Saved: visualizations/component_similarity.png")
        plt.close()

    # =================================================================
    # ANALYSIS 4: Difference between CONNECTED curve and DISTRACTOR curves
    # =================================================================
    print("\n=== ANALYSIS 4: Connected curve vs distractors ===")

    # The two largest components should be: (1) connected curve with dots, (2) distractors
    # In connected case, both dots are on the SAME component
    # Let's identify which component has the dots

    # Find dot locations (brightest points in the image)
    gray = pos_display_small.mean(axis=-1)
    # The dots are the brightest local maxima
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(gray, size=5) == gray
    dot_candidates = gray * local_max
    dot_threshold = np.percentile(dot_candidates[dot_candidates > 0], 90)
    dot_mask = dot_candidates > dot_threshold

    # Which component(s) contain dots?
    dot_components = set()
    for comp_id in range(1, n_comp_pos + 1):
        comp_mask = labeled_pos == comp_id
        if (comp_mask & dot_mask).sum() > 0:
            dot_components.add(comp_id)

    print(f"Components containing dots: {dot_components}")

    if len(dot_components) == 1:
        print("Both dots on SAME component -> This is the CONNECTED curve")
        connected_comp = list(dot_components)[0]

        # Mask for connected curve vs distractors
        connected_mask = (labeled_pos == connected_comp).astype(float)
        distractor_mask = ((labeled_pos > 0) & (labeled_pos != connected_comp)).astype(float)

        # Compare activations
        fig, axes = plt.subplots(3, 8, figsize=(24, 9))

        for t in range(8):
            A_t = A_pos[t]

            # Row 0: Full activation
            A_mag = np.abs(A_t).mean(axis=-1)
            axes[0, t].imshow(pos_display_small, alpha=0.3)
            axes[0, t].imshow(A_mag, cmap='hot', alpha=0.7)
            if t == 0:
                axes[0, t].set_ylabel('Full |A|', fontsize=11)
            axes[0, t].set_title(f't={t}')
            axes[0, t].axis('off')

            # Row 1: Activation on CONNECTED curve only
            A_connected = A_mag * connected_mask
            axes[1, t].imshow(pos_display_small, alpha=0.3)
            axes[1, t].imshow(A_connected, cmap='Greens', alpha=0.8)
            if t == 0:
                axes[1, t].set_ylabel('Connected curve', fontsize=11)
            axes[1, t].axis('off')

            # Row 2: Activation on DISTRACTORS only
            A_distractor = A_mag * distractor_mask
            axes[2, t].imshow(pos_display_small, alpha=0.3)
            axes[2, t].imshow(A_distractor, cmap='Reds', alpha=0.8)
            if t == 0:
                axes[2, t].set_ylabel('Distractors', fontsize=11)
            axes[2, t].axis('off')

        plt.suptitle('Connected Curve vs Distractors: Does attention differ?', fontsize=14)
        plt.tight_layout()
        fig.savefig('visualizations/connected_vs_distractors.png', dpi=150, bbox_inches='tight')
        print("Saved: visualizations/connected_vs_distractors.png")
        plt.close()

        # Quantitative: mean activation on connected vs distractor over time
        connected_means = []
        distractor_means = []
        for t in range(8):
            A_mag = np.abs(A_pos[t]).mean(axis=-1)
            if connected_mask.sum() > 0:
                connected_means.append(A_mag[connected_mask > 0.5].mean())
            if distractor_mask.sum() > 0:
                distractor_means.append(A_mag[distractor_mask > 0.5].mean())

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(8), connected_means, 'g-o', linewidth=2, label='Connected curve')
        ax.plot(range(8), distractor_means, 'r-o', linewidth=2, label='Distractors')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Mean |A| activation')
        ax.set_title('Attention on Connected Curve vs Distractors Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig('visualizations/connected_vs_distractor_timecourse.png', dpi=150, bbox_inches='tight')
        print("Saved: visualizations/connected_vs_distractor_timecourse.png")
        plt.close()

    # =================================================================
    # ANALYSIS 5: Look at A state DIFFERENCE between pos and neg at same location
    # =================================================================
    print("\n=== ANALYSIS 5: Position-matched pos vs neg comparison ===")

    # Find a pos/neg pair where dots are in similar locations
    # Then compare A activations at those locations

    # For simplicity, just compare at the dot locations of the positive example
    dot_locs = np.where(dot_mask)
    if len(dot_locs[0]) >= 2:
        print(f"Dot locations: {list(zip(dot_locs[0][:2], dot_locs[1][:2]))}")

        fig, axes = plt.subplots(2, 8, figsize=(24, 6))

        for t in range(8):
            # A state at dots for positive
            A_pos_at_dots = A_pos[t][dot_mask].mean(axis=0)  # (C,)
            A_neg_at_dots = A_neg[t][dot_mask].mean(axis=0)  # (C,) - same locations in neg image

            # This doesn't make sense - neg image has dots in different places
            # Instead, let's look at the A state difference map
            A_diff = A_pos[t].mean(axis=-1) - A_neg[t].mean(axis=-1)
            vmax = np.abs(A_diff).max()

            axes[0, t].imshow(pos_display_small, alpha=0.3)
            axes[0, t].imshow(A_diff, cmap='RdBu_r', alpha=0.7, vmin=-vmax, vmax=vmax)
            axes[0, t].set_title(f't={t}')
            axes[0, t].axis('off')

            # Also show where the curves differ
            curve_diff = pos_mask - extract_contour_mask(neg_display_small, threshold=0.3)
            axes[1, t].imshow(curve_diff, cmap='RdBu_r')
            axes[1, t].axis('off')

        axes[0, 0].set_ylabel('A(pos) - A(neg)', fontsize=11)
        axes[1, 0].set_ylabel('Curve diff', fontsize=11)

        plt.suptitle('A State Difference: Connected - Disconnected', fontsize=14)
        plt.tight_layout()
        fig.savefig('visualizations/pos_neg_A_difference.png', dpi=150, bbox_inches='tight')
        print("Saved: visualizations/pos_neg_A_difference.png")
        plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
