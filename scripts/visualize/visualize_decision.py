#!/usr/bin/env python3
"""
Analyze what the model actually uses to make the connected/disconnected decision.

Look at:
1. Gradients from the output w.r.t. the final CSSM state
2. What spatial locations matter most for the decision
3. How does the decision feature differ between pos and neg?
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp
import flax.linen as nn

sys.path.insert(0, '/media/data_cifs/projects/prj_video_imagenet/CSSM')
from src.models.simple_cssm import SimpleCSSM


def main():
    print("Loading checkpoint...")
    checkpoint_path = 'checkpoints/KQV_64/epoch_20/checkpoint.pkl'
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']

    print("\nLoading model...")
    model = SimpleCSSM(
        num_classes=2,
        embed_dim=128,
        depth=1,
        cssm_type='transformer',
        kernel_size=15,
        pos_embed='spatiotemporal',
        seq_len=8,
    )

    print("\nLoading images...")
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
    for img, label in ds:
        if label.numpy() == 1 and len(pos_imgs) < 4:
            pos_imgs.append(img.numpy())
        elif label.numpy() == 0 and len(neg_imgs) < 4:
            neg_imgs.append(img.numpy())
        if len(pos_imgs) >= 4 and len(neg_imgs) >= 4:
            break

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    def denormalize(img):
        return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)

    os.makedirs('visualizations', exist_ok=True)

    # =================================================================
    # ANALYSIS 1: Gradient of output w.r.t. input - what pixels matter?
    # =================================================================
    print("\n=== Gradient analysis: what input pixels matter for decision? ===")

    def compute_saliency(img, target_class):
        """Compute gradient of logit w.r.t. input image."""
        x = jnp.array(img)[None, ...]  # (1, H, W, C)

        def forward(x):
            logits = model.apply({'params': params}, x, training=False)
            return logits[0, target_class]

        grad = jax.grad(forward)(x)
        return np.array(grad[0])

    # Compute saliency for positive examples (target=1=connected)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i in range(4):
        pos_display = denormalize(pos_imgs[i])
        neg_display = denormalize(neg_imgs[i])

        # Saliency for "connected" class
        pos_saliency = compute_saliency(pos_imgs[i], target_class=1)
        neg_saliency = compute_saliency(neg_imgs[i], target_class=1)

        # Magnitude
        pos_sal_mag = np.abs(pos_saliency).sum(axis=-1)
        neg_sal_mag = np.abs(neg_saliency).sum(axis=-1)

        # Downsample display to 56x56 for comparison
        pos_display_small = zoom(pos_display, (56/224, 56/224, 1), order=1)
        neg_display_small = zoom(neg_display, (56/224, 56/224, 1), order=1)
        pos_sal_small = zoom(pos_sal_mag, (56/224, 56/224), order=1)
        neg_sal_small = zoom(neg_sal_mag, (56/224, 56/224), order=1)

        vmax = max(pos_sal_small.max(), neg_sal_small.max())

        axes[i, 0].imshow(pos_display_small)
        axes[i, 0].set_title('Connected' if i == 0 else '')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(pos_display_small, alpha=0.3)
        axes[i, 1].imshow(pos_sal_small, cmap='hot', alpha=0.7, vmin=0, vmax=vmax)
        axes[i, 1].set_title('Saliency (→Connected)' if i == 0 else '')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(neg_display_small)
        axes[i, 2].set_title('Disconnected' if i == 0 else '')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(neg_display_small, alpha=0.3)
        axes[i, 3].imshow(neg_sal_small, cmap='hot', alpha=0.7, vmin=0, vmax=vmax)
        axes[i, 3].set_title('Saliency (→Connected)' if i == 0 else '')
        axes[i, 3].axis('off')

    plt.suptitle('Input Saliency: Which pixels affect the "Connected" prediction?', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/input_saliency.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/input_saliency.png")
    plt.close()

    # =================================================================
    # ANALYSIS 2: Logit difference decomposition
    # =================================================================
    print("\n=== What's the difference in logits? ===")

    def get_logits(img):
        x = jnp.array(img)[None, ...]
        logits = model.apply({'params': params}, x, training=False)
        return np.array(logits[0])

    print("Logits for examples:")
    print("  Pos examples (should be [-, +] for connected):")
    for i, img in enumerate(pos_imgs[:4]):
        logits = get_logits(img)
        pred = "Connected" if logits[1] > logits[0] else "Disconnected"
        print(f"    #{i+1}: [{logits[0]:.3f}, {logits[1]:.3f}] -> {pred}")

    print("  Neg examples (should be [+, -] for disconnected):")
    for i, img in enumerate(neg_imgs[:4]):
        logits = get_logits(img)
        pred = "Connected" if logits[1] > logits[0] else "Disconnected"
        print(f"    #{i+1}: [{logits[0]:.3f}, {logits[1]:.3f}] -> {pred}")

    # =================================================================
    # ANALYSIS 3: Intermediate feature analysis - what does the model pool?
    # =================================================================
    print("\n=== Analyzing pooled features ===")

    # The SimpleCSSM does: CSSM -> take last timestep -> pool -> classify
    # Let's look at what the pooled feature looks like

    def get_pre_pool_features(img):
        """Get features before global pooling."""
        x = jnp.array(img)[None, ...]  # (1, H, W, 3)
        x = jnp.repeat(x[:, None, ...], 8, axis=1)  # (1, 8, H, W, 3)

        # Run through model manually to extract intermediate
        # This is hacky but let's see what we can get
        from src.models.simple_cssm import SimpleCSSM, CSSM_REGISTRY, HGRUBilinearCSSM
        from src.models.cssm import TransformerCSSM

        # Recompute features
        B, T, H, W, C = x.shape

        # Stem
        x_flat = x.reshape(B * T, H, W, C)

        # Conv1
        conv1_w = jnp.array(params['conv1']['kernel'])
        conv1_b = jnp.array(params['conv1']['bias'])
        x_flat = jax.lax.conv_general_dilated(
            x_flat.transpose(0, 3, 1, 2),
            conv1_w.transpose(3, 2, 0, 1),
            window_strides=(1, 1),
            padding='SAME'
        ).transpose(0, 2, 3, 1) + conv1_b
        x_flat = jax.nn.softplus(x_flat)

        ln1_scale = jnp.array(params['norm1']['scale'])
        ln1_bias = jnp.array(params['norm1']['bias'])
        x_flat = (x_flat - x_flat.mean(axis=-1, keepdims=True)) / (x_flat.std(axis=-1, keepdims=True) + 1e-6)
        x_flat = x_flat * ln1_scale + ln1_bias

        x_flat = x_flat.reshape(x_flat.shape[0], x_flat.shape[1]//2, 2, x_flat.shape[2]//2, 2, x_flat.shape[3])
        x_flat = x_flat.max(axis=(2, 4))

        # Conv2
        conv2_w = jnp.array(params['conv2']['kernel'])
        conv2_b = jnp.array(params['conv2']['bias'])
        x_flat = jax.lax.conv_general_dilated(
            x_flat.transpose(0, 3, 1, 2),
            conv2_w.transpose(3, 2, 0, 1),
            window_strides=(1, 1),
            padding='SAME'
        ).transpose(0, 2, 3, 1) + conv2_b
        x_flat = jax.nn.softplus(x_flat)

        ln2_scale = jnp.array(params['norm2']['scale'])
        ln2_bias = jnp.array(params['norm2']['bias'])
        x_flat = (x_flat - x_flat.mean(axis=-1, keepdims=True)) / (x_flat.std(axis=-1, keepdims=True) + 1e-6)
        x_flat = x_flat * ln2_scale + ln2_bias

        x_flat = x_flat.reshape(x_flat.shape[0], x_flat.shape[1]//2, 2, x_flat.shape[2]//2, 2, x_flat.shape[3])
        x_flat = x_flat.max(axis=(2, 4))

        return np.array(x_flat)  # (8, 56, 56, 128) for seq_len=8

    # Get stem features for one pos and one neg
    pos_stem = get_pre_pool_features(pos_imgs[0])
    neg_stem = get_pre_pool_features(neg_imgs[0])

    print(f"Stem output shape: {pos_stem.shape}")

    # Compare: are the stem outputs different?
    stem_diff = np.abs(pos_stem - neg_stem).mean()
    print(f"Mean absolute difference in stem features: {stem_diff:.4f}")

    # =================================================================
    # ANALYSIS 4: Per-pixel contribution to final decision
    # =================================================================
    print("\n=== Per-pixel contribution to decision ===")

    # Compute: for each spatial location, how much does it contribute to the final logit?
    # This requires looking at the classifier weights applied to the pooled features

    # The classifier takes the pooled (mean over H,W) features
    # classifier: (128*3,) -> (2,)
    classifier_w = np.array(params['head']['kernel'])  # (128*3, 2)
    classifier_b = np.array(params['head']['bias'])  # (2,)

    print(f"Classifier weight shape: {classifier_w.shape}")

    # For connected class (class 1), the weight is classifier_w[:, 1]
    connected_weights = classifier_w[:, 1]  # (384,)
    disconnected_weights = classifier_w[:, 0]  # (384,)

    # The pooled features are mean(CSSM_output, axis=(H,W))
    # Each pixel contributes (1/H/W) * (CSSM_pixel @ classifier_weights)
    # So the per-pixel contribution is CSSM_pixel @ classifier_weights

    # But we need the CSSM output, not just the stem
    # Let's use the full model with a modified forward to get pre-pool CSSM output

    # For now, let's just show that the classifier weights tell us what features matter
    print("\nClassifier weight analysis:")
    print(f"  Sum of |weights| for 'connected': {np.abs(connected_weights).sum():.2f}")
    print(f"  Sum of |weights| for 'disconnected': {np.abs(disconnected_weights).sum():.2f}")

    # Which feature dimensions matter most?
    weight_diff = connected_weights - disconnected_weights
    top_dims = np.argsort(np.abs(weight_diff))[-10:]
    print(f"\nTop 10 feature dimensions by weight difference:")
    for d in top_dims[::-1]:
        print(f"  Dim {d}: connected={connected_weights[d]:.3f}, disconnected={disconnected_weights[d]:.3f}")

    # =================================================================
    # ANALYSIS 5: Compare final-timestep features directly
    # =================================================================
    print("\n=== Comparing CSSM output features ===")

    # Get CSSM output at final timestep
    from visualize_qka_states import apply_stem, extract_qka_states

    cssm_params = params['cssm_0']
    embed_dim = 128

    x_pos = jnp.array(pos_imgs[0])[None, None, ...]
    x_pos = jnp.repeat(x_pos, 8, axis=1)
    x_neg = jnp.array(neg_imgs[0])[None, None, ...]
    x_neg = jnp.repeat(x_neg, 8, axis=1)

    x_pos_embed = apply_stem(params, x_pos, embed_dim=embed_dim)
    x_neg_embed = apply_stem(params, x_neg, embed_dim=embed_dim)

    Q_pos, K_pos, A_pos = extract_qka_states(cssm_params, x_pos_embed, channels=embed_dim)
    Q_neg, K_neg, A_neg = extract_qka_states(cssm_params, x_neg_embed, channels=embed_dim)

    # Stack Q, K, A for final timestep
    pos_final = np.concatenate([
        np.array(Q_pos[0, -1]),
        np.array(K_pos[0, -1]),
        np.array(A_pos[0, -1])
    ], axis=-1)  # (56, 56, 384)

    neg_final = np.concatenate([
        np.array(Q_neg[0, -1]),
        np.array(K_neg[0, -1]),
        np.array(A_neg[0, -1])
    ], axis=-1)  # (56, 56, 384)

    # Compute per-pixel contribution to connected logit
    pos_contrib = (pos_final @ connected_weights).squeeze()  # (56, 56)
    neg_contrib = (neg_final @ connected_weights).squeeze()  # (56, 56)

    pos_display_small = zoom(denormalize(pos_imgs[0]), (56/224, 56/224, 1), order=1)
    neg_display_small = zoom(denormalize(neg_imgs[0]), (56/224, 56/224, 1), order=1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    vmax = max(np.abs(pos_contrib).max(), np.abs(neg_contrib).max())

    # Positive example
    axes[0, 0].imshow(pos_display_small)
    axes[0, 0].set_title('Connected')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pos_display_small, alpha=0.3)
    im1 = axes[0, 1].imshow(pos_contrib, cmap='RdBu_r', alpha=0.7, vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title('Per-pixel contribution\nto "Connected" logit')
    axes[0, 1].axis('off')

    # Positive contribution to connected class
    pos_pos_contrib = np.maximum(pos_contrib, 0)
    axes[0, 2].imshow(pos_display_small, alpha=0.3)
    axes[0, 2].imshow(pos_pos_contrib, cmap='Reds', alpha=0.7)
    axes[0, 2].set_title('Positive contributions\n(evidence FOR connected)')
    axes[0, 2].axis('off')

    # Negative contribution to connected class
    pos_neg_contrib = -np.minimum(pos_contrib, 0)
    axes[0, 3].imshow(pos_display_small, alpha=0.3)
    axes[0, 3].imshow(pos_neg_contrib, cmap='Blues', alpha=0.7)
    axes[0, 3].set_title('Negative contributions\n(evidence AGAINST connected)')
    axes[0, 3].axis('off')

    # Negative example
    axes[1, 0].imshow(neg_display_small)
    axes[1, 0].set_title('Disconnected')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(neg_display_small, alpha=0.3)
    axes[1, 1].imshow(neg_contrib, cmap='RdBu_r', alpha=0.7, vmin=-vmax, vmax=vmax)
    axes[1, 1].axis('off')

    neg_pos_contrib = np.maximum(neg_contrib, 0)
    axes[1, 2].imshow(neg_display_small, alpha=0.3)
    axes[1, 2].imshow(neg_pos_contrib, cmap='Reds', alpha=0.7)
    axes[1, 2].axis('off')

    neg_neg_contrib = -np.minimum(neg_contrib, 0)
    axes[1, 3].imshow(neg_display_small, alpha=0.3)
    axes[1, 3].imshow(neg_neg_contrib, cmap='Blues', alpha=0.7)
    axes[1, 3].axis('off')

    plt.suptitle('Per-Pixel Contribution to "Connected" Decision\n(Red=evidence FOR, Blue=evidence AGAINST)', fontsize=14)
    plt.tight_layout()
    fig.savefig('visualizations/decision_contribution.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/decision_contribution.png")
    plt.close()

    # Print summary statistics
    print(f"\nPer-pixel contribution summary:")
    print(f"  Connected example: mean={pos_contrib.mean():.4f}, sum={pos_contrib.sum():.2f}")
    print(f"  Disconnected example: mean={neg_contrib.mean():.4f}, sum={neg_contrib.sum():.2f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
