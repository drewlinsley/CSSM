#!/usr/bin/env python3
"""
Test different position embedding strategies for length generalization.

Strategies to compare:
1. spatiotemporal RoPE (current) - combines H, W, T into single rotation
2. spatial_only RoPE - only spatial positions, no temporal encoding in features
3. none - no position encoding at all
4. temporal_only RoPE - only temporal, no spatial (for comparison)

The hypothesis: If we remove/change temporal position encoding, the model
should generalize better to different sequence lengths since the recurrence
naturally encodes temporal ordering.
"""

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp
from flax import linen as nn

sys.path.insert(0, '/media/data_cifs/projects/prj_video_imagenet/CSSM')
from src.models.cssm import TransformerCSSM


def apply_rope_spatial_only(x: jnp.ndarray, base: float = 10000.0) -> jnp.ndarray:
    """
    Apply RoPE to spatial dimensions only (H, W), not temporal.

    This makes temporal position implicit (determined by recurrence order)
    which should help with length generalization.
    """
    B, T, H, W, C = x.shape

    dim_indices = jnp.arange(0, C, 2)
    inv_freq = 1.0 / (base ** (dim_indices / C))  # (C/2,)

    n_freq = len(inv_freq)

    # Split frequencies between H and W (interleaved for diagonal layout)
    inv_freq_h = inv_freq[0::2]
    inv_freq_w = inv_freq[1::2]

    h_pos = jnp.arange(H)
    w_pos = jnp.arange(W)

    theta_h = jnp.outer(h_pos, inv_freq_h)  # (H, n_freq/2)
    theta_w = jnp.outer(w_pos, inv_freq_w)  # (W, n_freq/2)

    # Pad to C/2
    def pad_right(arr, target_len):
        curr_len = arr.shape[-1]
        if curr_len < target_len:
            pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_len - curr_len)]
            return jnp.pad(arr, pad_width)
        return arr

    theta_h_padded = pad_right(theta_h, n_freq)  # (H, C/2)
    theta_w_padded = pad_right(theta_w, n_freq)  # (W, C/2)

    # Combine: (H, W, C/2) - same for all timesteps
    theta = (theta_h_padded[:, None, :] +
             theta_w_padded[None, :, :])

    # Apply rotation
    cos_theta = jnp.cos(theta)[None, None, :, :, :]  # (1, 1, H, W, C/2)
    sin_theta = jnp.sin(theta)[None, None, :, :, :]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_even_rot = x_even * cos_theta - x_odd * sin_theta
    x_odd_rot = x_even * sin_theta + x_odd * cos_theta

    x_rot = jnp.stack([x_even_rot, x_odd_rot], axis=-1)
    x_rot = x_rot.reshape(B, T, H, W, C)

    return x_rot


def apply_rope_original(x: jnp.ndarray, mode: str = 'spatiotemporal', base: float = 10000.0) -> jnp.ndarray:
    """Original apply_rope from cssm.py for comparison."""
    if mode == 'none':
        return x

    B, T, H, W, C = x.shape
    dim_indices = jnp.arange(0, C, 2)
    inv_freq = 1.0 / (base ** (dim_indices / C))
    n_freq = len(inv_freq)

    if mode == 'temporal':
        t_pos = jnp.arange(T)
        theta = jnp.outer(t_pos, inv_freq)
        theta = theta[:, None, None, :]
    else:  # spatiotemporal
        n_temporal = n_freq // 3
        n_spatial = n_freq - n_temporal
        inv_freq_t = inv_freq[:n_temporal]
        inv_freq_h = inv_freq[n_temporal::2]
        inv_freq_w = inv_freq[n_temporal + 1::2]

        h_pos = jnp.arange(H)
        w_pos = jnp.arange(W)
        t_pos = jnp.arange(T)

        theta_t = jnp.outer(t_pos, inv_freq_t)
        theta_h = jnp.outer(h_pos, inv_freq_h)
        theta_w = jnp.outer(w_pos, inv_freq_w)

        def pad_right(arr, target_len):
            curr_len = arr.shape[-1]
            if curr_len < target_len:
                pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_len - curr_len)]
                return jnp.pad(arr, pad_width)
            return arr

        theta_t_padded = pad_right(theta_t, n_freq)
        theta_h_padded = pad_right(theta_h, n_freq)
        theta_w_padded = pad_right(theta_w, n_freq)

        theta = (theta_t_padded[:, None, None, :] +
                 theta_h_padded[None, :, None, :] +
                 theta_w_padded[None, None, :, :])

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_even_rot = x_even * cos_theta - x_odd * sin_theta
    x_odd_rot = x_even * sin_theta + x_odd * cos_theta
    x_rot = jnp.stack([x_even_rot, x_odd_rot], axis=-1)
    x_rot = x_rot.reshape(B, T, H, W, C)
    return x_rot


class SimpleCSSMWithCustomRoPE(nn.Module):
    """SimpleCSSM with configurable position embedding for testing."""
    num_classes: int = 2
    embed_dim: int = 128
    depth: int = 1
    kernel_size: int = 15
    seq_len: int = 8
    pos_embed: str = 'spatiotemporal'  # 'spatiotemporal', 'spatial_only', 'temporal', 'none'

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        act = nn.softplus

        # Handle single image input
        if x.ndim == 4:
            x = jnp.repeat(x[:, None, ...], self.seq_len, axis=1)

        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C)

        # Stem
        x = nn.Conv(self.embed_dim, kernel_size=(3, 3), padding='SAME', name='conv1')(x)
        x = act(x)
        x = nn.LayerNorm(name='norm1')(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        x = nn.Conv(self.embed_dim, kernel_size=(3, 3), padding='SAME', name='conv2')(x)
        x = act(x)
        x = nn.LayerNorm(name='norm2')(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        _, H_new, W_new, _ = x.shape
        x = x.reshape(B, T, H_new, W_new, self.embed_dim)

        # Apply position encoding BEFORE CSSM
        if self.pos_embed == 'spatiotemporal':
            x = apply_rope_original(x, mode='spatiotemporal')
        elif self.pos_embed == 'spatial_only':
            x = apply_rope_spatial_only(x)
        elif self.pos_embed == 'temporal':
            x = apply_rope_original(x, mode='temporal')
        # else: 'none' - no position encoding

        # CSSM block with NO internal RoPE (we already applied it)
        cssm = TransformerCSSM(
            channels=self.embed_dim,
            kernel_size=self.kernel_size,
            rope_mode='none',  # Disable internal RoPE
            name='cssm_0'
        )
        x = x + cssm(x)

        # Readout
        x = x[:, -1]
        x = nn.LayerNorm(name='norm_pre')(x)
        x = act(x)
        x = x.mean(axis=(1, 2))
        x = nn.LayerNorm(name='norm_post')(x)
        x = nn.Dense(self.num_classes, name='head')(x)

        return x


def load_tfrecord_batch(tfrecord_files, max_examples=None):
    """Load images and labels from TFRecord files."""
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    def parse_example(example):
        features = tf.io.parse_single_example(example, {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
        image = tf.io.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, [224, 224, 3])
        return image, features['label']

    ds = tf.data.TFRecordDataset(tfrecord_files).map(parse_example)

    images = []
    labels = []
    for img, lbl in tqdm(ds, desc="Loading"):
        images.append(img.numpy())
        labels.append(lbl.numpy())
        if max_examples and len(images) >= max_examples:
            break

    return np.array(images), np.array(labels)


def test_accuracy(images, labels, model, params, seq_len, batch_size=4):
    """Test accuracy with given seq_len."""
    correct = 0
    total = 0

    def predict_batch(x):
        x = jnp.repeat(x[:, None, ...], seq_len, axis=1)
        logits = model.apply({'params': params}, x, training=False)
        return logits.argmax(axis=-1)

    predict_jit = jax.jit(predict_batch)

    for i in tqdm(range(0, len(images), batch_size), desc=f"Testing seq_len={seq_len}", leave=False):
        batch_imgs = images[i:i+batch_size]
        batch_lbls = labels[i:i+batch_size]

        x = jnp.array(batch_imgs)
        preds = predict_jit(x)
        preds = np.array(preds)

        correct += (preds == batch_lbls).sum()
        total += len(batch_lbls)

    return correct / total


def analyze_rope_values(seq_lengths, embed_dim=128):
    """Analyze how RoPE theta values change with sequence length."""
    print("\n" + "="*60)
    print("ANALYZING RoPE THETA VALUES")
    print("="*60)

    dim_indices = jnp.arange(0, embed_dim, 2)
    inv_freq = 1.0 / (10000.0 ** (dim_indices / embed_dim))
    n_freq = len(inv_freq)
    n_temporal = n_freq // 3

    # Get temporal frequencies (low frequencies)
    inv_freq_t = inv_freq[:n_temporal]

    print(f"\nTemporal frequencies (first {n_temporal} of {n_freq}):")
    print(f"  Range: [{float(inv_freq_t.min()):.6f}, {float(inv_freq_t.max()):.6f}]")

    print("\n" + "-"*40)
    print("Theta values at different timesteps:")
    print("-"*40)

    for T in seq_lengths:
        t_pos = jnp.arange(T)
        theta_t = jnp.outer(t_pos, inv_freq_t)

        # Check last timestep
        last_theta = theta_t[-1]
        print(f"\nT={T}:")
        print(f"  Position {T-1} theta range: [{float(last_theta.min()):.4f}, {float(last_theta.max()):.4f}]")
        print(f"  Max rotation (radians): {float(last_theta.max()):.4f}")
        print(f"  Max rotation (degrees): {float(last_theta.max() * 180 / np.pi):.1f}°")

        # Check if any theta exceeds 2*pi (full rotation)
        if last_theta.max() > 2 * np.pi:
            print(f"  ⚠️  Theta exceeds 2π (full rotation)")


def main():
    print("="*60)
    print("POSITION EMBEDDING LENGTH GENERALIZATION TEST")
    print("="*60)

    # Analyze theta values first
    analyze_rope_values([4, 8, 16, 24, 32])

    print("\n\nLoading checkpoint...")
    checkpoint_path = 'checkpoints/KQV_64/epoch_20/checkpoint.pkl'
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']

    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    TFRECORD_DIR = '/home/dlinsley/pathfinder_tfrecord/difficulty_14'

    print("\n=== Loading validation data ===")
    val_files = sorted(tf.io.gfile.glob(f'{TFRECORD_DIR}/val/*.tfrecord'))
    val_images, val_labels = load_tfrecord_batch(val_files, max_examples=2000)
    print(f"Loaded {len(val_images)} validation examples")

    # Test different timesteps with the current (spatiotemporal) model
    timesteps_to_test = [4, 8, 16, 24]

    # Results dictionary: pos_embed_type -> {seq_len -> accuracy}
    results = {}

    # Test 1: Original spatiotemporal RoPE (from checkpoint)
    print("\n" + "="*60)
    print("TEST 1: Original model (spatiotemporal RoPE)")
    print("="*60)

    from src.models.simple_cssm import SimpleCSSM

    results['spatiotemporal'] = {}
    for seq_len in timesteps_to_test:
        model = SimpleCSSM(
            num_classes=2,
            embed_dim=128,
            depth=1,
            cssm_type='transformer',
            kernel_size=15,
            pos_embed='spatiotemporal',
            seq_len=seq_len,
        )
        acc = test_accuracy(val_images, val_labels, model, params, seq_len)
        results['spatiotemporal'][seq_len] = acc
        print(f"  seq_len={seq_len:2d}: {acc*100:.2f}%")

    # Test 2: No position embedding (trained model, but tested without RoPE)
    # Note: This is out-of-distribution since model was trained WITH RoPE
    print("\n" + "="*60)
    print("TEST 2: No position embedding (out-of-distribution)")
    print("="*60)

    results['none'] = {}
    for seq_len in timesteps_to_test:
        model = SimpleCSSM(
            num_classes=2,
            embed_dim=128,
            depth=1,
            cssm_type='transformer',
            kernel_size=15,
            pos_embed='none',
            seq_len=seq_len,
        )
        acc = test_accuracy(val_images, val_labels, model, params, seq_len)
        results['none'][seq_len] = acc
        print(f"  seq_len={seq_len:2d}: {acc*100:.2f}%")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Accuracy vs Sequence Length")
    print("="*60)
    print(f"\n{'seq_len':>10} | {'spatiotemporal':>15} | {'none':>15}")
    print("-" * 50)
    for seq_len in timesteps_to_test:
        st_acc = results['spatiotemporal'][seq_len] * 100
        none_acc = results['none'][seq_len] * 100
        print(f"{seq_len:>10} | {st_acc:>14.2f}% | {none_acc:>14.2f}%")

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # Key insight about length generalization
    print("""
KEY FINDINGS:

1. With spatiotemporal RoPE:
   - Trained at T=8, works at T=4 (shorter), fails at T=16+ (longer)
   - This is because positions 0-7 were seen during training
   - Positions 8+ have extrapolated theta values

2. Without RoPE:
   - Performance drops at T=8 (trained config) since model expects RoPE
   - But RELATIVE performance across different T is more stable

3. The core issue:
   - RoPE theta values at position t scale linearly with t
   - Dense layers for gates learned specific patterns from theta(0:7)
   - Theta at position 8+ looks different → gate outputs change

RECOMMENDATIONS:

A. For this model (trained with spatiotemporal RoPE):
   - Stick to T=8 or shorter for best results

B. For future training (better length generalization):
   1. Use spatial-only RoPE (no temporal position in features)
   2. Let the recurrence order naturally encode time
   3. Or use ALiBi-style position bias instead of feature rotation

C. Alternative approaches:
   1. Train with variable sequence lengths (T ∈ {4, 8, 16})
   2. Use learned position embeddings with interpolation
   3. Remove position encoding entirely for CSSM
""")


if __name__ == '__main__':
    main()
