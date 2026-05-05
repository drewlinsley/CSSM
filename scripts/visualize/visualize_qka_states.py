#!/usr/bin/env python3
"""
Visualize Q, K, A internal states of TransformerCSSM over timesteps.

This shows how attention "spreads along the contour" over time -
the key insight is that we need to look at the INTERNAL states,
not gradients w.r.t. input (which is the same at all timesteps).
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# JAX setup
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.numpy as jnp
import flax.linen as nn

# Add CSSM to path
sys.path.insert(0, '/media/data_cifs/projects/prj_video_imagenet/CSSM')

from src.models.cssm import to_goom, from_goom
from src.models.math import cssm_3x3_matrix_scan_op


def apply_stem(params, x, embed_dim=128):
    """
    Apply the SimpleCSSM stem: 2x (Conv -> softplus -> LayerNorm -> maxpool).

    Args:
        params: Full model parameters dict
        x: Input (B, T, H, W, 3)
        embed_dim: Embedding dimension

    Returns:
        Embedded tensor (B, T, H/4, W/4, embed_dim)
    """
    B, T, H, W, C = x.shape
    x = x.reshape(B * T, H, W, C)

    # Conv1 -> softplus -> LayerNorm -> maxpool
    conv1_w = jnp.array(params['conv1']['kernel'])
    conv1_b = jnp.array(params['conv1']['bias'])
    x = jax.lax.conv_general_dilated(
        x.transpose(0, 3, 1, 2),  # NHWC -> NCHW
        conv1_w.transpose(3, 2, 0, 1),  # HWIO -> OIHW
        window_strides=(1, 1),
        padding='SAME'
    ).transpose(0, 2, 3, 1) + conv1_b  # NCHW -> NHWC
    x = jax.nn.softplus(x)

    # LayerNorm
    ln1_scale = jnp.array(params['norm1']['scale'])
    ln1_bias = jnp.array(params['norm1']['bias'])
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
    x = x * ln1_scale + ln1_bias

    # Maxpool 2x2
    x = x.reshape(x.shape[0], x.shape[1]//2, 2, x.shape[2]//2, 2, x.shape[3])
    x = x.max(axis=(2, 4))

    # Conv2 -> softplus -> LayerNorm -> maxpool
    conv2_w = jnp.array(params['conv2']['kernel'])
    conv2_b = jnp.array(params['conv2']['bias'])
    x = jax.lax.conv_general_dilated(
        x.transpose(0, 3, 1, 2),
        conv2_w.transpose(3, 2, 0, 1),
        window_strides=(1, 1),
        padding='SAME'
    ).transpose(0, 2, 3, 1) + conv2_b
    x = jax.nn.softplus(x)

    ln2_scale = jnp.array(params['norm2']['scale'])
    ln2_bias = jnp.array(params['norm2']['bias'])
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
    x = x * ln2_scale + ln2_bias

    x = x.reshape(x.shape[0], x.shape[1]//2, 2, x.shape[2]//2, 2, x.shape[3])
    x = x.max(axis=(2, 4))

    # Reshape back to (B, T, H', W', embed_dim)
    _, H_new, W_new, _ = x.shape
    x = x.reshape(B, T, H_new, W_new, embed_dim)

    return x


def extract_qka_states(cssm_params, x, channels=128, kernel_size=15):
    """
    Extract Q, K, A states at each timestep.

    This is a simplified forward pass that returns intermediate states
    instead of the final output.

    Args:
        cssm_params: CSSM parameters (cssm_0 sub-dict)
        x: Input tensor (B, T, H, W, C) - ALREADY embedded by stem
        channels: Number of channels (embed_dim)
        kernel_size: Spatial kernel size

    Returns:
        Q, K, A: States at each timestep, shape (B, T, H, W, C)
    """
    B, T, H, W, C = x.shape
    W_freq = W // 2 + 1

    # Get spatial kernel
    k_spatial = jnp.array(cssm_params['kernel'])  # (C, k, k)

    # Pad kernel to image size
    pad_h = max(0, (H - kernel_size) // 2)
    pad_w = max(0, (W - kernel_size) // 2)
    pad_h_after = H - kernel_size - pad_h
    pad_w_after = W - kernel_size - pad_w

    if kernel_size > H or kernel_size > W:
        start_h = (kernel_size - H) // 2
        start_w = (kernel_size - W) // 2
        k_padded = k_spatial[:, start_h:start_h+H, start_w:start_w+W]
    else:
        k_padded = jnp.pad(k_spatial, ((0, 0), (pad_h, max(0, pad_h_after)),
                                        (pad_w, max(0, pad_w_after))), mode='constant')

    K_hat_raw = jnp.fft.rfft2(k_padded, axes=(1, 2))

    # Stable spectral magnitude
    rho = 0.999
    mag = jnp.abs(K_hat_raw)
    max_mag = jnp.maximum(mag, 1e-8)
    scale = jnp.minimum(1.0, rho / max_mag)
    K_hat = K_hat_raw * scale

    # Project input to Q, K, A
    x_flat = x.reshape(B * T, H, W, C)

    # Manual dense layer for input projection: (B*T, H, W, C) @ (C, 3C) = (B*T, H, W, 3C)
    input_kernel = jnp.array(cssm_params['input_proj']['kernel'])
    input_bias = jnp.array(cssm_params['input_proj']['bias'])
    qka_proj = jnp.einsum('bijk,kl->bijl', x_flat, input_kernel) + input_bias
    qka_proj = qka_proj.reshape(B, T, H, W, 3 * C)

    q_input = qka_proj[..., :C]
    k_input = qka_proj[..., C:2*C]
    a_input = qka_proj[..., 2*C:]

    # FFT inputs
    U_Q_hat = jnp.fft.rfft2(q_input.transpose(0, 1, 4, 2, 3), axes=(3, 4))
    U_K_hat = jnp.fft.rfft2(k_input.transpose(0, 1, 4, 2, 3), axes=(3, 4))
    U_A_hat = jnp.fft.rfft2(a_input.transpose(0, 1, 4, 2, 3), axes=(3, 4))

    # Compute gates from context
    ctx = x.mean(axis=(2, 3))  # (B, T, C)
    n_gate = H * W_freq

    def apply_gate(gate_params, ctx, bound_low=0.0, bound_range=1.0):
        """Apply a dense gate with optional bounding."""
        kernel = jnp.array(gate_params['kernel'])
        bias = jnp.array(gate_params['bias'])
        out = ctx @ kernel + bias
        out = jax.nn.sigmoid(out)
        if bound_low > 0 or bound_range < 1:
            out = bound_low + bound_range * out
        return out.reshape(B, T, 1, H, W_freq)

    # Decay gates (bounded 0.1-0.99)
    decay_Q = apply_gate(cssm_params['decay_Q'], ctx, 0.1, 0.89)
    decay_K = apply_gate(cssm_params['decay_K'], ctx, 0.1, 0.89)
    decay_A = apply_gate(cssm_params['decay_A'], ctx, 0.1, 0.89)

    # Coupling gates
    w_qk = apply_gate(cssm_params['w_qk'], ctx)
    alpha = apply_gate(cssm_params['alpha'], ctx)
    gamma = apply_gate(cssm_params['gamma'], ctx)

    # I/O gates
    B_Q = apply_gate(cssm_params['B_Q'], ctx)
    B_K = apply_gate(cssm_params['B_K'], ctx)
    B_A = apply_gate(cssm_params['B_A'], ctx)
    C_gate = apply_gate(cssm_params['C_gate'], ctx)

    # Build 3x3 transition matrix
    K_b = K_hat[None, None, ...].astype(jnp.complex64)
    ones = jnp.ones_like(K_b)
    zeros = jnp.zeros_like(decay_Q.astype(jnp.complex64) * ones)

    decay_Q_c = decay_Q.astype(jnp.complex64)
    decay_K_c = decay_K.astype(jnp.complex64)
    decay_A_c = decay_A.astype(jnp.complex64)

    # Row 0: Q update
    A_00 = decay_Q_c * ones
    A_01 = w_qk * K_b
    A_02 = alpha * K_b

    # Row 1: K update
    A_10 = w_qk * K_b
    A_11 = decay_K_c * ones
    A_12 = zeros

    # Row 2: A update
    A_20 = gamma * ones
    A_21 = gamma * ones
    A_22 = decay_A_c * ones

    row0 = jnp.stack([A_00, A_01, A_02], axis=-1)
    row1 = jnp.stack([A_10, A_11, A_12], axis=-1)
    row2 = jnp.stack([A_20, A_21, A_22], axis=-1)
    K_mat = jnp.stack([row0, row1, row2], axis=-2)

    # Apply input gates
    U_Q_gated = U_Q_hat * B_Q
    U_K_gated = U_K_hat * B_K
    U_A_gated = U_A_hat * B_A
    U_vec = jnp.stack([U_Q_gated, U_K_gated, U_A_gated], axis=-1)

    # GOOM scan
    K_log = to_goom(K_mat)
    U_log = to_goom(U_vec)

    _, State_log = jax.lax.associative_scan(
        cssm_3x3_matrix_scan_op, (K_log, U_log), axis=1
    )

    QKA_hat = from_goom(State_log)  # (B, T, C, H, W_freq, 3)
    QKA_hat_gated = QKA_hat * C_gate[..., None]

    # Extract individual states and convert to spatial domain
    Q_hat = QKA_hat_gated[..., 0]  # (B, T, C, H, W_freq)
    K_hat_state = QKA_hat_gated[..., 1]
    A_hat = QKA_hat_gated[..., 2]

    Q = jnp.fft.irfft2(Q_hat, s=(H, W), axes=(3, 4)).transpose(0, 1, 3, 4, 2)
    K_state = jnp.fft.irfft2(K_hat_state, s=(H, W), axes=(3, 4)).transpose(0, 1, 3, 4, 2)
    A = jnp.fft.irfft2(A_hat, s=(H, W), axes=(3, 4)).transpose(0, 1, 3, 4, 2)

    return Q, K_state, A


def visualize_states_over_time(Q, K, A, img_display, title_prefix=""):
    """
    Visualize Q, K, A states as they evolve over time.

    Args:
        Q, K, A: State tensors of shape (T, H, W, C)
        img_display: Original image for overlay (H, W, 3) in [0,1]
        title_prefix: Prefix for plot title
    """
    T = Q.shape[0]

    # Compute magnitude across channels for visualization
    Q_mag = np.abs(Q).mean(axis=-1)  # (T, H, W)
    K_mag = np.abs(K).mean(axis=-1)
    A_mag = np.abs(A).mean(axis=-1)

    # Normalize each timestep independently for better visibility
    def normalize_frames(frames):
        normed = np.zeros_like(frames)
        for t in range(frames.shape[0]):
            f = frames[t]
            fmin, fmax = f.min(), f.max()
            if fmax - fmin > 1e-8:
                normed[t] = (f - fmin) / (fmax - fmin)
            else:
                normed[t] = f - fmin
        return normed

    Q_norm = normalize_frames(Q_mag)
    K_norm = normalize_frames(K_mag)
    A_norm = normalize_frames(A_mag)

    fig, axes = plt.subplots(T, 4, figsize=(16, 2*T))

    for t in range(T):
        # Original image
        axes[t, 0].imshow(img_display)
        axes[t, 0].set_title(f't={t} Original' if t == 0 else f't={t}')
        axes[t, 0].axis('off')

        # Q state
        axes[t, 1].imshow(img_display, alpha=0.3)
        axes[t, 1].imshow(Q_norm[t], cmap='hot', alpha=0.7)
        axes[t, 1].set_title('Q (Query)' if t == 0 else '')
        axes[t, 1].axis('off')

        # K state
        axes[t, 2].imshow(img_display, alpha=0.3)
        axes[t, 2].imshow(K_norm[t], cmap='cool', alpha=0.7)
        axes[t, 2].set_title('K (Key)' if t == 0 else '')
        axes[t, 2].axis('off')

        # A state
        axes[t, 3].imshow(img_display, alpha=0.3)
        axes[t, 3].imshow(A_norm[t], cmap='viridis', alpha=0.7)
        axes[t, 3].set_title('A (Attention)' if t == 0 else '')
        axes[t, 3].axis('off')

    plt.suptitle(f'{title_prefix}TransformerCSSM Internal States Over Time', fontsize=14)
    plt.tight_layout()
    return fig


def main():
    print("Loading checkpoint...")
    checkpoint_path = 'checkpoints/KQV_64/epoch_20/checkpoint.pkl'
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']
    cssm_params = params['cssm_0']

    embed_dim = cssm_params['kernel'].shape[0]
    print(f"Loaded checkpoint (embed_dim={embed_dim})")

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

    # Get one positive and one negative example
    pos_img, neg_img = None, None
    for img, label in ds:
        if label.numpy() == 1 and pos_img is None:
            pos_img = img.numpy()
        elif label.numpy() == 0 and neg_img is None:
            neg_img = img.numpy()
        if pos_img is not None and neg_img is not None:
            break

    print(f"Image range: [{pos_img.min():.2f}, {pos_img.max():.2f}] (already ImageNet normalized)")

    # Denormalize for display
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    def denormalize(img):
        return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)

    pos_display = denormalize(pos_img)
    neg_display = denormalize(neg_img)

    print("\nApplying stem and extracting Q, K, A states...")

    # Prepare input: repeat image across timesteps
    seq_len = 8
    x_pos = jnp.array(pos_img)[None, None, ...]  # (1, 1, H, W, C)
    x_pos = jnp.repeat(x_pos, seq_len, axis=1)   # (1, 8, H, W, 3)

    x_neg = jnp.array(neg_img)[None, None, ...]
    x_neg = jnp.repeat(x_neg, seq_len, axis=1)

    # Apply stem to get embedded representation
    x_pos_embed = apply_stem(params, x_pos, embed_dim=embed_dim)
    x_neg_embed = apply_stem(params, x_neg, embed_dim=embed_dim)
    print(f"After stem: {x_pos_embed.shape} (was {x_pos.shape})")

    # Extract states from CSSM
    Q_pos, K_pos, A_pos = extract_qka_states(cssm_params, x_pos_embed, channels=embed_dim)
    Q_neg, K_neg, A_neg = extract_qka_states(cssm_params, x_neg_embed, channels=embed_dim)

    print(f"State shapes: Q={Q_pos.shape}, K={K_pos.shape}, A={A_pos.shape}")

    # Convert to numpy and remove batch dimension
    Q_pos, K_pos, A_pos = np.array(Q_pos[0]), np.array(K_pos[0]), np.array(A_pos[0])
    Q_neg, K_neg, A_neg = np.array(Q_neg[0]), np.array(K_neg[0]), np.array(A_neg[0])

    print("\nGenerating visualizations...")

    # Create output directory
    os.makedirs('visualizations', exist_ok=True)

    # Visualize positive example
    fig_pos = visualize_states_over_time(Q_pos, K_pos, A_pos, pos_display, "POSITIVE: ")
    fig_pos.savefig('visualizations/qka_states_positive.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/qka_states_positive.png")

    # Visualize negative example
    fig_neg = visualize_states_over_time(Q_neg, K_neg, A_neg, neg_display, "NEGATIVE: ")
    fig_neg.savefig('visualizations/qka_states_negative.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/qka_states_negative.png")

    # Also create a difference visualization: how do states differ between pos/neg?
    fig_diff, axes = plt.subplots(1, 4, figsize=(16, 4))

    # At final timestep, show difference
    t = 7
    Q_diff = np.abs(Q_pos[t]).mean(axis=-1) - np.abs(Q_neg[t]).mean(axis=-1)
    K_diff = np.abs(K_pos[t]).mean(axis=-1) - np.abs(K_neg[t]).mean(axis=-1)
    A_diff = np.abs(A_pos[t]).mean(axis=-1) - np.abs(A_neg[t]).mean(axis=-1)

    vmax = max(np.abs(Q_diff).max(), np.abs(K_diff).max(), np.abs(A_diff).max())

    axes[0].imshow(pos_display)
    axes[0].set_title('Positive Example')
    axes[0].axis('off')

    im1 = axes[1].imshow(Q_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title(f'Q diff (t={t})')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(K_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2].set_title(f'K diff (t={t})')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(A_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[3].set_title(f'A diff (t={t})')
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.suptitle('State Differences: Positive - Negative (final timestep)', fontsize=14)
    plt.tight_layout()
    fig_diff.savefig('visualizations/qka_states_difference.png', dpi=150, bbox_inches='tight')
    print("Saved: visualizations/qka_states_difference.png")

    plt.close('all')
    print("\nDone! Check the visualizations/ folder.")


if __name__ == '__main__':
    main()
