#!/usr/bin/env python3
"""
Visualize X, Y, Z states from HGRUBilinearCSSM separately.

Shows how excitatory (X), inhibitory (Y), and interaction (Z) channels
evolve over temporal recurrence steps.

Usage:
    python scripts/visualize_xyz_states.py \
        --checkpoint checkpoints/pf14_vit_hgru_bi_d1_e32/epoch_30 \
        --output_dir visualizations/pf14_xyz_states \
        --cssm hgru_bi --difficulty 14
"""

import argparse
import os
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cssm_vit import CSSMViT
from src.pathfinder_data import get_pathfinder_datasets, IMAGENET_MEAN, IMAGENET_STD
import orbax.checkpoint as ocp


def extract_xyz_from_cssm_params(params, x, args):
    """
    Extract X, Y, Z states by running the CSSM forward pass manually.

    This replicates HGRUBilinearCSSM forward but returns individual states.
    """
    from src.models.cssm import _stable_spectral_magnitude, apply_rope, apply_temporal_rope_to_context
    from src.models.goom import to_goom, from_goom
    from src.models.math import cssm_3x3_matrix_scan_op
    from src.models.cssm_vit import ConvStem

    B, T, H, W, C = x.shape

    # === STEM ===
    stem_params = params['stem']
    # Apply stem manually - use ConvStem forward
    x_flat = x.reshape(B * T, H, W, C)

    # First conv -> norm1 -> gelu
    conv1_w = stem_params['conv1']['kernel']
    conv1_b = stem_params['conv1']['bias']
    x_flat = jax.lax.conv_general_dilated(
        x_flat, conv1_w, window_strides=(2, 2), padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    ) + conv1_b
    # LayerNorm 1
    norm1_scale = stem_params['norm1']['scale']
    norm1_bias = stem_params['norm1']['bias']
    x_flat = (x_flat - x_flat.mean(axis=-1, keepdims=True)) / (x_flat.std(axis=-1, keepdims=True) + 1e-6)
    x_flat = x_flat * norm1_scale + norm1_bias
    x_flat = jax.nn.gelu(x_flat)

    # Second conv -> norm2
    conv2_w = stem_params['conv2']['kernel']
    conv2_b = stem_params['conv2']['bias']
    x_flat = jax.lax.conv_general_dilated(
        x_flat, conv2_w, window_strides=(2, 2), padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    ) + conv2_b
    # LayerNorm 2
    norm2_scale = stem_params['norm2']['scale']
    norm2_bias = stem_params['norm2']['bias']
    x_flat = (x_flat - x_flat.mean(axis=-1, keepdims=True)) / (x_flat.std(axis=-1, keepdims=True) + 1e-6)
    x_flat = x_flat * norm2_scale + norm2_bias

    _, H_p, W_p, C_dim = x_flat.shape
    x = x_flat.reshape(B, T, H_p, W_p, C_dim)

    # === POSITION EMBEDDING ===
    if 'pos_embed' in params:
        x = x + params['pos_embed']

    # === PRE-NORM (block0/norm1) ===
    norm1_params = params['block0']['norm1']
    x_normed = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
    x_normed = x_normed * norm1_params['scale'] + norm1_params['bias']

    # === Apply RoPE ===
    if args.rope_mode != 'none':
        x_normed = apply_rope(x_normed, mode=args.rope_mode, base=10000.0)

    # === CSSM FORWARD ===
    cssm_params = params['block0']['cssm']
    W_freq = W_p // 2 + 1

    # Spatial kernels
    k_exc = cssm_params['k_exc']
    k_inh = cssm_params['k_inh']

    kernel_size = k_exc.shape[1]
    pad_h = max(0, (H_p - kernel_size) // 2)
    pad_w = max(0, (W_p - kernel_size) // 2)
    pad_h_after = H_p - kernel_size - pad_h
    pad_w_after = W_p - kernel_size - pad_w

    def pad_kernel(k):
        if kernel_size > H_p or kernel_size > W_p:
            start_h = (kernel_size - H_p) // 2
            start_w = (kernel_size - W_p) // 2
            return k[:, start_h:start_h+H_p, start_w:start_w+W_p]
        return jnp.pad(k, ((0, 0), (pad_h, max(0, pad_h_after)),
                           (pad_w, max(0, pad_w_after))), mode='constant')

    K_E_raw = jnp.fft.rfft2(pad_kernel(k_exc), axes=(1, 2))
    K_I_raw = jnp.fft.rfft2(pad_kernel(k_inh), axes=(1, 2))
    K_E = _stable_spectral_magnitude(K_E_raw, rho=0.999)
    K_I = _stable_spectral_magnitude(K_I_raw, rho=0.999)

    # Input projection
    input_proj_w = cssm_params['input_proj']['kernel']
    input_proj_b = cssm_params['input_proj']['bias']
    x_flat = x_normed.reshape(B * T, H_p, W_p, C_dim)
    xyz_proj = x_flat @ input_proj_w + input_proj_b
    xyz_proj = xyz_proj.reshape(B, T, H_p, W_p, 3 * C_dim)

    x_input = xyz_proj[..., :C_dim]
    y_input = xyz_proj[..., C_dim:2*C_dim]
    z_input = xyz_proj[..., 2*C_dim:]

    # FFT
    x_perm = x_input.transpose(0, 1, 4, 2, 3)
    y_perm = y_input.transpose(0, 1, 4, 2, 3)
    z_perm = z_input.transpose(0, 1, 4, 2, 3)
    U_X_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))
    U_Y_hat = jnp.fft.rfft2(y_perm, axes=(3, 4))
    U_Z_hat = jnp.fft.rfft2(z_perm, axes=(3, 4))

    # Gates from context
    ctx = x_normed.mean(axis=(2, 3))
    ctx = apply_temporal_rope_to_context(ctx, base=10000.0)
    n_gate_feats = H_p * W_freq

    def apply_gate(gate_params, ctx, n_out):
        w, b = gate_params['kernel'], gate_params['bias']
        raw = ctx @ w + b
        return nn.sigmoid(raw).reshape(B, T, 1, H_p, W_freq)

    decay_x_freq = 0.1 + 0.89 * apply_gate(cssm_params['decay_x_gate'], ctx, n_gate_feats)
    decay_y_freq = 0.1 + 0.89 * apply_gate(cssm_params['decay_y_gate'], ctx, n_gate_feats)
    decay_z_freq = 0.1 + 0.89 * apply_gate(cssm_params['decay_z_gate'], ctx, n_gate_feats)

    mu_inhib = apply_gate(cssm_params['mu_inhib_gate'], ctx, n_gate_feats)
    alpha_inhib = apply_gate(cssm_params['alpha_inhib_gate'], ctx, n_gate_feats)
    mu_excit = apply_gate(cssm_params['mu_excit_gate'], ctx, n_gate_feats)
    alpha_excit = apply_gate(cssm_params['alpha_excit_gate'], ctx, n_gate_feats)
    gamma = apply_gate(cssm_params['gamma_gate'], ctx, n_gate_feats)
    delta = apply_gate(cssm_params['delta_gate'], ctx, n_gate_feats)

    B_gate = apply_gate(cssm_params['B_gate'], ctx, n_gate_feats)
    C_gate = apply_gate(cssm_params['C_gate'], ctx, n_gate_feats)
    D_gate = apply_gate(cssm_params['D_gate'], ctx, n_gate_feats)
    E_gate = apply_gate(cssm_params['E_gate'], ctx, n_gate_feats)

    # Build 3x3 matrix
    K_E_b = K_E[None, None, ...]
    K_I_b = K_I[None, None, ...]
    ones = jnp.ones_like(K_E_b)
    decay_x_c = decay_x_freq.astype(jnp.complex64)
    decay_y_c = decay_y_freq.astype(jnp.complex64)
    decay_z_c = decay_z_freq.astype(jnp.complex64)

    A_00 = decay_x_c * ones
    A_01 = -1.0 * mu_inhib * K_I_b
    A_02 = -1.0 * alpha_inhib * K_I_b
    A_10 = mu_excit * K_E_b
    A_11 = decay_y_c * ones
    A_12 = alpha_excit * K_E_b
    A_20 = gamma * ones
    A_21 = delta * ones
    A_22 = decay_z_c * ones

    row0 = jnp.stack([A_00, A_01, A_02], axis=-1)
    row1 = jnp.stack([A_10, A_11, A_12], axis=-1)
    row2 = jnp.stack([A_20, A_21, A_22], axis=-1)
    K_mat = jnp.stack([row0, row1, row2], axis=-2)

    U_X_gated = U_X_hat * B_gate
    U_Y_gated = U_Y_hat * D_gate
    U_Z_gated = U_Z_hat * E_gate
    U_vec = jnp.stack([U_X_gated, U_Y_gated, U_Z_gated], axis=-1)

    K_log = to_goom(K_mat)
    U_log = to_goom(U_vec)
    _, State_log = jax.lax.associative_scan(cssm_3x3_matrix_scan_op, (K_log, U_log), axis=1)

    # Extract XYZ
    XYZ_hat = from_goom(State_log)
    XYZ_hat_gated = XYZ_hat * C_gate[..., None]

    X_hat = XYZ_hat_gated[..., 0]
    Y_hat = XYZ_hat_gated[..., 1]
    Z_hat = XYZ_hat_gated[..., 2]

    X_spatial = jnp.fft.irfft2(X_hat, s=(H_p, W_p), axes=(3, 4)).transpose(0, 1, 3, 4, 2)
    Y_spatial = jnp.fft.irfft2(Y_hat, s=(H_p, W_p), axes=(3, 4)).transpose(0, 1, 3, 4, 2)
    Z_spatial = jnp.fft.irfft2(Z_hat, s=(H_p, W_p), axes=(3, 4)).transpose(0, 1, 3, 4, 2)

    # NOTE: NO GELU inside CSSM - the model does:
    # 1. XYZ states (raw, no nonlinearity)
    # 2. Concatenate [X, Y, Z]
    # 3. output_proj (Dense)
    # 4. GELU at readout (in CSSMViT, not CSSM)

    # Concatenate XYZ and apply output_proj (what CSSM outputs)
    xyz_concat = jnp.concatenate([X_spatial, Y_spatial, Z_spatial], axis=-1)  # (B, T, H', W', 3C)
    output_proj_w = cssm_params['output_proj']['kernel']
    output_proj_b = cssm_params['output_proj']['bias']
    cssm_output = xyz_concat @ output_proj_w + output_proj_b  # (B, T, H', W', C)

    # Apply GELU (what goes to readout_proj in CSSMViT)
    readout_input = jax.nn.gelu(cssm_output)

    # Compute magnitudes (mean across channels)
    # Raw XYZ states (before any nonlinearity)
    X_mag = jnp.abs(X_spatial).mean(axis=-1)
    Y_mag = jnp.abs(Y_spatial).mean(axis=-1)
    Z_mag = jnp.abs(Z_spatial).mean(axis=-1)
    # After output_proj + GELU (what readout sees)
    readout_mag = jnp.abs(readout_input).mean(axis=-1)

    return X_mag, Y_mag, Z_mag, readout_mag


def load_model_and_params(checkpoint_path, args):
    """Load model and params."""
    model = CSSMViT(
        num_classes=2,
        embed_dim=args.embed_dim,
        depth=args.depth,
        patch_size=args.patch_size,
        cssm_type=args.cssm,
        kernel_size=args.kernel_size,
        stem_mode=args.stem_mode,
        use_pos_embed=not args.no_pos_embed,
        rope_mode=args.rope_mode,
        block_size=args.block_size,
        gate_rank=args.gate_rank,
        output_act=args.output_act,
        use_dwconv=args.use_dwconv,
    )

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(os.path.abspath(checkpoint_path))

    if hasattr(restored, 'params'):
        params = restored.params
    else:
        params = restored['params']

    print(f"Loaded: {checkpoint_path}")
    return model, params


def visualize_sample(model, params, image, label, sample_idx, output_dir, seq_len, args):
    """Generate visualization showing X, Y, Z state evolution."""
    x = jnp.array(image)[jnp.newaxis]
    B, T, H, W, C = x.shape

    # Get prediction using standard model
    final_logits, perpixel_logits = model.apply(
        {'params': params}, x, training=False, return_spatial=True
    )
    pred = int(jnp.argmax(final_logits, axis=-1)[0])
    correct = pred == label

    # Extract XYZ states and readout features
    X_mag, Y_mag, Z_mag, readout_mag = extract_xyz_from_cssm_params(params, x, args)

    X_np = np.array(X_mag[0])  # (T, H', W')
    Y_np = np.array(Y_mag[0])
    Z_np = np.array(Z_mag[0])
    readout_np = np.array(readout_mag[0])
    perpixel_np = np.array(perpixel_logits[0])  # (T, H', W', 2)

    frames = []
    base_image = np.array(image[0])
    base_denorm = base_image * IMAGENET_STD + IMAGENET_MEAN
    base_denorm = np.clip(base_denorm, 0, 1)

    _, H_feat, W_feat = X_np.shape

    # Compute global min/max across all timesteps for consistent color ranges
    xyz_global_min = min(X_np.min(), Y_np.min(), Z_np.min())
    xyz_global_max = max(X_np.max(), Y_np.max(), Z_np.max())
    readout_global_min = readout_np.min()
    readout_global_max = readout_np.max()
    logits_diff_all = perpixel_np[:, :, :, 1] - perpixel_np[:, :, :, 0]
    logits_global_max = max(abs(logits_diff_all.min()), abs(logits_diff_all.max()), 1e-3)

    from scipy.ndimage import zoom
    scale_h = H / H_feat
    scale_w = W / W_feat

    for t in range(min(seq_len, X_np.shape[0])):
        X_t = X_np[t]
        Y_t = Y_np[t]
        Z_t = Z_np[t]
        readout_t = readout_np[t]
        logits_diff = perpixel_np[t, :, :, 1] - perpixel_np[t, :, :, 0]

        # Upsample
        X_up = zoom(X_t, (scale_h, scale_w), order=1)
        Y_up = zoom(Y_t, (scale_h, scale_w), order=1)
        Z_up = zoom(Z_t, (scale_h, scale_w), order=1)
        readout_up = zoom(readout_t, (scale_h, scale_w), order=1)
        diff_up = zoom(logits_diff, (scale_h, scale_w), order=1)

        # Create 2x4 figure: Input, X, Y, Z on top; Readout, X-Y, Logits, Overlay on bottom
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Row 1: Input, X (Excitatory), Y (Inhibitory), Z (Interaction)
        axes[0, 0].imshow(base_denorm)
        axes[0, 0].set_title(f'Input (GT: {"Conn" if label == 1 else "Disc"})')
        axes[0, 0].axis('off')

        im_x = axes[0, 1].imshow(X_up, cmap='Reds', vmin=xyz_global_min, vmax=xyz_global_max)
        axes[0, 1].set_title(f'X (Excitatory) T={t+1}/{seq_len}')
        axes[0, 1].axis('off')
        plt.colorbar(im_x, ax=axes[0, 1], fraction=0.046)

        im_y = axes[0, 2].imshow(Y_up, cmap='Blues', vmin=xyz_global_min, vmax=xyz_global_max)
        axes[0, 2].set_title(f'Y (Inhibitory)')
        axes[0, 2].axis('off')
        plt.colorbar(im_y, ax=axes[0, 2], fraction=0.046)

        im_z = axes[0, 3].imshow(Z_up, cmap='Purples', vmin=xyz_global_min, vmax=xyz_global_max)
        axes[0, 3].set_title(f'Z (Interaction)')
        axes[0, 3].axis('off')
        plt.colorbar(im_z, ax=axes[0, 3], fraction=0.046)

        # Row 2: Readout features (concat→proj→GELU), X-Y diff, Logit diff, Overlay
        im_readout = axes[1, 0].imshow(readout_up, cmap='viridis', vmin=readout_global_min, vmax=readout_global_max)
        axes[1, 0].set_title(f'GELU(output_proj([X,Y,Z]))')
        axes[1, 0].axis('off')
        plt.colorbar(im_readout, ax=axes[1, 0], fraction=0.046)

        # X - Y (net excitation) - use global XYZ range for consistency
        xy_diff = X_up - Y_up
        xy_max = xyz_global_max - xyz_global_min
        im_xy = axes[1, 1].imshow(xy_diff, cmap='RdBu_r', vmin=-xy_max, vmax=xy_max)
        axes[1, 1].set_title(f'X - Y (Net Excitation)')
        axes[1, 1].axis('off')
        plt.colorbar(im_xy, ax=axes[1, 1], fraction=0.046)

        # Logit difference - use global range
        im_diff = axes[1, 2].imshow(diff_up, cmap='RdBu_r', vmin=-logits_global_max, vmax=logits_global_max)
        axes[1, 2].set_title(f'Logit[Conn]-[Disc]')
        axes[1, 2].axis('off')
        plt.colorbar(im_diff, ax=axes[1, 2], fraction=0.046)

        # Overlay - readout on image
        readout_norm = (readout_up - readout_global_min) / (readout_global_max - readout_global_min + 1e-8)
        cmap = plt.colormaps.get_cmap('hot')
        overlay_colored = cmap(readout_norm)[:, :, :3]
        overlay = 0.4 * base_denorm + 0.6 * overlay_colored
        overlay = np.clip(overlay, 0, 1)
        axes[1, 3].imshow(overlay)
        axes[1, 3].set_title(f'Pred: {"Conn" if pred==1 else "Disc"} {"✓" if correct else "✗"}')
        axes[1, 3].axis('off')

        plt.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3].copy() / 255.0
        frames.append(frame)
        plt.close(fig)

    # Save GIF
    status = "correct" if correct else "wrong"
    label_str = "connected" if label == 1 else "disconnected"

    gif_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_xyz.gif"
    pil_frames = [Image.fromarray((f * 255).astype(np.uint8)) for f in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                       duration=args.frame_duration, loop=0)

    png_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_xyz_final.png"
    Image.fromarray((frames[-1] * 255).astype(np.uint8)).save(png_path)

    return correct, pred, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='visualizations/xyz_states')
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--cssm', type=str, default='hgru_bi')
    parser.add_argument('--kernel_size', type=int, default=15)
    parser.add_argument('--stem_mode', type=str, default='conv')
    parser.add_argument('--no_pos_embed', action='store_true', default=False)
    parser.add_argument('--rope_mode', type=str, default='spatiotemporal')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--block_size', type=int, default=1)
    parser.add_argument('--gate_rank', type=int, default=0)
    parser.add_argument('--output_act', type=str, default='none')
    parser.add_argument('--use_dwconv', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str,
                        default='/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025')
    parser.add_argument('--difficulty', type=str, default='14')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--frame_duration', type=int, default=500)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, params = load_model_and_params(args.checkpoint, args)

    print("Loading data...")
    _, _, test_dataset = get_pathfinder_datasets(
        root=args.data_dir,
        difficulty=args.difficulty,
        image_size=args.image_size,
        num_frames=args.seq_len,
    )

    np.random.seed(42)
    indices = np.random.choice(len(test_dataset), min(args.num_samples, len(test_dataset)), replace=False)

    print(f"Generating {len(indices)} visualizations...")
    correct_count = 0

    for i, idx in enumerate(tqdm(indices)):
        image, label = test_dataset[idx]
        correct, pred, label = visualize_sample(
            model, params, image, label, i, output_dir, args.seq_len, args
        )
        if correct:
            correct_count += 1

    print(f"\nAccuracy: {correct_count}/{len(indices)} ({100*correct_count/len(indices):.1f}%)")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
