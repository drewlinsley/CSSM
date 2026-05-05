#!/usr/bin/env python3
"""
Gradient-based visualization for CSSM models.

Creates two types of gradient visualizations:
1. d(logit_t)/d(Image_t) - gradient of logit at each timestep w.r.t. input at that timestep
2. d(logit_final)/d(Image_t) - gradient of final logit w.r.t. input at each timestep

Usage:
    python scripts/visualize_gradients.py \
        --checkpoint checkpoints/hgru_CSSM_t8_y/epoch_35 \
        --output_dir visualizations/gradients \
        --cssm hgru_bi --difficulty 14
"""

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
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


def compute_per_timestep_gradients(model, params, x, label):
    """
    Compute d(logit_t)/d(Image_t) for each timestep.

    Returns gradients of shape (T, H, W, 3)
    """
    def forward_spatial(x):
        # Get per-pixel logits at all timesteps
        _, perpixel_logits = model.apply(
            {'params': params}, x[jnp.newaxis], training=False, return_spatial=True
        )
        # perpixel_logits: (1, T, H', W', 2)
        # Global pool and return logit for correct class at each timestep
        logits_per_t = perpixel_logits[0].mean(axis=(1, 2))  # (T, 2)
        return logits_per_t[:, label]  # (T,) - logit for correct class at each t

    # Compute gradient w.r.t. input
    grad_fn = jax.jacobian(forward_spatial)
    grads = grad_fn(x)  # (T, T, H, W, 3) - grad of each output t w.r.t. each input t

    # Extract diagonal: d(logit_t)/d(Image_t)
    T = x.shape[0]
    per_timestep_grads = jnp.array([grads[t, t] for t in range(T)])  # (T, H, W, 3)

    return per_timestep_grads


def compute_final_to_all_gradients(model, params, x, label):
    """
    Compute d(logit_final)/d(Image_t) for each timestep.

    Returns gradients of shape (T, H, W, 3)
    """
    def forward_final(x):
        # Get final logits
        logits = model.apply(
            {'params': params}, x[jnp.newaxis], training=False
        )
        return logits[0, label]  # scalar - logit for correct class

    # Compute gradient w.r.t. input
    grad_fn = jax.grad(forward_final)
    grads = grad_fn(x)  # (T, H, W, 3)

    return grads


def visualize_sample_gradients(model, params, image, label, sample_idx, output_dir, seq_len, args):
    """Generate gradient visualizations for a single sample."""
    x = jnp.array(image)  # (T, H, W, 3)
    T, H, W, C = x.shape

    # Get prediction
    logits = model.apply({'params': params}, x[jnp.newaxis], training=False)
    pred = int(jnp.argmax(logits, axis=-1)[0])
    correct = pred == label

    # Compute gradients
    print(f"  Computing per-timestep gradients...")
    per_t_grads = compute_per_timestep_gradients(model, params, x, label)

    print(f"  Computing final-to-all gradients...")
    final_grads = compute_final_to_all_gradients(model, params, x, label)

    per_t_grads_np = np.array(per_t_grads)  # (T, H, W, 3)
    final_grads_np = np.array(final_grads)  # (T, H, W, 3)

    # Denormalize input for display
    base_denorm = np.array(x) * IMAGENET_STD + IMAGENET_MEAN
    base_denorm = np.clip(base_denorm, 0, 1)

    # === Visualization 1: d(logit_t)/d(Image_t) ===
    frames_per_t = []
    # Compute global range across all timesteps
    per_t_mag = np.abs(per_t_grads_np).sum(axis=-1)  # (T, H, W)
    per_t_vmax = per_t_mag.max()

    for t in range(min(seq_len, T)):
        grad_t = per_t_grads_np[t]  # (H, W, 3)
        grad_mag = np.abs(grad_t).sum(axis=-1)  # (H, W)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Input at timestep t
        axes[0].imshow(base_denorm[t])
        axes[0].set_title(f'Input T={t+1}/{seq_len}')
        axes[0].axis('off')

        # Gradient magnitude
        im = axes[1].imshow(grad_mag, cmap='hot', vmin=0, vmax=per_t_vmax)
        axes[1].set_title(f'd(logit_t)/d(Image_t) magnitude')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Overlay on image
        grad_norm = grad_mag / (per_t_vmax + 1e-8)
        cmap = plt.colormaps.get_cmap('hot')
        overlay_colored = cmap(grad_norm)[:, :, :3]
        overlay = 0.4 * base_denorm[t] + 0.6 * overlay_colored
        overlay = np.clip(overlay, 0, 1)
        axes[2].imshow(overlay)
        axes[2].set_title(f'GT: {"Conn" if label==1 else "Disc"} | Pred: {"Conn" if pred==1 else "Disc"} {"OK" if correct else "X"}')
        axes[2].axis('off')

        plt.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3].copy() / 255.0
        frames_per_t.append(frame)
        plt.close(fig)

    # === Visualization 2: d(logit_final)/d(Image_t) ===
    frames_final = []
    # Compute global range
    final_mag = np.abs(final_grads_np).sum(axis=-1)  # (T, H, W)
    final_vmax = final_mag.max()

    for t in range(min(seq_len, T)):
        grad_t = final_grads_np[t]  # (H, W, 3)
        grad_mag = np.abs(grad_t).sum(axis=-1)  # (H, W)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Input at timestep t
        axes[0].imshow(base_denorm[t])
        axes[0].set_title(f'Input T={t+1}/{seq_len}')
        axes[0].axis('off')

        # Gradient magnitude
        im = axes[1].imshow(grad_mag, cmap='hot', vmin=0, vmax=final_vmax)
        axes[1].set_title(f'd(logit_final)/d(Image_t) magnitude')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Overlay on image
        grad_norm = grad_mag / (final_vmax + 1e-8)
        cmap = plt.colormaps.get_cmap('hot')
        overlay_colored = cmap(grad_norm)[:, :, :3]
        overlay = 0.4 * base_denorm[t] + 0.6 * overlay_colored
        overlay = np.clip(overlay, 0, 1)
        axes[2].imshow(overlay)
        axes[2].set_title(f'GT: {"Conn" if label==1 else "Disc"} | Pred: {"Conn" if pred==1 else "Disc"} {"OK" if correct else "X"}')
        axes[2].axis('off')

        plt.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3].copy() / 255.0
        frames_final.append(frame)
        plt.close(fig)

    # Save GIFs and PNGs
    status = "correct" if correct else "wrong"
    label_str = "connected" if label == 1 else "disconnected"

    # Per-timestep gradients
    gif_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_grad_per_t.gif"
    pil_frames = [Image.fromarray((f * 255).astype(np.uint8)) for f in frames_per_t]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                       duration=args.frame_duration, loop=0)

    png_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_grad_per_t_final.png"
    Image.fromarray((frames_per_t[-1] * 255).astype(np.uint8)).save(png_path)

    # Final-to-all gradients
    gif_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_grad_final.gif"
    pil_frames = [Image.fromarray((f * 255).astype(np.uint8)) for f in frames_final]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                       duration=args.frame_duration, loop=0)

    png_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_grad_final_final.png"
    Image.fromarray((frames_final[-1] * 255).astype(np.uint8)).save(png_path)

    return correct, pred, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='visualizations/gradients')
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
    parser.add_argument('--num_samples', type=int, default=5)
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

    print(f"Generating {len(indices)} gradient visualizations...")
    correct_count = 0

    for i, idx in enumerate(tqdm(indices)):
        image, label = test_dataset[idx]
        correct, pred, label = visualize_sample_gradients(
            model, params, image, label, i, output_dir, args.seq_len, args
        )
        if correct:
            correct_count += 1

    print(f"\nAccuracy: {correct_count}/{len(indices)} ({100*correct_count/len(indices):.1f}%)")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
