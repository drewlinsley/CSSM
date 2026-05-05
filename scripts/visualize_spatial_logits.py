#!/usr/bin/env python3
"""
Visualize per-pixel classification logits by applying readout head spatially.

The key idea:
1. Run forward pass with return_spatial=True
2. Get per-pixel logits: (B, T, H', W', num_classes)
3. Visualize logits[correct_class] at each spatial position over time

This shows WHERE in space the model sees evidence for "connected" vs "disconnected".

Usage:
    python scripts/visualize_spatial_logits.py \
        --checkpoint checkpoints/pf14_vit_hgru_bi_d1_e32/epoch_30 \
        --output_dir visualizations/pf14_spatial_logits \
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
    """Load model with architecture that supports spatial logit extraction."""
    # Create the model
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

    # Initialize
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, args.seq_len, args.image_size, args.image_size, 3))
    _ = model.init({'params': rng, 'dropout': rng}, dummy, training=False)

    # Load checkpoint
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(os.path.abspath(checkpoint_path))

    if hasattr(restored, 'params'):
        params = restored.params
    else:
        params = restored['params']

    print(f"Loaded: {checkpoint_path}")
    return model, params


def visualize_sample(model, params, image, label, sample_idx, output_dir, seq_len, args):
    """Generate visualization showing spatial logits evolution."""
    x = jnp.array(image)[jnp.newaxis]
    B, T, H, W, C = x.shape

    # Get global prediction AND per-pixel logits using return_spatial=True
    final_logits, perpixel_logits = model.apply(
        {'params': params}, x, training=False, return_spatial=True
    )
    # final_logits: (B, num_classes)
    # perpixel_logits: (B, T, H', W', num_classes)

    pred = int(jnp.argmax(final_logits, axis=-1)[0])
    correct = pred == label

    # Create frames showing logit evolution
    frames = []
    base_image = np.array(image[0])
    base_denorm = base_image * IMAGENET_STD + IMAGENET_MEAN
    base_denorm = np.clip(base_denorm, 0, 1)

    perpixel_np = np.array(perpixel_logits[0])  # (T, H', W', 2)
    _, H_feat, W_feat, _ = perpixel_np.shape

    for t in range(min(seq_len, perpixel_np.shape[0])):
        spatial_logits = perpixel_np[t]  # (H', W', 2)

        # Get logit for the correct class (ground truth label)
        logits_gt = spatial_logits[:, :, label]  # (H', W')

        # Also get the difference (connected - disconnected)
        logits_diff = spatial_logits[:, :, 1] - spatial_logits[:, :, 0]  # (H', W')

        # Upsample to original image size for visualization
        from scipy.ndimage import zoom
        scale_h = H / logits_gt.shape[0]
        scale_w = W / logits_gt.shape[1]
        logits_gt_up = zoom(logits_gt, (scale_h, scale_w), order=1)
        logits_diff_up = zoom(logits_diff, (scale_h, scale_w), order=1)

        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Original image
        axes[0].imshow(base_denorm)
        axes[0].set_title(f'Input (GT: {"Connected" if label == 1 else "Disconnected"})')
        axes[0].axis('off')

        # Logit for GT class
        vmax = max(abs(logits_gt_up.min()), abs(logits_gt_up.max()), 1e-3)
        im1 = axes[1].imshow(logits_gt_up, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1].set_title(f'Logit[GT class] at T={t+1}/{seq_len}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # Logit difference (connected - disconnected)
        vmax_diff = max(abs(logits_diff_up.min()), abs(logits_diff_up.max()), 1e-3)
        im2 = axes[2].imshow(logits_diff_up, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
        axes[2].set_title(f'Logit[Connected] - Logit[Disconnected]')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        # Overlay - positive logit[GT] on image
        logits_gt_norm = (logits_gt_up - logits_gt_up.min()) / (logits_gt_up.max() - logits_gt_up.min() + 1e-8)
        cmap = plt.colormaps.get_cmap('hot')
        overlay_colored = cmap(logits_gt_norm)[:, :, :3]
        overlay = 0.4 * base_denorm + 0.6 * overlay_colored
        overlay = np.clip(overlay, 0, 1)

        axes[3].imshow(overlay)
        axes[3].set_title(f'Pred: {"Connected" if pred == 1 else "Disconnected"} {"✓" if correct else "✗"}')
        axes[3].axis('off')

        plt.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3].copy() / 255.0
        frames.append(frame)
        plt.close(fig)

    # Save GIF
    status = "correct" if correct else "wrong"
    label_str = "connected" if label == 1 else "disconnected"

    gif_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_spatial.gif"
    pil_frames = [Image.fromarray((f * 255).astype(np.uint8)) for f in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                       duration=args.frame_duration, loop=0)

    png_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_spatial_final.png"
    Image.fromarray((frames[-1] * 255).astype(np.uint8)).save(png_path)

    return correct, pred, label


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='visualizations/spatial_logits')

    # Model architecture (must match training)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--cssm', type=str, default='opponent')
    parser.add_argument('--kernel_size', type=int, default=15)
    parser.add_argument('--stem_mode', type=str, default='conv')
    parser.add_argument('--no_pos_embed', action='store_true', default=False)
    parser.add_argument('--rope_mode', type=str, default='none')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--block_size', type=int, default=1)
    parser.add_argument('--gate_rank', type=int, default=0)
    parser.add_argument('--output_act', type=str, default='gelu')
    parser.add_argument('--use_dwconv', action='store_true', default=False)

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025')
    parser.add_argument('--difficulty', type=str, default='9')
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
