#!/usr/bin/env python3
"""
Visualize per-pixel class evidence evolution on Pathfinder.

Applies the classifier head to each spatial position at each timestep,
showing how the model's per-pixel "decision" evolves over temporal recurrence.

This reveals whether the model does contour-tracing (activity spreading along
the path) or global pattern matching (focusing only on endpoints).

Usage:
    python scripts/visualize_pathfinder_perpixel.py \
        --checkpoint checkpoints/pf9_vit_opponent_d1_e32/epoch_25 \
        --output_dir visualizations/pf9_perpixel \
        --num_samples 10
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

# Import model and data
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cssm_vit import CSSMViT
from src.pathfinder_data import get_pathfinder_datasets, IMAGENET_MEAN, IMAGENET_STD

import orbax.checkpoint as ocp


def load_model_and_checkpoint(checkpoint_path: str, args):
    """Load model architecture and restore checkpoint."""

    model = CSSMViT(
        num_classes=2,
        embed_dim=args.embed_dim,
        depth=args.depth,
        patch_size=args.patch_size,
        cssm_type=args.cssm,
        kernel_size=args.kernel_size,
        use_dwconv=args.use_dwconv,
        output_act=args.output_act,
        stem_mode=args.stem_mode,
        use_pos_embed=not args.no_pos_embed,
        rope_mode=args.rope_mode,
        block_size=args.block_size,
        gate_rank=args.gate_rank,
    )

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, args.seq_len, args.image_size, args.image_size, 3))
    variables = model.init({'params': rng, 'dropout': rng}, dummy_input, training=False)

    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_path = os.path.abspath(checkpoint_path)
    restored = checkpointer.restore(checkpoint_path)

    if hasattr(restored, 'params'):
        params = restored.params
        epoch = getattr(restored, 'epoch', 'unknown')
    elif isinstance(restored, dict) and 'params' in restored:
        params = restored['params']
        epoch = restored.get('epoch', 'unknown')
    else:
        raise ValueError(f"Could not find params in checkpoint")

    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")

    return model, params


def get_perpixel_predictions(model, params, x):
    """
    Get per-pixel class predictions at each timestep using model's return_spatial option.

    Returns:
        perpixel_logits: (B, T, H', W', num_classes)
        final_logits: (B, num_classes)
    """
    final_logits, perpixel_logits = model.apply(
        {'params': params}, x, training=False, return_spatial=True
    )
    return perpixel_logits, final_logits


def create_perpixel_visualization(image, delta_logit, timestep, total_timesteps,
                                   label, pred, correct, vmax, alpha=0.7):
    """
    Create visualization showing per-pixel Δlogit[connected] from T=0.

    Args:
        image: Original image (H, W, C)
        delta_logit: Per-pixel Δlogit[conn] from T=0 at this timestep (H', W')
        timestep: Current timestep (0-indexed)
        total_timesteps: Total number of timesteps
        label: Ground truth label
        pred: Predicted label
        correct: Whether prediction is correct
        vmax: Max value for colorbar (symmetric around 0)
    """
    # Denormalize image
    img_denorm = image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img_denorm = np.clip(img_denorm, 0, 1)

    H_img, W_img = image.shape[0], image.shape[1]
    H_feat, W_feat = delta_logit.shape

    # Resize delta_logit map to match image using float interpolation
    if H_feat != H_img or W_feat != W_img:
        from scipy.ndimage import zoom
        scale_h = H_img / H_feat
        scale_w = W_img / W_feat
        delta_resized = zoom(delta_logit, (scale_h, scale_w), order=1)
    else:
        delta_resized = delta_logit

    # Normalize to [0, 1] for colormap (centered at 0.5)
    delta_normalized = (delta_resized / (2 * vmax)) + 0.5
    delta_normalized = np.clip(delta_normalized, 0, 1)

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original image
    axes[0].imshow(img_denorm)
    axes[0].set_title(f'Input (GT: {"Connected" if label == 1 else "Disconnected"})', fontsize=12)
    axes[0].axis('off')

    # Panel 2: Overlay - blend image with delta_logit map
    # Red = positive Δlogit (more connected), Blue = negative (less connected)
    cmap = plt.colormaps.get_cmap('RdBu_r')
    delta_colored = cmap(delta_normalized)[:, :, :3]
    overlay = (1 - alpha) * img_denorm + alpha * delta_colored
    overlay = np.clip(overlay, 0, 1)

    axes[1].imshow(overlay)
    axes[1].set_title(f'Δlogit(conn) from T=0 @ T={timestep+1}/{total_timesteps}', fontsize=12)
    axes[1].axis('off')

    # Panel 3: Raw delta_logit map with colorbar
    im = axes[2].imshow(delta_resized, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    status_symbol = "✓" if correct else "✗"
    axes[2].set_title(f'Δlogit map | Pred: {"Conn" if pred == 1 else "Disc"} {status_symbol}', fontsize=12)
    axes[2].axis('off')

    # Add colorbar
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Δlogit(conn)', fontsize=10)

    plt.tight_layout()

    # Convert figure to numpy array
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frame = buf[:, :, :3].copy()
    plt.close(fig)

    return frame / 255.0


def create_gif(frames, output_path, duration=400):
    """Create GIF from list of frames."""
    pil_frames = []
    for frame in frames:
        frame_uint8 = (frame * 255).astype(np.uint8)
        pil_frames.append(Image.fromarray(frame_uint8))

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )


def visualize_sample(model, params, image, label, sample_idx, output_dir, seq_len, args):
    """Generate per-pixel visualization for a single sample."""
    x = jnp.array(image)[jnp.newaxis]  # (1, T, H, W, C)

    # Get per-pixel predictions at each timestep
    perpixel_logits, final_logits = get_perpixel_predictions(model, params, x)
    # perpixel_logits: (1, T, H', W', 2)

    # Compute logit difference: logit[connected] - logit[disconnected]
    perpixel_logits_np = np.array(perpixel_logits[0])  # (T, H', W', 2)
    logit_diff = perpixel_logits_np[:, :, :, 1] - perpixel_logits_np[:, :, :, 0]  # (T, H', W')

    # Compute Δlogit from T=0 (this shows temporal evolution!)
    delta_logit = logit_diff - logit_diff[0:1]  # (T, H', W'), T=0 is all zeros

    # Final prediction
    pred = int(jnp.argmax(final_logits, axis=-1)[0])
    correct = pred == label

    # Compute vmax for consistent colorbar across all timesteps
    vmax = max(np.abs(delta_logit).max(), 1.0)  # At least 1.0 for reasonable scaling

    # Create frames for each timestep
    frames = []
    base_image = np.array(image[0])  # First frame of video (they're all the same)

    for t in range(seq_len):
        frame = create_perpixel_visualization(
            base_image,
            delta_logit[t],
            timestep=t,
            total_timesteps=seq_len,
            label=label,
            pred=pred,
            correct=correct,
            vmax=vmax,
            alpha=args.overlay_alpha
        )
        frames.append(frame)

    # Save GIF
    status = "correct" if correct else "wrong"
    label_str = "connected" if label == 1 else "disconnected"
    gif_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}.gif"
    create_gif(frames, gif_path, duration=args.frame_duration)

    # Save first, middle, and last frames as PNGs
    for t_idx, t_name in [(0, 'first'), (seq_len//2, 'mid'), (seq_len-1, 'last')]:
        png_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_{t_name}.png"
        Image.fromarray((frames[t_idx] * 255).astype(np.uint8)).save(png_path)

    # Also save a summary figure showing evolution
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Top row: Δlogit maps at different timesteps
    timesteps_to_show = [0, 2, 4, 7] if seq_len == 8 else [0, seq_len//4, seq_len//2, seq_len-1]
    for i, t in enumerate(timesteps_to_show):
        if t < seq_len:
            im = axes[0, i].imshow(delta_logit[t], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[0, i].set_title(f'Δlogit @ T={t+1}')
            axes[0, i].axis('off')

    # Add colorbar to last plot in top row
    plt.colorbar(im, ax=axes[0, -1], fraction=0.046, pad=0.04)

    # Bottom row: Show original image and raw logit_diff at different timesteps
    img_denorm = base_image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img_denorm = np.clip(img_denorm, 0, 1)
    axes[1, 0].imshow(img_denorm)
    axes[1, 0].set_title(f'Input (GT: {"Conn" if label==1 else "Disc"})')
    axes[1, 0].axis('off')

    # Show raw logit_diff at T=1, T=4, T=8
    logit_vmax = max(np.abs(logit_diff).max(), 1.0)
    for i, t in enumerate([0, 3, 7] if seq_len == 8 else [0, seq_len//2, seq_len-1]):
        if t < seq_len:
            im = axes[1, i+1].imshow(logit_diff[t], cmap='RdBu_r', vmin=-logit_vmax, vmax=logit_vmax)
            axes[1, i+1].set_title(f'logit_diff T={t+1}')
            axes[1, i+1].axis('off')

    plt.colorbar(im, ax=axes[1, -1], fraction=0.046, pad=0.04, label='logit(conn)-logit(disc)')

    plt.suptitle(f'Sample {sample_idx}: {"Connected" if label==1 else "Disconnected"} | '
                 f'Pred: {"Connected" if pred==1 else "Disconnected"} {"✓" if correct else "✗"} | '
                 f'Δlogit_max={delta_logit[-1].max():.1f}',
                 fontsize=14)
    plt.tight_layout()

    summary_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    return correct, pred, label, delta_logit


def main():
    parser = argparse.ArgumentParser(description='Visualize per-pixel class evidence on Pathfinder')

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='visualizations/pathfinder_perpixel')

    # Model architecture
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--cssm', type=str, default='opponent')
    parser.add_argument('--kernel_size', type=int, default=15)
    parser.add_argument('--use_dwconv', action='store_true', default=True)
    parser.add_argument('--output_act', type=str, default='gelu')
    parser.add_argument('--stem_mode', type=str, default='conv')
    parser.add_argument('--no_pos_embed', action='store_true', default=False)
    parser.add_argument('--rope_mode', type=str, default='none')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--block_size', type=int, default=1)
    parser.add_argument('--gate_rank', type=int, default=0)

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025')
    parser.add_argument('--difficulty', type=str, default='9')
    parser.add_argument('--num_samples', type=int, default=10)

    # Visualization
    parser.add_argument('--frame_duration', type=int, default=500,
                        help='Duration of each frame in ms')
    parser.add_argument('--overlay_alpha', type=float, default=0.7,
                        help='Alpha for probability overlay')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("Loading model...")
    model, params = load_model_and_checkpoint(args.checkpoint, args)

    print("Loading test data...")
    _, _, test_dataset = get_pathfinder_datasets(
        root=args.data_dir,
        difficulty=args.difficulty,
        image_size=args.image_size,
        num_frames=args.seq_len,
    )

    print(f"Test set size: {len(test_dataset)}")

    num_samples = min(args.num_samples, len(test_dataset))
    np.random.seed(42)  # Reproducible samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    print(f"Generating {num_samples} per-pixel visualizations...")
    correct_count = 0
    all_probs = []

    for i, idx in enumerate(tqdm(indices)):
        image, label = test_dataset[idx]

        correct, pred, label, delta_logit = visualize_sample(
            model, params, image, label, i, output_dir, args.seq_len, args
        )

        if correct:
            correct_count += 1
        all_probs.append((label, pred, correct, delta_logit))

    print(f"\nDone! Accuracy on visualized samples: {correct_count}/{num_samples} ({100*correct_count/num_samples:.1f}%)")
    print(f"GIFs and summaries saved to: {output_dir}")

    # Print some statistics about the delta_logit evolution
    print("\n--- Per-pixel Δlogit evolution statistics ---")
    for i, (label, pred, correct, delta_logit) in enumerate(all_probs[:5]):
        # delta_logit[0] should be all zeros (by definition)
        max_delta = [delta_logit[t].max() for t in range(len(delta_logit))]
        min_delta = [delta_logit[t].min() for t in range(len(delta_logit))]
        status = "✓" if correct else "✗"
        print(f"Sample {i}: GT={'Conn' if label==1 else 'Disc'} {status}, "
              f"Δlogit max: {max_delta[0]:.1f}→{max_delta[2]:.1f}→{max_delta[-1]:.1f}, "
              f"min: {min_delta[0]:.1f}→{min_delta[2]:.1f}→{min_delta[-1]:.1f}")


if __name__ == '__main__':
    main()
