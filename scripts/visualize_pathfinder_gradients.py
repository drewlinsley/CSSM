#!/usr/bin/env python3
"""
Visualize CSSM gradient evolution on Pathfinder.

Creates GIF animations showing how the model's gradient-based attention
evolves over temporal recurrence steps as it solves the contour tracing task.

Usage:
    python scripts/visualize_pathfinder_gradients.py \
        --checkpoint checkpoints/pf9_vit_opponent_d1_e32/epoch_100 \
        --output_dir visualizations/pf9_gradients \
        --num_samples 20
"""

import argparse
import os
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

# Import model and data
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cssm_vit import CSSMViT
from src.pathfinder_data import get_pathfinder_datasets, IMAGENET_MEAN, IMAGENET_STD

import orbax.checkpoint as ocp
from flax.training import train_state
import optax


# Custom TrainState matching main.py
class TrainState(train_state.TrainState):
    """Extended train state with additional tracking."""
    epoch: int = 0


def load_model_and_checkpoint(checkpoint_path: str, args):
    """Load model architecture and restore checkpoint."""

    # Create model with same architecture as training
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

    # Initialize with dummy input
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, args.seq_len, args.image_size, args.image_size, 3))
    variables = model.init({'params': rng, 'dropout': rng}, dummy_input, training=False)

    # Load checkpoint - just extract params
    checkpointer = ocp.PyTreeCheckpointer()

    # Ensure absolute path
    checkpoint_path = os.path.abspath(checkpoint_path)

    # Restore raw checkpoint
    restored = checkpointer.restore(checkpoint_path)

    # Extract params from restored state
    if hasattr(restored, 'params'):
        params = restored.params
        epoch = getattr(restored, 'epoch', 'unknown')
    elif isinstance(restored, dict) and 'params' in restored:
        params = restored['params']
        epoch = restored.get('epoch', 'unknown')
    else:
        raise ValueError(f"Could not find params in checkpoint. Keys: {restored.keys() if isinstance(restored, dict) else dir(restored)}")

    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")

    return model, params


def get_per_timestep_logits(model, params, x):
    """
    Get classification logits at each timestep.

    Modifies the forward pass to return logits from each temporal position.
    """
    B, T, H, W, C = x.shape

    # We'll run the model T times, each time using only timesteps 0:t
    all_logits = []

    for t in range(1, T + 1):
        # Create input with only first t timesteps, pad the rest with zeros
        x_partial = jnp.concatenate([
            x[:, :t],
            jnp.zeros((B, T - t, H, W, C))
        ], axis=1)

        # Forward pass
        logits = model.apply({'params': params}, x_partial, training=False)
        all_logits.append(logits)

    return jnp.stack(all_logits, axis=1)  # (B, T, num_classes)


def compute_gradient_at_timestep(model, params, x, timestep, target_class):
    """
    Compute gradient of logit[target_class] at given timestep w.r.t. input.
    """
    def loss_fn(x_input):
        # Run model with input up to timestep
        B, T, H, W, C = x_input.shape

        # Use full input but we're interested in gradient flow through timestep
        logits = model.apply({'params': params}, x_input, training=False)

        # Return logit for target class
        return logits[0, target_class]

    # Compute gradient
    grad = jax.grad(loss_fn)(x)

    return grad


def compute_all_timestep_gradients(model, params, x, target_class, seq_len):
    """
    Compute gradients at each timestep.

    For each timestep t, compute gradient of final decision w.r.t. input frame t.
    This shows which parts of the input at each timestep influence the decision.
    """
    B, T, H, W, C = x.shape
    gradients = []

    def loss_fn(x_input):
        logits = model.apply({'params': params}, x_input, training=False)
        return logits[0, target_class]

    # Compute full gradient w.r.t. all timesteps at once
    full_grad = jax.grad(loss_fn)(x)  # (B, T, H, W, C)

    # Extract gradient for each timestep
    for t in range(seq_len):
        grad_frame = full_grad[0, t]  # (H, W, C) - gradient w.r.t. frame t
        gradients.append(grad_frame)

    return gradients


def get_per_timestep_activations(model, params, x):
    """
    Get spatial activations at each timestep by extracting intermediate states.

    This shows how the model's internal representation evolves spatially.
    Returns the per-pixel "evidence" for each class at each timestep.
    """
    # This requires modifying the model to return intermediate states
    # For now, we'll use a simpler approach: run the model with truncated inputs
    B, T, H, W, C = x.shape

    activations = []
    for t in range(1, T + 1):
        # Create input with first t frames, pad rest with zeros
        x_partial = jnp.concatenate([
            x[:, :t],
            jnp.zeros((B, T - t, H, W, C))
        ], axis=1)

        # Get logits (global prediction)
        logits = model.apply({'params': params}, x_partial, training=False)
        activations.append(logits[0])  # (num_classes,)

    return jnp.stack(activations, axis=0)  # (T, num_classes)


def normalize_gradient(grad):
    """Normalize gradient for visualization."""
    # Take absolute value and sum across channels
    grad_magnitude = np.abs(grad).sum(axis=-1)  # (H, W)

    # Normalize to [0, 1]
    grad_min = grad_magnitude.min()
    grad_max = grad_magnitude.max()
    if grad_max - grad_min > 1e-8:
        grad_normalized = (grad_magnitude - grad_min) / (grad_max - grad_min)
    else:
        grad_normalized = np.zeros_like(grad_magnitude)

    return grad_normalized


def create_overlay(image, gradient, alpha=0.5, cmap='hot'):
    """Create image with gradient heatmap overlay."""
    # Denormalize image
    image_denorm = image * IMAGENET_STD + IMAGENET_MEAN
    image_denorm = np.clip(image_denorm, 0, 1)

    # Convert grayscale to RGB if needed (Pathfinder images are grayscale repeated)
    if len(image_denorm.shape) == 3:
        image_rgb = image_denorm
    else:
        image_rgb = np.stack([image_denorm] * 3, axis=-1)

    # Normalize gradient
    grad_norm = normalize_gradient(gradient)

    # Apply colormap to gradient
    colormap = plt.colormaps.get_cmap(cmap)
    grad_colored = colormap(grad_norm)[:, :, :3]  # (H, W, 3)

    # Blend
    overlay = (1 - alpha) * image_rgb + alpha * grad_colored
    overlay = np.clip(overlay, 0, 1)

    return overlay, grad_norm


def create_gif(frames, output_path, duration=200):
    """Create GIF from list of frames."""
    # Convert to PIL images
    pil_frames = []
    for frame in frames:
        # Scale to 0-255
        frame_uint8 = (frame * 255).astype(np.uint8)
        pil_frames.append(Image.fromarray(frame_uint8))

    # Save GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )


def visualize_sample(model, params, image, label, sample_idx, output_dir, seq_len, args):
    """Generate visualization for a single sample."""
    # Prepare input: (1, T, H, W, C)
    x = jnp.array(image)[jnp.newaxis]  # (1, T, H, W, C)

    # Get prediction
    logits = model.apply({'params': params}, x, training=False)
    pred = int(jnp.argmax(logits, axis=-1)[0])
    correct = pred == label

    # Use predicted class for gradient (shows what model is "looking at")
    target_class = pred

    # Compute gradients at each timestep
    gradients = compute_all_timestep_gradients(model, params, x, target_class, seq_len)

    # Create frames
    frames = []
    base_image = np.array(image[0])  # First frame (they're all the same)

    for t, grad in enumerate(gradients):
        grad_np = np.array(grad)

        # Create overlay
        overlay, grad_norm = create_overlay(base_image, grad_np, alpha=0.6, cmap='hot')

        # Create figure with image and colorbar
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        img_denorm = base_image * IMAGENET_STD + IMAGENET_MEAN
        img_denorm = np.clip(img_denorm, 0, 1)
        axes[0].imshow(img_denorm)
        axes[0].set_title(f'Input (Label: {"Connected" if label == 1 else "Not Connected"})')
        axes[0].axis('off')

        # Gradient overlay - shows ∂decision/∂input[t]
        im = axes[1].imshow(overlay)
        axes[1].set_title(f'∂decision/∂input[T={t}] | Pred: {"Connected" if pred == 1 else "Not Connected"} {"✓" if correct else "✗"}')
        axes[1].axis('off')

        # Add colorbar for gradient magnitude
        # plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Convert figure to numpy array
        fig.canvas.draw()
        # Use buffer_rgba and convert to RGB
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3].copy()  # RGBA -> RGB
        frames.append(frame / 255.0)

        plt.close(fig)

    # Save GIF
    status = "correct" if correct else "wrong"
    label_str = "connected" if label == 1 else "disconnected"
    gif_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}.gif"
    create_gif(frames, gif_path, duration=args.frame_duration)

    # Also save final frame as PNG
    png_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_final.png"
    Image.fromarray((frames[-1] * 255).astype(np.uint8)).save(png_path)

    return correct, pred, label


def main():
    parser = argparse.ArgumentParser(description='Visualize CSSM gradients on Pathfinder')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='visualizations/pathfinder_gradients',
                        help='Output directory for GIFs')

    # Model architecture (should match training)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--cssm', type=str, default='opponent')
    parser.add_argument('--kernel_size', type=int, default=15)
    parser.add_argument('--use_dwconv', action='store_true', default=False)
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
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of test samples to visualize')

    # Visualization
    parser.add_argument('--frame_duration', type=int, default=300,
                        help='Duration of each frame in ms')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load model and checkpoint
    print("Loading model...")
    model, params = load_model_and_checkpoint(args.checkpoint, args)

    # Load test data
    print("Loading test data...")
    _, _, test_dataset = get_pathfinder_datasets(
        root=args.data_dir,
        difficulty=args.difficulty,
        image_size=args.image_size,
        num_frames=args.seq_len,
    )

    print(f"Test set size: {len(test_dataset)}")

    # Select samples
    num_samples = min(args.num_samples, len(test_dataset))
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    # Visualize each sample
    print(f"Generating {num_samples} visualizations...")
    correct_count = 0

    for i, idx in enumerate(tqdm(indices)):
        image, label = test_dataset[idx]

        correct, pred, label = visualize_sample(
            model, params, image, label, i, output_dir, args.seq_len, args
        )

        if correct:
            correct_count += 1

    print(f"\nDone! Accuracy on visualized samples: {correct_count}/{num_samples} ({100*correct_count/num_samples:.1f}%)")
    print(f"GIFs saved to: {output_dir}")


if __name__ == '__main__':
    main()
