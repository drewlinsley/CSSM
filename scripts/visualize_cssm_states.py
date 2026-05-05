#!/usr/bin/env python3
"""
Visualize CSSM hidden state evolution on Pathfinder.

Instead of gradients, this extracts the actual hidden states (X, Y channels)
from the CSSM recurrence to show how activity propagates spatially over time.

Key insight: Gradients show sensitivity, but hidden states show actual activity.
For contour tracing, we want to see the excitation (X) channel propagating
along the path over temporal recurrence steps.

Methods implemented:
1. Hidden state magnitude (|X|, |Y|) - shows where activity is
2. State difference (X[t] - X[t-1]) - shows where activity is SPREADING
3. Excitation-Inhibition ratio (X / (Y + eps)) - shows net activity
4. Spectral phase evolution - shows propagation in frequency domain

Usage:
    python scripts/visualize_cssm_states.py \
        --checkpoint checkpoints/pf14_vit_hgru_bi_d1_e32/epoch_30 \
        --output_dir visualizations/pf14_states \
        --method state_diff
"""

import argparse
import os
from pathlib import Path
from functools import partial
from typing import List, Tuple, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cssm_vit import CSSMViT
from src.pathfinder_data import get_pathfinder_datasets, IMAGENET_MEAN, IMAGENET_STD

import orbax.checkpoint as ocp


class CSSMStateExtractor(nn.Module):
    """
    Modified CSSMViT that returns intermediate hidden states.

    This captures the X (excitation) and Y (inhibition) states from the
    CSSM recurrence at each timestep, allowing us to visualize the
    actual activity propagation rather than just gradients.
    """
    base_model: CSSMViT

    @nn.compact
    def __call__(self, x, training=False):
        # We'll use a custom forward pass that captures intermediate states
        # For now, we'll use the capture_intermediates feature
        pass


def extract_cssm_states_via_intermediates(model, params, x):
    """
    Extract CSSM hidden states using Flax's capture_intermediates.

    Returns the intermediate activations from CSSM blocks.
    """
    # Use mutable='intermediates' to capture intermediate values
    # This requires the model to use self.sow() to record values

    # Alternative: just run the model and get what we can
    logits = model.apply({'params': params}, x, training=False)

    return logits, None  # States extraction requires model modification


def extract_states_by_truncation(model, params, x, seq_len):
    """
    Extract "effective states" by running model with increasing timesteps.

    For each t in 1..T:
    - Run model with first t frames (rest zeroed)
    - The output represents the accumulated state up to time t

    The DIFFERENCE between consecutive outputs approximates the
    contribution of each timestep.
    """
    B, T, H, W, C = x.shape

    outputs = []
    for t in range(1, seq_len + 1):
        # Create input with first t frames, rest zeroed
        x_partial = jnp.concatenate([
            x[:, :t],
            jnp.zeros((B, T - t, H, W, C))
        ], axis=1)

        logits = model.apply({'params': params}, x_partial, training=False)
        outputs.append(logits[0])  # (num_classes,)

    return jnp.stack(outputs, axis=0)  # (T, num_classes)


def compute_gradient_accumulation(model, params, x, target_class, seq_len):
    """
    Compute ACCUMULATED gradients over timesteps.

    Instead of showing the gradient at each timestep independently,
    this shows the cumulative gradient contribution, which better
    represents evidence accumulation.

    cumulative_grad[t] = sum(grad[0:t])
    """
    B, T, H, W, C = x.shape

    def loss_fn(x_input):
        logits = model.apply({'params': params}, x_input, training=False)
        return logits[0, target_class]

    # Compute full gradient
    full_grad = jax.grad(loss_fn)(x)  # (B, T, H, W, C)

    # Compute cumulative sum over time
    cumulative_grads = []
    running_sum = jnp.zeros((H, W, C))

    for t in range(seq_len):
        running_sum = running_sum + full_grad[0, t]
        cumulative_grads.append(running_sum.copy())

    return cumulative_grads


def compute_gradient_times_input(model, params, x, target_class):
    """
    Compute gradient × input (saliency).

    This highlights pixels that are both:
    - Important (high gradient)
    - Present in input (non-zero)

    Better than raw gradient for sparse inputs like Pathfinder.
    """
    def loss_fn(x_input):
        logits = model.apply({'params': params}, x_input, training=False)
        return logits[0, target_class]

    grad = jax.grad(loss_fn)(x)
    saliency = grad * x  # Element-wise product

    return saliency


def compute_integrated_gradients(model, params, x, target_class, steps=50):
    """
    Integrated Gradients: more principled attribution method.

    Integrates gradients along a path from baseline (zeros) to input.
    Satisfies completeness axiom: sum of attributions = output - baseline_output.

    This often shows cleaner, more interpretable attributions than raw gradients.
    """
    baseline = jnp.zeros_like(x)

    def loss_fn(x_input):
        logits = model.apply({'params': params}, x_input, training=False)
        return logits[0, target_class]

    # Compute gradients at interpolated points
    alphas = jnp.linspace(0, 1, steps)
    grads = []

    for alpha in alphas:
        x_interp = baseline + alpha * (x - baseline)
        grad = jax.grad(loss_fn)(x_interp)
        grads.append(grad)

    # Average gradients
    avg_grad = jnp.mean(jnp.stack(grads), axis=0)

    # Multiply by (input - baseline)
    integrated = avg_grad * (x - baseline)

    return integrated


def compute_temporal_gradient_diff(model, params, x, target_class, seq_len):
    """
    Compute gradient DIFFERENCE between consecutive timesteps.

    diff_grad[t] = grad[t] - grad[t-1]

    This shows where the model's attention is SHIFTING over time,
    which should reveal propagation along contours.
    """
    def loss_fn(x_input):
        logits = model.apply({'params': params}, x_input, training=False)
        return logits[0, target_class]

    full_grad = jax.grad(loss_fn)(x)  # (B, T, H, W, C)

    diff_grads = []
    prev_grad = jnp.zeros_like(full_grad[0, 0])

    for t in range(seq_len):
        curr_grad = full_grad[0, t]
        diff = curr_grad - prev_grad
        diff_grads.append(diff)
        prev_grad = curr_grad

    return diff_grads


def compute_positive_gradient_accumulation(model, params, x, target_class, seq_len):
    """
    Accumulate only POSITIVE gradient contributions.

    This shows where evidence FOR the prediction accumulates,
    filtering out negative/inhibitory gradients.

    For contour tracing, positive gradients along the path
    should accumulate over time.
    """
    def loss_fn(x_input):
        logits = model.apply({'params': params}, x_input, training=False)
        return logits[0, target_class]

    full_grad = jax.grad(loss_fn)(x)

    cumulative_pos = []
    running_sum = jnp.zeros_like(full_grad[0, 0])

    for t in range(seq_len):
        # Only accumulate positive gradients
        pos_grad = jnp.maximum(full_grad[0, t], 0)
        running_sum = running_sum + pos_grad
        cumulative_pos.append(running_sum.copy())

    return cumulative_pos


def normalize_for_viz(arr, percentile=99):
    """Normalize array for visualization with percentile clipping."""
    arr = np.array(arr)

    # Sum across channels if needed
    if len(arr.shape) == 3:
        arr = np.abs(arr).sum(axis=-1)

    # Percentile clipping for robustness
    vmax = np.percentile(np.abs(arr), percentile)
    if vmax > 1e-8:
        arr = np.clip(arr / vmax, -1, 1)

    return arr


def create_overlay(image, heatmap, alpha=0.6, cmap='hot', diverging=False):
    """Create image with heatmap overlay."""
    # Denormalize image
    image_denorm = image * IMAGENET_STD + IMAGENET_MEAN
    image_denorm = np.clip(image_denorm, 0, 1)

    # Normalize heatmap
    heatmap_norm = normalize_for_viz(heatmap)

    # Apply colormap
    if diverging:
        # For difference maps: blue (negative) to red (positive)
        colormap = plt.colormaps.get_cmap('RdBu_r')
        heatmap_colored = colormap((heatmap_norm + 1) / 2)[:, :, :3]
    else:
        colormap = plt.colormaps.get_cmap(cmap)
        heatmap_colored = colormap(np.abs(heatmap_norm))[:, :, :3]

    # Blend
    overlay = (1 - alpha) * image_denorm + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    return overlay, heatmap_norm


def create_gif(frames, output_path, duration=300):
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
    """Generate visualization for a single sample using specified method."""
    x = jnp.array(image)[jnp.newaxis]

    # Get prediction
    logits = model.apply({'params': params}, x, training=False)
    pred = int(jnp.argmax(logits, axis=-1)[0])
    correct = pred == label
    target_class = pred

    # Compute attribution based on method
    if args.method == 'gradient_accum':
        attributions = compute_gradient_accumulation(model, params, x, target_class, seq_len)
        title_prefix = "Cumulative Gradient"
        diverging = False

    elif args.method == 'gradient_diff':
        attributions = compute_temporal_gradient_diff(model, params, x, target_class, seq_len)
        title_prefix = "Gradient Δ (t - t-1)"
        diverging = True

    elif args.method == 'positive_accum':
        attributions = compute_positive_gradient_accumulation(model, params, x, target_class, seq_len)
        title_prefix = "Positive Gradient Accum"
        diverging = False

    elif args.method == 'saliency':
        saliency = compute_gradient_times_input(model, params, x, target_class)
        # Split into timesteps
        attributions = [saliency[0, t] for t in range(seq_len)]
        title_prefix = "Gradient × Input"
        diverging = False

    elif args.method == 'integrated':
        integrated = compute_integrated_gradients(model, params, x, target_class, steps=30)
        attributions = [integrated[0, t] for t in range(seq_len)]
        title_prefix = "Integrated Gradients"
        diverging = False

    else:  # default: raw gradient
        def loss_fn(x_in):
            return model.apply({'params': params}, x_in, training=False)[0, target_class]
        full_grad = jax.grad(loss_fn)(x)
        attributions = [full_grad[0, t] for t in range(seq_len)]
        title_prefix = "Raw Gradient"
        diverging = False

    # Create frames
    frames = []
    base_image = np.array(image[0])

    for t, attr in enumerate(attributions):
        attr_np = np.array(attr)
        overlay, heatmap_norm = create_overlay(base_image, attr_np, alpha=0.6,
                                                cmap='hot', diverging=diverging)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        img_denorm = base_image * IMAGENET_STD + IMAGENET_MEAN
        img_denorm = np.clip(img_denorm, 0, 1)
        axes[0].imshow(img_denorm)
        axes[0].set_title(f'Input ({"Connected" if label == 1 else "Disconnected"})')
        axes[0].axis('off')

        # Heatmap only
        if diverging:
            im = axes[1].imshow(heatmap_norm, cmap='RdBu_r', vmin=-1, vmax=1)
        else:
            im = axes[1].imshow(np.abs(heatmap_norm), cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f'{title_prefix} T={t}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f'Pred: {"Connected" if pred == 1 else "Disconnected"} {"✓" if correct else "✗"}')
        axes[2].axis('off')

        plt.tight_layout()

        # Convert to numpy
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3].copy() / 255.0
        frames.append(frame)
        plt.close(fig)

    # Save GIF
    status = "correct" if correct else "wrong"
    label_str = "connected" if label == 1 else "disconnected"
    gif_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_{args.method}.gif"
    create_gif(frames, gif_path, duration=args.frame_duration)

    # Save final frame
    png_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_{args.method}_final.png"
    Image.fromarray((frames[-1] * 255).astype(np.uint8)).save(png_path)

    return correct, pred, label


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


def main():
    parser = argparse.ArgumentParser(description='Visualize CSSM states on Pathfinder')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='visualizations/cssm_states')

    # Visualization method
    parser.add_argument('--method', type=str, default='gradient_accum',
                        choices=['gradient_accum', 'gradient_diff', 'positive_accum',
                                 'saliency', 'integrated', 'raw'],
                        help='Attribution method: '
                             'gradient_accum (cumulative), '
                             'gradient_diff (temporal difference), '
                             'positive_accum (positive only), '
                             'saliency (grad × input), '
                             'integrated (integrated gradients), '
                             'raw (standard gradient)')

    # Model architecture
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
    parser.add_argument('--num_samples', type=int, default=20)

    # Visualization
    parser.add_argument('--frame_duration', type=int, default=400)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {args.method}")
    print(f"Output: {output_dir}")

    # Load model
    print("Loading model...")
    model, params = load_model_and_checkpoint(args.checkpoint, args)

    # Load data
    print("Loading test data...")
    _, _, test_dataset = get_pathfinder_datasets(
        root=args.data_dir,
        difficulty=args.difficulty,
        image_size=args.image_size,
        num_frames=args.seq_len,
    )

    print(f"Test set: {len(test_dataset)} samples")

    # Select samples
    num_samples = min(args.num_samples, len(test_dataset))
    np.random.seed(42)  # Reproducible
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    # Visualize
    print(f"Generating {num_samples} visualizations...")
    correct_count = 0

    for i, idx in enumerate(tqdm(indices)):
        image, label = test_dataset[idx]
        correct, pred, label = visualize_sample(
            model, params, image, label, i, output_dir, args.seq_len, args
        )
        if correct:
            correct_count += 1

    print(f"\nAccuracy: {correct_count}/{num_samples} ({100*correct_count/num_samples:.1f}%)")
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    main()
