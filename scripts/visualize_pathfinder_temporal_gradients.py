#!/usr/bin/env python3
"""
Visualize per-timestep gradient evolution on Pathfinder.

Two visualization modes:
1. d(logit_connected_t) / d(image) - What input drives decision at timestep t
2. d(logit_t - logit_{t-1}) / d(image) - What input drives the CHANGE at timestep t

This reveals how the model's attention evolves over temporal recurrence.

Usage:
    python scripts/visualize_pathfinder_temporal_gradients.py \
        --checkpoint checkpoints/pf9_vit_opponent_d1_e32/epoch_25 \
        --output_dir visualizations/pf9_temporal_grads \
        --num_samples 10
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


def get_per_timestep_logits(model, params, x):
    """
    Get classification logits at each timestep by applying GAP+readout to features at each t.

    Returns:
        logits_per_t: (T,) array of logit[connected] at each timestep
        final_pred: final prediction (0 or 1)
    """
    # Get per-pixel logits at all timesteps
    final_logits, perpixel_logits = model.apply(
        {'params': params}, x, training=False, return_spatial=True
    )
    # perpixel_logits: (1, T, H', W', 2)

    # Apply LogSumExp pooling (same as the model's readout) per timestep
    # This simulates what the model would output if we read out at each timestep
    T = perpixel_logits.shape[1]
    logits_per_t = []

    for t in range(T):
        # Get logits at timestep t: (1, H', W', 2)
        logits_t = perpixel_logits[:, t, :, :, :]
        # Flatten spatial dims: (1, H'*W', 2)
        logits_flat = logits_t.reshape(1, -1, 2)
        # LogSumExp pool: (1, 2)
        pooled = jax.scipy.special.logsumexp(logits_flat, axis=1)
        # Get logit for connected class
        logit_conn = pooled[0, 1]
        logits_per_t.append(logit_conn)

    logits_per_t = jnp.stack(logits_per_t)  # (T,)
    final_pred = int(jnp.argmax(final_logits, axis=-1)[0])

    return logits_per_t, final_pred


def compute_timestep_gradients(model, params, x, seq_len, target_class):
    """
    Compute d(logit_target_t) / d(image) for each timestep t.

    Args:
        target_class: 0 for disconnected, 1 for connected (use GT label)

    Returns:
        grads: list of (H, W, C) gradients, one per timestep
        logits: (T,) array of logit[target_class] at each timestep
    """
    def logit_at_timestep(x_input, t, cls):
        """Get logit[cls] at timestep t."""
        _, perpixel_logits = model.apply(
            {'params': params}, x_input, training=False, return_spatial=True
        )
        # perpixel_logits: (1, T, H', W', 2)
        logits_t = perpixel_logits[:, t, :, :, :]  # (1, H', W', 2)
        logits_flat = logits_t.reshape(1, -1, 2)
        pooled = jax.scipy.special.logsumexp(logits_flat, axis=1)
        return pooled[0, cls]  # logit for target class

    grads = []
    logits = []

    for t in range(seq_len):
        # Compute gradient of logit_t w.r.t. input
        logit_t, grad_t = jax.value_and_grad(lambda x: logit_at_timestep(x, t, target_class))(x)
        grads.append(np.array(grad_t[0, 0]))  # (H, W, C) - gradient w.r.t. first frame
        logits.append(float(logit_t))

    return grads, np.array(logits)


def compute_delta_gradients(model, params, x, seq_len, target_class):
    """
    Compute d(logit_t - logit_{t-1}) / d(image) for each timestep t.

    This shows what input drives the CHANGE in decision at each timestep.

    Args:
        target_class: 0 for disconnected, 1 for connected (use GT label)

    Returns:
        grads: list of (H, W, C) gradients, one per timestep (t=0 is zeros)
        deltas: (T,) array of logit changes
    """
    def logit_delta_at_timestep(x_input, t, cls):
        """Get logit[cls]_t - logit[cls]_{t-1}."""
        _, perpixel_logits = model.apply(
            {'params': params}, x_input, training=False, return_spatial=True
        )

        def get_pooled_logit(logits_spatial):
            logits_flat = logits_spatial.reshape(1, -1, 2)
            pooled = jax.scipy.special.logsumexp(logits_flat, axis=1)
            return pooled[0, cls]

        logit_t = get_pooled_logit(perpixel_logits[:, t, :, :, :])
        if t == 0:
            return logit_t * 0.0  # No change at t=0
        logit_tm1 = get_pooled_logit(perpixel_logits[:, t-1, :, :, :])
        return logit_t - logit_tm1

    grads = []
    deltas = []

    for t in range(seq_len):
        delta_t, grad_t = jax.value_and_grad(lambda x: logit_delta_at_timestep(x, t, target_class))(x)
        grads.append(np.array(grad_t[0, 0]))  # (H, W, C)
        deltas.append(float(delta_t))

    return grads, np.array(deltas)


def normalize_gradient(grad):
    """Normalize gradient magnitude for visualization."""
    # Sum absolute value across channels
    grad_mag = np.abs(grad).sum(axis=-1)  # (H, W)

    # Normalize to [0, 1]
    vmin, vmax = grad_mag.min(), grad_mag.max()
    if vmax - vmin > 1e-8:
        return (grad_mag - vmin) / (vmax - vmin)
    return np.zeros_like(grad_mag)


def create_gradient_frame(image, gradient, timestep, total_timesteps, logit_val,
                          label, pred, correct, title_prefix, alpha=0.6):
    """Create visualization frame for gradient at a timestep."""
    # Denormalize image
    img_denorm = image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img_denorm = np.clip(img_denorm, 0, 1)

    # Normalize gradient
    grad_norm = normalize_gradient(gradient)

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original image
    axes[0].imshow(img_denorm)
    axes[0].set_title(f'Input (GT: {"Connected" if label == 1 else "Disconnected"})', fontsize=12)
    axes[0].axis('off')

    # Panel 2: Overlay
    cmap = plt.colormaps.get_cmap('hot')
    grad_colored = cmap(grad_norm)[:, :, :3]
    overlay = (1 - alpha) * img_denorm + alpha * grad_colored
    overlay = np.clip(overlay, 0, 1)

    axes[1].imshow(overlay)
    axes[1].set_title(f'{title_prefix} @ T={timestep+1}/{total_timesteps}', fontsize=12)
    axes[1].axis('off')

    # Panel 3: Raw gradient magnitude
    im = axes[2].imshow(grad_norm, cmap='hot', vmin=0, vmax=1)
    status = "✓" if correct else "✗"
    axes[2].set_title(f'Grad magnitude | logit={logit_val:.1f} | Pred: {"C" if pred==1 else "D"} {status}', fontsize=12)
    axes[2].axis('off')

    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frame = buf[:, :, :3].copy()
    plt.close(fig)

    return frame / 255.0


def create_gif(frames, output_path, duration=400):
    """Create GIF from frames."""
    pil_frames = [Image.fromarray((f * 255).astype(np.uint8)) for f in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )


def visualize_sample(model, params, image, label, sample_idx, output_dir, seq_len, args):
    """Generate temporal gradient visualizations for a single sample."""
    x = jnp.array(image)[jnp.newaxis]  # (1, T, H, W, C)

    # Get prediction
    final_logits = model.apply({'params': params}, x, training=False)
    pred = int(jnp.argmax(final_logits, axis=-1)[0])
    correct = pred == label

    base_image = np.array(image[0])
    status = "correct" if correct else "wrong"
    label_str = "connected" if label == 1 else "disconnected"
    class_str = "conn" if label == 1 else "disc"

    # Use GT label as target class for gradients
    target_class = label

    # === Mode 1: d(logit_t) / d(image) ===
    grads_logit, logits = compute_timestep_gradients(model, params, x, seq_len, target_class)

    frames_logit = []
    for t in range(seq_len):
        frame = create_gradient_frame(
            base_image, grads_logit[t], t, seq_len, logits[t],
            label, pred, correct, f"∂logit({class_str})_t/∂image", alpha=0.6
        )
        frames_logit.append(frame)

    gif_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_logit_grad.gif"
    create_gif(frames_logit, gif_path, duration=args.frame_duration)

    # === Mode 2: d(logit_t - logit_{t-1}) / d(image) ===
    grads_delta, deltas = compute_delta_gradients(model, params, x, seq_len, target_class)

    frames_delta = []
    for t in range(seq_len):
        frame = create_gradient_frame(
            base_image, grads_delta[t], t, seq_len, deltas[t],
            label, pred, correct, f"∂Δlogit({class_str})_t/∂image", alpha=0.6
        )
        frames_delta.append(frame)

    gif_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_delta_grad.gif"
    create_gif(frames_delta, gif_path, duration=args.frame_duration)

    # === Summary figure ===
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 0: Original image + logit evolution
    img_denorm = base_image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img_denorm = np.clip(img_denorm, 0, 1)
    axes[0, 0].imshow(img_denorm)
    axes[0, 0].set_title(f'Input (GT: {"Conn" if label==1 else "Disc"})')
    axes[0, 0].axis('off')

    # Logit evolution plot
    axes[0, 1].plot(range(1, seq_len+1), logits, 'b-o', label='logit(conn)')
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('logit(connected)')
    axes[0, 1].set_title('Logit evolution over time')
    axes[0, 1].legend()

    # Delta evolution plot
    axes[0, 2].bar(range(1, seq_len+1), deltas, color='green', alpha=0.7)
    axes[0, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel('Timestep')
    axes[0, 2].set_ylabel('Δlogit')
    axes[0, 2].set_title('Logit change per timestep')

    axes[0, 3].axis('off')

    # Row 1: ∂logit_t/∂image at different timesteps
    timesteps = [0, 2, 4, 7] if seq_len == 8 else [0, seq_len//4, seq_len//2, seq_len-1]
    for i, t in enumerate(timesteps):
        if t < seq_len:
            grad_norm = normalize_gradient(grads_logit[t])
            axes[1, i].imshow(grad_norm, cmap='hot', vmin=0, vmax=1)
            axes[1, i].set_title(f'∂logit/∂img T={t+1}')
            axes[1, i].axis('off')

    # Row 2: ∂Δlogit_t/∂image at different timesteps
    for i, t in enumerate(timesteps):
        if t < seq_len:
            grad_norm = normalize_gradient(grads_delta[t])
            axes[2, i].imshow(grad_norm, cmap='hot', vmin=0, vmax=1)
            axes[2, i].set_title(f'∂Δlogit/∂img T={t+1}')
            axes[2, i].axis('off')

    plt.suptitle(f'Sample {sample_idx}: {"Connected" if label==1 else "Disconnected"} | '
                 f'Pred: {"Connected" if pred==1 else "Disconnected"} {"✓" if correct else "✗"}',
                 fontsize=14)
    plt.tight_layout()

    summary_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    return correct, pred, label, logits, deltas


def main():
    parser = argparse.ArgumentParser(description='Visualize per-timestep gradients on Pathfinder')

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='visualizations/pathfinder_temporal_grads')

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
    parser.add_argument('--frame_duration', type=int, default=500)

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
    np.random.seed(42)
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    print(f"Generating {num_samples} temporal gradient visualizations...")
    correct_count = 0
    all_results = []

    for i, idx in enumerate(tqdm(indices)):
        image, label = test_dataset[idx]

        correct, pred, label, logits, deltas = visualize_sample(
            model, params, image, label, i, output_dir, args.seq_len, args
        )

        if correct:
            correct_count += 1
        all_results.append((label, pred, correct, logits, deltas))

    print(f"\nDone! Accuracy: {correct_count}/{num_samples} ({100*correct_count/num_samples:.1f}%)")
    print(f"GIFs and summaries saved to: {output_dir}")

    # Print statistics
    print("\n--- Logit evolution statistics ---")
    for i, (label, pred, correct, logits, deltas) in enumerate(all_results[:5]):
        status = "✓" if correct else "✗"
        print(f"Sample {i}: GT={'Conn' if label==1 else 'Disc'} {status}")
        print(f"  logits: {logits[0]:.1f} → {logits[3]:.1f} → {logits[-1]:.1f}")
        print(f"  deltas: {deltas[1]:.2f}, {deltas[3]:.2f}, {deltas[-1]:.2f}")


if __name__ == '__main__':
    main()
