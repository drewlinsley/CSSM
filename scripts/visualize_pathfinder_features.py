#!/usr/bin/env python3
"""
Visualize spatial feature evolution on Pathfinder.

Shows how the model's internal features evolve spatially over temporal recurrence.
Instead of gradients, this shows the actual feature activations before GAP pooling.

Usage:
    python scripts/visualize_pathfinder_features.py \
        --checkpoint checkpoints/pf9_vit_opponent_d1_e32/epoch_15 \
        --output_dir visualizations/pf9_features \
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
    model.init({'params': rng, 'dropout': rng}, dummy_input, training=False)

    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_path = os.path.abspath(checkpoint_path)
    restored = checkpointer.restore(checkpoint_path)

    if isinstance(restored, dict) and 'params' in restored:
        params = restored['params']
        epoch = restored.get('epoch', 'unknown')
    else:
        params = restored.params
        epoch = getattr(restored, 'epoch', 'unknown')

    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    return model, params


def get_spatial_features(model, params, x):
    """
    Get spatial features at each timestep before GAP pooling.

    Returns:
        perpixel_logits: (B, T, H', W', 2) - per-pixel class logits
        features_norm: (B, T, H', W') - L2 norm of features at each position
    """
    final_logits, perpixel_logits = model.apply(
        {'params': params}, x, training=False, return_spatial=True
    )

    # perpixel_logits shape: (B, T, H', W', num_classes)
    # Compute feature "energy" as the norm or max logit
    logit_diff = perpixel_logits[..., 1] - perpixel_logits[..., 0]  # (B, T, H', W')

    return perpixel_logits, logit_diff, final_logits


def create_feature_frame(image, feature_map, timestep, total_timesteps,
                         label, pred, correct, vmin, vmax, alpha=0.6):
    """Create visualization frame for feature map at a timestep."""
    # Denormalize image
    img_denorm = image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img_denorm = np.clip(img_denorm, 0, 1)

    H_img, W_img = image.shape[0], image.shape[1]
    H_feat, W_feat = feature_map.shape

    # Resize feature map to image size
    if H_feat != H_img or W_feat != W_img:
        from scipy.ndimage import zoom
        scale_h = H_img / H_feat
        scale_w = W_img / W_feat
        feature_resized = zoom(feature_map, (scale_h, scale_w), order=1)
    else:
        feature_resized = feature_map

    # Normalize for colormap
    feature_norm = (feature_resized - vmin) / (vmax - vmin + 1e-8)
    feature_norm = np.clip(feature_norm, 0, 1)

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original image
    axes[0].imshow(img_denorm)
    axes[0].set_title(f'Input (GT: {"Connected" if label == 1 else "Disconnected"})', fontsize=12)
    axes[0].axis('off')

    # Panel 2: Overlay
    cmap = plt.colormaps.get_cmap('hot')
    feature_colored = cmap(feature_norm)[:, :, :3]
    overlay = (1 - alpha) * img_denorm + alpha * feature_colored
    overlay = np.clip(overlay, 0, 1)

    axes[1].imshow(overlay)
    axes[1].set_title(f'logit(conn)-logit(disc) @ T={timestep+1}/{total_timesteps}', fontsize=12)
    axes[1].axis('off')

    # Panel 3: Raw feature map with colorbar
    im = axes[2].imshow(feature_resized, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    status = "✓" if correct else "✗"
    axes[2].set_title(f'Pred: {"Conn" if pred==1 else "Disc"} {status}', fontsize=12)
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
    """Generate feature visualization for a single sample."""
    x = jnp.array(image)[jnp.newaxis]

    # Get spatial features
    perpixel_logits, logit_diff, final_logits = get_spatial_features(model, params, x)

    pred = int(jnp.argmax(final_logits, axis=-1)[0])
    correct = pred == label

    # logit_diff: (1, T, H', W')
    logit_diff_np = np.array(logit_diff[0])  # (T, H', W')

    # Compute delta from T=0
    delta_logit = logit_diff_np - logit_diff_np[0:1]  # (T, H', W')

    base_image = np.array(image[0])
    status = "correct" if correct else "wrong"
    label_str = "connected" if label == 1 else "disconnected"

    # === Absolute logit_diff visualization ===
    vmin = logit_diff_np.min()
    vmax = logit_diff_np.max()

    frames_abs = []
    for t in range(seq_len):
        frame = create_feature_frame(
            base_image, logit_diff_np[t], t, seq_len,
            label, pred, correct, vmin, vmax, alpha=0.6
        )
        frames_abs.append(frame)

    gif_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_logit_diff.gif"
    create_gif(frames_abs, gif_path, duration=args.frame_duration)

    # === Delta logit_diff from T=0 visualization ===
    delta_vmax = max(abs(delta_logit.min()), abs(delta_logit.max()), 1.0)

    frames_delta = []
    for t in range(seq_len):
        frame = create_feature_frame(
            base_image, delta_logit[t], t, seq_len,
            label, pred, correct, -delta_vmax, delta_vmax, alpha=0.6
        )
        frames_delta.append(frame)

    gif_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_delta_logit.gif"
    create_gif(frames_delta, gif_path, duration=args.frame_duration)

    # === Summary figure ===
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    img_denorm = base_image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img_denorm = np.clip(img_denorm, 0, 1)

    # Row 0: logit_diff at different timesteps
    axes[0, 0].imshow(img_denorm)
    axes[0, 0].set_title(f'Input (GT: {"C" if label==1 else "D"})')
    axes[0, 0].axis('off')

    timesteps = [0, 1, 3, 5, 7] if seq_len == 8 else [0, 1, seq_len//2, seq_len*3//4, seq_len-1]
    for i, t in enumerate(timesteps[:4]):
        if t < seq_len:
            im = axes[0, i+1].imshow(logit_diff_np[t], cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[0, i+1].set_title(f'logit_diff T={t+1}')
            axes[0, i+1].axis('off')

    # Row 1: delta from T=0
    axes[1, 0].axis('off')
    for i, t in enumerate(timesteps[:4]):
        if t < seq_len:
            im = axes[1, i+1].imshow(delta_logit[t], cmap='RdBu_r', vmin=-delta_vmax, vmax=delta_vmax)
            axes[1, i+1].set_title(f'Δ from T=0 @ T={t+1}')
            axes[1, i+1].axis('off')

    plt.suptitle(f'Sample {sample_idx}: {"Connected" if label==1 else "Disconnected"} | '
                 f'Pred: {"Connected" if pred==1 else "Disconnected"} {"✓" if correct else "✗"}',
                 fontsize=14)
    plt.tight_layout()

    summary_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Print stats
    print(f"  Sample {sample_idx}: logit_diff range [{logit_diff_np.min():.1f}, {logit_diff_np.max():.1f}], "
          f"delta range [{delta_logit.min():.1f}, {delta_logit.max():.1f}]")

    return correct, pred, label, logit_diff_np, delta_logit


def main():
    parser = argparse.ArgumentParser(description='Visualize spatial features on Pathfinder')

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='visualizations/pathfinder_features')

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

    print(f"Generating {num_samples} feature visualizations...")
    correct_count = 0

    for i, idx in enumerate(tqdm(indices)):
        image, label = test_dataset[idx]
        correct, pred, label, logit_diff, delta_logit = visualize_sample(
            model, params, image, label, i, output_dir, args.seq_len, args
        )
        if correct:
            correct_count += 1

    print(f"\nDone! Accuracy: {correct_count}/{num_samples} ({100*correct_count/num_samples:.1f}%)")
    print(f"GIFs and summaries saved to: {output_dir}")


if __name__ == '__main__':
    main()
