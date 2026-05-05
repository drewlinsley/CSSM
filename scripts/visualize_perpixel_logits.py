#!/usr/bin/env python3
"""
Visualize per-pixel class evidence over temporal recurrence.

Instead of gradients, apply the classification readout at every spatial location
to see where the model detects "connected" vs "disconnected" evidence.

This is like dense prediction - treating each spatial position as potentially
contributing to the final classification.

For each timestep t:
1. Run the model up to timestep t
2. Instead of global pooling, keep spatial dimensions
3. Apply readout (1x1 conv style) at each location
4. Visualize logit[correct_class] across space

This should show:
- Where evidence for "connected" accumulates along the path
- How that evidence propagates/spreads over recurrence steps

Usage:
    python scripts/visualize_perpixel_logits.py \
        --checkpoint checkpoints/pf14_vit_hgru_bi_d1_e32/epoch_30 \
        --output_dir visualizations/pf14_perpixel \
        --cssm hgru_bi --difficulty 14
"""

import argparse
import os
from pathlib import Path
from functools import partial
from typing import List, Tuple, Dict, Any, Callable

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

from src.pathfinder_data import get_pathfinder_datasets, IMAGENET_MEAN, IMAGENET_STD
import orbax.checkpoint as ocp


def get_spatial_features_and_head(model, params, x, seq_len):
    """
    Extract spatial features BEFORE global pooling, and get the classification head.

    We need to intercept the forward pass to get features with spatial dimensions.

    Returns:
        features_per_timestep: List of (H', W', C) feature maps for each timestep
        head_params: Parameters of the classification head
    """
    # This requires modifying the model or using a hook-like mechanism
    # For now, we'll use a workaround: manually reconstruct the forward pass

    # The model structure is roughly:
    # 1. Patch embedding (stem)
    # 2. CSSM blocks
    # 3. Global average pooling
    # 4. Layer norm
    # 5. Classification head (Dense)

    # We need to get the features AFTER CSSM blocks but BEFORE pooling
    pass


def create_perpixel_predictor(model, params):
    """
    Create a function that applies the classification head at every spatial location.

    The model's head is a Dense layer: (B, C) -> (B, num_classes)
    We want to apply it as: (B, H, W, C) -> (B, H, W, num_classes)

    This is equivalent to a 1x1 convolution.
    """
    # Extract head parameters
    # The head is typically at params['head'] or params['fc'] or similar
    head_params = None

    # Try different possible locations for the head
    if 'head' in params:
        head_params = params['head']
    elif 'Dense_0' in params:
        head_params = params['Dense_0']

    if head_params is None:
        # Search for the classification head
        def find_head(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    result = find_head(v, f"{prefix}/{k}")
                    if result:
                        return result
                elif 'kernel' in k and prefix.endswith(('head', 'fc', 'classifier')):
                    return d
            return None

        head_params = find_head(params)

    return head_params


def extract_features_with_intermediates(model, params, x, seq_len):
    """
    Run model and extract intermediate features at each CSSM output.

    This requires knowing the model structure. For CSSMViT:
    - stem -> (B, T, H', W', C)
    - blocks (CSSM) -> (B, T, H', W', C)
    - final: pool -> norm -> head

    We want features after blocks but before pooling, for each timestep.
    """
    B, T, H, W, C = x.shape

    # Run full model to get structure
    logits = model.apply({'params': params}, x, training=False)

    # For each timestep, run with truncated input
    features_per_t = []

    for t in range(1, seq_len + 1):
        # Input with first t frames
        x_partial = jnp.concatenate([
            x[:, :t],
            jnp.zeros((B, T - t, H, W, C))
        ], axis=1)

        # We need to get intermediate features
        # Use a modified forward that returns features before pooling
        features_per_t.append(None)  # Placeholder

    return features_per_t


def apply_head_spatially(features, head_kernel, head_bias=None):
    """
    Apply classification head at every spatial location.

    features: (B, H, W, C) or (H, W, C)
    head_kernel: (C, num_classes)
    head_bias: (num_classes,) optional

    Returns: (B, H, W, num_classes) or (H, W, num_classes)
    """
    # This is just a matrix multiply at each spatial location
    # Equivalent to 1x1 conv
    logits = jnp.einsum('...c,cd->...d', features, head_kernel)
    if head_bias is not None:
        logits = logits + head_bias
    return logits


def get_intermediate_features_by_modification(model_apply_fn, params, x, seq_len):
    """
    Get intermediate features by creating a modified forward pass.

    Since we can't easily hook into Flax models, we'll compute features
    using gradient-based approach: the gradient of each spatial output
    w.r.t. the head input shows us the spatial contribution.

    Alternative: Manually reconstruct the forward pass steps.
    """
    pass


class FeatureExtractor(nn.Module):
    """
    Wrapper that extracts features before the final pooling.

    We manually implement the forward pass, stopping before global pooling.
    """
    base_config: Dict[str, Any]

    @nn.compact
    def __call__(self, x, training=False):
        from src.models.cssm_vit import Stem, CSSMBlock

        B, T, H, W, C = x.shape

        # Stem
        if self.base_config.get('stem_mode') == 'conv':
            # Conv stem
            x = nn.Conv(self.base_config['embed_dim'], (7, 7), strides=(4, 4),
                       padding='SAME', name='stem_conv')(x.reshape(B*T, H, W, C))
            x = nn.LayerNorm(name='stem_norm')(x)
            x = nn.gelu(x)
            # Reshape back
            _, H_new, W_new, C_new = x.shape
            x = x.reshape(B, T, H_new, W_new, C_new)
        else:
            # Patch embed
            patch_size = self.base_config.get('patch_size', 16)
            x = nn.Conv(self.base_config['embed_dim'],
                       (patch_size, patch_size),
                       strides=(patch_size, patch_size),
                       name='patch_embed')(x.reshape(B*T, H, W, C))
            _, H_new, W_new, C_new = x.shape
            x = x.reshape(B, T, H_new, W_new, C_new)

        # Return features with spatial dimensions intact
        # These are the features we want to apply the head to
        return x


def compute_perpixel_evidence_via_gradient(model, params, x, seq_len):
    """
    Compute per-pixel evidence by looking at how each spatial location
    contributes to the final logits.

    Method: For each spatial location (i, j), compute:
    d(logit[class]) / d(feature[i, j])

    This tells us how much each location contributes to each class prediction.

    Then multiply by the feature value to get the actual contribution.
    """
    B, T, H, W, C = x.shape

    def get_class_logit(x_input, target_class):
        logits = model.apply({'params': params}, x_input, training=False)
        return logits[0, target_class]

    # Get prediction
    logits = model.apply({'params': params}, x, training=False)
    pred = int(jnp.argmax(logits, axis=-1)[0])

    # For each timestep, compute gradients w.r.t. that timestep's input
    perpixel_evidence = []

    for t in range(seq_len):
        # Create mask that isolates timestep t
        def masked_forward(x_input, t_idx=t):
            # Zero out all timesteps except t_idx
            mask = jnp.zeros((1, seq_len, 1, 1, 1))
            mask = mask.at[0, t_idx, :, :, :].set(1.0)
            x_masked = x_input * mask

            # Add back the full input except timestep t
            full_mask = jnp.ones((1, seq_len, 1, 1, 1))
            full_mask = full_mask.at[0, t_idx, :, :, :].set(0.0)
            x_combined = x_input * full_mask + x_masked

            return model.apply({'params': params}, x_input, training=False)[0, pred]

        # Compute gradient w.r.t. input at timestep t
        grad_fn = jax.grad(lambda x_in: get_class_logit(x_in, pred))
        grad = grad_fn(x)  # (B, T, H, W, C)

        # Extract gradient for timestep t and compute magnitude
        grad_t = grad[0, t]  # (H, W, C)
        evidence_t = jnp.abs(grad_t).sum(axis=-1)  # (H, W)

        perpixel_evidence.append(evidence_t)

    return perpixel_evidence, pred, logits[0]


def compute_perpixel_logits_via_head_application(model, params, x, seq_len):
    """
    More direct approach: Apply the classification head to feature maps
    at each spatial location.

    This requires knowing the model structure to extract:
    1. Features before global pooling (H', W', C)
    2. The classification head weights

    For each timestep:
    - Get spatial features (before pooling)
    - Apply head at each location: features @ head_weights
    - This gives logits at each spatial position
    """
    # We need to manually extract features and head
    # This depends on the specific model architecture

    # For CSSMViT, the structure is:
    # 1. Stem (patch embed or conv stem)
    # 2. Position embedding (optional)
    # 3. CSSM blocks (depth layers)
    # 4. Final temporal mean (across T)
    # 5. Global spatial mean
    # 6. LayerNorm
    # 7. Head (Dense)

    # We want features AFTER step 4 (temporal mean) but BEFORE step 5 (spatial mean)

    # The challenge is Flax doesn't easily expose intermediates
    # We'll use a reconstruction approach

    pass


def visualize_gradient_evidence(model, params, image, label, sample_idx, output_dir, seq_len, args):
    """
    Visualize per-pixel gradient-based evidence for each class.
    """
    x = jnp.array(image)[jnp.newaxis]
    B, T, H, W, C = x.shape

    # Get prediction
    logits = model.apply({'params': params}, x, training=False)
    pred = int(jnp.argmax(logits, axis=-1)[0])
    correct = pred == label

    # Compute per-pixel evidence for the CORRECT class
    # This shows where evidence for the ground truth accumulates
    target_class = label  # Use ground truth, not prediction

    def get_logit(x_input):
        return model.apply({'params': params}, x_input, training=False)[0, target_class]

    # Compute gradient w.r.t. input
    grad = jax.grad(get_logit)(x)  # (B, T, H, W, C)

    # For each timestep, show cumulative evidence
    frames = []
    base_image = np.array(image[0])

    cumulative_evidence = np.zeros((H, W))

    for t in range(seq_len):
        grad_t = np.array(grad[0, t])  # (H, W, C)

        # Evidence = gradient magnitude (shows where model is sensitive)
        evidence_t = np.abs(grad_t).sum(axis=-1)

        # Accumulate evidence over time
        cumulative_evidence = cumulative_evidence + evidence_t

        # Normalize for visualization
        vmax = np.percentile(cumulative_evidence, 99) + 1e-8
        evidence_norm = np.clip(cumulative_evidence / vmax, 0, 1)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        img_denorm = base_image * IMAGENET_STD + IMAGENET_MEAN
        img_denorm = np.clip(img_denorm, 0, 1)
        axes[0].imshow(img_denorm)
        axes[0].set_title(f'Input (GT: {"Connected" if label == 1 else "Disconnected"})')
        axes[0].axis('off')

        # Cumulative evidence heatmap
        im = axes[1].imshow(evidence_norm, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f'Cumulative Evidence for GT Class (T=0:{t})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Overlay
        colormap = plt.colormaps.get_cmap('hot')
        evidence_colored = colormap(evidence_norm)[:, :, :3]
        overlay = 0.4 * img_denorm + 0.6 * evidence_colored
        overlay = np.clip(overlay, 0, 1)

        axes[2].imshow(overlay)
        pred_str = "Connected" if pred == 1 else "Disconnected"
        axes[2].set_title(f'Pred: {pred_str} {"✓" if correct else "✗"} | Logits: [{logits[0,0]:.2f}, {logits[0,1]:.2f}]')
        axes[2].axis('off')

        plt.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3].copy() / 255.0
        frames.append(frame)
        plt.close(fig)

    # Save GIF
    status = "correct" if correct else "wrong"
    label_str = "connected" if label == 1 else "disconnected"

    gif_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_evidence.gif"

    pil_frames = [Image.fromarray((f * 255).astype(np.uint8)) for f in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                       duration=args.frame_duration, loop=0)

    # Save final
    png_path = output_dir / f"sample_{sample_idx:03d}_{label_str}_{status}_evidence_final.png"
    Image.fromarray((frames[-1] * 255).astype(np.uint8)).save(png_path)

    return correct, pred, label


def load_model_and_checkpoint(checkpoint_path: str, args):
    """Load model and checkpoint."""
    from src.models.cssm_vit import CSSMViT

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
    dummy = jnp.ones((1, args.seq_len, args.image_size, args.image_size, 3))
    model.init({'params': rng, 'dropout': rng}, dummy, training=False)

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(os.path.abspath(checkpoint_path))

    if hasattr(restored, 'params'):
        params = restored.params
    else:
        params = restored['params']

    print(f"Loaded: {checkpoint_path}")
    return model, params


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='visualizations/perpixel')

    # Model
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
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--frame_duration', type=int, default=400)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, params = load_model_and_checkpoint(args.checkpoint, args)

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
        correct, pred, label = visualize_gradient_evidence(
            model, params, image, label, i, output_dir, args.seq_len, args
        )
        if correct:
            correct_count += 1

    print(f"\nAccuracy: {correct_count}/{len(indices)} ({100*correct_count/len(indices):.1f}%)")
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    main()
