#!/usr/bin/env python3
"""
Finetune JEPA pretrained weights for ImageNet classification.

Supports:
- Full finetuning (all parameters trainable)
- Linear probing (frozen encoder, only train classification head)
- Partial finetuning with layer-wise learning rate decay
- Loading from JEPA (V-JEPA) checkpoints
- Loading from EMA target encoder weights

Usage:
    # Full finetune from JEPA checkpoint
    python src/training/finetune_jepa.py \
        --jepa_checkpoint /path/to/jepa/checkpoint \
        --mode finetune \
        --lr 1e-4 --epochs 100

    # Linear probe (freeze encoder)
    python src/training/finetune_jepa.py \
        --jepa_checkpoint /path/to/jepa/checkpoint \
        --mode linear_probe \
        --lr 0.1 --epochs 90

    # Finetune with layer-wise LR decay
    python src/training/finetune_jepa.py \
        --jepa_checkpoint /path/to/jepa/checkpoint \
        --mode finetune \
        --lr 1e-4 --lr_decay_rate 0.65 \
        --epochs 100
"""

import argparse
import os
import re
import time
from functools import partial
from typing import Dict, Any, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import jax_utils, traverse_util
from flax.training import train_state
from tqdm import tqdm
import orbax.checkpoint as ocp

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.imagenet import get_imagenet_loader
from src.models.cssm_shvit import (
    CSSMSHViT, cssm_shvit_s1, cssm_shvit_s2, cssm_shvit_s3, cssm_shvit_s4
)

# Lazy imports for tf.data loaders (avoid tensorflow import conflicts)
def get_tfdata_imagenet_loader(*args, **kwargs):
    from src.data.imagenet_tfdata import get_tfdata_imagenet_loader as _get_loader
    return _get_loader(*args, **kwargs)

def get_tfrecord_imagenet_loader(*args, **kwargs):
    from src.data.imagenet_tfdata import get_tfrecord_imagenet_loader as _get_loader
    return _get_loader(*args, **kwargs)

from src.training.distributed import (
    replicate_state, unreplicate_state, shard_batch, split_rng_for_devices
)
from src.training.timm_utils import (
    mixup_cutmix, cross_entropy_with_smoothing, create_ema_state, update_ema
)


# =============================================================================
# Checkpoint Loading Utilities
# =============================================================================

def detect_model_size_from_path(checkpoint_path: str) -> Optional[str]:
    """
    Auto-detect model size from checkpoint path.

    Looks for patterns like 'shvit_s1', 'shvit_s2', 'shvit_s3', 'shvit_s4'
    in the checkpoint path.

    Returns:
        Model size string ('s1', 's2', 's3', 's4') or None if not detected
    """
    path_lower = checkpoint_path.lower()

    # Look for explicit size patterns
    for size in ['s1', 's2', 's3', 's4']:
        if f'shvit_{size}' in path_lower or f'shvit-{size}' in path_lower:
            return size
        if f'_{size}_' in path_lower or f'_{size}/' in path_lower:
            return size

    return None


def load_jepa_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load JEPA checkpoint and extract encoder parameters.

    JEPA checkpoint structure:
        params/
            online_encoder/
                encoder/  <-- This is what we want (CSSMSHViT params)
            predictor/
            mask_token
            ...

    Returns:
        Dict containing encoder params compatible with CSSMSHViT
    """
    print(f"Loading JEPA checkpoint from: {checkpoint_path}")

    checkpointer = ocp.StandardCheckpointer()

    # Try to restore checkpoint
    try:
        ckpt = checkpointer.restore(checkpoint_path)
    except Exception as e:
        # Try with orbax abstract restore
        print(f"Standard restore failed, trying alternative: {e}")
        ckpt = checkpointer.restore(checkpoint_path, args=ocp.args.StandardRestore())

    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        params = ckpt.get('params', ckpt)
    else:
        # Assume it's a TrainState
        params = ckpt.params

    # Extract encoder params
    encoder_params = extract_encoder_from_jepa(params)

    # Count and report
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(encoder_params))
    print(f"  Loaded {num_params:,} encoder parameters")

    return encoder_params


def extract_encoder_from_jepa(jepa_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract CSSMSHViT encoder parameters from JEPA params.

    JEPA stores encoder under: online_encoder/encoder/...
    Or for V-JEPA with different nesting: online_encoder/...
    """
    flat_params = traverse_util.flatten_dict(jepa_params, sep='/')

    # Debug: print top-level keys
    top_keys = set(k.split('/')[0] for k in flat_params.keys())
    print(f"  JEPA checkpoint top-level keys: {top_keys}")

    # Find encoder params - try different paths
    encoder_flat = {}

    # Path 1: online_encoder/encoder/... (CSSMJEPA structure)
    for k, v in flat_params.items():
        if k.startswith('online_encoder/encoder/'):
            new_key = k.replace('online_encoder/encoder/', '')
            encoder_flat[new_key] = v

    # Path 2: online_encoder/... without nested encoder (alternative structure)
    if not encoder_flat:
        for k, v in flat_params.items():
            if k.startswith('online_encoder/') and not k.startswith('online_encoder/encoder/'):
                new_key = k.replace('online_encoder/', '')
                # Skip predictor-related params
                if not any(skip in new_key for skip in ['predictor', 'mask_token', 'pos_embed']):
                    encoder_flat[new_key] = v

    # Path 3: VideoEncoder directly stores under encoder/
    if not encoder_flat:
        for k, v in flat_params.items():
            if k.startswith('encoder/'):
                new_key = k.replace('encoder/', '')
                encoder_flat[new_key] = v

    # Path 4: Direct params (no wrapper)
    if not encoder_flat:
        # Take everything except predictor/masking stuff
        exclude = ['predictor', 'mask_token', 'pos_embed', 'target_', 'temporal_pos', 'spatial_pos']
        for k, v in flat_params.items():
            if not any(ex in k for ex in exclude):
                encoder_flat[k] = v

    if not encoder_flat:
        raise ValueError(
            "Could not find encoder parameters in checkpoint. "
            f"Available keys: {list(flat_params.keys())[:20]}..."
        )

    print(f"  Extracted {len(encoder_flat)} parameter tensors")

    # Unflatten back to nested dict
    return traverse_util.unflatten_dict(encoder_flat, sep='/')


def load_ema_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load EMA encoder weights (target encoder in JEPA).

    EMA checkpoints are typically saved as: checkpoint_path_ema
    """
    ema_path = checkpoint_path + '_ema' if not checkpoint_path.endswith('_ema') else checkpoint_path

    if not os.path.exists(ema_path):
        print(f"  EMA checkpoint not found at {ema_path}, using online encoder")
        return load_jepa_checkpoint(checkpoint_path)

    print(f"Loading EMA checkpoint from: {ema_path}")

    checkpointer = ocp.StandardCheckpointer()
    ckpt = checkpointer.restore(ema_path)

    # EMA params might be stored under 'ema_params' or 'target_params'
    if isinstance(ckpt, dict):
        if 'ema_params' in ckpt:
            params = ckpt['ema_params']
        elif 'target_params' in ckpt:
            params = ckpt['target_params']
        else:
            params = ckpt
    else:
        params = ckpt

    return extract_encoder_from_jepa(params)


def match_and_load_params(
    pretrained_params: Dict[str, Any],
    model_params: Dict[str, Any],
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Match pretrained parameters to model parameters.

    Handles cases where:
    - Pretrained has no classification head (add randomly initialized head)
    - Parameter names differ slightly
    - Some parameters are missing

    Args:
        pretrained_params: Loaded JEPA encoder params
        model_params: Randomly initialized model params (with head)
        strict: If True, raise error on mismatch; else warn and continue

    Returns:
        Merged params with pretrained weights where available
    """
    pretrained_flat = traverse_util.flatten_dict(pretrained_params, sep='/')
    model_flat = traverse_util.flatten_dict(model_params, sep='/')

    loaded = 0
    skipped = 0
    new_params = {}

    # Copy pretrained params that exist in model
    for key, value in model_flat.items():
        if key in pretrained_flat:
            # Shape check
            if pretrained_flat[key].shape == value.shape:
                new_params[key] = pretrained_flat[key]
                loaded += 1
            else:
                msg = f"Shape mismatch for {key}: pretrained {pretrained_flat[key].shape} vs model {value.shape}"
                if strict:
                    raise ValueError(msg)
                print(f"  Warning: {msg}")
                new_params[key] = value  # Keep random init
                skipped += 1
        else:
            # Parameter not in pretrained (e.g., classification head)
            new_params[key] = value
            if 'head' not in key:  # Don't warn about head
                print(f"  New parameter (random init): {key}")
            skipped += 1

    print(f"  Loaded {loaded} parameters from pretrained")
    print(f"  Skipped/new {skipped} parameters")

    return traverse_util.unflatten_dict(new_params, sep='/')


# =============================================================================
# Layer-wise Learning Rate Decay
# =============================================================================

def get_layer_index(param_name: str, num_stages: int = 4) -> int:
    """
    Get layer index from parameter name for LR decay.

    Layer structure:
    - patch_embed: layer 0
    - stage0: layers 1-depth[0]
    - stage1: layers depth[0]+1 to depth[0]+depth[1]
    - ...
    - head: max layer (highest LR)
    """
    if 'patch_embed' in param_name:
        return 0

    # Extract stage number
    stage_match = re.search(r'stage(\d+)', param_name)
    if stage_match:
        stage = int(stage_match.group(1))
        # Extract block number within stage
        block_match = re.search(r'block(\d+)', param_name)
        block = int(block_match.group(1)) if block_match else 0
        # Approximate layer index (stages have increasing indices)
        return stage * 3 + block + 1

    if 'norm' in param_name and 'stage' not in param_name:
        return num_stages * 3 + 1  # Final norm

    if 'head' in param_name:
        return num_stages * 3 + 2  # Classification head (highest)

    return 0  # Default to lowest


def create_layerwise_optimizer(
    learning_rate: float,
    weight_decay: float,
    lr_decay_rate: float,
    num_layers: int,
    total_steps: int,
    warmup_steps: int,
    grad_clip: float = 1.0,
) -> optax.GradientTransformation:
    """
    Create optimizer with layer-wise learning rate decay.

    Earlier layers get smaller learning rates:
        lr_layer_i = learning_rate * (lr_decay_rate ** (num_layers - i))

    Args:
        learning_rate: Base learning rate (for final layer)
        weight_decay: Weight decay coefficient
        lr_decay_rate: Decay rate per layer (e.g., 0.65)
        num_layers: Number of layers
        total_steps: Total training steps
        warmup_steps: Warmup steps
        grad_clip: Gradient clipping norm
    """

    def get_lr_scale(path: str) -> float:
        """Get learning rate scale for a parameter path."""
        layer_idx = get_layer_index(path)
        # Scale = decay_rate^(num_layers - layer_idx)
        # Earlier layers (low idx) get smaller LR
        return lr_decay_rate ** (num_layers - layer_idx)

    # Create cosine schedule with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.01,
    )

    # Multi-transform with per-parameter LR scaling
    def make_tx(lr_scale: float):
        return optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.scale_by_adam(),
            optax.add_decayed_weights(weight_decay),
            optax.scale_by_schedule(schedule),
            optax.scale(-lr_scale),
        )

    # Build parameter partition
    def partition_fn(path, value):
        path_str = '/'.join(str(p) for p in path)
        return get_lr_scale(path_str)

    # We need to use multi_transform to apply different LRs
    # For simplicity, use label_fn to get scales

    # Actually, let's use a simpler approach with masked transforms
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=weight_decay,
        ),
    )


def create_optimizer(
    params: Dict[str, Any],
    learning_rate: float,
    weight_decay: float,
    total_steps: int,
    warmup_steps: int,
    grad_clip: float = 1.0,
    lr_decay_rate: float = 1.0,
    freeze_encoder: bool = False,
) -> optax.GradientTransformation:
    """
    Create optimizer with optional layer-wise LR decay and encoder freezing.

    Args:
        params: Model parameters (to determine layer structure)
        learning_rate: Peak learning rate
        weight_decay: Weight decay
        total_steps: Total training steps
        warmup_steps: Warmup steps
        grad_clip: Gradient clipping
        lr_decay_rate: Layer-wise LR decay (1.0 = no decay)
        freeze_encoder: If True, only train classification head

    Returns:
        Optax optimizer
    """
    # Cosine schedule with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.01,
    )

    if freeze_encoder:
        # Linear probe: only train head
        # Create label tree matching params structure
        flat_params = traverse_util.flatten_dict(params, sep='/')
        flat_labels = {}
        for key in flat_params.keys():
            flat_labels[key] = 'head' if 'head' in key else 'frozen'
        labels = traverse_util.unflatten_dict(flat_labels, sep='/')

        head_tx = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.sgd(learning_rate=schedule, momentum=0.9),
        )

        frozen_tx = optax.set_to_zero()

        return optax.multi_transform(
            {'head': head_tx, 'frozen': frozen_tx},
            labels,
        )

    elif lr_decay_rate < 1.0:
        # Layer-wise LR decay
        # Get layer structure
        flat_params = traverse_util.flatten_dict(params, sep='/')
        num_layers = max(get_layer_index(k) for k in flat_params.keys()) + 1
        print(f"  Detected {num_layers} layers for LR decay")

        # Create per-layer optimizers
        layer_txs = {}
        for layer_idx in range(num_layers + 1):
            lr_scale = lr_decay_rate ** (num_layers - layer_idx)
            layer_txs[layer_idx] = optax.chain(
                optax.clip_by_global_norm(grad_clip),
                optax.scale_by_adam(),
                optax.add_decayed_weights(weight_decay),
                optax.scale_by_schedule(schedule),
                optax.scale(-lr_scale),
            )

        # Create label tree matching params structure
        flat_labels = {}
        for key in flat_params.keys():
            flat_labels[key] = get_layer_index(key, num_stages=4)
        labels = traverse_util.unflatten_dict(flat_labels, sep='/')

        return optax.multi_transform(layer_txs, labels)

    else:
        # Standard AdamW
        return optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=weight_decay,
            ),
        )


# =============================================================================
# Training State and Steps
# =============================================================================

class TrainState(train_state.TrainState):
    """Extended train state with batch_stats for BatchNorm."""
    batch_stats: Optional[Dict[str, Any]] = None


def create_train_state(
    rng: jax.Array,
    model: CSSMSHViT,
    params: Dict[str, Any],
    batch_stats: Optional[Dict[str, Any]],
    optimizer: optax.GradientTransformation,
) -> TrainState:
    """Create training state with pretrained params and batch_stats."""
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
    )


def make_train_step(model: CSSMSHViT, num_classes: int = 1000):
    """Create training step function with BatchNorm support."""

    def train_step(state, batch, rng, axis_name='batch'):
        images, labels = batch

        def loss_fn(params):
            logits, updates = model.apply(
                {'params': params, 'batch_stats': state.batch_stats},
                images, training=True,
                rngs={'dropout': rng},
                mutable=['batch_stats'],
            )
            # Cross-entropy loss
            one_hot = jax.nn.one_hot(labels, num_classes)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss, (logits, updates)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (logits, updates)), grads = grad_fn(state.params)

        # Sync gradients and batch_stats across devices
        grads = jax.lax.pmean(grads, axis_name)
        loss = jax.lax.pmean(loss, axis_name)

        # Update batch_stats (average across devices)
        new_batch_stats = jax.lax.pmean(updates['batch_stats'], axis_name)

        # Update state
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=new_batch_stats)

        # Compute accuracy
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        acc = jax.lax.pmean(acc, axis_name)

        metrics = {'loss': loss, 'acc': acc}
        return state, metrics

    return train_step


def make_train_step_mixup(model: CSSMSHViT, num_classes: int = 1000, label_smoothing: float = 0.1):
    """Create training step with mixup (soft labels) and BatchNorm support."""

    def train_step(state, batch, rng, axis_name='batch'):
        images, labels = batch  # labels are already one-hot/soft from mixup

        def loss_fn(params):
            logits, updates = model.apply(
                {'params': params, 'batch_stats': state.batch_stats},
                images, training=True,
                rngs={'dropout': rng},
                mutable=['batch_stats'],
            )
            # Cross-entropy with soft labels
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            loss = -jnp.sum(labels * log_probs, axis=-1).mean()
            return loss, (logits, updates)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (logits, updates)), grads = grad_fn(state.params)

        grads = jax.lax.pmean(grads, axis_name)
        loss = jax.lax.pmean(loss, axis_name)
        new_batch_stats = jax.lax.pmean(updates['batch_stats'], axis_name)

        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=new_batch_stats)

        # For accuracy, use argmax of both
        preds = jnp.argmax(logits, axis=-1)
        targets = jnp.argmax(labels, axis=-1)
        acc = jnp.mean(preds == targets)
        acc = jax.lax.pmean(acc, axis_name)

        metrics = {'loss': loss, 'acc': acc}
        return state, metrics

    return train_step


def make_eval_step(model: CSSMSHViT, num_classes: int = 1000):
    """Create evaluation step function with BatchNorm support."""

    def eval_step(state, batch, axis_name='batch'):
        images, labels = batch

        logits = model.apply(
            {'params': state.params, 'batch_stats': state.batch_stats},
            images, training=False,
        )

        # Loss
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        loss = jax.lax.pmean(loss, axis_name)

        # Accuracy
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        acc = jax.lax.pmean(acc, axis_name)

        return {'loss': loss, 'acc': acc}

    return eval_step


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Finetune JEPA for ImageNet classification')

    # Checkpoint
    parser.add_argument('--jepa_checkpoint', type=str, required=True,
                        help='Path to JEPA pretrained checkpoint')
    parser.add_argument('--use_ema', action='store_true',
                        help='Load EMA (target encoder) weights instead of online encoder')

    # Model
    parser.add_argument('--model_size', type=str, default='s4',
                        choices=['s1', 's2', 's3', 's4'],
                        help='CSSM-SHViT model size')
    parser.add_argument('--cssm_type', type=str, default='gated',
                        choices=['standard', 'gated', 'opponent'],
                        help='CSSM type')
    parser.add_argument('--num_timesteps', type=int, default=8,
                        help='Number of CSSM recurrence timesteps')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')

    # Finetuning mode
    parser.add_argument('--mode', type=str, default='finetune',
                        choices=['finetune', 'linear_probe'],
                        help='Finetuning mode: finetune (all params) or linear_probe (freeze encoder)')
    parser.add_argument('--lr_decay_rate', type=float, default=1.0,
                        help='Layer-wise LR decay rate (e.g., 0.65). 1.0 = no decay')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of finetuning epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Per-device batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')

    # Augmentation
    parser.add_argument('--mixup_alpha', type=float, default=0.8,
                        help='Mixup alpha (0 to disable)')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='CutMix alpha (0 to disable)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable augmentation (for linear probe)')

    # Data
    parser.add_argument('--data_loader', type=str,
                        choices=['streaming', 'tfdata', 'tfrecord'],
                        default='streaming',
                        help='Data loader: streaming (cv2+threads), tfdata (tf.data on JPEGs), tfrecord (fastest)')
    parser.add_argument('--data_dir', type=str,
                        default='/gpfs/data/shared/imagenet/ILSVRC2012',
                        help='ImageNet data directory (for streaming/tfdata)')
    parser.add_argument('--tfrecord_dir', type=str,
                        default='~/scratch/imagenet_tfrecords',
                        help='TFRecord directory (for tfrecord loader)')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Data loading workers (streaming loader only)')

    # Logging
    parser.add_argument('--project', type=str, default='cssm-finetune',
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/finetune',
                        help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Auto-detect model size from checkpoint path if not specified
    detected_size = detect_model_size_from_path(args.jepa_checkpoint)
    if detected_size:
        if args.model_size == 's4':  # Default value, user didn't specify
            print(f"Auto-detected model size from checkpoint path: {detected_size}")
            args.model_size = detected_size
        elif args.model_size != detected_size:
            print(f"WARNING: Specified model_size={args.model_size} but checkpoint path suggests {detected_size}")
            print(f"         Make sure this is intentional to avoid shape mismatches!")

    # Linear probe specific defaults
    if args.mode == 'linear_probe' and not args.no_augmentation:
        print("Linear probe mode: disabling mixup/cutmix, using simpler augmentation")
        args.mixup_alpha = 0.0
        args.cutmix_alpha = 0.0
        args.label_smoothing = 0.0
        if args.lr == 1e-4:  # Default, adjust for linear probe
            args.lr = 0.1

    use_mixup = args.mixup_alpha > 0 or args.cutmix_alpha > 0

    # Setup
    num_devices = jax.device_count()
    global_batch_size = args.batch_size * num_devices

    print("=" * 60)
    print("JEPA → ImageNet Finetuning")
    print("=" * 60)
    print(f"JEPA checkpoint: {args.jepa_checkpoint}")
    print(f"Model: cssm_shvit_{args.model_size}")
    print(f"Mode: {args.mode}")
    print(f"Layer-wise LR decay: {args.lr_decay_rate}")
    print(f"Devices: {num_devices}")
    print(f"Global batch size: {global_batch_size}")
    print(f"Data loader: {args.data_loader}")
    if args.data_loader == 'tfrecord':
        print(f"TFRecord dir: {args.tfrecord_dir}")
    print("-" * 60)

    # Generate run name
    if args.run_name is None:
        args.run_name = f"finetune_shvit_{args.model_size}_{args.mode}"
        if args.lr_decay_rate < 1.0:
            args.run_name += f"_lrd{args.lr_decay_rate}"

    # Initialize W&B
    if not args.no_wandb:
        wandb.init(
            project=args.project,
            name=args.run_name,
            config=vars(args),
        )

    # Create model
    model_kwargs = {
        'num_classes': 1000,
        'cssm_type': args.cssm_type,
        'num_timesteps': args.num_timesteps,
    }

    if args.model_size == 's1':
        model = cssm_shvit_s1(**model_kwargs)
    elif args.model_size == 's2':
        model = cssm_shvit_s2(**model_kwargs)
    elif args.model_size == 's3':
        model = cssm_shvit_s3(**model_kwargs)
    else:
        model = cssm_shvit_s4(**model_kwargs)

    # Initialize model to get parameter structure
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    dummy_input = jnp.ones((1, args.image_size, args.image_size, 3))
    variables = model.init({'params': init_rng, 'dropout': init_rng}, dummy_input, training=True)
    model_params = variables['params']
    model_batch_stats = variables.get('batch_stats', None)

    if model_batch_stats is not None:
        print("  Model uses BatchNorm (batch_stats initialized)")

    # Load pretrained weights
    print("\nLoading pretrained weights...")
    if args.use_ema:
        pretrained_params = load_ema_checkpoint(args.jepa_checkpoint)
    else:
        pretrained_params = load_jepa_checkpoint(args.jepa_checkpoint)

    # Match and load parameters
    params = match_and_load_params(pretrained_params, model_params, strict=False)

    # Use freshly initialized batch_stats (JEPA doesn't save them typically,
    # and they need to adapt to ImageNet distribution anyway)
    batch_stats = model_batch_stats

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"\nTotal model parameters: {num_params:,}")

    # Create data loaders
    print(f"\nLoading data (loader: {args.data_loader})...")

    if args.data_loader == 'tfrecord':
        train_loader = get_tfrecord_imagenet_loader(
            tfrecord_dir=args.tfrecord_dir,
            split='train',
            batch_size=global_batch_size,
            image_size=args.image_size,
        )
        val_loader = get_tfrecord_imagenet_loader(
            tfrecord_dir=args.tfrecord_dir,
            split='val',
            batch_size=global_batch_size,
            image_size=args.image_size,
        )
    elif args.data_loader == 'tfdata':
        train_loader = get_tfdata_imagenet_loader(
            data_dir=args.data_dir,
            split='train',
            batch_size=global_batch_size,
            image_size=args.image_size,
        )
        val_loader = get_tfdata_imagenet_loader(
            data_dir=args.data_dir,
            split='val',
            batch_size=global_batch_size,
            image_size=args.image_size,
        )
    else:  # streaming (default)
        train_loader = get_imagenet_loader(
            data_dir=args.data_dir,
            split='train',
            batch_size=global_batch_size,
            image_size=args.image_size,
            sequence_length=args.num_timesteps,
            num_workers=args.num_workers,
            augmentation_type='randaugment' if not args.no_augmentation else 'none',
        )
        val_loader = get_imagenet_loader(
            data_dir=args.data_dir,
            split='val',
            batch_size=global_batch_size,
            image_size=args.image_size,
            sequence_length=args.num_timesteps,
            num_workers=args.num_workers,
        )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")

    # Create optimizer
    print("\nCreating optimizer...")
    optimizer = create_optimizer(
        params=params,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        grad_clip=args.grad_clip,
        lr_decay_rate=args.lr_decay_rate,
        freeze_encoder=(args.mode == 'linear_probe'),
    )

    # Create training state
    state = create_train_state(init_rng, model, params, batch_stats, optimizer)

    # Create training/eval steps
    if use_mixup:
        train_step_fn = make_train_step_mixup(model, num_classes=1000, label_smoothing=args.label_smoothing)
    else:
        train_step_fn = make_train_step(model, num_classes=1000)

    eval_step_fn = make_eval_step(model, num_classes=1000)

    p_train_step = jax.pmap(train_step_fn, axis_name='batch', donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step_fn, axis_name='batch')

    # Mixup function
    @jax.jit
    def apply_mixup_cutmix(images, labels, rng):
        return mixup_cutmix(
            images, labels, rng,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mixup_prob=0.5,
            cutmix_prob=0.5,
            num_classes=1000,
        )

    # Replicate state
    state = replicate_state(state)

    # Checkpointing
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    best_val_acc = 0.0
    global_step = 0

    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True) as pbar:
            for batch in pbar:
                rng, step_rng, mixup_rng = jax.random.split(rng, 3)
                images, labels = batch

                # Apply mixup/cutmix (returns soft labels)
                if use_mixup:
                    images, labels = apply_mixup_cutmix(images, labels, mixup_rng)
                # else: keep integer labels, train_step will convert to one-hot

                # Shard batch
                batch = shard_batch((images, labels), num_devices)
                step_rngs = split_rng_for_devices(step_rng)

                # Training step
                state, metrics = p_train_step(state, batch, step_rngs)

                loss = float(metrics['loss'][0])
                acc = float(metrics['acc'][0])

                epoch_loss += loss
                epoch_acc += acc
                num_batches += 1
                global_step += 1

                pbar.set_postfix({'loss': f"{loss:.4f}", 'acc': f"{acc:.4f}"})

                if not args.no_wandb and global_step % 10 == 0:
                    wandb.log({
                        'train/loss': loss,
                        'train/acc': acc,
                    }, step=global_step)

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, train_acc={avg_acc:.4f}")

        # Validation
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0

        for batch in tqdm(val_loader, desc="Validation", leave=False, dynamic_ncols=True):
            batch = shard_batch(batch, num_devices)
            metrics = p_eval_step(state, batch)

            val_loss += float(metrics['loss'][0])
            val_acc += float(metrics['acc'][0])
            val_batches += 1

        val_loss /= max(val_batches, 1)
        val_acc /= max(val_batches, 1)

        print(f"  Val: loss={val_loss:.4f}, acc={val_acc:.4f}")

        if not args.no_wandb:
            wandb.log({
                'val/loss': val_loss,
                'val/acc': val_acc,
                'epoch': epoch + 1,
            }, step=global_step)

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  New best: {best_val_acc:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            state_to_save = unreplicate_state(state)
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}')
            try:
                checkpointer.save(ckpt_path, state_to_save)
                print(f"  Saved checkpoint to {ckpt_path}")
            except Exception as e:
                print(f"  Warning: Failed to save checkpoint: {e}")

    # Finish
    checkpointer.wait_until_finished()

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}")

    if not args.no_wandb:
        wandb.log({'best_val_acc': best_val_acc})
        wandb.finish()


if __name__ == '__main__':
    main()
