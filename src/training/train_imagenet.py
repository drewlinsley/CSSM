#!/usr/bin/env python3
"""
ImageNet training script with TIMM and DeiT III training recipes.

Supports:
- Multi-GPU training via JAX pmap
- Multiple model architectures (SHViT, TinyViT, DeiT3, and CSSM variants)
- Two training recipes:
  - TIMM: AdamW, RandAugment, Mixup/CutMix, label smoothing, EMA
  - DeiT III: LAMB, 3-Augment, BCE loss, no label smoothing
- Cosine LR with warmup
- Gradient clipping
- W&B logging and checkpointing

Usage:
    # TIMM recipe (AdamW, RandAugment, label smoothing)
    python src/training/train_imagenet.py --model cssm_shvit --timm_recipe

    # DeiT III recipe (LAMB, 3-Augment, BCE loss)
    python src/training/train_imagenet.py --model cssm_shvit --deit3_recipe

    # Custom settings
    python src/training/train_imagenet.py --model cssm_shvit \\
        --optimizer lamb --lr 3e-3 --loss bce --augmentation 3aug
"""

import argparse
import os
import random
import re
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import jax_utils
from flax.training import train_state
from tqdm import tqdm
import orbax.checkpoint as ocp

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.imagenet import get_imagenet_loader, get_imagenet_info

# Lazy imports for tf.data loaders (avoid tensorflow import conflicts)
def get_tfdata_imagenet_loader(*args, **kwargs):
    from src.data.imagenet_tfdata import get_tfdata_imagenet_loader as _get_loader
    return _get_loader(*args, **kwargs)

def get_tfrecord_imagenet_loader(*args, **kwargs):
    from src.data.imagenet_tfdata import get_tfrecord_imagenet_loader as _get_loader
    return _get_loader(*args, **kwargs)


def get_gpu_memory_stats():
    """Get GPU memory usage for all devices."""
    stats = {}
    for i, device in enumerate(jax.local_devices()):
        try:
            mem = device.memory_stats()
            if mem:
                # Convert bytes to GB
                stats[f'gpu{i}/mem_used_gb'] = mem.get('bytes_in_use', 0) / 1e9
                stats[f'gpu{i}/mem_peak_gb'] = mem.get('peak_bytes_in_use', 0) / 1e9
        except Exception:
            pass
    return stats
from src.models.deit3 import DeiT3Large, deit3_large_patch16_384, deit3_base_patch16_224
from src.models.cssm_deit3 import CSSMDeiT3Large, cssm_deit3_large_patch16_384, cssm_deit3_base_patch16_224
from src.models.tiny_vit import TinyViT, tiny_vit_21m_224, tiny_vit_5m_224, tiny_vit_11m_224
from src.models.cssm_tiny_vit import CSSMTinyViT, cssm_tiny_vit_21m_224, cssm_tiny_vit_5m_224, cssm_tiny_vit_11m_224
from src.models.shvit import SHViT, shvit_s1, shvit_s2, shvit_s3, shvit_s4
from src.models.cssm_shvit import CSSMSHViT, cssm_shvit_s1, cssm_shvit_s2, cssm_shvit_s3, cssm_shvit_s4
from src.training.distributed import (
    replicate_state,
    unreplicate_state,
    shard_batch,
    create_parallel_train_step,
    split_rng_for_devices,
    make_train_step,
    make_train_step_mixup,
    make_train_step_bce,
    make_eval_step,
)
from src.training.timm_utils import (
    mixup_cutmix,
    mixup,
    cutmix,
    cross_entropy_with_smoothing,
    smooth_labels,
    create_ema_state,
    update_ema,
    random_erasing,
    three_augment_np,
    three_augment_jax,
)
from src.training.optimizers import create_optimizer


class TrainState(train_state.TrainState):
    """Extended train state with epoch tracking."""
    epoch: int = 0


def create_train_state(
    rng: jax.Array,
    model,
    learning_rate: float,
    weight_decay: float,
    total_steps: int,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
    grad_clip_mode: str = 'norm',
    agc_clip_factor: float = 0.02,
    min_lr: float = None,
    image_size: int = 384,
    optimizer_name: str = 'adamw',
):
    """Create training state with optimizer. Returns (state, batch_stats)."""
    # Dummy input for initialization
    dummy_input = jnp.ones((1, image_size, image_size, 3))

    variables = model.init(
        {'params': rng, 'dropout': rng},
        dummy_input,
        training=True  # Use training=True to init batch_stats
    )
    params = variables['params']
    batch_stats = variables.get('batch_stats', None)

    # Create optimizer using factory
    tx = create_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        grad_clip=grad_clip,
        grad_clip_mode=grad_clip_mode,
        agc_clip_factor=agc_clip_factor,
        min_lr=min_lr,
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    return state, batch_stats


def main():
    parser = argparse.ArgumentParser(description='ImageNet training for DeiT3/CSSM-DeiT3')

    # Model
    parser.add_argument('--model', type=str,
                        choices=['deit3', 'cssm_deit3', 'tiny_vit', 'cssm_tiny_vit', 'shvit', 'cssm_shvit'],
                        default='cssm_shvit', help='Model architecture')
    parser.add_argument('--model_size', type=str,
                        choices=['s1', 's2', 's3', 's4', '5m', '11m', '21m', 'base', 'large'],
                        default='s4', help='Model size')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (224 or 384)')

    # CSSM options
    parser.add_argument('--cssm_type', type=str, choices=['gated', 'gdn'],
                        default='gated', help='CSSM type: gated (Spectral Mamba), gdn (GatedDeltaNet with RMSNorm)')
    parser.add_argument('--delta_key_dim', type=int, default=2,
                        help='Key dimension for GDN CSSM (2 or 3)')
    parser.add_argument('--output_norm', type=str, default='rms',
                        choices=['rms', 'layer', 'none'],
                        help='Output normalization for GDN CSSM')
    parser.add_argument('--gate_type_cssm', type=str, default='factored',
                        choices=['factored', 'channel', 'dense', 'scalar'],
                        help='Gate parameterization for GDN CSSM')
    parser.add_argument('--num_timesteps', type=int, default=8,
                        help='Number of CSSM recurrence timesteps (used for eval if --variable_timesteps set)')
    parser.add_argument('--variable_timesteps', type=str, default=None,
                        help='Comma-separated timesteps to sample from during training, e.g., "1,2,4,8". Eval uses --num_timesteps')
    parser.add_argument('--dense_mixing', action='store_true',
                        help='Use dense mixing in CSSM (LMME channel mixing for --cssm_type gated)')
    parser.add_argument('--block_size', type=int, default=32,
                        help='Block size for LMME channel mixing (only with --cssm_type gated --dense_mixing)')
    parser.add_argument('--mixing_rank', type=int, default=0,
                        help='Low-rank channel mixing rank (0=full LMME, >0=low-rank). Recommended: 8-16')
    parser.add_argument('--gate_activation', type=str, default='softplus',
                        choices=['softplus', 'sigmoid', 'softplus_clamped', 'tanh_scaled', 'exp'],
                        help='Gate activation for CSSM (softplus recommended for gated, sigmoid for opponent)')
    parser.add_argument('--spectral_rho', type=float, default=0.999,
                        help='Maximum spectral magnitude for CSSM stability (should be < 1)')
    parser.add_argument('--rope_mode', type=str, default='none',
                        choices=['none', 'temporal', 'spatiotemporal'],
                        help='RoPE position encoding mode (VideoRoPE style)')
    parser.add_argument('--use_dwconv', action='store_true', default=True,
                        help='Use DWConv in MLP (matches SHViT, adds params)')
    parser.add_argument('--no_dwconv', action='store_true',
                        help='Disable DWConv in MLP')
    parser.add_argument('--output_act', type=str, default='none',
                        choices=['none', 'gelu', 'silu'],
                        help='Output activation after CSSM (adds nonlinearity)')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='CSSM spectral kernel size (1=no spatial kernel, all spatial mixing via FFT gates)')
    parser.add_argument('--short_conv_spatial_size', type=int, default=3,
                        help='Spatial depthwise conv kernel size in CSSM (0=disabled)')
    parser.add_argument('--short_conv_size', type=int, default=4,
                        help='Temporal causal conv kernel size in CSSM (0=disabled)')
    parser.add_argument('--block_norm', type=str, default='global_layer',
                        choices=['layer', 'global_layer', 'temporal_layer'],
                        help='Normalization in CSSM blocks: layer (per-frame C), global_layer (T,H,W,C), temporal_layer (T,C)')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Per-device batch size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--grad_clip_mode', type=str, default='norm',
                        choices=['norm', 'agc'],
                        help='Gradient clipping mode: norm (standard) or agc (adaptive gradient clipping)')
    parser.add_argument('--agc_clip_factor', type=float, default=0.02,
                        help='AGC clipping factor (only used with --grad_clip_mode agc)')
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help='Stochastic depth rate')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum learning rate for cosine schedule')
    parser.add_argument('--repeated_aug', action='store_true',
                        help='Use repeated augmentation (sample each image 3x per epoch with different augmentations)')

    # Training Recipe Presets
    parser.add_argument('--timm_recipe', action='store_true',
                        help='Use TIMM recipe: AdamW, RandAugment, Mixup/CutMix, label smoothing, EMA')
    parser.add_argument('--shvit_recipe', action='store_true',
                        help='Use SHViT recipe: AdamW, RandAugment, AGC, repeated aug, EMA 0.99996')
    parser.add_argument('--deit3_recipe', action='store_true',
                        help='Use DeiT III recipe: LAMB, 3-Augment, BCE loss, no label smoothing')

    # Optimizer Selection
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'lamb', 'sgd'],
                        help='Optimizer: adamw (default), lamb (DeiT III), sgd')

    # Loss Function
    parser.add_argument('--loss', type=str, default='ce',
                        choices=['ce', 'bce'],
                        help='Loss: ce (softmax cross-entropy), bce (binary cross-entropy, DeiT III)')

    # Augmentation
    parser.add_argument('--augmentation', type=str, default='randaugment',
                        choices=['randaugment', '3aug', 'none'],
                        help='Augmentation: randaugment (TIMM), 3aug (DeiT III), none')
    parser.add_argument('--mixup_alpha', type=float, default=0.8,
                        help='Mixup alpha (0 to disable)')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='CutMix alpha (0 to disable)')
    parser.add_argument('--mixup_prob', type=float, default=0.5,
                        help='Probability of applying mixup (vs cutmix)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (0 to disable)')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate (0 to disable)')
    parser.add_argument('--random_erase_prob', type=float, default=0.25,
                        help='Random erasing probability (0 to disable)')
    parser.add_argument('--randaugment_num_ops', type=int, default=2,
                        help='RandAugment number of operations')
    parser.add_argument('--randaugment_magnitude', type=int, default=9,
                        help='RandAugment magnitude (0-10)')
    parser.add_argument('--color_jitter', type=float, default=0.3,
                        help='Color jitter factor for 3-Augment')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable all augmentations (for debugging)')

    # Data
    parser.add_argument('--data_loader', type=str,
                        choices=['streaming', 'tfdata', 'tfrecord'],
                        default='streaming',
                        help='Data loader: streaming (cv2+threads), tfdata (tf.data on JPEGs), tfrecord (fastest, requires conversion)')
    parser.add_argument('--data_dir', type=str,
                        default='/gpfs/data/shared/imagenet/ILSVRC2012',
                        help='ImageNet data directory (for streaming/tfdata)')
    parser.add_argument('--tfrecord_dir', type=str,
                        default='~/scratch/imagenet_tfrecords',
                        help='TFRecord directory (for tfrecord loader)')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Data loading workers (streaming loader only)')

    # Logging
    parser.add_argument('--project', type=str, default='cssm-imagenet',
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')

    # Checkpoint resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoints/run_name/epoch_190)')
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='Override epoch to resume from (default: auto-detect from checkpoint path)')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Apply DeiT III recipe defaults
    if args.deit3_recipe:
        args.optimizer = 'lamb'
        args.lr = 3e-3
        args.epochs = 800
        args.loss = 'bce'
        args.augmentation = '3aug'
        args.label_smoothing = 0.0
        args.random_erase_prob = 0.0
        args.ema_decay = 0.0  # DeiT III doesn't use EMA
        print("Using DeiT III recipe: LAMB optimizer, 3-Augment, BCE loss")

    # Apply SHViT recipe defaults (from published SHViT paper)
    elif args.shvit_recipe:
        args.optimizer = 'adamw'
        args.lr = 1e-3
        args.weight_decay = 0.025
        args.augmentation = 'randaugment'
        args.randaugment_num_ops = 2
        args.randaugment_magnitude = 9
        args.mixup_alpha = 0.8
        args.cutmix_alpha = 1.0
        args.label_smoothing = 0.1
        args.ema_decay = 0.99996
        args.random_erase_prob = 0.25
        args.color_jitter = 0.4
        args.grad_clip_mode = 'agc'
        args.grad_clip = 0.02
        args.min_lr = 1e-5
        args.repeated_aug = True
        print("Using SHViT recipe: AdamW, RandAugment, AGC, repeated aug, EMA 0.99996, wd=0.025")

    # Apply TIMM recipe defaults
    elif args.timm_recipe:
        args.optimizer = 'adamw'
        args.lr = 1e-3
        args.augmentation = 'randaugment'
        args.mixup_alpha = 0.8
        args.cutmix_alpha = 1.0
        args.label_smoothing = 0.1
        args.ema_decay = 0.9998
        args.random_erase_prob = 0.25
        print("Using TIMM recipe: AdamW optimizer, RandAugment, label smoothing")

    # Disable augmentations if requested
    if args.no_augmentation:
        args.mixup_alpha = 0.0
        args.cutmix_alpha = 0.0
        args.label_smoothing = 0.0
        args.random_erase_prob = 0.0
        args.augmentation = 'none'

    # Determine if we're using mixup/cutmix
    use_mixup = args.mixup_alpha > 0 or args.cutmix_alpha > 0
    use_bce = args.loss == 'bce'

    # Parse variable timesteps
    variable_timesteps = None
    if args.variable_timesteps:
        variable_timesteps = [int(t) for t in args.variable_timesteps.split(',')]
        print(f"Variable timesteps enabled: {variable_timesteps}")
        print(f"  Training: sample from {variable_timesteps}")
        print(f"  Eval: fixed at {args.num_timesteps}")

    # Setup
    num_devices = jax.device_count()
    global_batch_size = args.batch_size * num_devices

    print("=" * 60)
    print("ImageNet Training")
    print("=" * 60)
    print(f"Model: {args.model} ({args.model_size})")
    print(f"Image size: {args.image_size}")
    if 'cssm' in args.model:
        print(f"CSSM type: {args.cssm_type}")
        print(f"CSSM timesteps: {args.num_timesteps}")
        if variable_timesteps:
            print(f"CSSM variable timesteps: {variable_timesteps}")
        print(f"CSSM spectral_rho: {args.spectral_rho}")
        print(f"CSSM rope_mode: {args.rope_mode}")
    print(f"Devices: {num_devices}")
    print(f"Per-device batch size: {args.batch_size}")
    print(f"Global batch size: {global_batch_size}")
    print(f"Data loader: {args.data_loader}")
    print("-" * 60)
    print("Training:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Loss: {args.loss}")
    print("-" * 60)
    print("Augmentation:")
    print(f"  Type: {args.augmentation}")
    print(f"  Mixup alpha: {args.mixup_alpha}")
    print(f"  CutMix alpha: {args.cutmix_alpha}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Random erase prob: {args.random_erase_prob}")
    print(f"  EMA decay: {args.ema_decay}")
    print("=" * 60)

    # Generate run name if not provided
    if args.run_name is None:
        if 'cssm' in args.model:
            args.run_name = f"{args.model}_{args.model_size}_{args.cssm_type}_t{args.num_timesteps}"
        else:
            args.run_name = f"{args.model}_{args.model_size}_baseline"

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
        'drop_path_rate': args.drop_path_rate,
    }

    # Handle --no_dwconv override
    use_dwconv = args.use_dwconv and not args.no_dwconv

    cssm_kwargs = {
        'cssm_type': args.cssm_type,
        'num_timesteps': args.num_timesteps,
        'dense_mixing': args.dense_mixing,
        'block_size': args.block_size,
        'mixing_rank': args.mixing_rank,
        'gate_activation': args.gate_activation,
        'spectral_rho': args.spectral_rho,
        'rope_mode': args.rope_mode,
        'use_dwconv': use_dwconv,
        'output_act': args.output_act,
        'short_conv_size': args.short_conv_size,
        'short_conv_spatial_size': args.short_conv_spatial_size,
        'delta_key_dim': args.delta_key_dim,
        'output_norm': args.output_norm,
        'gate_type': args.gate_type_cssm,
        'block_norm': args.block_norm,
    }

    if args.model == 'cssm_shvit':
        model_kwargs.update(cssm_kwargs)
        # Apply uniform kernel_size to all CSSM stages
        model_kwargs['kernel_sizes'] = (args.kernel_size,) * 4
        if args.model_size == 's1':
            model = cssm_shvit_s1(**model_kwargs)
        elif args.model_size == 's2':
            model = cssm_shvit_s2(**model_kwargs)
        elif args.model_size == 's3':
            model = cssm_shvit_s3(**model_kwargs)
        else:  # s4 (default)
            model = cssm_shvit_s4(**model_kwargs)

    elif args.model == 'shvit':
        if args.model_size == 's1':
            model = shvit_s1(**model_kwargs)
        elif args.model_size == 's2':
            model = shvit_s2(**model_kwargs)
        elif args.model_size == 's3':
            model = shvit_s3(**model_kwargs)
        else:  # s4 (default)
            model = shvit_s4(**model_kwargs)

    elif args.model == 'cssm_tiny_vit':
        model_kwargs.update(cssm_kwargs)
        if args.model_size == '5m':
            model = cssm_tiny_vit_5m_224(**model_kwargs)
        elif args.model_size == '11m':
            model = cssm_tiny_vit_11m_224(**model_kwargs)
        else:  # 21m (default)
            model = cssm_tiny_vit_21m_224(**model_kwargs)

    elif args.model == 'tiny_vit':
        if args.model_size == '5m':
            model = tiny_vit_5m_224(**model_kwargs)
        elif args.model_size == '11m':
            model = tiny_vit_11m_224(**model_kwargs)
        else:  # 21m (default)
            model = tiny_vit_21m_224(**model_kwargs)

    elif args.model == 'cssm_deit3':
        model_kwargs.update(cssm_kwargs)
        if args.model_size == 'large':
            model = cssm_deit3_large_patch16_384(**model_kwargs) if args.image_size == 384 else \
                    CSSMDeiT3Large(embed_dim=1024, depth=24, num_heads=16, patch_size=16, **model_kwargs)
        else:
            model = cssm_deit3_base_patch16_224(**model_kwargs)

    else:  # deit3
        if args.model_size == 'large':
            model = deit3_large_patch16_384(**model_kwargs) if args.image_size == 384 else \
                    DeiT3Large(embed_dim=1024, depth=24, num_heads=16, patch_size=16, **model_kwargs)
        else:
            model = deit3_base_patch16_224(**model_kwargs)

    # Initialize
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    # Create data loaders
    print(f"\nLoading data (loader: {args.data_loader})...")

    if args.data_loader == 'tfrecord':
        train_loader = get_tfrecord_imagenet_loader(
            tfrecord_dir=args.tfrecord_dir,
            split='train',
            batch_size=global_batch_size,
            image_size=args.image_size,
            repeated_aug=args.repeated_aug,
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
        # Use sequence_length=1 for training if variable_timesteps (expand later)
        train_seq_len = 1 if variable_timesteps else args.num_timesteps
        train_loader = get_imagenet_loader(
            data_dir=args.data_dir,
            split='train',
            batch_size=global_batch_size,
            image_size=args.image_size,
            sequence_length=train_seq_len,
            num_workers=args.num_workers,
            augmentation_type=args.augmentation,
            randaugment_num_ops=args.randaugment_num_ops,
            randaugment_magnitude=args.randaugment_magnitude,
            color_jitter=args.color_jitter,
        )
        # Eval always uses fixed num_timesteps
        val_loader = get_imagenet_loader(
            data_dir=args.data_dir,
            split='val',
            batch_size=global_batch_size,
            image_size=args.image_size,
            sequence_length=args.num_timesteps,
            num_workers=args.num_workers,
        )
        if args.augmentation == 'randaugment':
            print(f"  RandAugment: {args.randaugment_num_ops} ops, magnitude {args.randaugment_magnitude}")
        elif args.augmentation == '3aug':
            print(f"  3-Augment: color_jitter={args.color_jitter}")

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    # Create training state
    state, batch_stats = create_train_state(
        rng=init_rng,
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        grad_clip=args.grad_clip,
        grad_clip_mode=args.grad_clip_mode,
        agc_clip_factor=args.agc_clip_factor,
        min_lr=args.min_lr,
        image_size=args.image_size,
        optimizer_name=args.optimizer,
    )

    has_batch_stats = batch_stats is not None
    if has_batch_stats:
        print("  Model uses BatchNorm (batch_stats enabled)")

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"\nModel parameters: {num_params:,}")

    if not args.no_wandb:
        wandb.config.update({'num_params': num_params})

    # Create parallel training step
    if use_bce:
        # DeiT III uses Binary Cross Entropy
        train_step_fn = make_train_step_bce(model, num_classes=1000)
        print("  Using BCE loss (DeiT III style)")
    elif use_mixup:
        # Use mixup-compatible train step (expects soft labels)
        train_step_fn = make_train_step_mixup(
            model, num_classes=1000, label_smoothing=args.label_smoothing
        )
        print("  Using mixup/cutmix training step with soft labels")
    else:
        train_step_fn = make_train_step(model, num_classes=1000)

    def parallel_train_step_wrapper(state, batch, rng, batch_stats):
        return train_step_fn(state, batch, rng, batch_stats=batch_stats, axis_name='batch')

    p_train_step = jax.pmap(parallel_train_step_wrapper, axis_name='batch', donate_argnums=(0,))

    # Create parallel eval step
    eval_step_fn = make_eval_step(model, num_classes=1000)

    def parallel_eval_step_wrapper(state, batch, batch_stats):
        return eval_step_fn(state, batch, batch_stats=batch_stats, axis_name='batch')

    p_eval_step = jax.pmap(parallel_eval_step_wrapper, axis_name='batch')

    # Create JIT-compiled augmentation functions for GPU
    @jax.jit
    def apply_mixup_cutmix(images, labels, rng):
        """Apply mixup or cutmix augmentation to batch."""
        return mixup_cutmix(
            images, labels, rng,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mixup_prob=args.mixup_prob,
            cutmix_prob=1.0 - args.mixup_prob,
            num_classes=1000,
        )

    @jax.jit
    def apply_random_erasing(images, rng):
        """Apply random erasing augmentation to batch."""
        return random_erasing(
            images, rng,
            probability=args.random_erase_prob,
        )

    @jax.jit
    def apply_three_augment(images, rng):
        """Apply GPU-accelerated 3-Augment to batch."""
        return three_augment_jax(
            images, rng,
            color_jitter=args.color_jitter,
        )

    # Determine if we should use GPU augmentation (for TFRecord loader)
    use_gpu_augment = (args.data_loader == 'tfrecord' and
                       args.augmentation in ['3aug', 'randaugment'] and
                       not args.no_augmentation)
    if use_gpu_augment:
        print(f"  Using GPU 3-Augment (TFRecord mode)")

    # Replicate state and batch_stats for multi-GPU
    state = replicate_state(state)
    if has_batch_stats:
        batch_stats = jax_utils.replicate(batch_stats)

    # Initialize EMA state (on unreplicated params)
    use_ema = args.ema_decay > 0
    ema_state = None
    if use_ema:
        ema_state = create_ema_state(unreplicate_state(state).params, decay=args.ema_decay)
        print(f"  Using EMA with decay={args.ema_decay}")

    # Checkpointing
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    best_is_ema = False
    global_step = 0

    if args.resume:
        resume_path = os.path.abspath(args.resume)
        if os.path.exists(resume_path):
            print(f"\n{'='*60}")
            print(f"Resuming from checkpoint: {resume_path}")
            print(f"{'='*60}\n")

            # Unreplicate state for restoration
            single_state = unreplicate_state(state)

            # Restore the checkpoint - restore without target to get raw checkpoint
            try:
                restored = checkpointer.restore(resume_path)

                # Handle different restore formats
                if isinstance(restored, train_state.TrainState):
                    # Restored as TrainState directly
                    restored_state = restored
                    print(f"  Checkpoint format: TrainState object")
                elif isinstance(restored, dict):
                    # Restored as dict - reconstruct TrainState
                    print(f"  Checkpoint format: dict with keys {list(restored.keys())}")
                    if 'params' in restored:
                        # Get step from checkpoint
                        ckpt_step = restored.get('step', 0)
                        if hasattr(ckpt_step, 'item'):
                            ckpt_step = int(ckpt_step.item())
                        else:
                            ckpt_step = int(ckpt_step)
                        print(f"  Checkpoint step: {ckpt_step}")

                        # Reconstruct TrainState with restored values
                        # Note: We keep the NEW optimizer (with correct schedule) but restore params
                        restored_state = single_state.replace(
                            params=restored['params'],
                            step=ckpt_step,
                        )
                        print(f"  Restored params and step={ckpt_step}")
                    else:
                        raise ValueError(f"Checkpoint dict missing 'params'. Keys: {restored.keys()}")
                else:
                    # Assume it's a TrainState-like object
                    restored_state = restored
                    print(f"  Checkpoint format: {type(restored)}")

                # Validate restoration by checking param shapes match
                orig_leaves = jax.tree_util.tree_leaves(single_state.params)
                restored_leaves = jax.tree_util.tree_leaves(restored_state.params)
                if len(orig_leaves) != len(restored_leaves):
                    raise ValueError(f"Param count mismatch: expected {len(orig_leaves)}, got {len(restored_leaves)}")

                # Check a few param values aren't all zeros/random
                sample_param = restored_leaves[0]
                print(f"  Restored params sample - mean: {float(jnp.mean(sample_param)):.6f}, std: {float(jnp.std(sample_param)):.6f}")

                # Re-replicate for multi-GPU
                state = replicate_state(restored_state)
                print(f"  Successfully restored and replicated state")

            except Exception as e:
                print(f"ERROR: Failed to restore checkpoint: {e}")
                print("Starting from scratch instead.")
                import traceback
                traceback.print_exc()

            # Restore EMA if it exists
            if use_ema:
                ema_ckpt_path = resume_path + '_ema'
                if os.path.exists(ema_ckpt_path):
                    try:
                        ema_restored = checkpointer.restore(ema_ckpt_path, {'ema_params': ema_state['ema_params']})
                        if isinstance(ema_restored, dict) and 'ema_params' in ema_restored:
                            ema_state['ema_params'] = ema_restored['ema_params']
                        else:
                            ema_state['ema_params'] = ema_restored
                        print(f"  Restored EMA params from {ema_ckpt_path}")
                    except Exception as e:
                        print(f"  Warning: Could not restore EMA checkpoint: {e}")

            # Determine starting epoch
            if args.resume_epoch is not None:
                start_epoch = args.resume_epoch
            else:
                # Try to extract epoch from checkpoint path (e.g., epoch_190)
                match = re.search(r'epoch_(\d+)', resume_path)
                if match:
                    start_epoch = int(match.group(1))
                else:
                    start_epoch = 0

            # Calculate global_step based on resumed epoch
            global_step = start_epoch * steps_per_epoch

            print(f"Resumed from epoch {start_epoch}, global_step {global_step}")
            print(f"Will continue training for epochs {start_epoch + 1} to {args.epochs}\n")
        else:
            print(f"WARNING: Checkpoint not found at {resume_path}, starting from scratch")

    # Pre-warmup: compile all T values upfront (if using variable timesteps)
    if variable_timesteps is not None and start_epoch == 0:
        print(f"\nPre-compiling for variable timesteps: {variable_timesteps}")
        dummy_images_base = jnp.ones((global_batch_size, args.image_size, args.image_size, 3))
        dummy_labels = jax.nn.one_hot(jnp.zeros(global_batch_size, dtype=jnp.int32), 1000)

        for T in variable_timesteps:
            print(f"  Compiling T={T}...", end=" ", flush=True)
            dummy_images = jnp.repeat(dummy_images_base[:, None, :, :, :], T, axis=1)
            dummy_batch = shard_batch((dummy_images, dummy_labels), num_devices)
            rng, warmup_rng = jax.random.split(rng)
            warmup_rngs = split_rng_for_devices(warmup_rng)
            # Run one forward+backward pass to trigger compilation
            state, _, _ = p_train_step(state, dummy_batch, warmup_rngs, batch_stats)
            jax.block_until_ready(state)
            print("done")
        print("Pre-compilation complete!\n")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Training
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        epoch_step_times = []

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True) as pbar:
            for batch in pbar:
                rng, step_rng, mixup_rng, erase_rng, aug_rng = jax.random.split(rng, 5)
                images, labels = batch

                # Apply GPU 3-Augment if using TFRecord (before mixup)
                if use_gpu_augment:
                    images = apply_three_augment(images, aug_rng)

                # Apply mixup/cutmix before sharding (on full batch)
                if use_mixup:
                    images, labels = apply_mixup_cutmix(images, labels, mixup_rng)
                else:
                    # Convert labels to one-hot for consistency
                    labels = jax.nn.one_hot(labels, 1000)

                # Apply random erasing (after mixup, before sharding)
                if args.random_erase_prob > 0:
                    images = apply_random_erasing(images, erase_rng)

                # Variable timesteps: sample T per batch and expand images to (B, T, H, W, C)
                if variable_timesteps is not None:
                    T = random.choice(variable_timesteps)  # Pure Python random, no JAX tracing
                    # Use numpy for repeat (faster outside JIT), convert to JAX array after
                    images_np = np.asarray(images)
                    if images_np.ndim == 4:
                        images = jnp.array(np.repeat(images_np[:, None, :, :, :], T, axis=1))
                    elif images_np.ndim == 5:
                        # Take last frame and expand to sampled T
                        images = jnp.array(np.repeat(images_np[:, -1:, :, :, :], T, axis=1))

                batch = (images, labels)

                # Shard batch across devices
                batch = shard_batch(batch, num_devices)

                # Split RNG for each device
                step_rngs = split_rng_for_devices(step_rng)

                # Training step with timing
                step_start = time.perf_counter()
                state, metrics, new_batch_stats = p_train_step(state, batch, step_rngs, batch_stats)
                if has_batch_stats and new_batch_stats is not None:
                    batch_stats = new_batch_stats
                jax.block_until_ready(metrics)
                step_time = time.perf_counter() - step_start
                epoch_step_times.append(step_time)

                # Update EMA
                if use_ema:
                    current_params = unreplicate_state(state).params
                    ema_state = update_ema(ema_state, current_params)

                # Get metrics from first device
                loss = float(metrics['loss'][0])
                acc = float(metrics['acc'][0])

                epoch_loss += loss
                epoch_acc += acc
                num_batches += 1
                global_step += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'acc': f"{acc:.4f}",
                    'ms/step': f"{step_time*1000:.1f}",
                })

                # Log to W&B
                if not args.no_wandb and global_step % 10 == 0:
                    log_dict = {
                        'train/loss': loss,
                        'train/acc': acc,
                        'timing/step_ms': step_time * 1000,
                        'timing/throughput': global_batch_size / step_time,
                    }
                    # Add GPU memory stats
                    log_dict.update(get_gpu_memory_stats())
                    wandb.log(log_dict, step=global_step)

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        avg_step_time = sum(epoch_step_times) / len(epoch_step_times) * 1000

        # Get GPU memory for epoch summary
        gpu_mem = get_gpu_memory_stats()
        mem_str = ", ".join([f"GPU{i}: {gpu_mem.get(f'gpu{i}/mem_used_gb', 0):.1f}GB"
                             for i in range(num_devices)])
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}, avg_step={avg_step_time:.1f}ms, {mem_str}")

        # Validation
        if (epoch + 1) % args.eval_every == 0:
            val_loss = 0.0
            val_acc = 0.0
            val_batches = 0

            for batch in tqdm(val_loader, desc="Validation", leave=False, dynamic_ncols=True):
                batch = shard_batch(batch, num_devices)
                metrics = p_eval_step(state, batch, batch_stats)

                val_loss += float(metrics['loss'][0])
                val_acc += float(metrics['acc'][0])
                val_batches += 1

            val_loss /= max(val_batches, 1)
            val_acc /= max(val_batches, 1)

            print(f"  Val: loss={val_loss:.4f}, acc={val_acc:.4f}")

            # EMA Validation (if enabled)
            ema_val_loss = 0.0
            ema_val_acc = 0.0
            if use_ema:
                # Create temporary state with EMA params for evaluation
                ema_params_replicated = jax_utils.replicate(ema_state['ema_params'])

                # Create a temporary state with EMA params
                ema_state_temp = state.replace(params=ema_params_replicated)

                ema_val_batches = 0
                for batch in tqdm(val_loader, desc="EMA Validation", leave=False, dynamic_ncols=True):
                    batch = shard_batch(batch, num_devices)
                    metrics = p_eval_step(ema_state_temp, batch, batch_stats)

                    ema_val_loss += float(metrics['loss'][0])
                    ema_val_acc += float(metrics['acc'][0])
                    ema_val_batches += 1

                ema_val_loss /= max(ema_val_batches, 1)
                ema_val_acc /= max(ema_val_batches, 1)

                print(f"  EMA Val: loss={ema_val_loss:.4f}, acc={ema_val_acc:.4f}")

            if not args.no_wandb:
                val_log = {
                    'val/loss': val_loss,
                    'val/acc': val_acc,
                    'epoch': epoch + 1,
                }
                if use_ema:
                    val_log['val/ema_loss'] = ema_val_loss
                    val_log['val/ema_acc'] = ema_val_acc
                val_log.update(get_gpu_memory_stats())
                wandb.log(val_log, step=global_step)

            # Track best model (use EMA accuracy if available, otherwise regular)
            check_acc = ema_val_acc if use_ema else val_acc
            if check_acc > best_val_acc:
                best_val_acc = check_acc
                best_is_ema = use_ema
                print(f"  New best validation accuracy: {best_val_acc:.4f}" +
                      (" (EMA)" if best_is_ema else ""))

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            # Unreplicate state for saving
            state_to_save = unreplicate_state(state)
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}')
            try:
                checkpointer.save(ckpt_path, state_to_save)
                print(f"  Saved checkpoint to {ckpt_path}")
            except Exception as e:
                print(f"  Warning: Failed to save checkpoint: {e}")
                # Fallback: save params as numpy
                fallback_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_params.npz')
                try:
                    flat_params = jax.tree_util.tree_leaves(state_to_save.params)
                    np.savez(fallback_path, *flat_params)
                    print(f"  Saved fallback checkpoint to {fallback_path}")
                except Exception as e2:
                    print(f"  Warning: Fallback save also failed: {e2}")

            # Save EMA params separately if enabled
            if use_ema:
                ema_ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_ema')
                try:
                    checkpointer.save(ema_ckpt_path, {'ema_params': ema_state['ema_params']})
                    print(f"  Saved EMA checkpoint to {ema_ckpt_path}")
                except Exception as e:
                    print(f"  Warning: Failed to save EMA checkpoint: {e}")

    # Wait for checkpointing to complete
    checkpointer.wait_until_finished()

    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}" +
          (" (EMA)" if best_is_ema else ""))
    print(f"{'='*60}")

    if not args.no_wandb:
        wandb.log({
            'best_val_acc': best_val_acc,
            'best_is_ema': best_is_ema,
        })
        wandb.finish()


if __name__ == '__main__':
    main()
