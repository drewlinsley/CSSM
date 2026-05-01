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

# Import imax.randaugment eagerly at module load time (before any JAX trace)
# so its module-level JAX values don't leak into pmap/vmap scopes.
from imax import randaugment as _imax_randaugment  # noqa: E402

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
    no_decay_names: tuple = (),
    probe_lr: float = None,
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
        no_decay_names=no_decay_names,
        probe_lr=probe_lr,
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
                        choices=['none', 'temporal', 'spatiotemporal', 'learned_t'],
                        help='Temporal position encoding mode. "spatiotemporal"/"temporal" '
                             'apply fixed-frequency RoPE rotations to gate context. '
                             '"learned_t" uses a learned (T, C) additive embedding on the gate '
                             'context — lets the model decide how strongly to vary gates per-t, '
                             'zero-init so starts identical to the rope=none case.')
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
    parser.add_argument('--static_image_fast_path', action='store_true',
                        help='Skip T-replication inside CSSM-SHViT GDN blocks for static image input '
                             '(~4× speedup at num_timesteps>1). Auto-coerces short_conv_size=0, '
                             'no_input_gates=True, rope_mode=none. Model is NOT weight-compatible '
                             'with standard training.')
    parser.add_argument('--no_input_gates', action='store_true',
                        help='Disable GatedDeltaNetCSSM B_k/B_v input gates. Required with --static_image_fast_path.')
    parser.add_argument('--gate_proj_bias_init', type=float, default=0.0,
                        help='Bias init for GDN gate_proj. Default 0.0 (matches tog9av97 75%% run). '
                             'Set to 1.0 to restore the regressed pre-bisect behavior.')
    parser.add_argument('--pos_conv', dest='use_pos_conv_cli', action='store_true', default=True,
                        help='Enable DWConv pos_conv before CSSM input (default, matches tog9av97).')
    parser.add_argument('--no_pos_conv', dest='use_pos_conv_cli', action='store_false',
                        help='Disable pos_conv (use for fast-warmup ablations).')
    parser.add_argument('--head_pool', type=str, default='max', choices=['max', 'mean'],
                        help='Global pooling before classifier head. Default "max" matches '
                             'tog9av97 75%% run. "mean" matches SHViT baseline.')
    parser.add_argument('--timestep_gate', action='store_true',
                        help='Replace x[:, -1] timestep selection with a data-dependent softmax '
                             'pooling over CSSM timesteps. Gate MLP takes mean-pool(input) concat '
                             'per-t mean-pool(output); softmax is prior-biased toward t=T-1 so the '
                             'block starts behaving like the un-gated version. Incompatible with '
                             '--static_image_fast_path (auto-coerced off).')
    # Recurrent-LeJEPA SSL loss (Balestriero & LeCun, arXiv:2511.08544)
    parser.add_argument('--ssl_temporal_loss', action='store_true',
                        help='SSL-only continuation: recurrent LeJEPA (pairwise predictive + SIGReg) '
                             'on per-CSSM-block temporal feature trajectories. Drops main '
                             'classification CE, adds a stop-grad linear probe head for val '
                             'tracking. Auto-coerces --static_image_fast_path off.')
    parser.add_argument('--ssl_loss_weight', type=float, default=1.0,
                        help='Weight on the pairwise predictive term.')
    parser.add_argument('--ssl_sigreg_weight', type=float, default=1.0,
                        help='Weight on SIGReg (the anti-collapse isotropic-Gaussian regularizer).')
    parser.add_argument('--ssl_pair_mode', type=str, default='all',
                        choices=['all', 'successive', 'to_mean'],
                        help='Which timestep pairs enter the predictive loss. '
                             '"all": every i<j pair. "successive": only (z_t, z_{t+1}). '
                             '"to_mean": every z_t pulled to the per-sample mean.')
    parser.add_argument('--ssl_proj_dim', type=int, default=64,
                        help='Projection dim P for SSL features (each CSSM block adds an MLP head).')
    parser.add_argument('--ssl_num_slices', type=int, default=1024,
                        help='Random 1D projections for SIGReg (LeJEPA default 1024).')
    parser.add_argument('--ssl_num_points', type=int, default=17,
                        help='Characteristic-function t-grid points for SIGReg (LeJEPA default 17).')
    parser.add_argument('--probe_lr', type=float, default=1e-3,
                        help='Constant LR for linear_probe_head/* (no warmup, no weight decay).')
    # Deprecated (old triplet SSL); accepted but warned.
    parser.add_argument('--ssl_var_weight', type=float, default=None,
                        help='DEPRECATED (old triplet SSL). No-op under LeJEPA.')
    parser.add_argument('--ssl_margin', type=float, default=None,
                        help='DEPRECATED (old triplet SSL). No-op under LeJEPA.')
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'bf16'],
                        help='Compute precision. "bf16" → bf16 activations/matmul with fp32 weight storage, '
                             'fp32 FFT, and fp32 loss. Default fp32 for full precision.')

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
                        choices=['streaming', 'tfdata', 'tfrecord', 'dali', 'ego4d'],
                        default='streaming',
                        help='Data loader: streaming (cv2+threads), tfdata (tf.data on JPEGs), '
                             'tfrecord (tf.data TFRecords, CPU decode), '
                             'dali (NVIDIA DALI TFRecords with GPU nvJPEG decode — fastest), '
                             'ego4d (decord video clips, for video-SSL training)')
    parser.add_argument('--ego4d_video_dir', type=str,
                        default='/cifs/data/tserre_lrs/projects/projects/prj_ego4d/ego4d_data/v2/down_scaled',
                        help='Ego4D video directory (only used with --data_loader ego4d)')
    parser.add_argument('--ego4d_manifest', type=str, default=None,
                        help='Ego4D JSON manifest (auto-built from video_dir if absent)')
    parser.add_argument('--ego4d_frame_stride', type=int, default=4,
                        help='Ego4D temporal stride in source frames between sampled frames')
    parser.add_argument('--data_dir', type=str,
                        default='/gpfs/data/shared/imagenet/ILSVRC2012',
                        help='ImageNet data directory (for streaming/tfdata)')
    parser.add_argument('--tfrecord_dir', type=str,
                        default='~/scratch/imagenet_tfrecords',
                        help='TFRecord directory (for tfrecord loader)')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Data loading workers (streaming loader only)')
    parser.add_argument('--tf_max_intraop', type=int, default=1,
                        help='tf.data max_intra_op_parallelism for TFRecord loader '
                             '(default 1 to avoid pthread exhaustion; bump to 4-8 '
                             'on nodes with high ulimit if data-loader bound)')
    parser.add_argument('--tf_parallel_reads', type=int, default=64,
                        help='tf.data num_parallel_reads. Drop to 4-8 on crowded nodes.')
    parser.add_argument('--tf_prefetch_batches', type=int, default=16,
                        help='tf.data prefetch_batches. Drop to 2-4 on crowded nodes.')
    parser.add_argument('--tf_threadpool_size', type=int, default=48,
                        help='tf.data private_threadpool_size. Drop to 4-8 on crowded nodes.')
    parser.add_argument('--prefetch_to_device', action='store_true',
                        help='Wrap the TFRecord loader with a background thread '
                             'that pre-shards each batch onto devices via '
                             'jax.device_put_sharded. Overlaps host→device copy '
                             'with GPU compute — worth ~100-200 ms/step when '
                             'data-loader bound. Requires --data_loader tfrecord.')
    parser.add_argument('--prefetch_depth', type=int, default=2,
                        help='Queue depth for --prefetch_to_device worker '
                             '(default 2, raise to 3-4 if still data-bound '
                             'and host RAM allows).')

    # Logging
    parser.add_argument('--project', type=str, default='cssm-imagenet-edge',
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
        # Match tog9av97 75% run: its code hardcoded 3-Augment regardless of the
        # `augmentation` flag, so the ACTUAL aug was 3aug not randaugment. The
        # bug was fixed later; keep 3aug here to preserve the known-good recipe.
        args.augmentation = '3aug'
        args.mixup_alpha = 0.8
        args.cutmix_alpha = 1.0
        args.label_smoothing = 0.1
        args.ema_decay = 0.9998
        args.random_erase_prob = 0.25
        print("Using TIMM recipe: AdamW optimizer, 3-Augment (matches tog9av97), label smoothing")

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

    # Static-image fast path: auto-coerce compatible options
    if args.static_image_fast_path:
        if variable_timesteps is not None:
            raise ValueError(
                "--static_image_fast_path is incompatible with --variable_timesteps: "
                "fast path requires a fixed scan length."
            )
        coercions = []
        if args.short_conv_size != 0:
            coercions.append(f"--short_conv_size {args.short_conv_size} → 0")
            args.short_conv_size = 0
        if not args.no_input_gates:
            coercions.append("--no_input_gates (was False) → True")
            args.no_input_gates = True
        if args.rope_mode != 'none':
            coercions.append(f"--rope_mode {args.rope_mode} → none")
            args.rope_mode = 'none'
        if coercions:
            print("=" * 60)
            print("--static_image_fast_path auto-coercions:")
            for c in coercions:
                print(f"  {c}")
            print("=" * 60)

    # Timestep gate: needs the full (B, T, H, W, C) tensor before the timestep
    # collapse, which the static fast path skips. Auto-coerce fast path off.
    if args.timestep_gate and args.static_image_fast_path:
        print("=" * 60)
        print("--timestep_gate auto-coercion:")
        print("  --static_image_fast_path → off (gate needs per-t features)")
        print("=" * 60)
        args.static_image_fast_path = False

    # Temporal SSL: auto-coerce incompatible options. SSL needs per-timestep
    # features, so the static fast path (which collapses to S_{T-1}) must be off.
    if args.ssl_temporal_loss:
        coercions = []
        if args.static_image_fast_path:
            coercions.append("--static_image_fast_path → off (SSL needs per-t features)")
            args.static_image_fast_path = False
        print("=" * 60)
        print("--ssl_temporal_loss enabled (recurrent-LeJEPA, SSL-only):")
        print(f"  pair_mode = {args.ssl_pair_mode}")
        print(f"  predictive weight = {args.ssl_loss_weight}, "
              f"sigreg weight = {args.ssl_sigreg_weight}")
        print(f"  proj dim = {args.ssl_proj_dim}, "
              f"num_slices = {args.ssl_num_slices}, num_points = {args.ssl_num_points}")
        print(f"  probe LR (constant, no warmup) = {args.probe_lr}")
        if args.ssl_margin is not None or args.ssl_var_weight is not None:
            print("  [deprecated] --ssl_margin / --ssl_var_weight are no-ops under LeJEPA")
        for c in coercions:
            print(f"  coercion: {c}")
        print("=" * 60)

    # Setup
    num_devices = jax.device_count()
    global_batch_size = args.batch_size * num_devices

    # SHViT recipe: linear LR scaling by global batch size (ref: SHViT main.py:312)
    #   effective_lr = base_lr * global_batch_size / 512
    if args.shvit_recipe:
        base_lr = args.lr
        args.lr = base_lr * global_batch_size / 512.0
        print(f"SHViT LR scaling: base={base_lr:.1e} × {global_batch_size}/512 → {args.lr:.2e}")
        if args.min_lr is not None:
            base_min_lr = args.min_lr
            args.min_lr = base_min_lr * global_batch_size / 512.0
            print(f"SHViT min_lr scaling: {base_min_lr:.1e} → {args.min_lr:.2e}")

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

    # Mixed precision: compute in bf16, store params as fp32 for optimizer stability
    compute_dtype = jnp.bfloat16 if args.precision == 'bf16' else jnp.float32
    param_dtype = jnp.float32  # always fp32 for weights

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
        'static_image_fast_path': args.static_image_fast_path,
        'use_input_gates': not args.no_input_gates,
        'dtype': compute_dtype,
        'param_dtype': param_dtype,
    }
    # SSL flags are CSSMSHViT-specific (other CSSM model variants don't have
    # these fields). Add them only for cssm_shvit below.
    cssm_shvit_ssl_kwargs = {
        'ssl_proj_dim': args.ssl_proj_dim if args.ssl_temporal_loss else 0,
        'linear_probe': args.ssl_temporal_loss,
        'timestep_gate': args.timestep_gate,
        'gate_proj_bias_init': args.gate_proj_bias_init,
        'use_pos_conv': args.use_pos_conv_cli,
        'head_pool': args.head_pool,
    }

    if args.model == 'cssm_shvit':
        model_kwargs.update(cssm_kwargs)
        model_kwargs.update(cssm_shvit_ssl_kwargs)
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

    if args.data_loader == 'ego4d':
        from src.data.ego4d_wrapper import create_ego4d_imagenet_loader
        # ego4d has no classification labels; the probe CE computed on dummy
        # zeros is harmless under --ssl_temporal_loss (stop-grad on probe head).
        # Val uses ImageNet tfrecord so we still have a supervised signal.
        train_loader = create_ego4d_imagenet_loader(
            video_dir=args.ego4d_video_dir,
            manifest_path=args.ego4d_manifest,
            batch_size=global_batch_size,
            num_frames=args.num_timesteps,
            frame_stride=args.ego4d_frame_stride,
            resolution=args.image_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = get_tfrecord_imagenet_loader(
            tfrecord_dir=args.tfrecord_dir,
            split='val',
            batch_size=global_batch_size,
            image_size=args.image_size,
            max_intraop=args.tf_max_intraop,
        )
        print(f"  Ego4D: {len(train_loader)} batches/epoch "
              f"(T={args.num_timesteps}, stride={args.ego4d_frame_stride})")
        print(f"  Val: ImageNet tfrecord (static, T-replicated)")
    elif args.data_loader == 'dali':
        from src.data.imagenet_dali import get_dali_imagenet_loader
        train_loader = get_dali_imagenet_loader(
            tfrecord_dir=args.tfrecord_dir,
            split='train',
            batch_size=global_batch_size,
            image_size=args.image_size,
            repeated_aug=args.repeated_aug,
        )
        val_loader = get_dali_imagenet_loader(
            tfrecord_dir=args.tfrecord_dir,
            split='val',
            batch_size=global_batch_size,
            image_size=args.image_size,
        )
        if args.prefetch_to_device:
            from src.data.imagenet_tfdata import DevicePrefetchLoader
            train_loader = DevicePrefetchLoader(
                train_loader, num_devices=num_devices,
                prefetch_depth=args.prefetch_depth,
            )
            val_loader = DevicePrefetchLoader(
                val_loader, num_devices=num_devices,
                prefetch_depth=args.prefetch_depth,
            )
            print(f"  DevicePrefetchLoader enabled (depth={args.prefetch_depth}, "
                  f"shard+device_put on background thread)")
    elif args.data_loader == 'tfrecord':
        tfr_kwargs = dict(
            num_parallel_reads=args.tf_parallel_reads,
            prefetch_batches=args.tf_prefetch_batches,
            threadpool_size=args.tf_threadpool_size,
        )
        train_loader = get_tfrecord_imagenet_loader(
            tfrecord_dir=args.tfrecord_dir,
            split='train',
            batch_size=global_batch_size,
            image_size=args.image_size,
            repeated_aug=args.repeated_aug,
            max_intraop=args.tf_max_intraop,
            **tfr_kwargs,
        )
        val_loader = get_tfrecord_imagenet_loader(
            tfrecord_dir=args.tfrecord_dir,
            split='val',
            batch_size=global_batch_size,
            image_size=args.image_size,
            max_intraop=args.tf_max_intraop,
            **tfr_kwargs,
        )
        if args.prefetch_to_device:
            from src.data.imagenet_tfdata import DevicePrefetchLoader
            train_loader = DevicePrefetchLoader(
                train_loader, num_devices=num_devices,
                prefetch_depth=args.prefetch_depth,
            )
            val_loader = DevicePrefetchLoader(
                val_loader, num_devices=num_devices,
                prefetch_depth=args.prefetch_depth,
            )
            print(f"  DevicePrefetchLoader enabled (depth={args.prefetch_depth}, "
                  f"shard+device_put on background thread)")
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

    # Create training state. SSL mode adds:
    #   1. WD mask excluding 'head' and 'linear_probe_head' — without it, the
    #      main head receives no gradient signal and AdamW silently decays it
    #      to zero, making the top1_main watchdog lie.
    #   2. A multi-transform optimizer with constant probe_lr (no warmup) for
    #      params under linear_probe_head/*.
    if args.ssl_temporal_loss:
        no_decay_names = ('head', 'linear_probe_head')
        probe_lr = args.probe_lr
    else:
        no_decay_names = ()
        probe_lr = None

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
        no_decay_names=no_decay_names,
        probe_lr=probe_lr,
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
        # Use mixup-compatible train step (expects soft labels). The mixup
        # train step is the only one wired for SSL today — when
        # --ssl_temporal_loss is set, the SHViT recipe's mixup is the path.
        train_step_fn = make_train_step_mixup(
            model, num_classes=1000, label_smoothing=args.label_smoothing,
            ssl_temporal_loss=args.ssl_temporal_loss,
            ssl_loss_weight=args.ssl_loss_weight,
            ssl_sigreg_weight=args.ssl_sigreg_weight,
            ssl_pair_mode=args.ssl_pair_mode,
            ssl_num_slices=args.ssl_num_slices,
            ssl_num_points=args.ssl_num_points,
        )
        print("  Using mixup/cutmix training step with soft labels"
              + (" + SSL temporal loss" if args.ssl_temporal_loss else ""))
    else:
        train_step_fn = make_train_step(model, num_classes=1000)

    use_ema_early = args.ema_decay > 0
    ema_decay_val = args.ema_decay

    # Mixed-precision compute dtype (bf16 or fp32). Used for on-device
    # uint8→float cast and ImageNet normalization inside pmap wrappers.
    _COMPUTE_DTYPE = jnp.bfloat16 if args.precision == 'bf16' else jnp.float32
    _IMAGENET_MEAN_DEV = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
    _IMAGENET_STD_DEV = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)

    def _uint8_to_normalized(images_u8):
        """Convert uint8 [0,255] to ImageNet-normalized compute dtype.
        The TF loader yields uint8 to make the host→device transfer 4× smaller;
        normalization runs on device where it's essentially free."""
        img_f = images_u8.astype(jnp.float32) * (1.0 / 255.0)
        img_f = (img_f - _IMAGENET_MEAN_DEV) / _IMAGENET_STD_DEV
        return img_f.astype(_COMPUTE_DTYPE)

    # Create parallel eval step
    eval_step_fn = make_eval_step(
        model, num_classes=1000, linear_probe=args.ssl_temporal_loss
    )

    def parallel_eval_step_wrapper(state, batch, batch_stats):
        images, labels = batch
        # Loader yields uint8 — cast+normalize on device.
        if images.dtype == jnp.uint8:
            images = _uint8_to_normalized(images)
        return eval_step_fn(state, (images, labels), batch_stats=batch_stats, axis_name='batch')

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

    # GPU RandAugment via imax library
    def _randaugment_single(image_01, rng):
        """Apply RandAugment to a single image. Expects input in [0, 1] float.

        Loader yields uint8; _augment_on_device casts to [0,1] float before calling
        this. Output is still [0, 1] float — ImageNet normalization happens once at
        the end of _augment_on_device, not round-tripped here.
        """
        # imax RandAugment: input [0, 1], output may be [0, 255] (some ops use uint8 internally)
        img_aug = _imax_randaugment.distort_image_with_randaugment(
            image_01,
            num_layers=args.randaugment_num_ops,
            magnitude=args.randaugment_magnitude,
            random_key=rng,
        )
        img_aug = jnp.where(img_aug > 1.5, img_aug / 255.0, img_aug)
        img_aug = jnp.clip(img_aug, 0.0, 1.0)
        return img_aug

    @jax.jit
    def apply_randaugment(images, rng):
        """Apply RandAugment to a batch of images on GPU."""
        batch_size = images.shape[0]
        rngs = jax.random.split(rng, batch_size)
        return jax.vmap(_randaugment_single)(images, rngs)

    # Determine if we should use GPU augmentation. Applies to any loader that
    # yields uint8 or float images — _augment_on_device converts to [0,1] fp32
    # before the augment block, so RandAugment / 3-Augment work regardless of
    # whether the data came from tfrecord, dali, or streaming.
    use_gpu_augment = (args.augmentation in ['3aug', 'randaugment'] and
                       not args.no_augmentation)
    use_randaugment = (use_gpu_augment and args.augmentation == 'randaugment')
    if use_gpu_augment:
        if use_randaugment:
            print(f"  Using GPU RandAugment (imax, {args.randaugment_num_ops} ops, mag {args.randaugment_magnitude}) — loader={args.data_loader}")
        else:
            print(f"  Using GPU 3-Augment — loader={args.data_loader}")

    # Raw (non-jitted) augmentation bodies — called inside pmap so each device
    # augments its own shard in parallel. The @jax.jit wrappers above remain
    # for code paths that still call them outside pmap (e.g. validation, probes).
    use_random_erase = args.random_erase_prob > 0

    def _augment_on_device(images, labels, rng):
        """Per-device augmentation pipeline: uint8→[0,1] → randaugment/3aug →
        mixup → random_erase → ImageNet-normalize → bf16/fp32 compute dtype.

        Runs inside pmap so each device processes only its local shard in
        parallel with the other device. `labels` starts as int class indices
        and comes out one-hot (soft if mixup is used).

        The loader yields uint8 in [0, 255]; all augments run in [0, 1] fp32
        space; ImageNet normalization + dtype cast happens exactly once at the
        end, replacing the old un-normalize/re-normalize round-trip.

        Video (5D) input (ego4d) skips image-level augmentations — frame-mixing
        mixup/cutmix + randaugment don't make sense on a temporally coherent
        clip. Just normalize per-frame and cast.
        """
        if images.ndim == 5:
            # (B, T, H, W, C) ego4d video path. Normalize + cast, no aug.
            if images.dtype == jnp.uint8:
                images = images.astype(jnp.float32) * (1.0 / 255.0)
            # ImageNet mean/std broadcast over T spatial: shape (1,1,1,1,3).
            mean = _IMAGENET_MEAN_DEV.reshape(1, 1, 1, 1, 3)
            std = _IMAGENET_STD_DEV.reshape(1, 1, 1, 1, 3)
            images = (images - mean) / std
            images = images.astype(_COMPUTE_DTYPE)
            labels = jax.nn.one_hot(labels, 1000)
            return images, labels

        rng_aug, rng_mixup, rng_erase = jax.random.split(rng, 3)

        # Normalize input to [0, 1] fp32 space for augmentation math.
        # Two supported inputs: uint8 (new hot-path from TFRecord loader) or
        # already-normalized fp32 (legacy streaming loader). Augments always
        # run in [0, 1], normalization happens once at the end.
        if images.dtype == jnp.uint8:
            images = images.astype(jnp.float32) * (1.0 / 255.0)
        else:
            images = images * _IMAGENET_STD_DEV + _IMAGENET_MEAN_DEV
            images = jnp.clip(images, 0.0, 1.0)

        if use_gpu_augment:
            if use_randaugment:
                bs = images.shape[0]
                rngs = jax.random.split(rng_aug, bs)
                images = jax.vmap(_randaugment_single)(images, rngs)
            else:
                images = three_augment_jax(
                    images, rng_aug, color_jitter=args.color_jitter
                )

        if use_mixup:
            images, labels = mixup_cutmix(
                images, labels, rng_mixup,
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
                mixup_prob=args.mixup_prob,
                cutmix_prob=1.0 - args.mixup_prob,
                num_classes=1000,
            )
        else:
            labels = jax.nn.one_hot(labels, 1000)

        if use_random_erase:
            # Erase to 0 in [0, 1] space = black; still valid before normalization.
            images = random_erasing(
                images, rng_erase, probability=args.random_erase_prob
            )

        # Final ImageNet normalization + cast to compute dtype, once.
        images = (images - _IMAGENET_MEAN_DEV) / _IMAGENET_STD_DEV
        images = images.astype(_COMPUTE_DTYPE)

        return images, labels

    def parallel_train_step_wrapper(state, batch, rng, batch_stats, ema_params):
        # Split device-local RNG: one for augmentation, one for the train step
        rng_aug, rng_step = jax.random.split(rng)

        images, labels = batch
        images, labels = _augment_on_device(images, labels, rng_aug)
        augmented_batch = (images, labels)

        new_state, metrics, new_batch_stats = train_step_fn(
            state, augmented_batch, rng_step, batch_stats=batch_stats, axis_name='batch'
        )
        # Fused EMA update inside the same pmap (no host round-trip)
        if use_ema_early:
            new_ema_params = jax.tree_util.tree_map(
                lambda e, p: ema_decay_val * e + (1.0 - ema_decay_val) * p,
                ema_params, new_state.params,
            )
        else:
            new_ema_params = ema_params
        return new_state, metrics, new_batch_stats, new_ema_params

    # donate state AND ema_params so XLA can reuse buffers
    p_train_step = jax.pmap(
        parallel_train_step_wrapper, axis_name='batch',
        donate_argnums=(0, 4) if use_ema_early else (0,),
    )

    # Replicate state and batch_stats for multi-GPU
    state = replicate_state(state)
    if has_batch_stats:
        batch_stats = jax_utils.replicate(batch_stats)

    # Initialize EMA params (replicated, updated inside fused pmap train step)
    use_ema = args.ema_decay > 0
    ema_state = None
    if use_ema:
        # Copy state.params to a separate buffer (not sharing with state)
        ema_state = {
            'ema_params': jax.tree_util.tree_map(lambda x: x + 0.0, state.params),
            'decay': args.ema_decay,
        }
        print(f"  Using EMA with decay={args.ema_decay} (fused into train step)")

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
                    # Restored as TrainState directly — fall back to old replace path.
                    restored_state = restored
                    print(f"  Checkpoint format: TrainState object")
                elif isinstance(restored, dict):
                    # Restored as dict — reconstruct TrainState via structural merge.
                    print(f"  Checkpoint format: dict with keys {list(restored.keys())}")
                    if 'params' not in restored:
                        raise ValueError(f"Checkpoint dict missing 'params'. Keys: {restored.keys()}")

                    # Get step from checkpoint
                    ckpt_step = restored.get('step', 0)
                    if hasattr(ckpt_step, 'item'):
                        ckpt_step = int(ckpt_step.item())
                    else:
                        ckpt_step = int(ckpt_step)
                    print(f"  Checkpoint step: {ckpt_step}")

                    # Structural merge: take checkpoint values where keys exist,
                    # keep freshly-initialized values for any new keys (SSL
                    # projection heads, linear_probe_head, etc.). This lets the
                    # SSL training resume from a supervised checkpoint that
                    # doesn't have those subtrees.
                    import flax
                    def _merge(target, source, path=()):
                        if isinstance(target, (dict, flax.core.FrozenDict)):
                            tgt = (flax.core.unfreeze(target)
                                   if isinstance(target, flax.core.FrozenDict)
                                   else target)
                            src = (flax.core.unfreeze(source)
                                   if isinstance(source, flax.core.FrozenDict)
                                   else source) if source is not None else {}
                            out = {}
                            fresh = []
                            for k, v in tgt.items():
                                if k in src:
                                    out[k] = _merge(v, src[k], path + (k,))
                                else:
                                    fresh.append('/'.join(path + (k,)))
                                    out[k] = v
                            for f in fresh:
                                print(f"  [resume] fresh-init: {f}")
                            return out
                        # Leaf
                        if hasattr(target, 'shape') and hasattr(source, 'shape'):
                            if target.shape == source.shape:
                                return source
                            print(f"  [resume] shape mismatch at {'/'.join(path)}: "
                                  f"ckpt {source.shape} vs init {target.shape}, keeping init")
                            return target
                        return source

                    merged_params = _merge(single_state.params, restored['params'])

                    # Step counter: SSL continuation needs a fresh schedule, so
                    # reset to 0. Without this, resuming a supervised checkpoint
                    # at step ~250k drops the LR schedule into its min_lr tail.
                    if args.ssl_temporal_loss:
                        new_step = 0
                        print(f"  [resume] SSL continuation: step counter reset to 0 "
                              f"(was {ckpt_step}) — fresh cosine schedule")
                    else:
                        new_step = ckpt_step

                    restored_state = single_state.replace(
                        params=merged_params,
                        step=new_step,
                    )
                    print(f"  Restored params (structural merge) and step={new_step}")
                else:
                    # Assume it's a TrainState-like object
                    restored_state = restored
                    print(f"  Checkpoint format: {type(restored)}")

                # Sanity check (no longer strict — merge may legitimately add params)
                orig_leaves = jax.tree_util.tree_leaves(single_state.params)
                restored_leaves = jax.tree_util.tree_leaves(restored_state.params)
                print(f"  Param leaves: init={len(orig_leaves)}, "
                      f"after-merge={len(restored_leaves)}")

                # Check a few param values aren't all zeros/random
                sample_param = restored_leaves[0]
                print(f"  Restored params sample - mean: {float(jnp.mean(sample_param)):.6f}, std: {float(jnp.std(sample_param)):.6f}")

                # Re-replicate for multi-GPU
                state = replicate_state(restored_state)
                print(f"  Successfully restored and replicated state")

                # Rebuild EMA from the MERGED params, not the fresh-init buffer
                # that was created before resume. Without this, EMA would track
                # random init for many epochs and pollute val/ema metrics.
                if use_ema:
                    ema_state['ema_params'] = jax.tree_util.tree_map(
                        lambda x: x + 0.0,
                        jax_utils.unreplicate(state.params),
                    )
                    ema_state['ema_params'] = jax_utils.replicate(ema_state['ema_params'])
                    print(f"  Rebuilt EMA params from merged checkpoint state")

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
                        # Checkpoint holds unreplicated params; restore then replicate
                        ema_target = jax_utils.unreplicate(ema_state['ema_params'])
                        ema_restored = checkpointer.restore(ema_ckpt_path, {'ema_params': ema_target})
                        if isinstance(ema_restored, dict) and 'ema_params' in ema_restored:
                            restored_params = ema_restored['ema_params']
                        else:
                            restored_params = ema_restored
                        ema_state['ema_params'] = jax_utils.replicate(restored_params)
                        print(f"  Restored EMA params from {ema_ckpt_path}")
                    except Exception as e:
                        print(f"  Warning: Could not restore EMA checkpoint: {e}")

            # Determine starting epoch
            if args.ssl_temporal_loss:
                # SSL continuation always restarts the epoch counter so the
                # cosine LR schedule, save cadence, and probe metrics start
                # fresh — regardless of which supervised checkpoint we resumed.
                start_epoch = 0
                print(f"  [resume] SSL continuation: start_epoch reset to 0")
            elif args.resume_epoch is not None:
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
        # Raw int labels — augment-fused pmap wrapper one-hot-encodes internally
        dummy_labels = jnp.zeros(global_batch_size, dtype=jnp.int32)

        for T in variable_timesteps:
            print(f"  Compiling T={T}...", end=" ", flush=True)
            dummy_images = jnp.repeat(dummy_images_base[:, None, :, :, :], T, axis=1)
            dummy_batch = shard_batch((dummy_images, dummy_labels), num_devices)
            rng, warmup_rng = jax.random.split(rng)
            warmup_rngs = split_rng_for_devices(warmup_rng)
            # Run one forward+backward pass to trigger compilation
            ema_in = ema_state['ema_params'] if use_ema else state.params
            state, _, _, new_ema = p_train_step(state, dummy_batch, warmup_rngs, batch_stats, ema_in)
            if use_ema:
                ema_state['ema_params'] = new_ema
            jax.block_until_ready(state)
            print("done")
        print("Pre-compilation complete!\n")

    LOG_EVERY = 50  # sync metrics to host every N steps

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Training
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        epoch_step_times = []
        pending_metrics = []
        step_start_window = time.perf_counter()

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True) as pbar:
            for batch in pbar:
                rng, step_rng = jax.random.split(rng)
                images, labels = batch

                # Variable timesteps: sample T per batch and expand images to (B, T, H, W, C)
                # (Still host-side — rare path, not on the hot loop.)
                if variable_timesteps is not None:
                    T = random.choice(variable_timesteps)
                    images_np = np.asarray(images)
                    if images_np.ndim == 4:
                        images = jnp.array(np.repeat(images_np[:, None, :, :, :], T, axis=1))
                    elif images_np.ndim == 5:
                        images = jnp.array(np.repeat(images_np[:, -1:, :, :, :], T, axis=1))

                batch = (images, labels)

                # Shard RAW batch across devices (no host-side augmentation —
                # augment + mixup + erasing run INSIDE the pmap wrapper, so each
                # device augments its own shard in parallel).
                # When --prefetch_to_device is active, the DevicePrefetchLoader
                # background thread has ALREADY sharded + device_put, so batch
                # is already a pair of ShardedDeviceArray and we skip here.
                if not args.prefetch_to_device:
                    batch = shard_batch(batch, num_devices)

                # Split RNG for each device (covers augment + train step inside pmap)
                step_rngs = split_rng_for_devices(step_rng)

                # Training step (augment + fwd + bwd + EMA all fused inside pmap)
                ema_in = ema_state['ema_params'] if use_ema else state.params
                state, metrics, new_batch_stats, new_ema = p_train_step(
                    state, batch, step_rngs, batch_stats, ema_in
                )
                if has_batch_stats and new_batch_stats is not None:
                    batch_stats = new_batch_stats
                if use_ema:
                    ema_state['ema_params'] = new_ema

                # Accumulate device arrays (no host sync yet)
                pending_metrics.append(metrics)
                num_batches += 1
                global_step += 1

                # Sync metrics every LOG_EVERY steps (amortizes host transfer cost)
                if global_step % LOG_EVERY == 0:
                    # Stack then reduce on device; single transfer for both values
                    stacked_loss = jnp.stack([m['loss'][0] for m in pending_metrics])
                    stacked_acc = jnp.stack([m['acc'][0] for m in pending_metrics])
                    loss = float(jnp.mean(stacked_loss))
                    acc = float(jnp.mean(stacked_acc))

                    ssl_metrics_summary = {}
                    if args.ssl_temporal_loss:
                        for k in ('pred_loss', 'sigreg_loss', 'probe_ce', 'probe_acc', 'min_z_var'):
                            stacked_k = jnp.stack([m[k][0] for m in pending_metrics])
                            # min_z_var uses pmin reduction, others mean
                            reducer = jnp.min if k == 'min_z_var' else jnp.mean
                            ssl_metrics_summary[k] = float(reducer(stacked_k))

                    step_time = (time.perf_counter() - step_start_window) / len(pending_metrics)
                    epoch_step_times.append(step_time)
                    epoch_loss += loss * len(pending_metrics)
                    epoch_acc += acc * len(pending_metrics)
                    pending_metrics = []
                    step_start_window = time.perf_counter()

                    postfix = {
                        'loss': f"{loss:.4f}",
                        'acc': f"{acc:.4f}",
                        'ms/step': f"{step_time*1000:.1f}",
                    }
                    if args.ssl_temporal_loss:
                        postfix['pred'] = f"{ssl_metrics_summary['pred_loss']:.3f}"
                        postfix['sig'] = f"{ssl_metrics_summary['sigreg_loss']:.4f}"
                        postfix['probe'] = f"{ssl_metrics_summary['probe_acc']:.3f}"
                        postfix['min_var'] = f"{ssl_metrics_summary['min_z_var']:.4f}"
                    pbar.set_postfix(postfix)

                    if not args.no_wandb:
                        log_dict = {
                            'train/loss': loss,
                            'train/acc': acc,
                            'timing/step_ms': step_time * 1000,
                            'timing/throughput': global_batch_size / step_time,
                        }
                        if args.ssl_temporal_loss:
                            log_dict.update({
                                'train/pred_loss': ssl_metrics_summary['pred_loss'],
                                'train/sigreg_loss': ssl_metrics_summary['sigreg_loss'],
                                'train/probe_ce': ssl_metrics_summary['probe_ce'],
                                'train/probe_acc': ssl_metrics_summary['probe_acc'],
                                'train/min_z_var': ssl_metrics_summary['min_z_var'],
                            })
                        log_dict.update(get_gpu_memory_stats())
                        wandb.log(log_dict, step=global_step)

        # Flush any pending metrics from the last incomplete window
        if pending_metrics:
            stacked_loss = jnp.stack([m['loss'][0] for m in pending_metrics])
            stacked_acc = jnp.stack([m['acc'][0] for m in pending_metrics])
            flush_loss = float(jnp.mean(stacked_loss))
            flush_acc = float(jnp.mean(stacked_acc))
            epoch_loss += flush_loss * len(pending_metrics)
            epoch_acc += flush_acc * len(pending_metrics)
            pending_metrics = []

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        avg_step_time = sum(epoch_step_times) / max(len(epoch_step_times), 1) * 1000

        # Get GPU memory for epoch summary
        gpu_mem = get_gpu_memory_stats()
        mem_str = ", ".join([f"GPU{i}: {gpu_mem.get(f'gpu{i}/mem_used_gb', 0):.1f}GB"
                             for i in range(num_devices)])
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}, avg_step={avg_step_time:.1f}ms, {mem_str}")

        # Validation
        if (epoch + 1) % args.eval_every == 0:
            val_loss = 0.0
            val_acc = 0.0
            val_probe_acc = 0.0  # only populated when args.ssl_temporal_loss
            val_batches = 0

            for batch in tqdm(val_loader, desc="Validation", leave=False, dynamic_ncols=True):
                if not args.prefetch_to_device:
                    batch = shard_batch(batch, num_devices)
                metrics = p_eval_step(state, batch, batch_stats)

                val_loss += float(metrics['loss'][0])
                val_acc += float(metrics['acc'][0])
                if args.ssl_temporal_loss:
                    val_probe_acc += float(metrics['probe_acc'][0])
                val_batches += 1

            val_loss /= max(val_batches, 1)
            val_acc /= max(val_batches, 1)
            val_probe_acc /= max(val_batches, 1)

            if args.ssl_temporal_loss:
                print(f"  Val: top1_main={val_acc:.4f} (frozen), "
                      f"top1_probe={val_probe_acc:.4f} (rising), loss={val_loss:.4f}")
            else:
                print(f"  Val: loss={val_loss:.4f}, acc={val_acc:.4f}")

            # EMA Validation (if enabled)
            ema_val_loss = 0.0
            ema_val_acc = 0.0
            ema_val_probe_acc = 0.0
            if use_ema:
                # EMA params are already replicated - use directly
                ema_state_temp = state.replace(params=ema_state['ema_params'])

                ema_val_batches = 0
                for batch in tqdm(val_loader, desc="EMA Validation", leave=False, dynamic_ncols=True):
                    if not args.prefetch_to_device:
                        batch = shard_batch(batch, num_devices)
                    metrics = p_eval_step(ema_state_temp, batch, batch_stats)

                    ema_val_loss += float(metrics['loss'][0])
                    ema_val_acc += float(metrics['acc'][0])
                    if args.ssl_temporal_loss:
                        ema_val_probe_acc += float(metrics['probe_acc'][0])
                    ema_val_batches += 1

                ema_val_loss /= max(ema_val_batches, 1)
                ema_val_acc /= max(ema_val_batches, 1)
                ema_val_probe_acc /= max(ema_val_batches, 1)

                if args.ssl_temporal_loss:
                    print(f"  EMA Val: top1_main={ema_val_acc:.4f}, "
                          f"top1_probe={ema_val_probe_acc:.4f}, loss={ema_val_loss:.4f}")
                else:
                    print(f"  EMA Val: loss={ema_val_loss:.4f}, acc={ema_val_acc:.4f}")

            if not args.no_wandb:
                val_log = {
                    'val/loss': val_loss,
                    'val/acc': val_acc,
                    'epoch': epoch + 1,
                }
                if args.ssl_temporal_loss:
                    val_log['val/top1_main'] = val_acc
                    val_log['val/top1_probe'] = val_probe_acc
                if use_ema:
                    val_log['val/ema_loss'] = ema_val_loss
                    val_log['val/ema_acc'] = ema_val_acc
                    if args.ssl_temporal_loss:
                        val_log['val/ema_top1_probe'] = ema_val_probe_acc
                val_log.update(get_gpu_memory_stats())
                wandb.log(val_log, step=global_step)

            # Track best model. In SSL mode, "best" = best probe accuracy
            # (the only signal that reflects backbone improvement).
            if args.ssl_temporal_loss:
                check_acc = ema_val_probe_acc if use_ema else val_probe_acc
            else:
                check_acc = ema_val_acc if use_ema else val_acc
            if check_acc > best_val_acc:
                best_val_acc = check_acc
                best_is_ema = use_ema
                metric_name = "probe accuracy" if args.ssl_temporal_loss else "validation accuracy"
                print(f"  New best {metric_name}: {best_val_acc:.4f}" +
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
                    checkpointer.save(ema_ckpt_path, {'ema_params': jax_utils.unreplicate(ema_state['ema_params'])})
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
