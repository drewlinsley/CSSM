"""
Main training script for CSSM models.

Supports two architectures:
- ConvNeXt-style: Pure/Hybrid CSSM blocks with ConvNeXt structure
- ViT-style: Clean pre-norm transformer blocks with CSSM replacing attention

Training features:
- Cosine learning rate scheduling
- Gradient clipping
- Validation evaluation
- Checkpointing (orbax)
- Weights & Biases logging
"""

# =============================================================================
# IMPORTANT: Configure TensorFlow to use CPU-only BEFORE importing it
# This prevents TensorFlow (used for TFRecord data loading) from conflicting
# with JAX's NCCL for multi-GPU training.
# =============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warning logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Don't preallocate if TF does use GPU
# =============================================================================

import argparse
import pickle
import re
import time
from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import linen as nn
from flax import jax_utils
from flax.training import train_state
from tqdm import tqdm

# =============================================================================
# Performance optimizations
# =============================================================================
def setup_jax_optimizations(use_cache: bool = True, use_bf16: bool = False):
    """Configure JAX for optimal performance."""
    # Enable persistent compilation cache (huge speedup on restarts)
    if use_cache:
        cache_dir = os.path.expanduser('~/.cache/jax_compilation_cache')
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update('jax_compilation_cache_dir', cache_dir)

    # Enable bfloat16 matrix multiplication (faster on modern GPUs)
    if use_bf16:
        jax.config.update('jax_default_matmul_precision', 'bfloat16')

    # Disable debug nans check in production (small speedup)
    jax.config.update('jax_debug_nans', False)


# =============================================================================
# Simple pickle-based checkpointer (NFS-safe, no TensorStore dependency)
# =============================================================================
class SimpleCheckpointer:
    """
    Simple pickle-based checkpointer that works reliably on NFS.

    Orbax/TensorStore use atomic renames which fail on many NFS setups.
    This saves only params/batch_stats/epoch (not optimizer state which has
    unpicklable closures from optax.apply_if_finite).
    """

    def save(self, path: str, state: Any) -> None:
        """Save state to pickle file (only picklable parts)."""
        os.makedirs(path, exist_ok=True)

        # Convert JAX arrays to numpy for pickling
        def to_numpy(x):
            if hasattr(x, 'numpy'):
                return np.array(x)
            elif isinstance(x, jnp.ndarray):
                return np.array(x)
            return x

        # Only save the parts we need (params, batch_stats, epoch, step)
        # Skip opt_state and apply_fn which have unpicklable closures
        checkpoint_data = {
            'params': jax.tree_util.tree_map(to_numpy, state.params),
            'epoch': int(state.epoch) if hasattr(state.epoch, 'item') else state.epoch,
            'step': int(state.step) if hasattr(state.step, 'item') else state.step,
        }
        if hasattr(state, 'batch_stats') and state.batch_stats is not None:
            checkpoint_data['batch_stats'] = jax.tree_util.tree_map(to_numpy, state.batch_stats)

        # Save to temp file first, then rename (more atomic)
        tmp_path = os.path.join(path, 'checkpoint.pkl.tmp')
        final_path = os.path.join(path, 'checkpoint.pkl')

        with open(tmp_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        # Rename (if this fails on NFS, just copy)
        try:
            os.rename(tmp_path, final_path)
        except OSError:
            import shutil
            shutil.copy2(tmp_path, final_path)
            os.remove(tmp_path)

    def restore(self, path: str, state: Any) -> Any:
        """Restore state from pickle file."""
        ckpt_file = os.path.join(path, 'checkpoint.pkl')

        with open(ckpt_file, 'rb') as f:
            checkpoint_data = pickle.load(f)

        # Convert numpy arrays back to JAX arrays
        def to_jax(x):
            if isinstance(x, np.ndarray):
                return jnp.array(x)
            return x

        # Restore into the existing state structure (keeps opt_state, apply_fn)
        restored_params = jax.tree_util.tree_map(to_jax, checkpoint_data['params'])
        restored_epoch = jnp.array(checkpoint_data.get('epoch', 0))

        # Update state with restored values
        state = state.replace(params=restored_params, epoch=restored_epoch)

        if 'batch_stats' in checkpoint_data and checkpoint_data['batch_stats'] is not None:
            restored_batch_stats = jax.tree_util.tree_map(to_jax, checkpoint_data['batch_stats'])
            state = state.replace(batch_stats=restored_batch_stats)

        return state

from src.models.cssm_vit import CSSMViT, cssm_vit_tiny, cssm_vit_small
from src.models.baseline_vit import BaselineViT, baseline_vit_tiny, baseline_vit_small
from src.models.simple_cssm import SimpleCSSM
from src.data import get_imagenette_video_loader, get_dataset_info, get_imagenet_loader, get_imagenet_info

# =============================================================================
# Configure TensorFlow to use CPU only BEFORE importing data loaders
# This prevents TensorFlow from conflicting with JAX's NCCL for multi-GPU
# =============================================================================
try:
    import tensorflow as tf
    # Hide all GPUs from TensorFlow - it only needs CPU for data loading
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    pass  # TensorFlow not installed
except RuntimeError:
    pass  # Virtual devices already initialized

from src.pathfinder_data import get_pathfinder_loader, get_pathfinder_info, get_pathfinder_tfrecord_loader
from src.cabc_data import get_cabc_loader, get_cabc_info, get_cabc_tfrecord_loader


# Multi-GPU utilities
def get_device_setup():
    """Get device information for multi-GPU training."""
    devices = jax.devices()
    num_devices = len(devices)
    return devices, num_devices, num_devices > 1


def pmean_complex_safe(x, axis_name='devices'):
    """
    pmean that handles complex-valued arrays by converting to real representation.

    NCCL doesn't support complex types directly. We stack real and imaginary
    parts into a single float array, do pmean, then unstack.
    """
    def _pmean_leaf(leaf):
        # Use explicit dtype comparison instead of issubdtype for traced context
        is_complex = (leaf.dtype == jnp.complex64 or leaf.dtype == jnp.complex128)
        if is_complex:
            # Stack real and imag into last dimension: (..., 2)
            stacked = jnp.stack([jnp.real(leaf), jnp.imag(leaf)], axis=-1)
            stacked = stacked.astype(jnp.float32)
            # Single pmean on the float array
            stacked_mean = jax.lax.pmean(stacked, axis_name=axis_name)
            # Unstack and recombine to complex
            real_mean = stacked_mean[..., 0]
            imag_mean = stacked_mean[..., 1]
            return (real_mean + 1j * imag_mean).astype(leaf.dtype)
        else:
            return jax.lax.pmean(leaf, axis_name=axis_name)

    return jax.tree_util.tree_map(_pmean_leaf, x)


class TrainState(train_state.TrainState):
    """Extended train state with additional tracking."""
    epoch: jnp.ndarray = None  # Use array for multi-GPU compatibility
    batch_stats: Any = None  # For BatchNorm running statistics


def create_train_state(
    rng: jax.Array,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    total_steps: int,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
) -> Tuple[TrainState, optax.GradientTransformation]:
    """
    Initialize training state with optimizer and LR schedule.

    Args:
        rng: Random key for initialization
        model: The model to train
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        total_steps: Total training steps for LR schedule
        warmup_steps: Number of warmup steps
        grad_clip: Maximum gradient norm

    Returns:
        Tuple of (train_state, optimizer)
    """
    # Create dummy input for initialization
    dummy_input = jnp.ones((1, 8, 224, 224, 3))
    variables = model.init({'params': rng, 'dropout': rng}, dummy_input, training=False)
    params = variables['params']
    batch_stats = variables.get('batch_stats', None)  # For BatchNorm

    # Learning rate schedule: warmup + cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=learning_rate * 0.01,
    )

    # Optimizer: AdamW with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )

    # Wrap with apply_if_finite to skip updates on NaN/Inf gradients
    # This prevents NaN propagation and helps debug where instability originates
    tx = optax.apply_if_finite(tx, max_consecutive_errors=5)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        epoch=jnp.array(0),  # Use JAX array for multi-GPU compatibility
        batch_stats=batch_stats,
    )

    return state, tx


@partial(jax.jit, static_argnums=(3, 4))
def train_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jax.Array,
    num_classes: int,
    has_batch_stats: bool = False,
) -> Tuple[TrainState, Dict[str, float]]:
    """
    Single training step.

    Args:
        state: Current training state
        batch: Tuple of (videos, labels)
        rng: Random key for dropout
        num_classes: Number of output classes
        has_batch_stats: Whether model uses BatchNorm

    Returns:
        Tuple of (updated_state, metrics_dict)

    Note: donate_argnums=(0,) allows JAX to reuse the state's memory buffer,
    reducing memory allocation overhead.
    """
    videos, labels = batch

    def loss_fn(params, batch_stats):
        variables = {'params': params}
        if has_batch_stats:
            variables['batch_stats'] = batch_stats
            output, mutated = state.apply_fn(
                variables,
                videos,
                training=True,
                rngs={'dropout': rng},
                mutable=['batch_stats'],
            )
            return optax.softmax_cross_entropy(
                output, jax.nn.one_hot(labels, num_classes)
            ).mean(), (output, mutated['batch_stats'])
        else:
            logits = state.apply_fn(
                variables,
                videos,
                training=True,
                rngs={'dropout': rng},
            )
            return optax.softmax_cross_entropy(
                logits, jax.nn.one_hot(labels, num_classes)
            ).mean(), (logits, None)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_batch_stats)), grads = grad_fn(state.params, state.batch_stats)

    # Update state
    state = state.apply_gradients(grads=grads)
    if has_batch_stats and new_batch_stats is not None:
        state = state.replace(batch_stats=new_batch_stats)

    # Compute metrics
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)

    metrics = {
        'train_loss': loss,
        'train_acc': acc,
    }

    return state, metrics


@partial(jax.jit, static_argnums=(2, 3))
def eval_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    num_classes: int,
    has_batch_stats: bool = False,
) -> Dict[str, float]:
    """
    Single evaluation step.

    Args:
        state: Current training state
        batch: Tuple of (videos, labels)
        num_classes: Number of output classes
        has_batch_stats: Whether model uses BatchNorm

    Returns:
        Dictionary of metrics
    """
    videos, labels = batch

    variables = {'params': state.params}
    if has_batch_stats:
        variables['batch_stats'] = state.batch_stats

    logits = state.apply_fn(
        variables,
        videos,
        training=False,
    )

    one_hot = jax.nn.one_hot(labels, num_classes)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)

    return {
        'val_loss': loss,
        'val_acc': acc,
    }


def evaluate(
    state: TrainState,
    val_loader,
    num_classes: int,
    num_batches: int = None,
    has_batch_stats: bool = False,
) -> Dict[str, float]:
    """
    Run full validation evaluation.

    Args:
        state: Current training state
        val_loader: Validation data iterator
        num_classes: Number of output classes
        num_batches: Max batches to evaluate (None = all)
        has_batch_stats: Whether model uses BatchNorm

    Returns:
        Dictionary of averaged metrics
    """
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    for i, batch in enumerate(val_loader):
        if num_batches is not None and i >= num_batches:
            break

        metrics = eval_step(state, batch, num_classes, has_batch_stats)
        total_loss += float(metrics['val_loss'])
        total_acc += float(metrics['val_acc'])
        count += 1

    return {
        'val_loss': total_loss / max(count, 1),
        'val_acc': total_acc / max(count, 1),
    }


# Multi-GPU versions using pmap
# Pattern matched to working ImageNet training in src/training/distributed.py
def create_pmap_train_step(num_classes: int, has_batch_stats: bool = False):
    """Create pmap'd train step for multi-GPU training."""

    def train_step_pmap(state, batch, rng):
        videos, labels = batch

        def loss_fn(params, batch_stats):
            variables = {'params': params}
            if has_batch_stats:
                variables['batch_stats'] = batch_stats
                output, mutated = state.apply_fn(
                    variables,
                    videos,
                    training=True,
                    rngs={'dropout': rng},
                    mutable=['batch_stats'],
                )
                return optax.softmax_cross_entropy(
                    output, jax.nn.one_hot(labels, num_classes)
                ).mean(), (output, mutated['batch_stats'])
            else:
                logits = state.apply_fn(
                    variables,
                    videos,
                    training=True,
                    rngs={'dropout': rng},
                )
                return optax.softmax_cross_entropy(
                    logits, jax.nn.one_hot(labels, num_classes)
                ).mean(), (logits, None)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (logits, new_batch_stats)), grads = grad_fn(state.params, state.batch_stats)

        # Average gradients across devices (plain pmean - matches ImageNet training)
        grads = jax.lax.pmean(grads, axis_name='batch')
        loss = jax.lax.pmean(loss, axis_name='batch')

        # Update state
        state = state.apply_gradients(grads=grads)
        if has_batch_stats and new_batch_stats is not None:
            # Average batch stats across devices
            new_batch_stats = jax.lax.pmean(new_batch_stats, axis_name='batch')
            state = state.replace(batch_stats=new_batch_stats)

        # Compute metrics
        acc = jnp.mean(jnp.argmax(logits, -1) == labels)
        acc = jax.lax.pmean(acc, axis_name='batch')

        metrics = {
            'train_loss': loss,
            'train_acc': acc,
        }

        return state, metrics

    # Match ImageNet training: use 'batch' axis_name and donate_argnums
    return jax.pmap(train_step_pmap, axis_name='batch', donate_argnums=(0,))


def create_pmap_eval_step(num_classes: int, has_batch_stats: bool = False):
    """Create pmap'd eval step for multi-GPU evaluation."""

    def eval_step_pmap(state, batch):
        videos, labels = batch

        variables = {'params': state.params}
        if has_batch_stats:
            variables['batch_stats'] = state.batch_stats

        logits = state.apply_fn(
            variables,
            videos,
            training=False,
        )

        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        acc = jnp.mean(jnp.argmax(logits, -1) == labels)

        # Average across devices
        loss = jax.lax.pmean(loss, axis_name='batch')
        acc = jax.lax.pmean(acc, axis_name='batch')

        return {
            'val_loss': loss,
            'val_acc': acc,
        }

    return jax.pmap(eval_step_pmap, axis_name='batch')


def evaluate_multi_gpu(
    state: TrainState,
    val_loader,
    eval_step_fn,
    num_devices: int,
    num_batches: int = None,
) -> Dict[str, float]:
    """Run multi-GPU validation evaluation."""
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    for i, batch in enumerate(val_loader):
        if num_batches is not None and i >= num_batches:
            break

        videos, labels = batch

        # Skip incomplete batches
        if videos.shape[0] % num_devices != 0:
            continue

        # Reshape for pmap: (B, ...) -> (num_devices, B//num_devices, ...)
        batch_per_device = videos.shape[0] // num_devices
        videos = videos.reshape(num_devices, batch_per_device, *videos.shape[1:])
        labels = labels.reshape(num_devices, batch_per_device, *labels.shape[1:])

        metrics = eval_step_fn(state, (videos, labels))

        # Get scalar from first device (all devices have same value after pmean)
        total_loss += float(metrics['val_loss'][0])
        total_acc += float(metrics['val_acc'][0])
        count += 1

    return {
        'val_loss': total_loss / max(count, 1),
        'val_acc': total_acc / max(count, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='Train CSSM models')

    # Architecture selection
    parser.add_argument('--arch', type=str,
                        choices=['simple', 'vit', 'baseline'],
                        default='simple',
                        help='Architecture: simple (clean CSSM arch), vit (CSSM-ViT), baseline (ViT)')

    # CSSM configuration
    parser.add_argument('--cssm', type=str,
                        choices=['hgru_bi', 'transformer', 'gated', 'kqv', 'standard', 'opponent', 'hgru', 'bilinear'],
                        default='hgru_bi',
                        help='CSSM type: hgru_bi, kqv (K*Q gating), transformer, gated, standard, opponent, hgru, bilinear')
    parser.add_argument('--mixing', type=str, choices=['dense', 'depthwise'], default='depthwise',
                        help='Mixing type: dense (multi-head) or depthwise')
    parser.add_argument('--no_concat_xy', action='store_true',
                        help='Disable [X,Y] concat+project in GatedOpponentCSSM')
    parser.add_argument('--gate_activation', type=str, default='sigmoid',
                        choices=['sigmoid', 'softplus_clamped', 'tanh_scaled'],
                        help='Gate activation for coupling (sigmoid=bounded [0,1], default)')

    # Model configuration (ViT-style and baseline)
    parser.add_argument('--embed_dim', type=int, default=384,
                        help='[vit/baseline] Embedding dimension (192=tiny, 384=small, 768=base)')
    parser.add_argument('--depth', type=int, default=12,
                        help='[vit/baseline] Number of transformer blocks')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='[vit/baseline] Patch size (only used with --stem_mode patch)')
    parser.add_argument('--stem_mode', type=str, default='conv',
                        choices=['patch', 'conv'],
                        help='[vit] Stem: patch (ViT-style) or conv (single conv + GELU + norm)')
    parser.add_argument('--stem_stride', type=int, default=4,
                        choices=[1, 2, 3, 4],
                        help='[vit] Stem downsampling stride (only for single-conv stem)')
    parser.add_argument('--stem_norm', type=str, default='layer',
                        choices=['layer', 'batch'],
                        help='[vit] Stem normalization: layer (LayerNorm) or batch (BatchNorm)')
    parser.add_argument('--no_pos_embed', action='store_true',
                        help='[vit/baseline] Disable position embeddings')
    parser.add_argument('--num_heads', type=int, default=6,
                        help='[baseline] Number of attention heads')
    parser.add_argument('--no_temporal_attn', action='store_true',
                        help='[baseline] Disable temporal attention blocks')
    parser.add_argument('--temporal_attn_every', type=int, default=3,
                        help='[baseline] Add temporal attention every N spatial blocks')
    parser.add_argument('--rope_mode', type=str, default='none',
                        choices=['spatiotemporal', 'temporal', 'none'],
                        help='[vit] RoPE position encoding mode')
    parser.add_argument('--block_size', type=int, default=1,
                        help='[vit] Block size for LMME channel mixing (1=depthwise, >1=channel mixing). Works with gated and opponent.')
    parser.add_argument('--kernel_size', type=int, default=11,
                        help='[vit] Spatial kernel size for CSSM excitation/inhibition kernels (11=original, 15=larger RF)')
    parser.add_argument('--position_independent_gates', action='store_true',
                        help='[cssm] Compute gates from raw input (before position encoding) for better length generalization')
    parser.add_argument('--use_dwconv', action='store_true',
                        help='[vit] Use DWConv in MLP (adds params, matches SHViT)')
    parser.add_argument('--output_act', type=str, default='none',
                        choices=['none', 'gelu', 'silu'],
                        help='[vit] Output activation after CSSM (adds nonlinearity)')
    parser.add_argument('--layer_scale_init', type=float, default=1e-6,
                        help='[vit] Layer scale initialization (1e-6 for stable deep training, 1.0 for shallow)')
    parser.add_argument('--gate_rank', type=int, default=0,
                        help='[vit] Low-rank gate bottleneck (0=full rank, try 16-64 to reduce gate params)')
    parser.add_argument('--readout_state', type=str, default='xyz',
                        choices=['xyz', 'x', 'y', 'z', 'xy', 'xz', 'yz'],
                        help='[hgru_bi] Which state(s) to use for output: x=excitatory, y=inhibitory, z=interaction')
    parser.add_argument('--pre_output_act', type=str, default='none',
                        choices=['none', 'gelu', 'silu'],
                        help='[hgru_bi] Activation before output_proj inside CSSM')
    parser.add_argument('--pooling_mode', type=str, default='mean',
                        choices=['mean', 'max', 'logsumexp'],
                        help='[vit] Global spatial pooling mode')
    parser.add_argument('--readout_act', type=str, default='gelu',
                        choices=['none', 'gelu', 'silu'],
                        help='[vit] Final activation before readout (after all blocks, before norm+pool)')

    # SimpleCSSM-specific arguments
    parser.add_argument('--frame_readout', type=str, default='last',
                        choices=['last', 'all'],
                        help='[simple] Use last frame or all frames for readout')
    parser.add_argument('--pos_embed', type=str, default='spatiotemporal',
                        choices=['spatiotemporal', 'spatial_only', 'separate', 'sinusoidal', 'temporal', 'learnable', 'none'],
                        help='[simple] Position embedding type: spatiotemporal (VideoRoPE), spatial_only (no temporal), separate (spatial RoPE + learned temporal), sinusoidal (spatial RoPE + sinusoidal temporal, best length extrapolation), temporal, learnable, or none')
    parser.add_argument('--act_type', type=str, default='softplus',
                        choices=['softplus', 'gelu', 'relu'],
                        help='[simple] Nonlinearity type in stem and readout')
    parser.add_argument('--pool_type', type=str, default='mean',
                        choices=['mean', 'max'],
                        help='[simple] Final spatial/spatiotemporal pooling type')
    parser.add_argument('--norm_type', type=str, default='layer',
                        choices=['layer', 'batch'],
                        help='[simple] Normalization type throughout model')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per device')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (0=sequential)')
    parser.add_argument('--prefetch_batches', type=int, default=2,
                        help='Number of batches to prefetch')
    parser.add_argument('--seq_len', type=int, default=8,
                        help='Sequence length (number of frames). Used as max when --variable_seq_len is set.')
    parser.add_argument('--variable_seq_len', action='store_true',
                        help='Enable variable-length training: randomly sample seq_len from [seq_len_min, seq_len] each batch')
    parser.add_argument('--seq_len_min', type=int, default=4,
                        help='Minimum sequence length when --variable_seq_len is enabled')
    parser.add_argument('--max_seq_len', type=int, default=32,
                        help='[simple] Maximum sequence length for learned temporal embeddings (used with pos_embed=separate)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay coefficient')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--force_multi_gpu', action='store_true',
                        help='Force multi-GPU even if NCCL test fails')

    # Performance optimizations
    parser.add_argument('--bf16', action='store_true',
                        help='Use bfloat16 matmul precision (faster on A100/H100)')
    parser.add_argument('--no_jit_cache', action='store_true',
                        help='Disable JAX compilation cache')
    parser.add_argument('--xla_flags', type=str, default=None,
                        help='Extra XLA flags (e.g., "--xla_gpu_triton_gemm_any=true")')

    # Logging and checkpointing
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name for checkpoints and wandb (default: auto-generated from config)')
    parser.add_argument('--project', type=str, default='cssm',
                        help='Wandb project name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log metrics every N steps')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--save_best_only', action='store_true', default=False,
                        help='Only save checkpoint with best val loss (deletes previous best)')
    parser.add_argument('--checkpointer', type=str, default='simple',
                        choices=['simple', 'pytree', 'standard'],
                        help='Checkpointer: simple (pickle, NFS-safe), pytree (orbax), standard (orbax async)')

    # Data
    parser.add_argument('--dataset', type=str, choices=['imagenette', 'pathfinder', 'cabc', 'imagenet'], default='imagenette',
                        help='Dataset: imagenette, pathfinder, cabc, or imagenet')
    parser.add_argument('--data_dir', type=str,
                        default='/media/data_cifs/projects/prj_video_imagenet/fftconv/data/imagenette2-320',
                        help='Path to dataset directory')
    parser.add_argument('--imagenet_dir', type=str,
                        default='/gpfs/data/shared/imagenet/ILSVRC2012',
                        help='Path to ImageNet directory')
    parser.add_argument('--pathfinder_difficulty', type=str, choices=['9', '14', '20'], default='9',
                        help='[pathfinder] Contour length difficulty (9=easy, 20=hard)')
    parser.add_argument('--cabc_difficulty', type=str, choices=['easy', 'medium', 'hard'], default='easy',
                        help='[cabc] Difficulty level (easy, medium, hard)')
    parser.add_argument('--tfrecord_dir', type=str, default=None,
                        help='[pathfinder] TFRecord directory (if set, uses TFRecord loader for max I/O perf)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (224 or 384 for DeiT3)')

    # Checkpoint resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoints/run_name/epoch_190)')
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='Override epoch to resume from (default: auto-detect from checkpoint)')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    args = parser.parse_args()

    # Setup JAX optimizations (must be before any JAX operations)
    setup_jax_optimizations(
        use_cache=not args.no_jit_cache,
        use_bf16=args.bf16,
    )
    if args.xla_flags:
        os.environ['XLA_FLAGS'] = args.xla_flags

    # Generate run name from config (or use provided name)
    if args.run_name:
        run_name = args.run_name
    else:
        if args.arch == 'simple':
            run_name = f"simple_{args.cssm}_d{args.depth}_e{args.embed_dim}"
        elif args.arch == 'vit':
            run_name = f"vit_{args.cssm}_d{args.depth}_e{args.embed_dim}"
        else:  # baseline
            temporal_str = f"_temp{args.temporal_attn_every}" if not args.no_temporal_attn else "_notime"
            run_name = f"baseline_d{args.depth}_e{args.embed_dim}_h{args.num_heads}{temporal_str}"

        # Add dataset to run name
        if args.dataset == 'pathfinder':
            run_name = f"pf{args.pathfinder_difficulty}_{run_name}"
        elif args.dataset == 'cabc':
            run_name = f"cabc{args.cabc_difficulty}_{run_name}"
    print(f"\n{'='*60}")
    print(f"Running Configuration: {run_name}")
    print(f"Architecture: {args.arch.upper()}")
    print(f"{'='*60}\n")

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
        )

    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    # Get dataset info
    if args.dataset == 'pathfinder':
        # Set default data dir for pathfinder if not specified
        if 'imagenette' in args.data_dir:
            args.data_dir = '/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025'
        dataset_info = get_pathfinder_info(
            difficulty=args.pathfinder_difficulty,
            root=args.data_dir,
            tfrecord_dir=args.tfrecord_dir,
        )
        dataset_name = f"Pathfinder (difficulty={args.pathfinder_difficulty})"
    elif args.dataset == 'cabc':
        # Set default data dir for cABC if not specified
        if 'imagenette' in args.data_dir:
            args.data_dir = '/media/data_cifs_lrs/projects/prj_LRA/cabc'
        dataset_info = get_cabc_info(
            difficulty=args.cabc_difficulty,
            root=args.data_dir,
            tfrecord_dir=args.tfrecord_dir,
        )
        dataset_name = f"cABC (difficulty={args.cabc_difficulty})"
    elif args.dataset == 'imagenet':
        dataset_info = get_imagenet_info()
        dataset_name = "ImageNet"
    else:
        dataset_info = get_dataset_info()
        dataset_name = "Imagenette"

    num_classes = dataset_info['num_classes']
    train_size = dataset_info['train_size']
    steps_per_epoch = train_size // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    print(f"Dataset: {dataset_name}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Train samples: {train_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}\n")

    # Create model based on architecture
    if args.arch == 'simple':
        model = SimpleCSSM(
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            cssm_type=args.cssm,
            kernel_size=args.kernel_size,
            block_size=args.block_size,
            frame_readout=args.frame_readout,
            norm_type=args.norm_type,
            pos_embed=args.pos_embed,
            act_type=args.act_type,
            pool_type=args.pool_type,
            seq_len=args.seq_len,
            max_seq_len=args.max_seq_len,
            position_independent_gates=args.position_independent_gates,
        )
    elif args.arch == 'vit':
        model = CSSMViT(
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            patch_size=args.patch_size,
            stem_mode=args.stem_mode,
            stem_stride=args.stem_stride,
            stem_norm=args.stem_norm,
            cssm_type=args.cssm,
            dense_mixing=(args.mixing == 'dense'),
            concat_xy=not args.no_concat_xy,
            gate_activation=args.gate_activation,
            use_pos_embed=not args.no_pos_embed,
            rope_mode=args.rope_mode,
            block_size=args.block_size,
            gate_rank=args.gate_rank,
            kernel_size=args.kernel_size,
            use_dwconv=args.use_dwconv,
            output_act=args.output_act,
            layer_scale_init=args.layer_scale_init,
            readout_state=args.readout_state,
            pre_output_act=args.pre_output_act,
            pooling_mode=args.pooling_mode,
            readout_act=args.readout_act,
        )
    else:  # baseline
        model = BaselineViT(
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            use_temporal_attn=not args.no_temporal_attn,
            temporal_attn_every=args.temporal_attn_every,
            use_pos_embed=not args.no_pos_embed,
        )

    # Initialize training state
    state, _ = create_train_state(
        rng=init_rng,
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        grad_clip=args.grad_clip,
    )

    # Check if model uses BatchNorm (has batch_stats)
    has_batch_stats = state.batch_stats is not None

    # Count parameters (before replication)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model parameters: {num_params:,}\n")

    if not args.no_wandb:
        wandb.config.update({'num_params': num_params})

    # Setup checkpointing (must be absolute path for orbax)
    # Use realpath to resolve symlinks - important for NFS mounts
    checkpoint_dir = os.path.realpath(os.path.join(args.checkpoint_dir, run_name))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Choose checkpointer type:
    # - simple (default): pickle-based, works reliably on NFS
    # - pytree: orbax PyTreeCheckpointer, faster but may fail on some NFS
    # - standard: orbax StandardCheckpointer, fastest but often fails on NFS
    if args.checkpointer == 'standard':
        import orbax.checkpoint as ocp
        checkpointer = ocp.StandardCheckpointer()
        print("Using Orbax StandardCheckpointer (may fail on NFS)")
    elif args.checkpointer == 'pytree':
        import orbax.checkpoint as ocp
        checkpointer = ocp.PyTreeCheckpointer()
        print("Using Orbax PyTreeCheckpointer")
    else:  # 'simple' (default)
        checkpointer = SimpleCheckpointer()
        print("Using SimpleCheckpointer (NFS-safe)")

    # Resume from checkpoint if specified (BEFORE multi-GPU replication)
    start_epoch = 0
    global_step = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_checkpoint_path = None

    if args.resume:
        resume_path = os.path.abspath(args.resume)
        if os.path.exists(resume_path):
            print(f"\n{'='*60}")
            print(f"Resuming from checkpoint: {resume_path}")
            print(f"{'='*60}\n")

            # Restore the state (params + optimizer state)
            state = checkpointer.restore(resume_path, state)

            # Determine starting epoch
            if args.resume_epoch is not None:
                start_epoch = args.resume_epoch
            else:
                # Try to extract epoch from checkpoint path (e.g., epoch_190)
                match = re.search(r'epoch_(\d+)', resume_path)
                if match:
                    start_epoch = int(match.group(1))
                else:
                    # Use epoch from state if available
                    start_epoch = int(state.epoch) if hasattr(state, 'epoch') else 0

            # Calculate global_step based on resumed epoch
            global_step = start_epoch * steps_per_epoch

            print(f"Resumed from epoch {start_epoch}, global_step {global_step}")
            print(f"Will continue training for epochs {start_epoch + 1} to {args.epochs}\n")
        else:
            print(f"WARNING: Checkpoint not found at {resume_path}, starting from scratch")

    # Setup multi-GPU (AFTER checkpoint restore)
    devices, num_devices, use_multi_gpu = get_device_setup()

    # Adjust batch size for multi-GPU
    if use_multi_gpu and args.batch_size % num_devices != 0:
        old_bs = args.batch_size
        args.batch_size = (args.batch_size // num_devices) * num_devices
        print(f"Adjusted batch_size {old_bs} -> {args.batch_size} for {num_devices} devices")

    batch_per_device = args.batch_size // num_devices if use_multi_gpu else args.batch_size

    print(f"Devices: {num_devices} ({[str(d) for d in devices]})")
    if use_multi_gpu:
        print(f"Multi-GPU training enabled: batch_size={args.batch_size}, per_device={batch_per_device}")

    # Create pmap'd functions for multi-GPU
    if use_multi_gpu:
        # Test NCCL with a simple operation first
        print("Testing NCCL collective operations...")
        nccl_ok = False
        try:
            test_data = jnp.ones((num_devices, 10), dtype=jnp.float32)
            simple_pmean = jax.pmap(lambda x: jax.lax.pmean(x, axis_name='batch'), axis_name='batch')
            result = simple_pmean(test_data)
            jax.block_until_ready(result)
            print(f"NCCL test PASSED: pmean result shape={result.shape}, value={float(result[0, 0]):.4f}")
            nccl_ok = True
        except Exception as e:
            print(f"NCCL test FAILED: {e}")
            if args.force_multi_gpu:
                print("--force_multi_gpu set, continuing with multi-GPU anyway...")
                nccl_ok = True  # Force it
            else:
                print("Multi-GPU may not work. Falling back to single GPU...")
                print("(Use --force_multi_gpu to bypass this test)")
                use_multi_gpu = False
                devices = [devices[0]]
                num_devices = 1

        if use_multi_gpu and nccl_ok:
            pmap_train_step = create_pmap_train_step(num_classes, has_batch_stats)
            pmap_eval_step = create_pmap_eval_step(num_classes, has_batch_stats)
            # Replicate state across devices using Flax's jax_utils (matches ImageNet training)
            state = jax_utils.replicate(state)
            print("State replicated across devices")

    # Create data loaders ONCE before training (not each epoch)
    # The loaders handle shuffling internally on each iteration
    print("\nCreating data loaders...")
    if args.dataset == 'pathfinder':
        if args.tfrecord_dir:
            train_loader = get_pathfinder_tfrecord_loader(
                tfrecord_dir=args.tfrecord_dir,
                difficulty=args.pathfinder_difficulty,
                batch_size=args.batch_size,
                num_frames=args.seq_len,
                split='train',
                shuffle=True,
                prefetch_batches=args.prefetch_batches,
            )
        else:
            train_loader = get_pathfinder_loader(
                root=args.data_dir,
                difficulty=args.pathfinder_difficulty,
                batch_size=args.batch_size,
                num_frames=args.seq_len,
                split='train',
                num_workers=args.num_workers,
                prefetch_batches=args.prefetch_batches,
            )
    elif args.dataset == 'cabc':
        if args.tfrecord_dir:
            train_loader = get_cabc_tfrecord_loader(
                tfrecord_dir=args.tfrecord_dir,
                difficulty=args.cabc_difficulty,
                batch_size=args.batch_size,
                num_frames=args.seq_len,
                split='train',
                shuffle=True,
                prefetch_batches=args.prefetch_batches,
            )
        else:
            train_loader = get_cabc_loader(
                root=args.data_dir,
                difficulty=args.cabc_difficulty,
                batch_size=args.batch_size,
                num_frames=args.seq_len,
                split='train',
                num_workers=args.num_workers,
                prefetch_batches=args.prefetch_batches,
            )
    elif args.dataset == 'imagenet':
        train_loader = get_imagenet_loader(
            data_dir=args.imagenet_dir,
            split='train',
            batch_size=args.batch_size,
            image_size=args.image_size,
            sequence_length=args.seq_len,
        )
    else:
        train_loader = get_imagenette_video_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            sequence_length=args.seq_len,
            split='train',
        )

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Update epoch in state
        if use_multi_gpu:
            # For multi-GPU, epoch is already replicated in state structure
            # Just update the unreplicated value - jax_utils handles this
            pass  # Don't modify epoch during training - it's just for logging
        else:
            state = state.replace(epoch=jnp.array(epoch))

        # Training
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        epoch_step_times = []

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True) as pbar:
            for batch in pbar:
                rng, step_rng = jax.random.split(rng)
                videos, labels = batch

                # Skip incomplete batches for multi-GPU
                if use_multi_gpu and videos.shape[0] % num_devices != 0:
                    continue

                # Variable sequence length training: randomly sample seq_len per batch
                if args.variable_seq_len:
                    rng, seq_rng = jax.random.split(rng)
                    # Sample seq_len uniformly from [seq_len_min, seq_len]
                    current_seq_len = int(jax.random.randint(seq_rng, (), args.seq_len_min, args.seq_len + 1))
                    # Slice videos to the sampled length (videos shape: B, T, H, W, C)
                    videos = videos[:, :current_seq_len]

                # Time the training step
                step_start = time.perf_counter()

                if use_multi_gpu:
                    # Reshape for pmap: (B, ...) -> (num_devices, B//num_devices, ...)
                    videos = videos.reshape(num_devices, batch_per_device, *videos.shape[1:])
                    labels = labels.reshape(num_devices, batch_per_device, *labels.shape[1:])

                    # Split RNG for each device
                    step_rngs = jax.random.split(step_rng, num_devices)

                    state, metrics = pmap_train_step(state, (videos, labels), step_rngs)

                    # Get scalar from first device (all same after pmean)
                    loss_val = float(metrics['train_loss'][0])
                    acc_val = float(metrics['train_acc'][0])
                else:
                    state, metrics = train_step(state, batch, step_rng, num_classes, has_batch_stats)
                    loss_val = float(metrics['train_loss'])
                    acc_val = float(metrics['train_acc'])

                # Block until computation completes for accurate timing
                jax.block_until_ready(metrics)
                step_time = time.perf_counter() - step_start
                epoch_step_times.append(step_time)

                epoch_loss += loss_val
                epoch_acc += acc_val
                num_batches += 1
                global_step += 1

                # Update progress bar with timing
                pbar.set_postfix({
                    'loss': f"{loss_val:.4f}",
                    'acc': f"{acc_val:.4f}",
                    'ms/step': f"{step_time*1000:.1f}",
                })

                # Log to wandb
                if not args.no_wandb and global_step % args.log_every == 0:
                    # Get current learning rate from optimizer state
                    lr = args.lr  # Simplified - actual LR from schedule
                    wandb.log({
                        'train/loss': loss_val,
                        'train/acc': acc_val,
                        'train/learning_rate': lr,
                        'train/step': global_step,
                        'timing/step_ms': step_time * 1000,
                        'timing/throughput_samples_per_sec': args.batch_size / step_time,
                    }, step=global_step)

        # Epoch summary
        avg_train_loss = epoch_loss / max(num_batches, 1)
        avg_train_acc = epoch_acc / max(num_batches, 1)

        # Timing statistics
        avg_step_time_ms = sum(epoch_step_times) / len(epoch_step_times) * 1000 if epoch_step_times else 0
        min_step_time_ms = min(epoch_step_times) * 1000 if epoch_step_times else 0
        max_step_time_ms = max(epoch_step_times) * 1000 if epoch_step_times else 0
        throughput = args.batch_size / (sum(epoch_step_times) / len(epoch_step_times)) if epoch_step_times else 0

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"  Timing: avg={avg_step_time_ms:.1f}ms/step, min={min_step_time_ms:.1f}ms, max={max_step_time_ms:.1f}ms, throughput={throughput:.1f} samples/sec")

        # Validation
        if (epoch + 1) % args.eval_every == 0:
            if args.dataset == 'pathfinder':
                if args.tfrecord_dir:
                    val_loader = get_pathfinder_tfrecord_loader(
                        tfrecord_dir=args.tfrecord_dir,
                        difficulty=args.pathfinder_difficulty,
                        batch_size=args.batch_size,
                        num_frames=args.seq_len,
                        split='val',
                        shuffle=False,
                        prefetch_batches=args.prefetch_batches,
                    )
                else:
                    val_loader = get_pathfinder_loader(
                        root=args.data_dir,
                        difficulty=args.pathfinder_difficulty,
                        batch_size=args.batch_size,
                        num_frames=args.seq_len,
                        split='val',
                        shuffle=False,
                        num_workers=args.num_workers,
                        prefetch_batches=args.prefetch_batches,
                    )
            elif args.dataset == 'cabc':
                # cABC uses 'test' split for validation
                if args.tfrecord_dir:
                    val_loader = get_cabc_tfrecord_loader(
                        tfrecord_dir=args.tfrecord_dir,
                        difficulty=args.cabc_difficulty,
                        batch_size=args.batch_size,
                        num_frames=args.seq_len,
                        split='test',
                        shuffle=False,
                        prefetch_batches=args.prefetch_batches,
                    )
                else:
                    val_loader = get_cabc_loader(
                        root=args.data_dir,
                        difficulty=args.cabc_difficulty,
                        batch_size=args.batch_size,
                        num_frames=args.seq_len,
                        split='test',
                        shuffle=False,
                        num_workers=args.num_workers,
                        prefetch_batches=args.prefetch_batches,
                    )
            elif args.dataset == 'imagenet':
                val_loader = get_imagenet_loader(
                    data_dir=args.imagenet_dir,
                    split='val',
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    sequence_length=args.seq_len,
                )
            else:
                val_loader = get_imagenette_video_loader(
                    data_dir=args.data_dir,
                    batch_size=args.batch_size,
                    sequence_length=args.seq_len,
                    split='val',
                )
            if use_multi_gpu:
                val_metrics = evaluate_multi_gpu(state, val_loader, pmap_eval_step, num_devices)
            else:
                val_metrics = evaluate(state, val_loader, num_classes, has_batch_stats=has_batch_stats)

            print(f"Epoch {epoch+1} - Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.4f}")

            if not args.no_wandb:
                wandb.log({
                    'val/loss': val_metrics['val_loss'],
                    'val/acc': val_metrics['val_acc'],
                    'epoch': epoch + 1,
                    'timing/epoch_avg_step_ms': avg_step_time_ms,
                    'timing/epoch_min_step_ms': min_step_time_ms,
                    'timing/epoch_max_step_ms': max_step_time_ms,
                    'timing/epoch_throughput': throughput,
                }, step=global_step)

            # Track best model
            if val_metrics['val_acc'] > best_val_acc:
                best_val_acc = val_metrics['val_acc']
                print(f"  New best validation accuracy: {best_val_acc:.4f}")

            # Save checkpoint based on best val loss (for pathfinder)
            if args.save_best_only and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                # Delete previous best checkpoint if it exists
                if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
                    import shutil
                    shutil.rmtree(best_checkpoint_path)
                    print(f"  Deleted previous checkpoint: {best_checkpoint_path}")
                # Save new best checkpoint
                best_checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}')
                # Unreplicate state for saving (take first device copy) and set epoch
                if use_multi_gpu:
                    save_state = jax.tree_util.tree_map(lambda x: x[0], state)
                    save_state = save_state.replace(epoch=jnp.array(epoch))
                else:
                    save_state = state
                checkpointer.save(best_checkpoint_path, save_state)
                print(f"  New best val_loss: {best_val_loss:.4f} - Saved to {best_checkpoint_path}")

        # Save checkpoint periodically (if not using best-only mode)
        if not args.save_best_only and (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}')
            # Unreplicate state for saving (take first device copy) and set epoch
            if use_multi_gpu:
                save_state = jax.tree_util.tree_map(lambda x: x[0], state)
                save_state = save_state.replace(epoch=jnp.array(epoch))
            else:
                save_state = state
            checkpointer.save(ckpt_path, save_state)
            print(f"  Saved checkpoint to {ckpt_path}")

    # Wait for any pending checkpoint operations to complete
    # (only StandardCheckpointer has async operations that need waiting)
    if hasattr(checkpointer, 'wait_until_finished'):
        checkpointer.wait_until_finished()

    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}\n")

    if not args.no_wandb:
        wandb.log({'best_val_acc': best_val_acc})
        wandb.finish()


if __name__ == '__main__':
    main()
