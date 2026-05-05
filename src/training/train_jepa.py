"""
CSSM-JEPA training script for video self-supervised learning.

Trains CSSM-JEPA on Ego4D or other video datasets with dual
spectral + feature space prediction loss.

Supports multi-GPU training with JAX pmap.
"""

import argparse
import os
import pickle
import time
from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import linen as nn
from flax.training import train_state
from flax.jax_utils import replicate, unreplicate
from flax import serialization
from tqdm import tqdm
import orbax.checkpoint as ocp

from ..jepa.model import CSSMJEPA, create_cssm_jepa, CSSMVJEPA, create_cssm_vjepa
from ..jepa.encoder import update_ema_params

# Import all data loaders
from ..data.ego4d import Ego4DDataset, create_ego4d_loader
from ..data.ego4d_fast import Ego4DFastLoader, create_ego4d_fast_loader
try:
    from ..data.ego4d_webdataset import create_ego4d_webdataset_loader
    HAS_WEBDATASET = True
except ImportError:
    HAS_WEBDATASET = False

# TFRecord loader is imported lazily to avoid TensorFlow/JAX NCCL conflicts
# TensorFlow can interfere with JAX's NCCL initialization if imported too early
HAS_TFRECORD = True  # Will check at runtime
_tfrecord_loader = None

def get_tfrecord_loader():
    """Lazy import of TFRecord loader to avoid TF/JAX NCCL conflicts."""
    global _tfrecord_loader
    if _tfrecord_loader is None:
        try:
            from ..data.ego4d_tfrecord import create_ego4d_tfrecord_loader
            _tfrecord_loader = create_ego4d_tfrecord_loader
        except ImportError:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
    return _tfrecord_loader


def get_num_devices():
    """Get number of available devices."""
    return jax.local_device_count()


def test_multi_gpu_communication(num_devices: int) -> bool:
    """
    Test basic multi-GPU communication to verify NCCL works.

    Returns True if test passes, raises exception otherwise.
    """
    if num_devices <= 1:
        return True

    print(f"\nTesting multi-GPU communication ({num_devices} devices)...")

    @partial(jax.pmap, axis_name='devices')
    def test_pmean(x):
        return jax.lax.pmean(x, axis_name='devices')

    @partial(jax.pmap, axis_name='devices')
    def test_psum(x):
        return jax.lax.psum(x, axis_name='devices')

    try:
        # Test 1: pmean with small array
        test_data = jnp.arange(num_devices, dtype=jnp.float32).reshape(num_devices, 1)
        result = test_pmean(test_data)
        jax.block_until_ready(result)
        print(f"  pmean test: PASSED (result: {result.flatten()})")

        # Test 2: psum with larger array
        test_data2 = jnp.ones((num_devices, 100, 100), dtype=jnp.float32)
        result2 = test_psum(test_data2)
        jax.block_until_ready(result2)
        expected_sum = num_devices
        if jnp.allclose(result2[0, 0, 0], expected_sum):
            print(f"  psum test: PASSED")
        else:
            print(f"  psum test: UNEXPECTED (got {result2[0, 0, 0]}, expected {expected_sum})")

        print("  Multi-GPU communication: OK\n")
        return True

    except Exception as e:
        print(f"\n  Multi-GPU communication test FAILED!")
        print(f"  Error: {e}")
        print("\n  Suggestions:")
        print("    1. Try --disable_nccl_p2p flag")
        print("    2. Try --single_gpu flag")
        print("    3. Set NCCL_P2P_DISABLE=1 environment variable")
        print()
        raise RuntimeError(f"Multi-GPU communication failed: {e}")


def safe_unreplicate(tree):
    """
    Safely unreplicate a pytree, avoiding NCCL issues.

    Uses jax.device_get to transfer to host first, then takes first replica.
    This avoids NCCL collective operations that can fail on some setups.
    """
    # First, get data to host (this uses device-to-host transfer, not NCCL)
    tree_on_host = jax.device_get(tree)
    # Then take first replica
    return jax.tree_util.tree_map(lambda x: x[0], tree_on_host)


def save_checkpoint_simple(path: str, state):
    """
    Save checkpoint using simple pickle serialization.
    Works on NFS/network filesystems where Orbax OCDBT fails.
    """
    # Convert state to serializable format
    state_dict = serialization.to_state_dict(state)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(state_dict, f)


def load_checkpoint_simple(path: str, state):
    """
    Load checkpoint from simple pickle format.
    """
    with open(path, 'rb') as f:
        state_dict = pickle.load(f)
    return serialization.from_state_dict(state, state_dict)


def convert_to_jax_arrays(tree):
    """
    Convert numpy scalars and arrays to JAX arrays for Orbax checkpointing.

    Orbax doesn't handle numpy.bool, numpy.int64, numpy.float64 etc.
    This converts them to JAX arrays which Orbax can serialize properly.
    """
    def _convert(x):
        # Handle all numpy scalar types
        if isinstance(x, np.generic):  # Catches numpy.bool_, numpy.int64, numpy.float64, etc.
            return jnp.asarray(x)
        # Handle numpy arrays
        elif isinstance(x, np.ndarray):
            return jnp.asarray(x)
        # Handle Python scalars (convert to 0-d JAX arrays)
        elif isinstance(x, bool):
            return jnp.asarray(x, dtype=jnp.bool_)
        elif isinstance(x, int):
            return jnp.asarray(x, dtype=jnp.int32)
        elif isinstance(x, float):
            return jnp.asarray(x, dtype=jnp.float32)
        # Keep JAX arrays and other types as-is
        return x

    return jax.tree_util.tree_map(_convert, tree)


def shard_batch(batch: jnp.ndarray, num_devices: int) -> jnp.ndarray:
    """Reshape batch for pmap: (B, ...) -> (num_devices, B//num_devices, ...)"""
    batch_size = batch.shape[0]
    per_device = batch_size // num_devices
    return batch.reshape(num_devices, per_device, *batch.shape[1:])


class JEPATrainState(train_state.TrainState):
    """Extended train state with target encoder params and EMA tracking."""
    target_params: Any = None
    batch_stats: Any = None  # For BatchNorm
    ema_decay: float = 0.996
    epoch: int = 0


def create_train_state(
    rng: jax.random.PRNGKey,
    model: CSSMJEPA,
    learning_rate: float,
    weight_decay: float,
    total_steps: int,
    warmup_steps: int = 1000,
    grad_clip: float = 1.0,
    ema_decay: float = 0.996,
    num_frames: int = 16,
    resolution: int = 224,
) -> JEPATrainState:
    """
    Initialize training state with optimizer and target encoder.

    Args:
        rng: Random key for initialization
        model: CSSM-JEPA model
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        total_steps: Total training steps
        warmup_steps: Warmup steps
        grad_clip: Gradient clipping norm
        ema_decay: EMA decay for target encoder
        num_frames: Number of video frames
        resolution: Video resolution

    Returns:
        Training state with online and target params
    """
    # Create dummy input for initialization
    rng, init_rng, mask_rng = jax.random.split(rng, 3)
    dummy_video = jnp.ones((2, num_frames, resolution, resolution, 3))

    # Initialize model
    variables = model.init(
        {'params': init_rng, 'dropout': init_rng},
        dummy_video,
        mask_rng,
        training=False,
    )
    params = variables['params']
    batch_stats = variables.get('batch_stats', None)

    # Target params start as copy of online params
    target_params = jax.tree_util.tree_map(lambda x: x, params)

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

    # Wrap with apply_if_finite
    tx = optax.apply_if_finite(tx, max_consecutive_errors=5)

    return JEPATrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        target_params=target_params,
        batch_stats=batch_stats,
        ema_decay=jnp.array(ema_decay, dtype=jnp.float32),
        epoch=jnp.array(0, dtype=jnp.int32),
    )


@partial(jax.jit, static_argnums=())
def train_step(
    state: JEPATrainState,
    batch: jnp.ndarray,
    rng: jax.random.PRNGKey,
) -> Tuple[JEPATrainState, Dict[str, float]]:
    """
    Single training step.

    Args:
        state: Current training state
        batch: Video batch (B, T, H, W, C)
        rng: Random key

    Returns:
        Updated state and metrics
    """
    rng, mask_rng, drop_rng = jax.random.split(rng, 3)

    def loss_fn(params):
        # Build variables dict with params and optional batch_stats
        variables = {'params': params}
        if state.batch_stats is not None:
            variables['batch_stats'] = state.batch_stats

        # Determine what collections are mutable
        mutable = ['batch_stats'] if state.batch_stats is not None else []

        if mutable:
            (loss, metrics), new_model_state = state.apply_fn(
                variables,
                batch,
                mask_rng,
                training=True,
                rngs={'dropout': drop_rng},
                mutable=mutable,
            )
            return (loss, (metrics, new_model_state))
        else:
            loss, metrics = state.apply_fn(
                variables,
                batch,
                mask_rng,
                training=True,
                rngs={'dropout': drop_rng},
            )
            return (loss, (metrics, None))

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, new_model_state)), grads = grad_fn(state.params)

    # Update online params
    state = state.apply_gradients(grads=grads)

    # Update batch_stats if we have them
    if new_model_state is not None and 'batch_stats' in new_model_state:
        state = state.replace(batch_stats=new_model_state['batch_stats'])

    # EMA update for target encoder
    new_target_params = update_ema_params(
        state.params,
        state.target_params,
        state.ema_decay,
    )
    state = state.replace(target_params=new_target_params)

    return state, metrics


@partial(jax.pmap, axis_name='devices')
def train_step_pmap(
    state: JEPATrainState,
    batch: jnp.ndarray,
    rng: jax.random.PRNGKey,
) -> Tuple[JEPATrainState, Dict[str, float]]:
    """
    Multi-GPU training step using pmap.

    Args:
        state: Replicated training state
        batch: Sharded video batch (per_device_batch, T, H, W, C)
        rng: Random key (different per device)

    Returns:
        Updated state and metrics (averaged across devices)
    """
    rng, mask_rng, drop_rng = jax.random.split(rng, 3)

    def loss_fn(params):
        variables = {'params': params}
        if state.batch_stats is not None:
            variables['batch_stats'] = state.batch_stats

        mutable = ['batch_stats'] if state.batch_stats is not None else []

        if mutable:
            (loss, metrics), new_model_state = state.apply_fn(
                variables,
                batch,
                mask_rng,
                training=True,
                rngs={'dropout': drop_rng},
                mutable=mutable,
            )
            return (loss, (metrics, new_model_state))
        else:
            loss, metrics = state.apply_fn(
                variables,
                batch,
                mask_rng,
                training=True,
                rngs={'dropout': drop_rng},
            )
            return (loss, (metrics, None))

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, new_model_state)), grads = grad_fn(state.params)

    # Debug: check grad shapes before pmean
    def check_for_nan(x):
        return jnp.any(jnp.isnan(x))
    has_nan = jax.tree_util.tree_reduce(
        lambda a, b: a | b,
        jax.tree_util.tree_map(check_for_nan, grads)
    )

    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='devices')

    # Update online params
    state = state.apply_gradients(grads=grads)

    # Update batch_stats if we have them (average across devices)
    if new_model_state is not None and 'batch_stats' in new_model_state:
        batch_stats = jax.lax.pmean(new_model_state['batch_stats'], axis_name='devices')
        state = state.replace(batch_stats=batch_stats)

    # EMA update for target encoder
    new_target_params = update_ema_params(
        state.params,
        state.target_params,
        state.ema_decay,
    )
    state = state.replace(target_params=new_target_params)

    # Average metrics across devices
    metrics = jax.lax.pmean(metrics, axis_name='devices')

    return state, metrics


def train_epoch(
    state: JEPATrainState,
    dataloader,
    rng: jax.random.PRNGKey,
    epoch: int,
    log_every: int = 10,
    use_wandb: bool = True,
    num_devices: int = 1,
    start_step: int = 0,
) -> Tuple[JEPATrainState, Dict[str, float], int]:
    """
    Train for one epoch.

    Args:
        state: Training state (replicated if multi-GPU)
        dataloader: Video data loader
        rng: Random key
        epoch: Current epoch number
        log_every: Log every N steps
        use_wandb: Whether to log to wandb
        num_devices: Number of devices for multi-GPU
        start_step: Global step count at start of epoch (for wandb logging)

    Returns:
        Updated state, epoch metrics, global step count
    """
    epoch_metrics = {}
    step_times = []
    global_step = start_step
    use_pmap = num_devices > 1

    with tqdm(dataloader, desc=f"Epoch {epoch+1}", dynamic_ncols=True) as pbar:
        for batch_data in pbar:
            # Unpack batch
            if isinstance(batch_data, tuple):
                videos, video_ids = batch_data
            else:
                videos = batch_data
                video_ids = None

            # Convert to JAX array
            videos = jnp.array(videos)

            # Check batch size is compatible with num_devices
            batch_size = videos.shape[0]
            if use_pmap and batch_size % num_devices != 0:
                # Pad or skip batch
                continue

            # Get RNG for this step
            rng, step_rng = jax.random.split(rng)

            # Training step
            step_start = time.perf_counter()

            if use_pmap:
                # Shard batch and RNG for pmap
                videos_sharded = shard_batch(videos, num_devices)
                # Create different RNG for each device
                step_rngs = jax.random.split(step_rng, num_devices)

                # Debug: check for NaN in input (only on first step)
                if global_step == 0:
                    print(f"DEBUG: videos shape={videos.shape}, sharded={videos_sharded.shape}")
                    print(f"DEBUG: videos min={videos.min():.3f}, max={videos.max():.3f}")

                try:
                    state, metrics = train_step_pmap(state, videos_sharded, step_rngs)
                except Exception as e:
                    error_msg = str(e)
                    if 'NCCL' in error_msg or 'AllReduce' in error_msg or 'invalid argument' in error_msg.lower():
                        print("\n" + "="*70)
                        print("NCCL ERROR DETECTED!")
                        print("="*70)
                        print("This may be due to B200/Blackwell GPU compatibility issues.")
                        print("\nTry one of these workarounds:")
                        print("  1. Add --disable_nccl_p2p flag")
                        print("  2. Set environment: NCCL_P2P_DISABLE=1")
                        print("  3. Use --single_gpu for single-GPU training")
                        print("  4. Try: NCCL_SOCKET_IFNAME=eth0 (use correct interface)")
                        print("="*70 + "\n")
                    raise
                # Get metrics from first device (they're averaged)
                metrics = jax.tree_util.tree_map(lambda x: x[0], metrics)
            else:
                state, metrics = train_step(state, videos, step_rng)

            jax.block_until_ready(metrics)
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

            global_step += 1

            # Accumulate metrics
            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(float(v))

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss_total']:.4f}",
                'ms/step': f"{step_time*1000:.1f}",
            })

            # Log to wandb
            if use_wandb and global_step % log_every == 0:
                log_dict = {f'train/{k}': float(v) for k, v in metrics.items()}
                log_dict['train/step_time_ms'] = step_time * 1000
                log_dict['train/step'] = global_step
                wandb.log(log_dict)

    # Average epoch metrics
    avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
    avg_metrics['step_time_ms'] = np.mean(step_times) * 1000

    return state, avg_metrics, global_step


def main():
    parser = argparse.ArgumentParser(description='Train CSSM-JEPA / V-JEPA')

    # Model type selection
    parser.add_argument('--model_type', type=str, default='vjepa',
                       choices=['jepa', 'vjepa'],
                       help='Model type: jepa (spatial masking) or vjepa (causal future prediction, recommended)')

    # Model configuration
    parser.add_argument('--encoder', type=str, default='cssm_shvit_s4',
                       choices=['cssm_shvit_s1', 'cssm_shvit_s2', 'cssm_shvit_s3', 'cssm_shvit_s4'],
                       help='Encoder model')
    parser.add_argument('--embed_dim', type=int, default=512,
                       help='Encoder output dimension')
    parser.add_argument('--predictor_depth', type=int, default=4,
                       help='Predictor transformer depth')
    parser.add_argument('--predictor_heads', type=int, default=8,
                       help='Predictor attention heads')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                       help='Mask ratio for target region')
    parser.add_argument('--rope_mode', type=str, default='spatiotemporal',
                       choices=['spatiotemporal', 'temporal', 'none'],
                       help='Position encoding mode')

    # V-JEPA specific (causal future prediction)
    parser.add_argument('--context_ratio', type=float, default=0.5,
                       help='[vjepa] Fraction of frames as context (past). e.g., 0.5 = first half is context')
    parser.add_argument('--num_mask_blocks', type=int, default=4,
                       help='[vjepa] Number of spatiotemporal blocks to mask in target region')
    parser.add_argument('--block_size_t', type=int, default=2,
                       help='[vjepa] Temporal size of masked blocks')
    parser.add_argument('--block_size_h', type=int, default=2,
                       help='[vjepa] Height of masked blocks')
    parser.add_argument('--block_size_w', type=int, default=2,
                       help='[vjepa] Width of masked blocks')

    # Loss configuration
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Spectral loss weight')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Feature loss weight')
    parser.add_argument('--vicreg_weight', type=float, default=0.0,
                       help='VICReg regularization weight')

    # EMA configuration
    parser.add_argument('--ema_decay', type=float, default=0.996,
                       help='EMA decay for target encoder')
    parser.add_argument('--ema_anneal_end', type=float, default=0.9999,
                       help='Final EMA decay after annealing')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of video frames')
    parser.add_argument('--frame_stride', type=int, default=4,
                       help='Temporal stride between frames')
    parser.add_argument('--resolution', type=int, default=224,
                       help='Video resolution')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping norm')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Warmup epochs')

    # Data configuration
    parser.add_argument('--data_loader', type=str, default='tfrecord',
                       choices=['basic', 'fast', 'webdataset', 'tfrecord'],
                       help='Data loader: basic (slow), fast (threaded), webdataset, tfrecord (fastest)')
    parser.add_argument('--ego4d_manifest', type=str, default=None,
                       help='Path to Ego4D manifest JSON/txt (for basic loader)')
    parser.add_argument('--video_dir', type=str, default=None,
                       help='Directory containing video files (for basic/fast loaders)')
    parser.add_argument('--webdataset_path', type=str, default=None,
                       help='WebDataset path pattern (for webdataset loader)')
    parser.add_argument('--tfrecord_dir', type=str, default=None,
                       help='TFRecord directory (for tfrecord loader - fastest)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--prefetch_batches', type=int, default=4,
                       help='Number of batches to prefetch')
    parser.add_argument('--shuffle_buffer', type=int, default=500,
                       help='Shuffle buffer size (default 500, each clip ~10MB so 500=5GB RAM)')

    # Logging and checkpointing
    parser.add_argument('--project', type=str, default='cssm-jepa',
                       help='Wandb project name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log_every', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--simple_checkpoint', action='store_true',
                       help='Use simple pickle checkpoints (works on NFS/network filesystems)')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--single_gpu', action='store_true',
                       help='Force single GPU training (avoids NCCL issues)')
    parser.add_argument('--nccl_debug', action='store_true',
                       help='Enable NCCL debug output')
    parser.add_argument('--disable_nccl_p2p', action='store_true',
                       help='Disable NCCL P2P (helps with B200/Blackwell GPUs)')
    parser.add_argument('--disable_nccl_ib', action='store_true',
                       help='Disable NCCL InfiniBand')

    args = parser.parse_args()

    # Enable NCCL debug if requested
    if args.nccl_debug:
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
        print("NCCL debug enabled")

    # NCCL workarounds for B200/Blackwell GPUs
    if args.disable_nccl_p2p:
        os.environ['NCCL_P2P_DISABLE'] = '1'
        print("NCCL P2P disabled (B200 compatibility mode)")

    if args.disable_nccl_ib:
        os.environ['NCCL_IB_DISABLE'] = '1'
        print("NCCL InfiniBand disabled")

    # Force single GPU if requested
    if args.single_gpu:
        # If CUDA_VISIBLE_DEVICES is set to multiple GPUs, take just the first
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        first_device = cuda_devices.split(',')[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = first_device
        print(f"Forcing single GPU mode: CUDA_VISIBLE_DEVICES={first_device}")

    # Run name
    if args.model_type == 'vjepa':
        run_name = f"vjepa_{args.encoder}_ctx{args.context_ratio}_m{args.mask_ratio}_blk{args.num_mask_blocks}"
    else:
        run_name = f"jepa_{args.encoder}_a{args.alpha}_b{args.beta}_m{args.mask_ratio}"
    print(f"\n{'='*60}")
    print(f"CSSM-{args.model_type.upper()} Training: {run_name}")
    print(f"{'='*60}\n")

    if args.model_type == 'vjepa':
        print(f"V-JEPA Configuration (Causal Future Prediction):")
        print(f"  Context ratio: {args.context_ratio} (first {args.context_ratio*100:.0f}% frames)")
        print(f"  Target mask ratio: {args.mask_ratio}")
        print(f"  Num mask blocks: {args.num_mask_blocks}")
        print(f"  Block size: ({args.block_size_t}, {args.block_size_h}, {args.block_size_w})")
        print()

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
        )

    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    # === IMPORTANT: Multi-GPU setup BEFORE data loader ===
    # This ensures JAX initializes NCCL before TensorFlow is imported.
    # TensorFlow can interfere with JAX's NCCL initialization on some GPUs (e.g., B200).
    num_devices = get_num_devices()
    print(f"Number of devices: {num_devices}")
    if num_devices > 1:
        print(f"Using multi-GPU training with {num_devices} devices")
        print(f"Effective batch size: {args.batch_size} (total across all devices)")

        # Test multi-GPU communication BEFORE importing TensorFlow
        # This helps catch NCCL issues early (especially on B200/Blackwell GPUs)
        test_multi_gpu_communication(num_devices)

    # Create data loader (may import TensorFlow for TFRecord - safe now after NCCL init)
    print(f"Loading Ego4D dataset with '{args.data_loader}' loader...")

    if args.data_loader == 'tfrecord':
        if not args.tfrecord_dir:
            raise ValueError("--tfrecord_dir required for tfrecord loader")
        # Use lazy import to avoid TF/JAX NCCL conflicts
        # TensorFlow is loaded here, AFTER JAX has initialized its GPU context
        create_tfrecord_loader = get_tfrecord_loader()
        dataloader = create_tfrecord_loader(
            tfrecord_dir=args.tfrecord_dir,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            resolution=args.resolution,
            shuffle_buffer=args.shuffle_buffer,  # Default 500 (~5GB), was 10000 (~100GB!)
            prefetch_batches=args.prefetch_batches,
            num_parallel_reads=16,
            augment=True,
        )
        print(f"  TFRecord: {args.tfrecord_dir}")
        print(f"  Shards: {dataloader.num_shards}, estimated samples: ~{dataloader.estimated_samples}")
        # Each clip is ~10MB (16 frames × 224² × 3 × 4 bytes)
        shuffle_mem_gb = args.shuffle_buffer * 10 / 1024
        print(f"  Shuffle buffer: {args.shuffle_buffer} clips (~{shuffle_mem_gb:.1f} GB RAM)")

    elif args.data_loader == 'webdataset':
        if not HAS_WEBDATASET:
            raise ImportError("webdataset not installed. Install with: pip install webdataset")
        if not args.webdataset_path:
            raise ValueError("--webdataset_path required for webdataset loader")
        dataloader = create_ego4d_webdataset_loader(
            dataset_path=args.webdataset_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_buffer=1000,
            augment=True,
            shard_id=0,
            num_shards=1,
        )
        print(f"  WebDataset: {args.webdataset_path}")
        print(f"  Estimated shards: {dataloader.num_tar_shards}, samples: ~{dataloader.estimated_samples}")

    elif args.data_loader == 'fast':
        if not args.video_dir:
            raise ValueError("--video_dir required for fast loader")
        dataloader = create_ego4d_fast_loader(
            video_dir=args.video_dir,
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            resolution=args.resolution,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_batches=args.prefetch_batches,
            shuffle=True,
            augment=True,
            shard_id=0,
            num_shards=1,
        )
        print(f"  Fast loader: {args.video_dir}")
        print(f"  Workers: {args.num_workers}, Prefetch: {args.prefetch_batches}")

    else:  # basic loader
        if not args.ego4d_manifest:
            raise ValueError("--ego4d_manifest required for basic loader")
        dataloader = create_ego4d_loader(
            manifest_path=args.ego4d_manifest,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            resolution=args.resolution,
            shuffle=True,
            video_dir=args.video_dir,
        )
        print(f"  Basic loader: {args.ego4d_manifest}")

    # Estimate steps per epoch (approximate since video loading is dynamic)
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    print(f"Steps per epoch: ~{steps_per_epoch}")
    print(f"Total steps: ~{total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Create model
    if args.model_type == 'vjepa':
        print("\nCreating CSSM-V-JEPA model (causal future prediction)...")
        model = create_cssm_vjepa(
            encoder=args.encoder,
            embed_dim=args.embed_dim,
            predictor_depth=args.predictor_depth,
            predictor_heads=args.predictor_heads,
            context_ratio=args.context_ratio,
            mask_ratio=args.mask_ratio,
            num_mask_blocks=args.num_mask_blocks,
            block_size=(args.block_size_t, args.block_size_h, args.block_size_w),
            alpha=args.alpha,
            beta=args.beta,
            ema_decay=args.ema_decay,
            rope_mode=args.rope_mode,
            vicreg_weight=args.vicreg_weight,
        )
    else:
        print("\nCreating CSSM-JEPA model (spatial masking)...")
        model = create_cssm_jepa(
            encoder=args.encoder,
            embed_dim=args.embed_dim,
            predictor_depth=args.predictor_depth,
            predictor_heads=args.predictor_heads,
            mask_ratio=args.mask_ratio,
            alpha=args.alpha,
            beta=args.beta,
            ema_decay=args.ema_decay,
            rope_mode=args.rope_mode,
            vicreg_weight=args.vicreg_weight,
        )

    # Initialize training state
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        rng=init_rng,
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        num_frames=args.num_frames,
        resolution=args.resolution,
    )

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model parameters: {num_params:,}")

    # Replicate state across devices (multi-GPU test already done earlier)
    if num_devices > 1:
        state = replicate(state)

    if use_wandb:
        wandb.config.update({'num_params': num_params, 'num_devices': num_devices})

    # Setup checkpointing
    checkpoint_dir = os.path.abspath(os.path.join(args.checkpoint_dir, run_name))
    os.makedirs(checkpoint_dir, exist_ok=True)
    use_simple_ckpt = args.simple_checkpoint
    if use_simple_ckpt:
        print(f"Using simple pickle checkpoints (NFS-compatible)")
        checkpointer = None  # Will use save_checkpoint_simple
    else:
        checkpointer = ocp.PyTreeCheckpointer()

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        # Anneal EMA decay
        progress = epoch / args.epochs
        ema_decay = args.ema_decay + (args.ema_anneal_end - args.ema_decay) * progress

        # Update state (handle replicated vs non-replicated)
        if num_devices > 1:
            # For replicated state, each device has its own copy of these scalars
            # Create properly shaped arrays that match the replicated structure
            epoch_arr = jnp.array([epoch] * num_devices, dtype=jnp.int32)
            ema_arr = jnp.array([ema_decay] * num_devices, dtype=jnp.float32)
            state = state.replace(epoch=epoch_arr, ema_decay=ema_arr)
        else:
            state = state.replace(
                epoch=jnp.array(epoch, dtype=jnp.int32),
                ema_decay=jnp.array(ema_decay, dtype=jnp.float32)
            )

        # Train epoch
        rng, epoch_rng = jax.random.split(rng)
        state, epoch_metrics, global_step = train_epoch(
            state=state,
            dataloader=dataloader,
            rng=epoch_rng,
            epoch=epoch,
            log_every=args.log_every,
            use_wandb=use_wandb,
            num_devices=num_devices,
            start_step=global_step,
        )

        # Log epoch metrics
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Loss: {epoch_metrics['loss_total']:.4f}")
        if args.alpha > 0:
            print(f"  Spectral: {epoch_metrics.get('spectral_total_loss', 0):.4f}")
        if args.beta > 0:
            print(f"  Feature: {epoch_metrics.get('feature_loss', 0):.4f}")
        print(f"  EMA decay: {ema_decay:.6f}")
        print(f"  Step time: {epoch_metrics['step_time_ms']:.1f} ms")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'epoch/loss': epoch_metrics['loss_total'],
                'epoch/ema_decay': ema_decay,
                'epoch/step_time_ms': epoch_metrics['step_time_ms'],
            }, step=global_step)

        # Save checkpoint (unreplicate if multi-GPU)
        if (epoch + 1) % args.save_every == 0:
            state_to_save = safe_unreplicate(state) if num_devices > 1 else state
            if use_simple_ckpt:
                ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pkl')
                save_checkpoint_simple(ckpt_path, state_to_save)
            else:
                ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}')
                # Convert numpy types (e.g. numpy.bool from optax) to JAX arrays for Orbax
                state_to_save = convert_to_jax_arrays(state_to_save)
                checkpointer.save(ckpt_path, state_to_save)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    state_to_save = safe_unreplicate(state) if num_devices > 1 else state
    if use_simple_ckpt:
        final_path = os.path.join(checkpoint_dir, 'final.pkl')
        save_checkpoint_simple(final_path, state_to_save)
    else:
        checkpointer.wait_until_finished()
        final_path = os.path.join(checkpoint_dir, 'final')
        # Convert numpy types (e.g. numpy.bool from optax) to JAX arrays for Orbax
        state_to_save = convert_to_jax_arrays(state_to_save)
        checkpointer.save(final_path, state_to_save)
    print(f"\nTraining complete! Final checkpoint: {final_path}")

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
