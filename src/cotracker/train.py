#!/usr/bin/env python3
"""
Training script for CSSM-CoTracker on Kubric dataset (JAX/Flax).

Usage:
    # Train with CSSM-SHViT encoder (can use JEPA weights)
    python src/cotracker/train.py --use_cssm_encoder --encoder_size s1

    # Train with pretrained JEPA weights
    python src/cotracker/train.py --use_cssm_encoder \
        --jepa_checkpoint /path/to/jepa/checkpoint

    # Basic training with ResNet encoder
    python src/cotracker/train.py --cssm_type opponent
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from functools import partial

# Prevent TensorFlow from seeing GPUs (before any TF import)
# TF is only used for data loading, doesn't need GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Add project root to path for absolute imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax


def parse_args():
    parser = argparse.ArgumentParser(description="Train CSSM-CoTracker (JAX)")

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='/media/data_cifs/projects/prj_video_imagenet/data/kubric',
                        help='Path to Kubric dataset (tar.gz files)')
    parser.add_argument('--tfrecord_dir', type=str, default=None,
                        help='Path to Kubric TFRecords (faster, use instead of data_dir)')
    parser.add_argument('--fast_tfrecord', action='store_true',
                        help='Use fast TFRecord format (from convert_kubric_tfrecords_fast.py)')
    parser.add_argument('--use_prefetch', action='store_true',
                        help='Use JAX device prefetcher for async data transfer')
    parser.add_argument('--shuffle_buffer', type=int, default=2000,
                        help='Shuffle buffer size for TFRecord loading')
    parser.add_argument('--num_parallel_reads', type=int, default=32,
                        help='Number of parallel TFRecord reads')
    parser.add_argument('--sequence_length', type=int, default=24,
                        help='Number of frames per training sample')
    parser.add_argument('--num_points', type=int, default=256,
                        help='Number of points to track per sample')
    parser.add_argument('--crop_size', type=int, default=384,
                        help='Training crop size')

    # Model
    parser.add_argument('--use_cssm', action='store_true',
                        help='Use CSSM variant instead of transformer')
    parser.add_argument('--cssm_type', type=str, default='opponent',
                        choices=['standard', 'opponent', 'hgru_bi'],
                        help='CSSM variant to use (only if --use_cssm)')
    parser.add_argument('--hidden_size', type=int, default=384,
                        help='Transformer/CSSM hidden dimension')
    parser.add_argument('--num_iters', type=int, default=4,
                        help='Number of refinement iterations')
    parser.add_argument('--space_depth', type=int, default=3,
                        help='Spatial attention depth')
    parser.add_argument('--time_depth', type=int, default=3,
                        help='Temporal attention/CSSM depth')
    parser.add_argument('--num_virtual_tracks', type=int, default=64,
                        help='Number of virtual tokens')

    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_steps', type=int, default=50000,
                        help='Total training steps')
    parser.add_argument('--warmup_steps', type=int, default=2500,
                        help='Learning rate warmup steps')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='Temporal weighting decay for loss')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective_batch = batch_size * grad_accum_steps)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing to reduce memory (slower but fits larger models)')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints/cssm_cotracker',
                        help='Checkpoint save directory')
    parser.add_argument('--jepa_checkpoint', type=str, default=None,
                        help='Path to pretrained JEPA checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--wandb_project', type=str, default='cssm-cotracker',
                        help='W&B project name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--log_every', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--save_every', type=int, default=5000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_every', type=int, default=1000,
                        help='Evaluate every N steps')

    return parser.parse_args()


def huber_loss(pred, target, mask=None, delta=1.0):
    """Huber loss for coordinate predictions."""
    diff = pred - target
    abs_diff = jnp.abs(diff)
    loss = jnp.where(abs_diff < delta,
                     0.5 * diff ** 2,
                     delta * (abs_diff - 0.5 * delta))
    if mask is not None:
        loss = loss * mask[..., None]
        return loss.sum() / (mask.sum() * 2 + 1e-6)  # *2 for x,y
    return loss.mean()


def sequence_loss(pred_coords, target_coords, visibility, gamma=0.8):
    """
    Sequence loss with temporal weighting.

    Later predictions are weighted more heavily (gamma^(T-t) weighting).
    """
    T = pred_coords.shape[1]

    total_loss = 0.0
    for t in range(T):
        weight = gamma ** (T - t - 1)
        t_loss = huber_loss(pred_coords[:, t], target_coords[:, t], visibility[:, t])
        total_loss = total_loss + weight * t_loss

    return total_loss / T


def visibility_loss(pred_vis, target_vis):
    """Binary cross-entropy loss for visibility."""
    eps = 1e-7
    pred_vis = jnp.clip(pred_vis, eps, 1 - eps)
    target_vis = target_vis[..., None].astype(jnp.float32)
    return -(target_vis * jnp.log(pred_vis) + (1 - target_vis) * jnp.log(1 - pred_vis)).mean()


def _train_step_single(state, video, queries, trajs, visibility, gamma, use_remat=False):
    """Single training step (not jitted, used inside pmap)."""

    def loss_fn(params):
        # Optionally use gradient checkpointing (remat) to reduce memory
        if use_remat:
            # Wrap apply_fn so training is a positional arg we can mark static
            def apply_with_static_training(params, video, queries, training):
                return state.apply_fn(params, video, queries, training=training)

            apply_fn = jax.checkpoint(
                apply_with_static_training,
                static_argnums=(3,)  # training is positional arg 3
            )
            outputs = apply_fn(params, video, queries, True)
        else:
            outputs = state.apply_fn(params, video, queries, training=True)

        coords = outputs['coords']
        vis = outputs['vis']

        # Coordinate loss with temporal weighting
        coord_loss = sequence_loss(coords, trajs, visibility, gamma)

        # Visibility loss
        vis_loss = visibility_loss(vis, visibility)

        total_loss = coord_loss + 0.1 * vis_loss

        return total_loss, {'coord_loss': coord_loss, 'vis_loss': vis_loss}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='devices')
    loss = jax.lax.pmean(loss, axis_name='devices')
    metrics = jax.lax.pmean(metrics, axis_name='devices')

    state = state.apply_gradients(grads=grads)

    return state, loss, metrics


# Single-GPU version (jitted)
@partial(jax.jit, static_argnums=(5, 6))
def train_step_single_gpu(state, video, queries, trajs, visibility, gamma, use_remat=False):
    """Single training step for single GPU."""
    return _train_step_single(state, video, queries, trajs, visibility, gamma, use_remat)


def create_train_step_multi_gpu(gamma, use_remat=False):
    """Create pmap'd train step for multi-GPU training."""
    def train_step_fn(state, video, queries, trajs, visibility):
        return _train_step_single(state, video, queries, trajs, visibility, gamma, use_remat)

    return jax.pmap(train_step_fn, axis_name='devices')


def compute_grads_single(state, video, queries, trajs, visibility, gamma, use_remat=False):
    """Compute gradients without applying them (for gradient accumulation)."""

    def loss_fn(params):
        if use_remat:
            # Wrap apply_fn so training is a positional arg we can mark static
            def apply_with_static_training(params, video, queries, training):
                return state.apply_fn(params, video, queries, training=training)

            apply_fn = jax.checkpoint(
                apply_with_static_training,
                static_argnums=(3,)  # training is positional arg 3
            )
            outputs = apply_fn(params, video, queries, True)
        else:
            outputs = state.apply_fn(params, video, queries, training=True)

        coords = outputs['coords']
        vis = outputs['vis']

        coord_loss = sequence_loss(coords, trajs, visibility, gamma)
        vis_loss = visibility_loss(vis, visibility)
        total_loss = coord_loss + 0.1 * vis_loss

        return total_loss, {'coord_loss': coord_loss, 'vis_loss': vis_loss}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return grads, loss, metrics


@partial(jax.jit, static_argnums=(5, 6))
def compute_grads_jit(state, video, queries, trajs, visibility, gamma, use_remat=False):
    """JIT-compiled gradient computation for single GPU."""
    return compute_grads_single(state, video, queries, trajs, visibility, gamma, use_remat)


def create_compute_grads_multi_gpu(gamma, use_remat=False):
    """Create pmap'd gradient computation for multi-GPU."""
    def compute_grads_fn(state, video, queries, trajs, visibility):
        grads, loss, metrics = compute_grads_single(
            state, video, queries, trajs, visibility, gamma, use_remat
        )
        # Average across devices
        grads = jax.lax.pmean(grads, axis_name='devices')
        loss = jax.lax.pmean(loss, axis_name='devices')
        metrics = jax.lax.pmean(metrics, axis_name='devices')
        return grads, loss, metrics

    return jax.pmap(compute_grads_fn, axis_name='devices')


@jax.jit
def apply_grads(state, grads):
    """Apply accumulated gradients to state."""
    return state.apply_gradients(grads=grads)


def apply_grads_multi_gpu(state, grads):
    """Apply gradients in multi-GPU setting."""
    # grads are already averaged, just apply
    return jax.pmap(lambda s, g: s.apply_gradients(grads=g))(state, grads)


def create_train_state(model, rng, args):
    """Create initial training state."""
    # Initialize with dummy input
    dummy_video = jnp.zeros((1, args.sequence_length, args.crop_size, args.crop_size, 3))
    dummy_queries = jnp.zeros((1, 10, 3))

    params = model.init(rng, dummy_video, dummy_queries, training=False)

    # Create optimizer with warmup + cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.num_steps,
        end_value=args.lr * 0.01,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    return state


def load_jepa_encoder_weights(state, jepa_checkpoint_path):
    """Load pretrained JEPA encoder weights into the model."""
    import orbax.checkpoint as ocp

    print(f"Loading JEPA weights from {jepa_checkpoint_path}")

    # Load JEPA checkpoint
    checkpointer = ocp.PyTreeCheckpointer()
    jepa_state = checkpointer.restore(jepa_checkpoint_path)

    # Extract encoder parameters
    # JEPA checkpoint structure: {'params': {'online_encoder': {...}}}
    if 'params' in jepa_state:
        jepa_params = jepa_state['params']
    else:
        jepa_params = jepa_state

    if 'online_encoder' in jepa_params:
        encoder_params = jepa_params['online_encoder']
    elif 'encoder' in jepa_params:
        encoder_params = jepa_params['encoder']
    else:
        print("Warning: Could not find encoder in JEPA checkpoint")
        return state

    # Match encoder params to model
    # This depends on the exact model structure
    # TODO: Implement proper parameter matching

    print("Loaded JEPA encoder weights")
    return state


def main():
    args = parse_args()

    # Setup multi-GPU
    devices = jax.devices()
    num_devices = len(devices)
    use_multi_gpu = num_devices > 1

    # Batch size must be divisible by number of devices
    if use_multi_gpu and args.batch_size % num_devices != 0:
        old_bs = args.batch_size
        args.batch_size = (args.batch_size // num_devices) * num_devices
        print(f"Warning: Adjusted batch_size {old_bs} -> {args.batch_size} for {num_devices} devices")

    batch_per_device = args.batch_size // num_devices if use_multi_gpu else args.batch_size

    # Setup
    print("=" * 60)
    print("CoTracker3 Training (JAX)")
    print("=" * 60)
    print(f"Use CSSM: {args.use_cssm}")
    if args.use_cssm:
        print(f"CSSM type: {args.cssm_type}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Devices: {devices}")
    print(f"Num devices: {num_devices}")
    print(f"Multi-GPU: {use_multi_gpu}")
    print(f"Total batch size: {args.batch_size}")
    if use_multi_gpu:
        print(f"Batch per device: {batch_per_device}")
    print(f"Data dir: {args.data_dir}")
    print("=" * 60)

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    if not args.no_wandb:
        try:
            import wandb
            run_name = f"cssm_{args.cssm_type}" if args.use_cssm else "transformer"
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=run_name,
            )
        except ImportError:
            print("W&B not available, logging disabled")
            args.no_wandb = True

    # Create model
    from src.cotracker.jax_impl.cotracker3 import create_cotracker3

    model = create_cotracker3(
        use_cssm=args.use_cssm,
        cssm_type=args.cssm_type,
        hidden_size=args.hidden_size,
        num_iters=args.num_iters,
        space_depth=args.space_depth,
        time_depth=args.time_depth,
        num_virtual_tracks=args.num_virtual_tracks,
    )

    # Initialize
    rng = jax.random.PRNGKey(42)
    state = create_train_state(model, rng, args)

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Parameters: {num_params / 1e6:.2f}M")

    # Load JEPA weights if provided (optional, for pretrained encoder)
    if args.jepa_checkpoint:
        state = load_jepa_encoder_weights(state, args.jepa_checkpoint)

    # Setup multi-GPU training
    use_grad_accum = args.grad_accum_steps > 1
    use_remat = args.gradient_checkpointing

    if use_remat:
        print(f"Gradient checkpointing enabled (reduces memory, slower)")
    if use_grad_accum:
        effective_batch = args.batch_size * args.grad_accum_steps
        print(f"Gradient accumulation: {args.grad_accum_steps} steps (effective batch={effective_batch})")

    if use_multi_gpu:
        # Replicate state across devices
        state = jax.device_put_replicated(state, devices)
        if use_grad_accum:
            compute_grads_fn = create_compute_grads_multi_gpu(args.gamma, use_remat)
        else:
            train_step_fn = create_train_step_multi_gpu(args.gamma, use_remat)
        print(f"State replicated across {num_devices} devices")
    else:
        # Create jitted single-GPU functions
        if use_grad_accum:
            def compute_grads_fn(state, video, queries, trajs, visibility):
                return compute_grads_jit(state, video, queries, trajs, visibility, args.gamma, use_remat)
        else:
            @jax.jit
            def train_step_fn(state, video, queries, trajs, visibility):
                return _train_step_single(state, video, queries, trajs, visibility, args.gamma, use_remat)

    # Create dataset
    use_tfrecord = args.tfrecord_dir is not None

    # Disable TensorFlow GPU BEFORE importing data loaders
    # This prevents TF from trying to compile CUDA kernels for data loading
    if use_tfrecord:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    if use_tfrecord:
        if args.fast_tfrecord:
            from src.cotracker.data.kubric import get_kubric_fast_loader, JAXPrefetcher
            print(f"\nLoading FAST Kubric TFRecords from {args.tfrecord_dir}...")
            try:
                train_loader = get_kubric_fast_loader(
                    tfrecord_dir=args.tfrecord_dir,
                    batch_size=args.batch_size,
                    shuffle_buffer=args.shuffle_buffer,
                    num_parallel_reads=args.num_parallel_reads,
                    prefetch_batches=8,
                    augment=True,
                )
            except Exception as e:
                print(f"Error loading fast TFRecords: {e}")
                print("Convert Kubric with fast converter first:")
                print("  python scripts/convert_kubric_tfrecords_fast.py --input_dir ... --output_dir ...")
                sys.exit(1)
        else:
            from src.cotracker.data.kubric import get_kubric_tfrecord_loader
            print(f"\nLoading Kubric TFRecords from {args.tfrecord_dir}...")
            try:
                train_loader = get_kubric_tfrecord_loader(
                    tfrecord_dir=args.tfrecord_dir,
                    sequence_length=args.sequence_length,
                    num_points=args.num_points,
                    crop_size=args.crop_size,
                    batch_size=args.batch_size,
                )
            except Exception as e:
                print(f"Error loading TFRecords: {e}")
                print("Convert Kubric to TFRecords first:")
                print("  python scripts/convert_kubric_tfrecords.py --input_dir ... --output_dir ...")
                sys.exit(1)
        print(f"Dataset: {len(train_loader) * args.batch_size} sequences")
    else:
        from src.cotracker.data.kubric import KubricDataset
        print(f"\nLoading Kubric tar.gz from {args.data_dir}...")
        try:
            train_dataset = KubricDataset(
                args.data_dir,
                split='train',
                sequence_length=args.sequence_length,
                num_points=args.num_points,
                crop_size=args.crop_size,
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Make sure to download Kubric first:")
            print("  python scripts/download_kubric.py")
            sys.exit(1)
        print(f"Dataset: {len(train_dataset)} sequences")

    # Training loop
    print(f"\nStarting training for {args.num_steps} steps...")

    use_prefetch = args.use_prefetch and use_tfrecord
    if use_prefetch:
        if use_multi_gpu:
            print(f"Using JAX device prefetcher with {num_devices}-way sharding")
        else:
            print("Using JAX device prefetcher for async data transfer")

    rng, data_rng = jax.random.split(rng)
    step = 0

    while step < args.num_steps:
        # Create iterator for this epoch
        if use_tfrecord:
            data_iter = iter(train_loader)
            # Wrap with JAX prefetcher for async GPU transfer
            if use_prefetch:
                from src.cotracker.data.kubric import JAXPrefetcher
                data_iter = iter(JAXPrefetcher(
                    data_iter,
                    prefetch_size=2,
                    num_devices=num_devices if use_multi_gpu else 1,
                    devices=devices if use_multi_gpu else None,
                ))
        else:
            # For tar.gz, create a simple batch iterator
            data_iter = None

        for batch_idx in range(len(train_loader) if use_tfrecord else len(train_dataset) // args.batch_size):
            if step >= args.num_steps:
                break

            # Get batch
            if use_tfrecord:
                try:
                    sample = next(data_iter)
                except StopIteration:
                    break
                # If using prefetcher, data is already on device as jax arrays
                if use_prefetch:
                    video = sample['video']
                    trajs = sample['trajs']
                    visibility = sample['visibility']
                else:
                    video = jnp.array(sample['video'])
                    trajs = jnp.array(sample['trajs'])
                    visibility = jnp.array(sample['visibility'])
            else:
                # Sample random batch from tar.gz dataset
                batch_videos, batch_trajs, batch_vis = [], [], []
                for _ in range(args.batch_size):
                    idx = np.random.randint(len(train_dataset))
                    s = train_dataset[idx]
                    batch_videos.append(s['video'])
                    batch_trajs.append(s['trajs'])
                    batch_vis.append(s['visibility'])

                video = jnp.array(np.stack(batch_videos).astype(np.float32) / 255.0)
                trajs = jnp.array(np.stack(batch_trajs))
                visibility = jnp.array(np.stack(batch_vis))

            # Create queries from first frame
            # Handle both sharded (num_devices, batch_per_device, ...) and flat (B, ...) formats
            if use_multi_gpu and use_prefetch:
                # Data already sharded: (num_devices, batch_per_device, T, N, 2)
                D, Bd, T, N, _ = trajs.shape
                query_frame = 0
                frame_idx = jnp.full((D, Bd, N, 1), query_frame)
                queries = jnp.concatenate([frame_idx, trajs[:, :, query_frame, :, :]], axis=-1)
            else:
                # Flat batch: (B, T, N, 2)
                B, T, N, _ = trajs.shape
                query_frame = 0
                frame_idx = jnp.full((B, N, 1), query_frame)
                queries = jnp.concatenate([frame_idx, trajs[:, query_frame, :, :]], axis=-1)

                # Reshape for multi-GPU if not using prefetcher
                if use_multi_gpu:
                    video = video.reshape(num_devices, batch_per_device, *video.shape[1:])
                    queries = queries.reshape(num_devices, batch_per_device, *queries.shape[1:])
                    trajs = trajs.reshape(num_devices, batch_per_device, *trajs.shape[1:])
                    visibility = visibility.reshape(num_devices, batch_per_device, *visibility.shape[1:])

            # Training step (with optional gradient accumulation)
            if use_grad_accum:
                # Gradient accumulation: compute grads, accumulate, then apply
                grads, loss, metrics = compute_grads_fn(state, video, queries, trajs, visibility)

                if use_multi_gpu:
                    loss = loss[0]
                    metrics = {k: v[0] for k, v in metrics.items()}

                # Initialize or accumulate gradients
                if batch_idx % args.grad_accum_steps == 0:
                    accum_grads = grads
                    accum_loss = loss
                    accum_metrics = metrics
                else:
                    accum_grads = jax.tree.map(lambda a, g: a + g, accum_grads, grads)
                    accum_loss = accum_loss + loss
                    accum_metrics = {k: accum_metrics[k] + metrics[k] for k in metrics}

                # Apply gradients after accumulation steps
                if (batch_idx + 1) % args.grad_accum_steps == 0:
                    # Average gradients
                    accum_grads = jax.tree.map(lambda g: g / args.grad_accum_steps, accum_grads)
                    accum_loss = accum_loss / args.grad_accum_steps
                    accum_metrics = {k: v / args.grad_accum_steps for k, v in accum_metrics.items()}

                    # Apply gradients
                    if use_multi_gpu:
                        state = jax.pmap(lambda s, g: s.apply_gradients(grads=g))(state, accum_grads)
                    else:
                        state = apply_grads(state, accum_grads)

                    loss = accum_loss
                    metrics = accum_metrics
                    step += 1
                else:
                    # Skip logging/checkpointing until we apply gradients
                    continue
            else:
                # Standard training step (no accumulation)
                state, loss, metrics = train_step_fn(state, video, queries, trajs, visibility)

                if use_multi_gpu:
                    # Get scalar values from first device (they're all the same after pmean)
                    loss = loss[0]
                    metrics = {k: v[0] for k, v in metrics.items()}

                step += 1

            # Logging
            if step % args.log_every == 0:
                print(f"Step {step:6d}/{args.num_steps} | "
                      f"Loss: {float(loss):.4f} | "
                      f"Coord: {float(metrics['coord_loss']):.4f} | "
                      f"Vis: {float(metrics['vis_loss']):.4f}")

                if not args.no_wandb:
                    import wandb
                    wandb.log({
                        'loss': float(loss),
                        'coord_loss': float(metrics['coord_loss']),
                        'vis_loss': float(metrics['vis_loss']),
                        'step': step,
                    })

            # Checkpointing
            if step > 0 and step % args.save_every == 0:
                save_path = Path(args.checkpoint_dir) / f"step_{step}"
                save_path.mkdir(parents=True, exist_ok=True)

                import orbax.checkpoint as ocp
                checkpointer = ocp.PyTreeCheckpointer()
                # Unreplicate state for saving (take first device's copy)
                save_state = jax.tree_util.tree_map(lambda x: x[0], state) if use_multi_gpu else state
                checkpointer.save(str(save_path), save_state)
                print(f"Saved checkpoint to {save_path}")

    # Final save
    save_path = Path(args.checkpoint_dir) / "final"
    save_path.mkdir(parents=True, exist_ok=True)
    import orbax.checkpoint as ocp
    checkpointer = ocp.PyTreeCheckpointer()
    # Unreplicate state for saving
    save_state = jax.tree_util.tree_map(lambda x: x[0], state) if use_multi_gpu else state
    checkpointer.save(str(save_path), save_state)
    print(f"Saved final checkpoint to {save_path}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
