"""Evaluate all sweep checkpoints on pathfinder test set.

Supports different batch norm eval modes:
  running   - Use running average statistics from training (default)
  online    - Use current eval batch statistics
  calibrate - Recompute statistics from training set, then eval
"""

import os
import sys
import pickle
import re
import gc
import argparse

# Force TF CPU-only before any other imports
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from src.models.simple_cssm import SimpleCSSM


def parse_sweep_name(name):
    """Parse hyperparameters from sweep checkpoint name."""
    # 128px sweep
    m = re.match(r'sweep_(pre|post)_(init|noinit)_(layer|batch|instance)_(mean|max)$', name)
    if m:
        return {
            'resolution': '128',
            'stem_layers': 1,
            'stem_norm_order': m.group(1),
            'learned_init': m.group(2) == 'init',
            'norm_type': m.group(3),
            'pool_type': m.group(4),
        }
    # 224px sweep
    m = re.match(r'sweep224_(pre|post)_(init|noinit)_(layer|batch|instance)_(mean|max)$', name)
    if m:
        return {
            'resolution': '224',
            'stem_layers': 2,
            'stem_norm_order': m.group(1),
            'learned_init': m.group(2) == 'init',
            'norm_type': m.group(3),
            'pool_type': m.group(4),
        }
    return None


def create_model(hp):
    """Create SimpleCSSM model matching sweep config."""
    return SimpleCSSM(
        num_classes=2,
        embed_dim=32,
        depth=1,
        cssm_type='add_kqv',
        kernel_size=11,
        norm_type=hp['norm_type'],
        stem_norm_order=hp['stem_norm_order'],
        pos_embed='separate',
        pool_type=hp['pool_type'],
        seq_len=8,
        gate_type='factored',
        stem_mode='default',
        stem_layers=hp['stem_layers'],
        learned_init=hp['learned_init'],
        use_complex32=True,
        use_goom=True,
    )


def load_checkpoint(ckpt_path):
    """Load a pickle checkpoint."""
    with open(os.path.join(ckpt_path, 'checkpoint.pkl'), 'rb') as f:
        return pickle.load(f)


def calibrate_batch_stats(model, params, batch_stats, train_loader):
    """Recompute batch norm statistics from training set.

    Does a full pass over training data, accumulating true mean/var
    (not exponential moving average).
    """
    # Get the batch_stats structure to know which layers have BN
    stat_keys = list(batch_stats.keys())

    # Accumulate sum and sum_of_squares for each BN layer
    accum = {}
    for key in stat_keys:
        accum[key] = {
            'mean_sum': jnp.zeros_like(batch_stats[key]['mean']),
            'var_sum': jnp.zeros_like(batch_stats[key]['var']),
            'count': 0,
        }

    # Run training data through model in training mode to get per-batch stats
    # We'll collect them and average
    @jax.jit
    def get_batch_stats(params, batch_stats, videos):
        variables = {'params': params, 'batch_stats': batch_stats}
        _, mutated = model.apply(
            variables, videos, training=True, mutable=['batch_stats']
        )
        return mutated['batch_stats']

    n_batches = 0
    for videos, labels in train_loader:
        new_stats = get_batch_stats(params, batch_stats, videos)
        for key in stat_keys:
            accum[key]['mean_sum'] = accum[key]['mean_sum'] + new_stats[key]['mean']
            accum[key]['var_sum'] = accum[key]['var_sum'] + new_stats[key]['var']
            accum[key]['count'] += 1
        n_batches += 1

    # Average to get true training set statistics
    calibrated = {}
    for key in stat_keys:
        calibrated[key] = {
            'mean': accum[key]['mean_sum'] / accum[key]['count'],
            'var': accum[key]['var_sum'] / accum[key]['count'],
        }

    print(f"    Calibrated BN stats from {n_batches} training batches")
    return calibrated


def evaluate_checkpoint(model, ckpt_data, val_loader, bn_mode='running',
                        train_loader=None):
    """Evaluate a single checkpoint.

    bn_mode:
      'running'   - use stored running average stats (default)
      'online'    - use current eval batch stats (training=True)
      'calibrate' - recompute stats from training set first
    """
    params = jax.tree_util.tree_map(jnp.array, ckpt_data['params'])
    batch_stats = None
    if 'batch_stats' in ckpt_data and ckpt_data['batch_stats'] is not None:
        batch_stats = jax.tree_util.tree_map(jnp.array, ckpt_data['batch_stats'])

    has_batch_stats = batch_stats is not None

    # Calibrate if requested
    if bn_mode == 'calibrate' and has_batch_stats and train_loader is not None:
        batch_stats = calibrate_batch_stats(model, params, batch_stats, train_loader)

    use_online = (bn_mode == 'online' and has_batch_stats)

    if use_online:
        @jax.jit
        def eval_step(params, batch_stats, videos, labels):
            variables = {'params': params, 'batch_stats': batch_stats}
            # training=True to use current batch statistics
            logits, _ = model.apply(
                variables, videos, training=True, mutable=['batch_stats']
            )
            one_hot = jax.nn.one_hot(labels, 2)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            acc = jnp.mean(jnp.argmax(logits, -1) == labels)
            return loss, acc
    else:
        @jax.jit
        def eval_step(params, batch_stats, videos, labels):
            variables = {'params': params}
            if has_batch_stats:
                variables['batch_stats'] = batch_stats
            logits = model.apply(variables, videos, training=False)
            one_hot = jax.nn.one_hot(labels, 2)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            acc = jnp.mean(jnp.argmax(logits, -1) == labels)
            return loss, acc

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for videos, labels in val_loader:
        bs = videos.shape[0]
        loss, acc = eval_step(params, batch_stats, videos, labels)
        total_loss += float(loss) * bs
        total_acc += float(acc) * bs
        total_samples += bs

    # Clean up
    del params, batch_stats
    jax.clear_caches()
    gc.collect()

    if total_samples == 0:
        return 0.0, 0.0, 0

    return total_loss / total_samples, total_acc / total_samples, total_samples


def main():
    parser = argparse.ArgumentParser(description='Evaluate sweep checkpoints')
    parser.add_argument('--tfrecord_dir', type=str,
                        default='/media/data_cifs/projects/prj_video_imagenet/pathfinder_tfrecords_128')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bn_mode', type=str, default='running',
                        choices=['running', 'online', 'calibrate', 'all'],
                        help='Batch norm eval mode (or "all" to test all 3)')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only evaluate checkpoints matching this substring')
    args = parser.parse_args()

    print(f"TFRecord dir: {args.tfrecord_dir}")
    print(f"Eval batch size: {args.batch_size}")

    bn_modes = ['running', 'online', 'calibrate'] if args.bn_mode == 'all' else [args.bn_mode]
    print(f"BN eval modes: {bn_modes}")

    # Load test data
    from src.pathfinder_data import get_pathfinder_tfrecord_loader
    val_loader = get_pathfinder_tfrecord_loader(
        tfrecord_dir=args.tfrecord_dir,
        difficulty='14',
        batch_size=args.batch_size,
        num_frames=8,
        split='test',
        shuffle=False,
    )
    print(f"Loaded TFRecord test set")

    # Load training data (only if calibrate mode needed)
    train_loader = None
    if 'calibrate' in bn_modes:
        train_loader = get_pathfinder_tfrecord_loader(
            tfrecord_dir=args.tfrecord_dir,
            difficulty='14',
            batch_size=args.batch_size,
            num_frames=8,
            split='train',
            shuffle=False,
        )
        print(f"Loaded TFRecord train set (for calibration)")

    # Find all sweep checkpoints
    checkpoint_dir = 'checkpoints'
    sweep_dirs = sorted([
        d for d in os.listdir(checkpoint_dir)
        if d.startswith('sweep') and os.path.isdir(os.path.join(checkpoint_dir, d))
    ])

    if args.filter:
        sweep_dirs = [d for d in sweep_dirs if args.filter in d]

    print(f"\nFound {len(sweep_dirs)} sweep checkpoints\n")

    results = []

    for sweep_name in sweep_dirs:
        hp = parse_sweep_name(sweep_name)
        if hp is None:
            print(f"  Skipping {sweep_name} (can't parse name)")
            continue

        # Find the latest epoch checkpoint
        sweep_path = os.path.join(checkpoint_dir, sweep_name)
        epochs = sorted([
            d for d in os.listdir(sweep_path)
            if d.startswith('epoch_')
        ], key=lambda x: int(x.split('_')[1]))

        if not epochs:
            print(f"  Skipping {sweep_name} (no epoch checkpoints)")
            continue

        last_epoch = epochs[-1]
        ckpt_path = os.path.join(sweep_path, last_epoch)

        for bn_mode in bn_modes:
            # Skip non-default BN modes for non-batch-norm models
            if bn_mode != 'running' and hp['norm_type'] != 'batch':
                continue

            mode_label = f" [bn={bn_mode}]" if bn_mode != 'running' else ""
            print(f"Evaluating {sweep_name} ({last_epoch}){mode_label}...")

            try:
                model = create_model(hp)
                ckpt_data = load_checkpoint(ckpt_path)
                loss, acc, n_samples = evaluate_checkpoint(
                    model, ckpt_data, val_loader,
                    bn_mode=bn_mode, train_loader=train_loader,
                )

                results.append({
                    'name': sweep_name,
                    'epoch': last_epoch,
                    'bn_mode': bn_mode,
                    **hp,
                    'val_loss': loss,
                    'val_acc': acc,
                    'n_samples': n_samples,
                })

                print(f"  Val Acc: {acc:.4f}  Val Loss: {loss:.4f}  (n={n_samples})")
            except Exception as e:
                print(f"  FAILED: {e}")

    # Print summary table
    if results:
        print("\n" + "=" * 110)
        print("SWEEP RESULTS SUMMARY")
        print("=" * 110)
        print(f"{'Name':<45} {'BN Mode':<12} {'Epoch':<10} {'Val Acc':>8} {'Val Loss':>9}")
        print("-" * 110)

        results.sort(key=lambda x: x['val_acc'], reverse=True)

        for r in results:
            print(f"{r['name']:<45} {r['bn_mode']:<12} {r['epoch']:<10} "
                  f"{r['val_acc']:>8.4f} {r['val_loss']:>9.4f}")

        print("-" * 110)

        # Analysis by hyperparameter
        print("\n--- Analysis by Hyperparameter ---\n")

        for hp_name in ['bn_mode', 'stem_norm_order', 'learned_init', 'norm_type', 'pool_type']:
            groups = {}
            for r in results:
                key = str(r[hp_name])
                if key not in groups:
                    groups[key] = []
                groups[key].append(r['val_acc'])

            print(f"{hp_name}:")
            for key in sorted(groups.keys()):
                accs = groups[key]
                print(f"  {key:<15} mean={np.mean(accs):.4f}  std={np.std(accs):.4f}  "
                      f"min={np.min(accs):.4f}  max={np.max(accs):.4f}  n={len(accs)}")
            print()

        best = results[0]
        print(f"Best: {best['name']} [bn={best['bn_mode']}]  Val Acc: {best['val_acc']:.4f}")


if __name__ == '__main__':
    main()
