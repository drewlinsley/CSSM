"""
Measure per-timestep variation of CSSM alpha_t / beta_t gates.

Loads a checkpoint, runs a forward pass on a small ImageNet val batch with
'intermediates' mutable, then reports:
  * per-t mean across (B, H, W)  → shape (T,)
  * per-t std  across (B, H, W)  → shape (T,)
  * range = max-min of the per-t means, normalized by their average magnitude
    (a "flat across t" gate has range/|mean| ≈ 0; t-varying gates are larger).

Usage:
    JAX_PLATFORMS=cpu python scripts/diagnostics/gate_t_variation.py \\
        --checkpoint /oscar/scratch/dlinsley/imagenet_checkpoints/cssm_gdn_T8_k3_v5cfg/epoch_90_ema \\
        --rope_mode spatiotemporal

    JAX_PLATFORMS=cpu python scripts/diagnostics/gate_t_variation.py \\
        --checkpoint /oscar/scratch/dlinsley/imagenet_checkpoints/cssm_gdn_T8_k3_v5cfg_learned_t/epoch_30_ema \\
        --rope_mode learned_t
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from src.models.cssm_shvit import cssm_shvit_s1
from src.data.imagenet_tfdata import TFRecordImageNetLoader

IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 1, 3)
IMAGENET_STD  = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 1, 3)


def build_model(args):
    return cssm_shvit_s1(
        num_classes=1000, num_timesteps=args.num_timesteps,
        cssm_type='gdn', delta_key_dim=2,
        kernel_sizes=(args.kernel_size,) * 4,
        output_norm='rms', gate_type='factored',
        short_conv_spatial_size=3, short_conv_size=4,
        block_norm='global_layer', rope_mode=args.rope_mode,
        use_pos_conv=True, gate_proj_bias_init=0.0,
        head_pool='mean',
    )


def load_checkpoint(ckpt_path, target_params):
    """Restore with explicit target tree so cross-device sharding remaps correctly.

    Checkpoints are saved as {'ema_params': pytree}; we wrap the init target to
    match that top-level key, then unwrap.
    """
    wrapped = {'ema_params': target_params}
    abs_target = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), wrapped)
    restored = ocp.StandardCheckpointer().restore(ckpt_path, target=abs_target)
    return restored['ema_params']


def prepare_batch(x, T):
    x = x.astype(jnp.float32) / 255.0
    if x.ndim == 4:
        x = jnp.broadcast_to(x[:, None], (x.shape[0], T) + x.shape[1:])
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


def find_all(node, key, prefix='', out=None):
    """Walk intermediates dict, collect every occurrence of `key` with its path."""
    if out is None:
        out = []
    if isinstance(node, dict):
        for k, v in node.items():
            new_prefix = f'{prefix}/{k}' if prefix else k
            if k == key:
                val = v[0] if isinstance(v, tuple) else v
                out.append((new_prefix, val))
            find_all(v, key, new_prefix, out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--rope_mode', required=True,
                   choices=['none', 'temporal', 'spatiotemporal', 'learned_t'])
    p.add_argument('--num_timesteps', type=int, default=8)
    p.add_argument('--kernel_size', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--tfrecord_dir', default='/oscar/scratch/dlinsley/imagenet_tfrecords')
    args = p.parse_args()

    print(f'Checkpoint: {args.checkpoint}')
    print(f'rope_mode: {args.rope_mode}, T={args.num_timesteps}, k={args.kernel_size}')

    model = build_model(args)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 224, 224, 3))
    init_vars = model.init({'params': rng, 'dropout': rng}, dummy, training=True)
    init_bs = init_vars.get('batch_stats', {})
    init_params = init_vars['params']

    bb_params = load_checkpoint(args.checkpoint, init_params)
    print(f'Backbone params: {sum(p.size for p in jax.tree_util.tree_leaves(bb_params)):,}')

    loader = TFRecordImageNetLoader(
        tfrecord_dir=args.tfrecord_dir, split='val',
        batch_size=args.batch_size, image_size=224, drop_remainder=True,
        num_parallel_reads=2, prefetch_batches=1, threadpool_size=2,
    )
    x_np, _ = next(iter(loader))
    x = prepare_batch(jnp.asarray(x_np), args.num_timesteps)
    print(f'Input shape: {x.shape}')

    _, new_state = model.apply(
        {'params': bb_params, 'batch_stats': init_bs}, x, training=True,
        rngs={'dropout': jax.random.PRNGKey(0)},
        mutable=['intermediates', 'batch_stats'],
    )
    interm = new_state['intermediates']

    # Report per-block, per-gate per-t variation.
    for gate_name in ('alpha_t', 'beta_t'):
        hits = find_all(interm, gate_name)
        print(f'\n=== {gate_name} ({len(hits)} blocks) ===')
        if not hits:
            print('  (not sown)')
            continue
        for path, val in hits:
            # Shape is (B, T, 1, H, W_freq) for factored gates; squeeze the singleton.
            val = jnp.asarray(val).astype(jnp.float32)
            # Collapse all non-T dims to one to compute per-t stats.
            B = val.shape[0]; T = val.shape[1]
            per_t = val.reshape(B, T, -1)                       # (B, T, HW)
            per_t_mean = per_t.mean(axis=(0, 2))                # (T,)  mean across B,H,W
            per_t_std  = per_t.std(axis=(0, 2))                 # (T,)
            avg_mag    = per_t_mean.mean()
            range_abs  = per_t_mean.max() - per_t_mean.min()
            range_rel  = float(range_abs / (abs(avg_mag) + 1e-8) * 100.)
            print(f'  [{path}]  per-t mean: {np.asarray(per_t_mean)}')
            print(f'    range across t: {float(range_abs):.5f}  '
                  f'({range_rel:.2f}% of mean {float(avg_mag):.4f})')
            print(f'    per-t std (across B,H,W): {np.asarray(per_t_std)}')


if __name__ == '__main__':
    main()
