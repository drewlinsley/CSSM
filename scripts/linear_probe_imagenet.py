"""
Frozen-backbone linear probe on ImageNet for SFA/LeJEPA evaluations.

Loads a CSSM-SHViT checkpoint, extracts pooled features via the sown
'pooled_features' intermediate, trains a fresh linear head, and reports
ImageNet val top-1. Single-GPU (head is cheap).

Usage:
    python scripts/linear_probe_imagenet.py \\
        --checkpoint /oscar/.../epoch_50_ema \\
        --num_timesteps 8 --kernel_size 3 --epochs 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import linen as nn
from tqdm import tqdm

from src.models.cssm_shvit import cssm_shvit_s1
from src.data.imagenet_tfdata import TFRecordImageNetLoader


IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 1, 3)
IMAGENET_STD  = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 1, 3)


def build_model(args):
    return cssm_shvit_s1(
        num_classes=1000, num_timesteps=args.num_timesteps,
        cssm_type=args.cssm_type, delta_key_dim=args.delta_key_dim,
        kernel_sizes=(args.kernel_size,) * 4,
        output_norm='rms', gate_type='factored',
        short_conv_spatial_size=0, short_conv_size=0,
        block_norm='global_layer', rope_mode='spatiotemporal',
        use_pos_conv=True, gate_proj_bias_init=0.0,
        head_pool=args.head_pool,
        linear_probe=args.linear_probe, ssl_proj_dim=args.ssl_proj_dim,
    )


def load_checkpoint(ckpt_path):
    restored = ocp.StandardCheckpointer().restore(ckpt_path)
    params = (restored.get('ema_params')
              or restored.get('params')
              or restored)
    return params


def prepare_batch(x, num_timesteps):
    x = x.astype(jnp.float32) / 255.0
    if x.ndim == 4:
        x = jnp.broadcast_to(x[:, None], (x.shape[0], num_timesteps) + x.shape[1:])
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cssm_type', default='gdn')
    parser.add_argument('--delta_key_dim', type=int, default=2)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--num_timesteps', type=int, default=8)
    parser.add_argument('--head_pool', default='mean')
    parser.add_argument('--linear_probe', action='store_true', default=True)
    parser.add_argument('--ssl_proj_dim', type=int, default=64)
    parser.add_argument('--tfrecord_dir', default='/oscar/scratch/dlinsley/imagenet_tfrecords')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--train_fraction', type=float, default=0.1,
                        help='Fraction of ImageNet train to cycle through per epoch.')
    parser.add_argument('--val_batches', type=int, default=1000,
                        help='Number of val batches to evaluate (~50k samples at bs=64).')
    args = parser.parse_args()

    print(f'Checkpoint: {args.checkpoint}')
    model = build_model(args)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 224, 224, 3))
    init_vars = model.init({'params': rng, 'dropout': rng}, dummy, training=True)
    init_bs = init_vars.get('batch_stats', {})

    bb_params = load_checkpoint(args.checkpoint)
    feat_dim = bb_params['head']['kernel'].shape[0]
    print(f'  Backbone params: {sum(p.size for p in jax.tree_util.tree_leaves(bb_params)):,}')
    print(f'  Feature dim: {feat_dim}')

    # Trainable linear head
    probe_w = jnp.zeros((feat_dim, 1000))
    probe_b = jnp.zeros((1000,))
    probe_params = {'kernel': probe_w, 'bias': probe_b}

    # Optimizer
    num_steps_per_epoch = int(1281167 * args.train_fraction / args.batch_size)
    total_steps = num_steps_per_epoch * args.epochs
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=max(total_steps // 20, 50),
        decay_steps=total_steps, end_value=args.lr * 0.01)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.sgd(schedule, momentum=0.9, nesterov=True),
    )
    opt_state = opt.init(probe_params)

    @jax.jit
    def extract_pooled(p, bs, x):
        x = prepare_batch(x, args.num_timesteps)
        _, new_state = model.apply(
            {'params': p, 'batch_stats': bs}, x, training=True,
            rngs={'dropout': jax.random.PRNGKey(0)},
            mutable=['intermediates', 'batch_stats'],
        )
        pooled = new_state['intermediates']
        # Walk to find pooled_features
        node = pooled
        while isinstance(node, dict):
            if 'pooled_features' in node:
                v = node['pooled_features']
                return v[0] if isinstance(v, tuple) else v
            node = next(iter(node.values()))
        raise RuntimeError('pooled_features not sown')

    @jax.jit
    def train_step(probe_params, opt_state, feats, labels):
        def loss_fn(p):
            logits = feats @ p['kernel'] + p['bias']
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, labels).mean()
        loss, grads = jax.value_and_grad(loss_fn)(probe_params)
        updates, opt_state = opt.update(grads, opt_state, probe_params)
        probe_params = optax.apply_updates(probe_params, updates)
        return probe_params, opt_state, loss

    @jax.jit
    def eval_step(probe_params, feats):
        return jnp.argmax(feats @ probe_params['kernel'] + probe_params['bias'], -1)

    loader_kwargs = dict(
        num_parallel_reads=4, prefetch_batches=4, threadpool_size=8,
    )

    def run_val():
        val_loader = TFRecordImageNetLoader(
            tfrecord_dir=args.tfrecord_dir, split='val',
            batch_size=64, image_size=224, drop_remainder=True,
            **loader_kwargs,
        )
        correct = total = 0
        for i, (x, y) in enumerate(val_loader):
            if i >= args.val_batches:
                break
            feats = extract_pooled(bb_params, init_bs, jnp.array(x))
            pred = eval_step(probe_params, feats)
            correct += int((pred == jnp.array(y)).sum())
            total += y.size
        del val_loader
        return correct / total, total

    train_loader = TFRecordImageNetLoader(
        tfrecord_dir=args.tfrecord_dir, split='train',
        batch_size=args.batch_size, image_size=224, drop_remainder=True,
        **loader_kwargs,
    )

    for epoch in range(args.epochs):
        t0 = time.time()
        pbar = tqdm(total=num_steps_per_epoch, desc=f'Epoch {epoch+1}')
        running_loss = 0.
        for step, (x, y) in enumerate(train_loader):
            if step >= num_steps_per_epoch:
                break
            feats = extract_pooled(bb_params, init_bs, jnp.array(x))
            probe_params, opt_state, loss = train_step(
                probe_params, opt_state, feats, jnp.array(y))
            running_loss += float(loss)
            pbar.update(1)
            if (step + 1) % 20 == 0:
                pbar.set_postfix(loss=f'{running_loss/(step+1):.3f}')
        pbar.close()
        val_acc, n = run_val()
        print(f'  Epoch {epoch+1}: val top-1 = {val_acc*100:.2f}% ({n} samples), '
              f'{(time.time()-t0):.0f}s')

    print('\nDone.')


if __name__ == '__main__':
    main()
