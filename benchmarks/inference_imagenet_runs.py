"""
Inference-only timing for the seven 300-ep ImageNet runs.

Forward-only, jit'd, on a single GPU. bs=64, image_size=224, fp32 params.

Usage:
    python benchmarks/inference_imagenet_runs.py [--batch_size 64] [--n_iter 50]
"""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import jax, jax.numpy as jnp
from src.models.cssm_shvit import cssm_shvit_s1
from src.models.shvit import shvit_s1


CONFIGS = [
    # (label, builder)
    ('shvit_s1_baseline', lambda: shvit_s1(num_classes=1000)),
    ('cssm_shvit_s1_gdn_dk2_k1x1 (T1,k1,sp)', lambda: cssm_shvit_s1(
        num_classes=1000, num_timesteps=1, cssm_type='gdn', delta_key_dim=2,
        kernel_sizes=(1,1,1,1), output_norm='rms', gate_type='factored',
        short_conv_spatial_size=3, short_conv_size=4,
        block_norm='global_layer', rope_mode='spatiotemporal',
        use_pos_conv=True, gate_proj_bias_init=0.0, head_pool='mean')),
    ('cssm_gdn_T1_k3_v5cfg (T1,k3,sp)', lambda: cssm_shvit_s1(
        num_classes=1000, num_timesteps=1, cssm_type='gdn', delta_key_dim=2,
        kernel_sizes=(3,3,3,3), output_norm='rms', gate_type='factored',
        short_conv_spatial_size=3, short_conv_size=4,
        block_norm='global_layer', rope_mode='spatiotemporal',
        use_pos_conv=True, gate_proj_bias_init=0.0, head_pool='mean')),
    ('cssm_gdn_T1_k1_v7 (T1,k1,sp)', lambda: cssm_shvit_s1(
        num_classes=1000, num_timesteps=1, cssm_type='gdn', delta_key_dim=2,
        kernel_sizes=(1,1,1,1), output_norm='rms', gate_type='factored',
        short_conv_spatial_size=3, short_conv_size=4,
        block_norm='global_layer', rope_mode='spatiotemporal',
        use_pos_conv=True, gate_proj_bias_init=0.0, head_pool='mean')),
    ('cssm_gdn_T4_k3_v5cfg (T4,k3,sp)', lambda: cssm_shvit_s1(
        num_classes=1000, num_timesteps=4, cssm_type='gdn', delta_key_dim=2,
        kernel_sizes=(3,3,3,3), output_norm='rms', gate_type='factored',
        short_conv_spatial_size=3, short_conv_size=4,
        block_norm='global_layer', rope_mode='spatiotemporal',
        use_pos_conv=True, gate_proj_bias_init=0.0, head_pool='mean')),
    ('cssm_gdn_T8_k3_v5cfg (T8,k3,sp)', lambda: cssm_shvit_s1(
        num_classes=1000, num_timesteps=8, cssm_type='gdn', delta_key_dim=2,
        kernel_sizes=(3,3,3,3), output_norm='rms', gate_type='factored',
        short_conv_spatial_size=3, short_conv_size=4,
        block_norm='global_layer', rope_mode='spatiotemporal',
        use_pos_conv=True, gate_proj_bias_init=0.0, head_pool='mean')),
    ('cssm_gdn_T4_k3_learned_t (T4,k3,learned_t)', lambda: cssm_shvit_s1(
        num_classes=1000, num_timesteps=4, cssm_type='gdn', delta_key_dim=2,
        kernel_sizes=(3,3,3,3), output_norm='rms', gate_type='factored',
        short_conv_spatial_size=3, short_conv_size=4,
        block_norm='global_layer', rope_mode='learned_t',
        use_pos_conv=True, gate_proj_bias_init=0.0, head_pool='mean')),
    ('cssm_gdn_T8_k3_learned_t (T8,k3,learned_t)', lambda: cssm_shvit_s1(
        num_classes=1000, num_timesteps=8, cssm_type='gdn', delta_key_dim=2,
        kernel_sizes=(3,3,3,3), output_norm='rms', gate_type='factored',
        short_conv_spatial_size=3, short_conv_size=4,
        block_norm='global_layer', rope_mode='learned_t',
        use_pos_conv=True, gate_proj_bias_init=0.0, head_pool='mean')),
]


def time_apply(fn, args, n_warmup=5, n_iter=50):
    for _ in range(n_warmup):
        out = fn(*args); jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args); jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n_iter * 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--n_iter', type=int, default=50)
    args = p.parse_args()

    print(f'Inference timing — bs={args.batch_size}, image={args.image_size}, fp32, '
          f'device={jax.devices()[0]}')
    print(f'{"model":<48} {"params":>10} {"fwd ms":>9} {"img/s":>9}')
    print('-' * 82)

    for label, build in CONFIGS:
        try:
            model = build()
            rng = jax.random.PRNGKey(0)
            x = jax.random.normal(rng, (args.batch_size, args.image_size, args.image_size, 3))
            vars_ = model.init(rng, x, training=False)
            params = vars_['params']
            bstats = vars_.get('batch_stats', {})
            n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

            def _apply(p, b, x):
                av = {'params': p}
                if b: av['batch_stats'] = b
                return model.apply(av, x, training=False)
            fwd = jax.jit(_apply)
            fwd_ms = time_apply(fwd, (params, bstats, x), n_iter=args.n_iter)
            ips = args.batch_size / (fwd_ms / 1000.0)
            print(f'{label:<48} {n_params/1e6:>9.2f}M {fwd_ms:>8.2f}  {ips:>8.0f}')
        except Exception as e:
            print(f'{label:<48} FAILED: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
