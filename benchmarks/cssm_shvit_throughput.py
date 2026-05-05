"""
End-to-end throughput benchmark for CSSM-SHViT with and without
`static_image_fast_path`.

Measures ms/step for forward-only and forward+backward for T in {1, 2, 4, 8}
at a fixed batch size, on a single GPU.

Usage:
    python benchmarks/cssm_shvit_throughput.py
    python benchmarks/cssm_shvit_throughput.py --batch_size 64 --image_size 224
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

from src.models.cssm_shvit import cssm_shvit_s1


def make_model(num_timesteps: int, fast_path: bool, dtype=jnp.float32):
    return cssm_shvit_s1(
        num_classes=1000,
        num_timesteps=num_timesteps,
        cssm_type='gdn',
        delta_key_dim=2,
        gate_type='factored',
        kernel_sizes=(3, 3, 3, 3),
        short_conv_size=0,
        short_conv_spatial_size=0,
        rope_mode='none',
        block_norm='global_layer',
        use_input_gates=False,
        static_image_fast_path=fast_path,
        dtype=dtype,
        param_dtype=jnp.float32,
    )


def time_step(fn, args, n_warmup=3, n_iter=20):
    for _ in range(n_warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    start = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args)
        jax.block_until_ready(out)
    end = time.perf_counter()
    return (end - start) / n_iter * 1000.0  # ms/step


def benchmark_one(batch_size: int, image_size: int, T: int, fast_path: bool,
                  seed: int = 0, dtype=jnp.float32):
    model = make_model(num_timesteps=T, fast_path=fast_path, dtype=dtype)
    rng = jax.random.PRNGKey(seed)
    init_rng, data_rng = jax.random.split(rng)
    x = jax.random.normal(data_rng, (batch_size, image_size, image_size, 3), dtype=jnp.float32)

    vars_ = model.init(init_rng, x, training=False)
    params = vars_['params']
    bstats = vars_.get('batch_stats', {})

    def _apply(params, bstats, x):
        apply_vars = {'params': params}
        if bstats:
            apply_vars['batch_stats'] = bstats
        return model.apply(apply_vars, x, training=False)

    fwd_jit = jax.jit(_apply)

    def _loss(params, bstats, x):
        out = _apply(params, bstats, x)
        return jnp.mean(out ** 2)

    grad_fn = jax.jit(jax.grad(_loss))

    fwd_ms = time_step(fwd_jit, (params, bstats, x))
    bwd_ms = time_step(grad_fn, (params, bstats, x))
    return fwd_ms, bwd_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--timesteps', type=str, default='1,2,4,8')
    parser.add_argument('--precisions', type=str, default='fp32,bf16',
                        help='Comma-separated: fp32, bf16')
    args = parser.parse_args()

    dtypes = []
    for p in args.precisions.split(','):
        p = p.strip().lower()
        if p == 'fp32':
            dtypes.append(('fp32', jnp.float32))
        elif p == 'bf16':
            dtypes.append(('bf16', jnp.bfloat16))

    timesteps = [int(t) for t in args.timesteps.split(',')]
    print(f"CSSM-SHViT-S1 throughput: batch={args.batch_size}, image={args.image_size}")
    print(f"Num devices: {jax.device_count()}  Device[0]: {jax.devices()[0]}")
    print()
    print(f"{'T':>3} {'prec':>6} {'mode':>8} {'fwd ms':>10} {'fwd+bwd ms':>14} "
          f"{'fwd img/s':>12} {'fwd+bwd img/s':>16}")
    print("-" * 80)

    results = {}
    for T in timesteps:
        for prec_name, dt in dtypes:
            for fast, label in [(False, 'slow'), (True, 'fast')]:
                try:
                    fwd_ms, bwd_ms = benchmark_one(
                        args.batch_size, args.image_size, T, fast, dtype=dt
                    )
                    results[(T, prec_name, label)] = (fwd_ms, bwd_ms)
                    fwd_ips = args.batch_size / (fwd_ms / 1000.0)
                    bwd_ips = args.batch_size / (bwd_ms / 1000.0)
                    print(f"{T:>3} {prec_name:>6} {label:>8} {fwd_ms:>10.2f} {bwd_ms:>14.2f} "
                          f"{fwd_ips:>12.1f} {bwd_ips:>16.1f}")
                except Exception as e:
                    print(f"{T:>3} {prec_name:>6} {label:>8} FAILED: {type(e).__name__}: {e}")
                    results[(T, prec_name, label)] = None

    print()
    print("Ratios (normalized to fp32 slow):")
    print(f"{'T':>3} {'variant':>15} {'fwd':>10} {'fwd+bwd':>12}")
    print("-" * 50)
    for T in timesteps:
        baseline = results.get((T, 'fp32', 'slow'))
        if baseline is None:
            continue
        for prec_name, _ in dtypes:
            for fast, label in [(False, 'slow'), (True, 'fast')]:
                res = results.get((T, prec_name, label))
                if res is None:
                    continue
                fwd_ratio = res[0] / baseline[0]
                bwd_ratio = res[1] / baseline[1]
                print(f"{T:>3} {prec_name + '_' + label:>15} "
                      f"{fwd_ratio:>10.3f}x {bwd_ratio:>11.3f}x")


if __name__ == '__main__':
    main()
