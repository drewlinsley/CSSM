"""
Benchmark: SSD chunked scan vs associative_scan for AdditiveCSSM.

Measures wall-clock time and peak GPU memory for:
1. linear_split_3x3_scan_op via associative_scan (current)
2. ssd_scan × 3 cascaded scalar (new SSD path)
3. cssm_3x3_matrix_scan_op via associative_scan (GOOM, slowest)

Sweeps over T={8, 16, 32, 64, 128} with realistic shapes.

Usage:
    CUDA_VISIBLE_DEVICES=7 python benchmarks/bench_ssd.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
import jax.numpy as jnp
import numpy as np

from src.models.math import (
    complex64_to_linear_split, linear_split_to_complex64,
    linear_split_3x3_scan_op,
    ssd_scan,
)


def benchmark_fn(fn, warmup=3, repeats=10):
    """Benchmark a function with warmup and timing."""
    # Warmup
    for _ in range(warmup):
        result = fn()
        jax.block_until_ready(result)

    # Timed runs
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
    }


def generate_3state_data(B, T, C, H, W_freq, key):
    """Generate realistic 3-state AdditiveCSSM data."""
    k1, k2, k3 = jax.random.split(key, 3)

    # Decay gates (0.3 to 0.95 magnitude)
    d_Q_mag = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.3, maxval=0.95)
    d_K_mag = jax.random.uniform(k2, (B, T, C, H, W_freq), minval=0.3, maxval=0.95)
    d_V_mag = jax.random.uniform(k3, (B, T, C, H, W_freq), minval=0.3, maxval=0.95)

    # Coupling gates
    w_kq_val = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.01, maxval=0.3)
    gamma_val = jax.random.uniform(k2, (B, T, C, H, W_freq), minval=0.01, maxval=0.3)

    # Kernel
    K_b = jax.random.uniform(k3, (C, H, W_freq), minval=0.1, maxval=0.8)
    K_b = K_b[None, None].astype(jnp.complex64)
    ones = jnp.ones_like(K_b)

    a_Q = (d_Q_mag * K_b).astype(jnp.complex64)
    a_K = (d_K_mag * K_b).astype(jnp.complex64)
    a_V = (d_V_mag * ones).astype(jnp.complex64)
    w_kq_c = (w_kq_val * K_b).astype(jnp.complex64)
    gamma_c = (gamma_val * ones).astype(jnp.complex64)

    # Inputs
    U_Q = (jax.random.normal(k1, (B, T, C, H, W_freq)) * 0.3 +
           1j * jax.random.normal(jax.random.PRNGKey(100), (B, T, C, H, W_freq)) * 0.3
          ).astype(jnp.complex64)
    U_K = (jax.random.normal(k2, (B, T, C, H, W_freq)) * 0.3 +
           1j * jax.random.normal(jax.random.PRNGKey(101), (B, T, C, H, W_freq)) * 0.3
          ).astype(jnp.complex64)
    U_V = (jax.random.normal(k3, (B, T, C, H, W_freq)) * 0.3 +
           1j * jax.random.normal(jax.random.PRNGKey(102), (B, T, C, H, W_freq)) * 0.3
          ).astype(jnp.complex64)

    return a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V


def run_assoc_scan_3x3(a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V):
    """Run 3×3 matrix associative scan (current production path)."""
    zeros = jnp.zeros_like(a_Q)
    K_mat = jnp.stack([
        jnp.stack([a_Q, zeros, zeros], axis=-1),
        jnp.stack([w_kq_c, a_K, zeros], axis=-1),
        jnp.stack([gamma_c, gamma_c, a_V], axis=-1),
    ], axis=-2)
    U_vec = jnp.stack([U_Q, U_K, U_V], axis=-1)

    K_re, K_im = complex64_to_linear_split(K_mat)
    U_re, U_im = complex64_to_linear_split(U_vec)
    _, _, U_re_out, U_im_out = jax.lax.associative_scan(
        linear_split_3x3_scan_op, (K_re, K_im, U_re, U_im), axis=1)
    QKV = linear_split_to_complex64(U_re_out, U_im_out)
    return QKV[..., 2]


def run_ssd_cascaded(a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V, chunk_size=8):
    """Run cascaded 3-scalar SSD (new path)."""
    Q_all = ssd_scan(a_Q, U_Q, chunk_size=chunk_size)
    Q_prev = jnp.concatenate([jnp.zeros_like(Q_all[:, :1]), Q_all[:, :-1]], axis=1)

    U_K_eff = w_kq_c * Q_prev + U_K
    K_all = ssd_scan(a_K, U_K_eff, chunk_size=chunk_size)
    K_prev = jnp.concatenate([jnp.zeros_like(K_all[:, :1]), K_all[:, :-1]], axis=1)

    U_V_eff = gamma_c * Q_prev + gamma_c * K_prev + U_V
    V = ssd_scan(a_V, U_V_eff, chunk_size=chunk_size)
    return V


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Benchmarking SSD chunked scan vs associative_scan")
    print()

    B, C, H = 8, 32, 32
    W = 33  # rfft2 of W=64: W_freq = 64//2+1 = 33
    # Actually for PathTracker: H=32, W=32, so W_freq = 32//2+1 = 17
    W_freq = 17

    print(f"Shape: B={B}, C={C}, H={H}, W_freq={W_freq}")
    print(f"Positions per timestep: {B * C * H * W_freq:,}")
    print()

    print(f"{'T':>4} | {'Method':<20} | {'Fwd (ms)':>10} | {'Bwd (ms)':>10} | {'Total (ms)':>10}")
    print(f"{'-'*4}-+-{'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for T in [8, 16, 32, 64, 128]:
        key = jax.random.PRNGKey(T)
        data = generate_3state_data(B, T, C, H, W_freq, key)
        a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V = data

        # --- Associative scan (3×3) ---
        def fwd_assoc():
            return run_assoc_scan_3x3(a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V)

        def fwd_bwd_assoc():
            def loss(U_Q, U_K, U_V):
                return jnp.sum(run_assoc_scan_3x3(
                    a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V).real)
            return jax.grad(loss, argnums=(0, 1, 2))(U_Q, U_K, U_V)

        try:
            fwd_time = benchmark_fn(fwd_assoc, warmup=2, repeats=5)
            bwd_time = benchmark_fn(fwd_bwd_assoc, warmup=2, repeats=5)
            total = fwd_time['mean_ms'] + bwd_time['mean_ms']
            print(f"{T:>4} | {'assoc_scan 3x3':<20} | {fwd_time['mean_ms']:>10.1f} | {bwd_time['mean_ms']:>10.1f} | {total:>10.1f}")
        except Exception as e:
            print(f"{T:>4} | {'assoc_scan 3x3':<20} | {'OOM' if 'resource' in str(e).lower() else 'ERR':>10} |")

        # --- SSD cascaded ---
        chunk_size = min(T, 8)

        def fwd_ssd():
            return run_ssd_cascaded(a_Q, a_K, a_V, w_kq_c, gamma_c,
                                    U_Q, U_K, U_V, chunk_size=chunk_size)

        def fwd_bwd_ssd():
            def loss(U_Q, U_K, U_V):
                return jnp.sum(run_ssd_cascaded(
                    a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V,
                    chunk_size=chunk_size).real)
            return jax.grad(loss, argnums=(0, 1, 2))(U_Q, U_K, U_V)

        try:
            fwd_time_ssd = benchmark_fn(fwd_ssd, warmup=2, repeats=5)
            bwd_time_ssd = benchmark_fn(fwd_bwd_ssd, warmup=2, repeats=5)
            total_ssd = fwd_time_ssd['mean_ms'] + bwd_time_ssd['mean_ms']
            print(f"{T:>4} | {'SSD cascaded L=' + str(chunk_size):<20} | {fwd_time_ssd['mean_ms']:>10.1f} | {bwd_time_ssd['mean_ms']:>10.1f} | {total_ssd:>10.1f}")
        except Exception as e:
            print(f"{T:>4} | {'SSD cascaded':<20} | {'OOM' if 'resource' in str(e).lower() else 'ERR':>10} |")

        print()


if __name__ == '__main__':
    main()
