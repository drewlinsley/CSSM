"""
Benchmark: scalar scan primitives — associative vs SSD vs quadratic.

Fair comparison: all three methods solve the SAME scalar recurrence
    h_t = A_t * h_{t-1} + X_t
using different algorithms.

Also benchmarks the full 3-state cascaded decomposition (3× scalar scan)
which is what AdditiveCSSM actually runs for SSD and quadratic modes.

The 3×3 matrix associative scan is shown separately since it's a different
formulation (not a cascaded decomposition), so not directly comparable.

Usage:
    CUDA_VISIBLE_DEVICES=7 python benchmarks/bench_scan_modes.py
    CUDA_VISIBLE_DEVICES=7 python benchmarks/bench_scan_modes.py --seq_lens 8,16 --embed_dim 32
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import jax
import jax.numpy as jnp
import numpy as np

from src.models.math import (
    complex64_to_linear_split, linear_split_to_complex64,
    linear_split_scalar_scan_op,
    linear_split_3x3_scan_op,
    ssd_scan,
    quadratic_scan, chunked_quadratic_scan,
)
from src.models.pallas_scan import pallas_scalar_scan


def benchmark_fn(fn, warmup=5, repeats=20):
    """Benchmark with generous warmup to ensure JIT is fully settled."""
    # JIT warmup — first call compiles, subsequent calls may still be settling
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
        'median_ms': np.median(times) * 1000,
    }


def generate_scalar_data(B, T, C, H, W_freq, key):
    """Generate data for a single scalar SSM: h_t = A_t * h_{t-1} + X_t."""
    k1, k2 = jax.random.split(key, 2)
    A = (jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.3, maxval=0.95)
         ).astype(jnp.complex64)
    X = (jax.random.normal(k2, (B, T, C, H, W_freq)) * 0.3 +
         1j * jax.random.normal(jax.random.PRNGKey(42), (B, T, C, H, W_freq)) * 0.3
         ).astype(jnp.complex64)
    return A, X


def generate_3state_data(B, T, C, H, W_freq, key):
    """Generate realistic 3-state AdditiveCSSM data."""
    k1, k2, k3 = jax.random.split(key, 3)

    d_Q_mag = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.3, maxval=0.95)
    d_K_mag = jax.random.uniform(k2, (B, T, C, H, W_freq), minval=0.3, maxval=0.95)
    d_V_mag = jax.random.uniform(k3, (B, T, C, H, W_freq), minval=0.3, maxval=0.95)

    w_kq_val = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.01, maxval=0.3)
    gamma_val = jax.random.uniform(k2, (B, T, C, H, W_freq), minval=0.01, maxval=0.3)

    K_b = jax.random.uniform(k3, (C, H, W_freq), minval=0.1, maxval=0.8)
    K_b = K_b[None, None].astype(jnp.complex64)
    ones = jnp.ones_like(K_b)

    a_Q = (d_Q_mag * K_b).astype(jnp.complex64)
    a_K = (d_K_mag * K_b).astype(jnp.complex64)
    a_V = (d_V_mag * ones).astype(jnp.complex64)
    w_kq_c = (w_kq_val * K_b).astype(jnp.complex64)
    gamma_c = (gamma_val * ones).astype(jnp.complex64)

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


# =========================================================================
# Scalar scan implementations (fair comparison: same recurrence, same data)
# =========================================================================

def scalar_assoc_scan(A, X):
    """Scalar associative scan via linear-split (bf16 re + bf16 im)."""
    A_re, A_im = complex64_to_linear_split(A)
    X_re, X_im = complex64_to_linear_split(X)
    _, _, X_re_out, X_im_out = jax.lax.associative_scan(
        linear_split_scalar_scan_op, (A_re, A_im, X_re, X_im), axis=1)
    return linear_split_to_complex64(X_re_out, X_im_out)


def scalar_ssd_scan(A, X, chunk_size=8):
    """Scalar SSD chunked scan."""
    return ssd_scan(A, X, chunk_size=chunk_size)


def scalar_quadratic_scan(A, X, chunk_size=8):
    """Scalar quadratic scan (pure matmul for T<=L, chunked otherwise)."""
    T = A.shape[1]
    if T <= chunk_size:
        return quadratic_scan(A, X)
    else:
        return chunked_quadratic_scan(A, X, chunk_size=chunk_size)


def scalar_pallas_scan(A, X):
    """Scalar scan via fused Pallas GPU kernel."""
    return pallas_scalar_scan(A, X)


# =========================================================================
# 3-state cascaded implementations
# =========================================================================

def cascaded_ssd(a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V, chunk_size=8):
    """Cascaded 3-scalar SSD scan."""
    Q_all = ssd_scan(a_Q, U_Q, chunk_size=chunk_size)
    Q_prev = jnp.concatenate([jnp.zeros_like(Q_all[:, :1]), Q_all[:, :-1]], axis=1)

    U_K_eff = w_kq_c * Q_prev + U_K
    K_all = ssd_scan(a_K, U_K_eff, chunk_size=chunk_size)
    K_prev = jnp.concatenate([jnp.zeros_like(K_all[:, :1]), K_all[:, :-1]], axis=1)

    U_V_eff = gamma_c * Q_prev + gamma_c * K_prev + U_V
    return ssd_scan(a_V, U_V_eff, chunk_size=chunk_size)


def cascaded_quadratic(a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V, chunk_size=8):
    """Cascaded 3-scalar quadratic scan."""
    T = a_Q.shape[1]
    if T <= chunk_size:
        _scan = quadratic_scan
    else:
        _scan = lambda A, X: chunked_quadratic_scan(A, X, chunk_size=chunk_size)

    Q_all = _scan(a_Q, U_Q)
    Q_prev = jnp.concatenate([jnp.zeros_like(Q_all[:, :1]), Q_all[:, :-1]], axis=1)

    U_K_eff = w_kq_c * Q_prev + U_K
    K_all = _scan(a_K, U_K_eff)
    K_prev = jnp.concatenate([jnp.zeros_like(K_all[:, :1]), K_all[:, :-1]], axis=1)

    U_V_eff = gamma_c * Q_prev + gamma_c * K_prev + U_V
    return _scan(a_V, U_V_eff)


def cascaded_3x3_assoc(a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V):
    """3×3 matrix associative scan (different formulation, shown for reference)."""
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


def sequential_scalar(A, X):
    """Naive sequential reference for a single scalar SSM."""
    T = A.shape[1]
    h = jnp.zeros_like(A[:, 0])
    ys = []
    for t in range(T):
        h = A[:, t] * h + X[:, t]
        ys.append(h)
    return jnp.stack(ys, axis=1)


def main():
    parser = argparse.ArgumentParser(description='Benchmark scan modes (fair scalar comparison)')
    parser.add_argument('--seq_lens', type=str, default='8,16,32,64',
                        help='Comma-separated sequence lengths to benchmark')
    parser.add_argument('--embed_dim', type=int, default=32, help='Embedding dimension (C)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, default=32, help='Spatial size H (after stem)')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations (for JIT)')
    parser.add_argument('--repeats', type=int, default=20, help='Timed iterations')
    parser.add_argument('--chunk_size', type=int, default=8, help='Chunk size for SSD/quadratic')
    parser.add_argument('--check_correctness', action='store_true',
                        help='Run correctness check vs sequential reference')
    parser.add_argument('--skip_3x3', action='store_true',
                        help='Skip the slow 3×3 matrix associative scan')
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(',')]
    B, C, H = args.batch_size, args.embed_dim, args.image_size
    W_freq = H // 2 + 1
    L = args.chunk_size

    print(f"JAX devices: {jax.devices()}")
    print(f"Shape: B={B}, C={C}, H={H}, W_freq={W_freq}")
    print(f"Elements per timestep: {B * C * H * W_freq:,} complex64")
    print(f"Chunk size: {L}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")
    print()

    # =====================================================================
    # Correctness check
    # =====================================================================
    if args.check_correctness:
        print("=" * 60)
        print("Correctness: scalar scan vs sequential reference")
        print("=" * 60)
        for T in seq_lens:
            key = jax.random.PRNGKey(T)
            A, X = generate_scalar_data(B, T, C, H, W_freq, key)
            Y_ref = sequential_scalar(A, X)
            Y_assoc = scalar_assoc_scan(A, X)
            Y_ssd = scalar_ssd_scan(A, X, chunk_size=L)
            Y_quad = scalar_quadratic_scan(A, X, chunk_size=L)

            try:
                Y_pallas = scalar_pallas_scan(A, X)
                pallas_err = f"pallas={jnp.abs(Y_pallas - Y_ref).max():.2e}"
            except Exception as e:
                pallas_err = f"pallas=ERR({e})"
            print(f"  T={T:>3}: assoc={jnp.abs(Y_assoc - Y_ref).max():.2e}  "
                  f"SSD={jnp.abs(Y_ssd - Y_ref).max():.2e}  "
                  f"quad={jnp.abs(Y_quad - Y_ref).max():.2e}  "
                  f"{pallas_err}")
        print()

    # =====================================================================
    # Part 1: Single scalar scan (fair apples-to-apples)
    # =====================================================================
    print("=" * 70)
    print("SINGLE SCALAR SCAN: h_t = A_t * h_{t-1} + X_t")
    print("  (same recurrence, same data, different algorithms)")
    print("=" * 70)

    header = f"{'T':>4} | {'Method':<28} | {'Fwd med (ms)':>12} | {'Fwd+Bwd med (ms)':>16}"
    print(header)
    print("-" * len(header))

    for T in seq_lens:
        key = jax.random.PRNGKey(T + 1000)
        A, X = generate_scalar_data(B, T, C, H, W_freq, key)

        scalar_methods = [
            ('assoc (linear-split)', lambda: scalar_assoc_scan(A, X)),
            (f'SSD (L={L})', lambda: scalar_ssd_scan(A, X, chunk_size=L)),
            (f'quadratic (L={L})', lambda: scalar_quadratic_scan(A, X, chunk_size=L)),
            ('pallas (fused kernel)', lambda: scalar_pallas_scan(A, X)),
        ]

        for name, fwd_fn in scalar_methods:
            def fwd_bwd_fn(fwd_fn=fwd_fn):
                def loss(X_):
                    return jnp.sum(fwd_fn().real)
                # Differentiate w.r.t. the closed-over X through fwd_fn
                # Better: pass X explicitly
                return None

            # Forward-only benchmark
            fwd_time = benchmark_fn(fwd_fn, warmup=args.warmup, repeats=args.repeats)

            # Forward+backward benchmark
            def make_fwd_bwd(A=A, X=X, name=name):
                if 'assoc' in name:
                    def loss(X_):
                        return jnp.sum(scalar_assoc_scan(A, X_).real)
                elif 'SSD' in name:
                    def loss(X_):
                        return jnp.sum(scalar_ssd_scan(A, X_, chunk_size=L).real)
                elif 'pallas' in name:
                    def loss(X_):
                        return jnp.sum(scalar_pallas_scan(A, X_).real)
                else:
                    def loss(X_):
                        return jnp.sum(scalar_quadratic_scan(A, X_, chunk_size=L).real)
                return lambda: jax.grad(loss)(X)

            bwd_fn = make_fwd_bwd()
            bwd_time = benchmark_fn(bwd_fn, warmup=args.warmup, repeats=args.repeats)

            print(f"{T:>4} | {name:<28} | {fwd_time['median_ms']:>12.2f} | {bwd_time['median_ms']:>16.2f}")

        print()

    # =====================================================================
    # Part 2: Full 3-state cascaded (what AdditiveCSSM actually runs)
    # =====================================================================
    print("=" * 70)
    print("3-STATE CASCADED: Q→K→V triangular decomposition (3× scalar scan)")
    print("  SSD and quadratic use same cascaded structure, 3×3 is different")
    print("=" * 70)

    header = f"{'T':>4} | {'Method':<28} | {'Fwd med (ms)':>12} | {'Fwd+Bwd med (ms)':>16}"
    print(header)
    print("-" * len(header))

    for T in seq_lens:
        key = jax.random.PRNGKey(T)
        data = generate_3state_data(B, T, C, H, W_freq, key)
        a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V = data

        cascaded_methods = []
        if not args.skip_3x3:
            cascaded_methods.append(
                ('3x3 assoc (reference)', lambda: cascaded_3x3_assoc(*data))
            )
        cascaded_methods.extend([
            (f'SSD cascaded (L={L})', lambda: cascaded_ssd(*data, chunk_size=L)),
            (f'quadratic cascaded (L={L})', lambda: cascaded_quadratic(*data, chunk_size=L)),
        ])

        for name, fwd_fn in cascaded_methods:
            fwd_time = benchmark_fn(fwd_fn, warmup=args.warmup, repeats=args.repeats)

            # Forward+backward
            def make_fwd_bwd(name=name):
                if '3x3' in name:
                    def loss(U_Q, U_K, U_V):
                        return jnp.sum(cascaded_3x3_assoc(
                            a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V).real)
                elif 'SSD' in name:
                    def loss(U_Q, U_K, U_V):
                        return jnp.sum(cascaded_ssd(
                            a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V, chunk_size=L).real)
                else:
                    def loss(U_Q, U_K, U_V):
                        return jnp.sum(cascaded_quadratic(
                            a_Q, a_K, a_V, w_kq_c, gamma_c, U_Q, U_K, U_V, chunk_size=L).real)
                return lambda: jax.grad(loss, argnums=(0, 1, 2))(U_Q, U_K, U_V)

            bwd_fn = make_fwd_bwd()
            bwd_time = benchmark_fn(bwd_fn, warmup=args.warmup, repeats=args.repeats)

            print(f"{T:>4} | {name:<28} | {fwd_time['median_ms']:>12.2f} | {bwd_time['median_ms']:>16.2f}")

        print()


if __name__ == '__main__':
    main()
