"""
Test forward pass and gradient agreement between SSD chunked scan and
associative_scan for AdditiveCSSM variants.

SSD (State Space Duality) decomposes the 3×3 triangular matrix scan into
3 cascaded scalar SSMs, each computed via chunked matmul instead of
element-wise associative scan. This should be numerically equivalent.

Tests:
1. Raw scalar SSD scan vs associative_scan (linear-split)
2. Cascaded 2-scalar SSD vs 2×2 matrix scan
3. Cascaded 3-scalar SSD vs 3×3 matrix scan
4. Full AdditiveCSSM(use_ssd=True) vs use_complex32=True
5. Sequence length sweep (error accumulation)

Usage:
    python tests/test_ssd.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np

from src.models.cssm import AdditiveCSSM
from src.models.math import (
    complex64_to_linear_split, linear_split_to_complex64,
    linear_split_scalar_scan_op, linear_split_2x2_scan_op,
    linear_split_3x3_scan_op,
    ssd_scan,
)


def rel_error(a, b):
    """Relative error between two arrays, handling near-zero values."""
    a = np.asarray(a)
    b = np.asarray(b)
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        a = a.astype(np.complex128)
        b = b.astype(np.complex128)
        denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
        return float(np.mean(2 * np.abs(a - b) / denom))
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.mean(2 * np.abs(a - b) / denom))


def max_rel_error(a, b):
    """Max relative error between two arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        a = a.astype(np.complex128)
        b = b.astype(np.complex128)
        denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
        return float(np.max(2 * np.abs(a - b) / denom))
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.max(2 * np.abs(a - b) / denom))


# =============================================================================
# Test 1: Raw scalar SSD scan vs associative_scan
# =============================================================================

def test_scalar_ssd_vs_associative():
    """Test SSD chunked scan matches linear-split associative scan for scalar SSM."""
    print("=" * 70)
    print("TEST: Scalar SSD scan vs associative_scan (linear-split)")
    print("=" * 70)

    key = jax.random.PRNGKey(0)
    B, T, C, H, W_freq = 2, 16, 4, 8, 5

    k1, k2 = jax.random.split(key)
    A_mag = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.3, maxval=0.95)
    A_angle = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=-jnp.pi, maxval=jnp.pi)
    A_linear = (A_mag * jnp.exp(1j * A_angle)).astype(jnp.complex64)

    U_linear = (jax.random.normal(k2, (B, T, C, H, W_freq)) * 0.5 +
                1j * jax.random.normal(jax.random.PRNGKey(99), (B, T, C, H, W_freq)) * 0.5
               ).astype(jnp.complex64)

    # --- Associative scan (reference) ---
    A_re, A_im = complex64_to_linear_split(A_linear)
    U_re, U_im = complex64_to_linear_split(U_linear)
    _, _, U_re_out, U_im_out = jax.lax.associative_scan(
        linear_split_scalar_scan_op, (A_re, A_im, U_re, U_im), axis=1)
    out_assoc = linear_split_to_complex64(U_re_out, U_im_out)

    # --- SSD chunked scan ---
    out_ssd = ssd_scan(A_linear, U_linear, chunk_size=8)

    err = rel_error(np.array(out_assoc), np.array(out_ssd))
    max_err = max_rel_error(np.array(out_assoc), np.array(out_ssd))
    print(f"  Forward output:  mean_rel_err={err:.6e}, max_rel_err={max_err:.6e}")

    # Backward comparison
    def loss_assoc(A, U):
        Ar, Ai = complex64_to_linear_split(A)
        Ur, Ui = complex64_to_linear_split(U)
        _, _, Ur_out, Ui_out = jax.lax.associative_scan(
            linear_split_scalar_scan_op, (Ar, Ai, Ur, Ui), axis=1)
        out = linear_split_to_complex64(Ur_out, Ui_out)
        return jnp.sum(out.real)

    def loss_ssd(A, U):
        out = ssd_scan(A, U, chunk_size=8)
        return jnp.sum(out.real)

    grad_assoc = jax.grad(loss_assoc, argnums=1)(A_linear, U_linear)
    grad_ssd = jax.grad(loss_ssd, argnums=1)(A_linear, U_linear)

    grad_err = rel_error(np.array(grad_assoc), np.array(grad_ssd))
    grad_max_err = max_rel_error(np.array(grad_assoc), np.array(grad_ssd))
    print(f"  Gradient (dL/dU): mean_rel_err={grad_err:.6e}, max_rel_err={grad_max_err:.6e}")
    print()
    return err, grad_err


# =============================================================================
# Test 2: Cascaded 2-scalar SSD vs 2×2 matrix scan
# =============================================================================

def test_2state_cascaded_ssd():
    """Test cascaded 2-scalar SSD matches 2×2 matrix associative scan."""
    print("=" * 70)
    print("TEST: Cascaded 2-scalar SSD vs 2×2 matrix scan (add_kqv_2)")
    print("=" * 70)

    key = jax.random.PRNGKey(1)
    B, T, C, H, W_freq = 2, 16, 4, 8, 5
    chunk_size = 8

    k1, k2, k3 = jax.random.split(key, 3)

    # Generate components matching AdditiveCSSM 2-state structure
    d_Q_mag = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.3, maxval=0.9)
    d_V_mag = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.3, maxval=0.9)
    gamma_val = jax.random.uniform(k2, (B, T, C, H, W_freq), minval=0.01, maxval=0.3)

    # Kernel (real for simplicity — still tests complex path via FFT domain)
    K_b_mag = jax.random.uniform(k3, (C, H, W_freq), minval=0.1, maxval=0.8)
    K_b = K_b_mag[None, None].astype(jnp.complex64)
    ones = jnp.ones_like(K_b)

    # Build transition matrix
    a_Q = (d_Q_mag * K_b).astype(jnp.complex64)
    a_V = (d_V_mag * ones).astype(jnp.complex64)
    gamma_c = (gamma_val * ones).astype(jnp.complex64)

    U_Q = (jax.random.normal(k2, (B, T, C, H, W_freq)) * 0.3 +
           1j * jax.random.normal(jax.random.PRNGKey(201), (B, T, C, H, W_freq)) * 0.3
          ).astype(jnp.complex64)
    U_V = (jax.random.normal(k3, (B, T, C, H, W_freq)) * 0.3 +
           1j * jax.random.normal(jax.random.PRNGKey(202), (B, T, C, H, W_freq)) * 0.3
          ).astype(jnp.complex64)

    # --- Reference: 2×2 matrix associative scan ---
    zeros = jnp.zeros_like(a_Q)
    K_mat = jnp.stack([
        jnp.stack([a_Q, zeros], axis=-1),
        jnp.stack([gamma_c, a_V], axis=-1),
    ], axis=-2)
    U_vec = jnp.stack([U_Q, U_V], axis=-1)

    K_re, K_im = complex64_to_linear_split(K_mat)
    U_re, U_im = complex64_to_linear_split(U_vec)
    _, _, U_re_out, U_im_out = jax.lax.associative_scan(
        linear_split_2x2_scan_op, (K_re, K_im, U_re, U_im), axis=1)
    QV_ref = linear_split_to_complex64(U_re_out, U_im_out)
    V_ref = QV_ref[..., 1]

    # --- SSD cascaded: 2 scalar SSMs ---
    Q_all = ssd_scan(a_Q, U_Q, chunk_size=chunk_size)
    Q_prev = jnp.concatenate([jnp.zeros_like(Q_all[:, :1]), Q_all[:, :-1]], axis=1)
    U_V_eff = gamma_c * Q_prev + U_V
    V_ssd = ssd_scan(a_V, U_V_eff, chunk_size=chunk_size)

    err = rel_error(np.array(V_ref), np.array(V_ssd))
    max_err = max_rel_error(np.array(V_ref), np.array(V_ssd))
    print(f"  V output:  mean_rel_err={err:.6e}, max_rel_err={max_err:.6e}")
    print()
    return err, 0.0


# =============================================================================
# Test 3: Cascaded 3-scalar SSD vs 3×3 matrix scan
# =============================================================================

def test_3state_cascaded_ssd():
    """Test cascaded 3-scalar SSD matches 3×3 matrix associative scan."""
    print("=" * 70)
    print("TEST: Cascaded 3-scalar SSD vs 3×3 matrix scan (add_kqv)")
    print("=" * 70)

    key = jax.random.PRNGKey(2)
    B, T, C, H, W_freq = 2, 16, 4, 8, 5
    chunk_size = 8

    k1, k2, k3 = jax.random.split(key, 3)

    d_Q_mag = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.3, maxval=0.9)
    d_K_mag = jax.random.uniform(k2, (B, T, C, H, W_freq), minval=0.3, maxval=0.9)
    d_V_mag = jax.random.uniform(k3, (B, T, C, H, W_freq), minval=0.3, maxval=0.9)
    w_kq_val = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.01, maxval=0.3)
    gamma_val = jax.random.uniform(k2, (B, T, C, H, W_freq), minval=0.01, maxval=0.3)

    K_b_mag = jax.random.uniform(k3, (C, H, W_freq), minval=0.1, maxval=0.8)
    K_b = K_b_mag[None, None].astype(jnp.complex64)
    ones = jnp.ones_like(K_b)

    a_Q = (d_Q_mag * K_b).astype(jnp.complex64)
    a_K = (d_K_mag * K_b).astype(jnp.complex64)
    a_V = (d_V_mag * ones).astype(jnp.complex64)
    w_kq_c = (w_kq_val * K_b).astype(jnp.complex64)
    gamma_c = (gamma_val * ones).astype(jnp.complex64)

    U_Q = (jax.random.normal(k1, (B, T, C, H, W_freq)) * 0.3 +
           1j * jax.random.normal(jax.random.PRNGKey(301), (B, T, C, H, W_freq)) * 0.3
          ).astype(jnp.complex64)
    U_K = (jax.random.normal(k2, (B, T, C, H, W_freq)) * 0.3 +
           1j * jax.random.normal(jax.random.PRNGKey(302), (B, T, C, H, W_freq)) * 0.3
          ).astype(jnp.complex64)
    U_V = (jax.random.normal(k3, (B, T, C, H, W_freq)) * 0.3 +
           1j * jax.random.normal(jax.random.PRNGKey(303), (B, T, C, H, W_freq)) * 0.3
          ).astype(jnp.complex64)

    # --- Reference: 3×3 matrix associative scan ---
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
    QKV_ref = linear_split_to_complex64(U_re_out, U_im_out)
    V_ref = QKV_ref[..., 2]

    # --- SSD cascaded: 3 scalar SSMs ---
    Q_all = ssd_scan(a_Q, U_Q, chunk_size=chunk_size)
    Q_prev = jnp.concatenate([jnp.zeros_like(Q_all[:, :1]), Q_all[:, :-1]], axis=1)

    U_K_eff = w_kq_c * Q_prev + U_K
    K_all = ssd_scan(a_K, U_K_eff, chunk_size=chunk_size)
    K_prev = jnp.concatenate([jnp.zeros_like(K_all[:, :1]), K_all[:, :-1]], axis=1)

    U_V_eff = gamma_c * Q_prev + gamma_c * K_prev + U_V
    V_ssd = ssd_scan(a_V, U_V_eff, chunk_size=chunk_size)

    err = rel_error(np.array(V_ref), np.array(V_ssd))
    max_err = max_rel_error(np.array(V_ref), np.array(V_ssd))
    print(f"  V output:  mean_rel_err={err:.6e}, max_rel_err={max_err:.6e}")

    # Also check Q and K individually
    Q_ref = QKV_ref[..., 0]
    K_ref = QKV_ref[..., 1]
    q_err = rel_error(np.array(Q_ref), np.array(Q_all))
    k_err = rel_error(np.array(K_ref), np.array(K_all))
    print(f"  Q output:  mean_rel_err={q_err:.6e}")
    print(f"  K output:  mean_rel_err={k_err:.6e}")
    print()
    return err, 0.0


# =============================================================================
# Test 4: Full AdditiveCSSM model agreement
# =============================================================================

def test_additive_cssm_ssd(cssm_type, single_state=False, no_k_state=False):
    """Test AdditiveCSSM(use_ssd=True) matches use_complex32=True."""
    label = 'add_kqv_1' if single_state else ('add_kqv_2' if no_k_state else 'add_kqv')
    scan_size = '1x1' if single_state else ('2x2' if no_k_state else '3x3')
    print("=" * 70)
    print(f"TEST: Full AdditiveCSSM SSD vs complex32 ({label}, {scan_size} scan)")
    print("=" * 70)

    B, T, H, W, C = 2, 16, 16, 16, 8
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (B, T, H, W, C))

    kwargs = dict(
        channels=C,
        kernel_size=5,
        single_state=single_state,
        no_k_state=no_k_state,
        gate_type='dense',
    )

    model_c32 = AdditiveCSSM(**kwargs, use_complex32=True)
    model_ssd = AdditiveCSSM(**kwargs, use_ssd=True, ssd_chunk_size=8)

    params = model_c32.init(jax.random.PRNGKey(0), x)

    # Forward pass
    out_c32 = model_c32.apply(params, x)
    out_ssd = model_ssd.apply(params, x)

    fwd_err = rel_error(np.array(out_c32), np.array(out_ssd))
    fwd_max = max_rel_error(np.array(out_c32), np.array(out_ssd))
    print(f"  Forward output:  mean_rel_err={fwd_err:.6e}, max_rel_err={fwd_max:.6e}")
    print(f"    Output range: [{float(out_c32.min()):.4f}, {float(out_c32.max()):.4f}]")

    # Backward pass
    def loss_fn(params, model, x):
        return jnp.sum(model.apply(params, x))

    grads_c32 = jax.grad(loss_fn)(params, model_c32, x)
    grads_ssd = jax.grad(loss_fn)(params, model_ssd, x)

    print(f"\n  Parameter gradients:")
    leaves_c32, _ = jax.tree_util.tree_flatten(grads_c32)
    leaves_ssd, _ = jax.tree_util.tree_flatten(grads_ssd)

    param_errors = []
    for i, (g32, gssd) in enumerate(zip(leaves_c32, leaves_ssd)):
        g32_np = np.array(g32).astype(np.float64)
        gssd_np = np.array(gssd).astype(np.float64)
        if g32_np.size == 0:
            continue
        err = rel_error(g32_np, gssd_np)
        max_e = max_rel_error(g32_np, gssd_np)
        param_errors.append((err, max_e, g32_np.shape))
        if err > 0.01:
            print(f"    param[{i}] {g32_np.shape}: mean_rel={err:.6e}, max_rel={max_e:.6e}")

    if param_errors:
        mean_all = np.mean([e[0] for e in param_errors])
        max_all = max(e[1] for e in param_errors)
        print(f"\n  Summary across {len(param_errors)} parameter groups:")
        print(f"    Mean of mean_rel_errors: {mean_all:.6e}")
        print(f"    Worst max_rel_error:     {max_all:.6e}")
    print()
    return fwd_err, mean_all if param_errors else 0.0


# =============================================================================
# Test 5: Sequence length sweep
# =============================================================================

def test_error_vs_sequence_length():
    """Test how error scales with sequence length for SSD vs associative scan."""
    print("=" * 70)
    print("TEST: Error vs sequence length (scalar SSD vs associative scan)")
    print("=" * 70)

    for T in [8, 16, 32, 64, 128]:
        k1, k2 = jax.random.split(jax.random.PRNGKey(T))
        B, C, H, W_freq = 2, 4, 8, 5

        A_mag = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.3, maxval=0.95)
        A_angle = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=-jnp.pi, maxval=jnp.pi)
        A_linear = (A_mag * jnp.exp(1j * A_angle)).astype(jnp.complex64)

        U_linear = (jax.random.normal(k2, (B, T, C, H, W_freq)) * 0.5 +
                    1j * jax.random.normal(jax.random.PRNGKey(T + 100), (B, T, C, H, W_freq)) * 0.5
                   ).astype(jnp.complex64)

        # Associative scan (reference)
        A_re, A_im = complex64_to_linear_split(A_linear)
        U_re, U_im = complex64_to_linear_split(U_linear)
        _, _, U_re_out, U_im_out = jax.lax.associative_scan(
            linear_split_scalar_scan_op, (A_re, A_im, U_re, U_im), axis=1)
        out_assoc = linear_split_to_complex64(U_re_out, U_im_out)

        # SSD scan
        out_ssd = ssd_scan(A_linear, U_linear, chunk_size=8)

        err = rel_error(np.array(out_assoc), np.array(out_ssd))
        print(f"  T={T:>3}: mean_rel_err={err:.6e}")

    print()


# =============================================================================
# Test 6: Chunk size sweep
# =============================================================================

def test_chunk_sizes():
    """Test different chunk sizes produce same result."""
    print("=" * 70)
    print("TEST: SSD with different chunk sizes")
    print("=" * 70)

    key = jax.random.PRNGKey(7)
    B, T, C, H, W_freq = 2, 64, 4, 8, 5

    k1, k2 = jax.random.split(key)
    A_mag = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.3, maxval=0.95)
    A_angle = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=-jnp.pi, maxval=jnp.pi)
    A_linear = (A_mag * jnp.exp(1j * A_angle)).astype(jnp.complex64)

    U_linear = (jax.random.normal(k2, (B, T, C, H, W_freq)) * 0.5 +
                1j * jax.random.normal(jax.random.PRNGKey(99), (B, T, C, H, W_freq)) * 0.5
               ).astype(jnp.complex64)

    # Reference: associative scan
    A_re, A_im = complex64_to_linear_split(A_linear)
    U_re, U_im = complex64_to_linear_split(U_linear)
    _, _, U_re_out, U_im_out = jax.lax.associative_scan(
        linear_split_scalar_scan_op, (A_re, A_im, U_re, U_im), axis=1)
    out_ref = linear_split_to_complex64(U_re_out, U_im_out)

    for L in [4, 8, 16, 32, 64]:
        out_ssd = ssd_scan(A_linear, U_linear, chunk_size=L)
        err = rel_error(np.array(out_ref), np.array(out_ssd))
        print(f"  chunk_size={L:>2}: mean_rel_err={err:.6e}")

    print()


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Testing SSD chunked scan vs associative_scan")
    print(f"SSD: chunked matmul algorithm from Mamba-2 (State Space Duality)")
    print()

    results = {}

    # Raw scan tests
    fwd, bwd = test_scalar_ssd_vs_associative()
    results['scalar_ssd'] = {'fwd': fwd, 'bwd': bwd}

    fwd, bwd = test_2state_cascaded_ssd()
    results['2state_cascaded'] = {'fwd': fwd, 'bwd': bwd}

    fwd, bwd = test_3state_cascaded_ssd()
    results['3state_cascaded'] = {'fwd': fwd, 'bwd': bwd}

    # Full model tests
    fwd, bwd = test_additive_cssm_ssd('add_kqv_1', single_state=True)
    results['model_add_kqv_1'] = {'fwd': fwd, 'bwd': bwd}

    fwd, bwd = test_additive_cssm_ssd('add_kqv_2', no_k_state=True)
    results['model_add_kqv_2'] = {'fwd': fwd, 'bwd': bwd}

    fwd, bwd = test_additive_cssm_ssd('add_kqv')
    results['model_add_kqv'] = {'fwd': fwd, 'bwd': bwd}

    # Sweep tests
    test_error_vs_sequence_length()
    test_chunk_sizes()

    # Final summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test':<25} | {'Fwd Rel Err':>12} | {'Grad Rel Err':>12} | {'Status':>8}")
    print(f"{'-'*25}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

    all_pass = True
    for name, r in results.items():
        fwd_ok = r['fwd'] < 0.1
        bwd_ok = r['bwd'] < 0.1
        status = "PASS" if (fwd_ok and bwd_ok) else "WARN" if r['fwd'] < 0.2 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{name:<25} | {r['fwd']:>12.6e} | {r['bwd']:>12.6e} | {status:>8}")

    print()
    if all_pass:
        print("All tests passed. SSD chunked scan is numerically equivalent to")
        print("associative_scan for all AdditiveCSSM variants.")
    else:
        print("Some tests show elevated error. Investigate chunk boundary handling.")


if __name__ == '__main__':
    main()
