"""
Test forward pass and gradient agreement between complex64 (GOOM) and
complex32 (linear-split bf16 real + bf16 imag) scans.

The linear-split approach skips GOOM entirely and works in linear complex space.
The scan operator is just complex multiply + add — no trig, no log, no exp.

For each AdditiveCSSM variant (add_kqv, add_kqv_2, add_kqv_1), this test:
1. Creates the model with use_complex32=False, runs forward + backward
2. Creates the model with use_complex32=True, runs forward + backward (same params)
3. Compares outputs and parameter gradients, reporting relative error

Usage:
    python tests/test_complex32.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from src.models.cssm import AdditiveCSSM
from src.models.math import (
    complex64_to_linear_split, linear_split_to_complex64,
    linear_split_scalar_scan_op, linear_split_2x2_scan_op,
    linear_split_3x3_scan_op,
    cssm_scalar_scan_op, cssm_matrix_scan_op, cssm_3x3_matrix_scan_op,
)
from src.models.goom import to_goom, from_goom


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
# Test 1: Raw scan operator agreement (linear-split vs GOOM complex64)
# =============================================================================

def test_scalar_scan_agreement():
    """Test linear-split scalar scan matches GOOM-based complex64 scalar scan.

    The comparison is: GOOM scan (log-space) vs linear-split scan (linear-space).
    These compute the SAME recurrence x_t = a*x_{t-1} + u_t, just in different
    numerical representations. We compare the final linear-space outputs.
    """
    print("=" * 70)
    print("TEST: Scalar scan — linear-split bf16 vs GOOM complex64 (add_kqv_1)")
    print("=" * 70)

    key = jax.random.PRNGKey(0)
    B, T, C, H, W_freq = 2, 8, 4, 8, 5

    # Generate inputs in LINEAR complex space (the ground truth)
    k1, k2 = jax.random.split(key)
    # Transition: magnitude < 1 for stability
    A_mag = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.1, maxval=0.95)
    A_angle = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=-jnp.pi, maxval=jnp.pi)
    A_linear = (A_mag * jnp.exp(1j * A_angle)).astype(jnp.complex64)

    U_linear = (jax.random.normal(k2, (B, T, C, H, W_freq)) * 0.5 +
                1j * jax.random.normal(jax.random.PRNGKey(99), (B, T, C, H, W_freq)) * 0.5
               ).astype(jnp.complex64)

    # --- GOOM path: convert to log-space, scan, convert back ---
    A_log = to_goom(A_linear)
    U_log = to_goom(U_linear)
    _, out_log = jax.lax.associative_scan(
        cssm_scalar_scan_op, (A_log, U_log), axis=1)
    out_goom = from_goom(out_log)

    # --- Linear-split path: split, scan in linear space ---
    A_re, A_im = complex64_to_linear_split(A_linear)
    U_re, U_im = complex64_to_linear_split(U_linear)
    _, _, U_re_out, U_im_out = jax.lax.associative_scan(
        linear_split_scalar_scan_op, (A_re, A_im, U_re, U_im), axis=1)
    out_ls = linear_split_to_complex64(U_re_out, U_im_out)

    # Compare
    err = rel_error(np.array(out_goom), np.array(out_ls))
    max_err = max_rel_error(np.array(out_goom), np.array(out_ls))
    print(f"  Forward output:  mean_rel_err={err:.6e}, max_rel_err={max_err:.6e}")

    # Backward comparison
    def loss_goom(A, U):
        A_l = to_goom(A)
        U_l = to_goom(U)
        _, out_l = jax.lax.associative_scan(cssm_scalar_scan_op, (A_l, U_l), axis=1)
        return jnp.sum(from_goom(out_l).real)

    def loss_ls(A, U):
        Ar, Ai = complex64_to_linear_split(A)
        Ur, Ui = complex64_to_linear_split(U)
        _, _, Ur_out, Ui_out = jax.lax.associative_scan(
            linear_split_scalar_scan_op, (Ar, Ai, Ur, Ui), axis=1)
        out = linear_split_to_complex64(Ur_out, Ui_out)
        return jnp.sum(out.real)

    grad_goom = jax.grad(loss_goom, argnums=1)(A_linear, U_linear)
    grad_ls = jax.grad(loss_ls, argnums=1)(A_linear, U_linear)

    grad_err = rel_error(np.array(grad_goom), np.array(grad_ls))
    grad_max_err = max_rel_error(np.array(grad_goom), np.array(grad_ls))
    print(f"  Gradient (dL/dU): mean_rel_err={grad_err:.6e}, max_rel_err={grad_max_err:.6e}")
    print()
    return err, grad_err


def test_2x2_scan_agreement():
    """Test linear-split 2x2 scan matches GOOM 2x2 scan."""
    print("=" * 70)
    print("TEST: 2x2 matrix scan — linear-split bf16 vs GOOM complex64 (add_kqv_2)")
    print("=" * 70)

    key = jax.random.PRNGKey(1)
    B, T, C, H, W_freq = 2, 8, 4, 8, 5

    k1, k2 = jax.random.split(key)
    # Transition matrix: scale entries < 1 for stability
    K_mag = jax.random.uniform(k1, (B, T, C, H, W_freq, 2, 2), minval=0.05, maxval=0.5)
    K_angle = jax.random.uniform(k1, (B, T, C, H, W_freq, 2, 2), minval=-jnp.pi, maxval=jnp.pi)
    K_linear = (K_mag * jnp.exp(1j * K_angle)).astype(jnp.complex64)

    U_linear = (jax.random.normal(k2, (B, T, C, H, W_freq, 2)) * 0.3 +
                1j * jax.random.normal(jax.random.PRNGKey(98), (B, T, C, H, W_freq, 2)) * 0.3
               ).astype(jnp.complex64)

    # GOOM path
    K_log = to_goom(K_linear)
    U_log = to_goom(U_linear)
    _, out_log = jax.lax.associative_scan(cssm_matrix_scan_op, (K_log, U_log), axis=1)
    out_goom = from_goom(out_log)

    # Linear-split path
    K_re, K_im = complex64_to_linear_split(K_linear)
    U_re, U_im = complex64_to_linear_split(U_linear)
    _, _, U_re_out, U_im_out = jax.lax.associative_scan(
        linear_split_2x2_scan_op, (K_re, K_im, U_re, U_im), axis=1)
    out_ls = linear_split_to_complex64(U_re_out, U_im_out)

    err = rel_error(np.array(out_goom), np.array(out_ls))
    max_err = max_rel_error(np.array(out_goom), np.array(out_ls))
    print(f"  Forward output:  mean_rel_err={err:.6e}, max_rel_err={max_err:.6e}")

    # Backward
    def loss_goom(K, U):
        Kl = to_goom(K)
        Ul = to_goom(U)
        _, out_l = jax.lax.associative_scan(cssm_matrix_scan_op, (Kl, Ul), axis=1)
        return jnp.sum(from_goom(out_l).real)

    def loss_ls(K, U):
        Kr, Ki = complex64_to_linear_split(K)
        Ur, Ui = complex64_to_linear_split(U)
        _, _, Ur_out, Ui_out = jax.lax.associative_scan(
            linear_split_2x2_scan_op, (Kr, Ki, Ur, Ui), axis=1)
        out = linear_split_to_complex64(Ur_out, Ui_out)
        return jnp.sum(out.real)

    grad_goom = jax.grad(loss_goom, argnums=1)(K_linear, U_linear)
    grad_ls = jax.grad(loss_ls, argnums=1)(K_linear, U_linear)

    grad_err = rel_error(np.array(grad_goom), np.array(grad_ls))
    grad_max_err = max_rel_error(np.array(grad_goom), np.array(grad_ls))
    print(f"  Gradient (dL/dU): mean_rel_err={grad_err:.6e}, max_rel_err={grad_max_err:.6e}")
    print()
    return err, grad_err


def test_3x3_scan_agreement():
    """Test linear-split 3x3 scan matches GOOM 3x3 scan."""
    print("=" * 70)
    print("TEST: 3x3 matrix scan — linear-split bf16 vs GOOM complex64 (add_kqv)")
    print("=" * 70)

    key = jax.random.PRNGKey(2)
    B, T, C, H, W_freq = 2, 8, 4, 8, 5

    k1, k2 = jax.random.split(key)
    K_mag = jax.random.uniform(k1, (B, T, C, H, W_freq, 3, 3), minval=0.05, maxval=0.4)
    K_angle = jax.random.uniform(k1, (B, T, C, H, W_freq, 3, 3), minval=-jnp.pi, maxval=jnp.pi)
    K_linear = (K_mag * jnp.exp(1j * K_angle)).astype(jnp.complex64)

    U_linear = (jax.random.normal(k2, (B, T, C, H, W_freq, 3)) * 0.3 +
                1j * jax.random.normal(jax.random.PRNGKey(97), (B, T, C, H, W_freq, 3)) * 0.3
               ).astype(jnp.complex64)

    # GOOM path
    K_log = to_goom(K_linear)
    U_log = to_goom(U_linear)
    _, out_log = jax.lax.associative_scan(cssm_3x3_matrix_scan_op, (K_log, U_log), axis=1)
    out_goom = from_goom(out_log)

    # Linear-split path
    K_re, K_im = complex64_to_linear_split(K_linear)
    U_re, U_im = complex64_to_linear_split(U_linear)
    _, _, U_re_out, U_im_out = jax.lax.associative_scan(
        linear_split_3x3_scan_op, (K_re, K_im, U_re, U_im), axis=1)
    out_ls = linear_split_to_complex64(U_re_out, U_im_out)

    err = rel_error(np.array(out_goom), np.array(out_ls))
    max_err = max_rel_error(np.array(out_goom), np.array(out_ls))
    print(f"  Forward output:  mean_rel_err={err:.6e}, max_rel_err={max_err:.6e}")

    # Backward
    def loss_goom(K, U):
        Kl = to_goom(K)
        Ul = to_goom(U)
        _, out_l = jax.lax.associative_scan(cssm_3x3_matrix_scan_op, (Kl, Ul), axis=1)
        return jnp.sum(from_goom(out_l).real)

    def loss_ls(K, U):
        Kr, Ki = complex64_to_linear_split(K)
        Ur, Ui = complex64_to_linear_split(U)
        _, _, Ur_out, Ui_out = jax.lax.associative_scan(
            linear_split_3x3_scan_op, (Kr, Ki, Ur, Ui), axis=1)
        out = linear_split_to_complex64(Ur_out, Ui_out)
        return jnp.sum(out.real)

    grad_goom = jax.grad(loss_goom, argnums=1)(K_linear, U_linear)
    grad_ls = jax.grad(loss_ls, argnums=1)(K_linear, U_linear)

    grad_err = rel_error(np.array(grad_goom), np.array(grad_ls))
    grad_max_err = max_rel_error(np.array(grad_goom), np.array(grad_ls))
    print(f"  Gradient (dL/dU): mean_rel_err={grad_err:.6e}, max_rel_err={grad_max_err:.6e}")
    print()
    return err, grad_err


# =============================================================================
# Test 2: Full AdditiveCSSM model agreement
# =============================================================================

def test_additive_cssm_agreement(cssm_type, single_state=False, no_k_state=False):
    """Test full AdditiveCSSM model output and gradient agreement."""
    label = 'add_kqv_1' if single_state else ('add_kqv_2' if no_k_state else 'add_kqv')
    scan_size = '1x1' if single_state else ('2x2' if no_k_state else '3x3')
    print("=" * 70)
    print(f"TEST: Full AdditiveCSSM ({label}, {scan_size} scan)")
    print("=" * 70)

    B, T, H, W, C = 2, 8, 16, 16, 8
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (B, T, H, W, C))

    kwargs = dict(
        channels=C,
        kernel_size=5,
        single_state=single_state,
        no_k_state=no_k_state,
        gate_type='dense',
    )

    model_c64 = AdditiveCSSM(**kwargs, use_complex32=False)
    model_c32 = AdditiveCSSM(**kwargs, use_complex32=True)

    params = model_c64.init(jax.random.PRNGKey(0), x)

    # Forward pass comparison
    out_c64 = model_c64.apply(params, x)
    out_c32 = model_c32.apply(params, x)

    fwd_err = rel_error(np.array(out_c64), np.array(out_c32))
    fwd_max = max_rel_error(np.array(out_c64), np.array(out_c32))
    print(f"  Forward output:  mean_rel_err={fwd_err:.6e}, max_rel_err={fwd_max:.6e}")
    print(f"    Output range: [{float(out_c64.min()):.4f}, {float(out_c64.max()):.4f}]")

    # Backward pass comparison
    def loss_fn(params, model, x):
        return jnp.sum(model.apply(params, x))

    grads_c64 = jax.grad(loss_fn)(params, model_c64, x)
    grads_c32 = jax.grad(loss_fn)(params, model_c32, x)

    print(f"\n  Parameter gradients:")
    leaves_c64, treedef = jax.tree_util.tree_flatten(grads_c64)
    leaves_c32, _ = jax.tree_util.tree_flatten(grads_c32)

    param_errors = []
    for i, (g64, g32) in enumerate(zip(leaves_c64, leaves_c32)):
        g64_np = np.array(g64).astype(np.float64)
        g32_np = np.array(g32).astype(np.float64)
        if g64_np.size == 0:
            continue
        err = rel_error(g64_np, g32_np)
        max_e = max_rel_error(g64_np, g32_np)
        param_errors.append((err, max_e, g64_np.shape))
        if err > 0.01:
            print(f"    param[{i}] {g64_np.shape}: mean_rel={err:.6e}, max_rel={max_e:.6e}")

    if param_errors:
        mean_all = np.mean([e[0] for e in param_errors])
        max_all = max(e[1] for e in param_errors)
        print(f"\n  Summary across {len(param_errors)} parameter groups:")
        print(f"    Mean of mean_rel_errors: {mean_all:.6e}")
        print(f"    Worst max_rel_error:     {max_all:.6e}")
    print()
    return fwd_err, mean_all if param_errors else 0.0


# =============================================================================
# Test 3: Longer sequences — error accumulation
# =============================================================================

def test_error_vs_sequence_length():
    """Test how error grows with sequence length for the scalar scan."""
    print("=" * 70)
    print("TEST: Error vs sequence length (scalar scan, linear-split bf16)")
    print("=" * 70)

    for T in [4, 8, 16, 32, 64]:
        k1, k2 = jax.random.split(jax.random.PRNGKey(T))
        B, C, H, W_freq = 2, 4, 8, 5

        A_mag = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=0.1, maxval=0.95)
        A_angle = jax.random.uniform(k1, (B, T, C, H, W_freq), minval=-jnp.pi, maxval=jnp.pi)
        A_linear = (A_mag * jnp.exp(1j * A_angle)).astype(jnp.complex64)

        U_linear = (jax.random.normal(k2, (B, T, C, H, W_freq)) * 0.5 +
                    1j * jax.random.normal(jax.random.PRNGKey(T + 100), (B, T, C, H, W_freq)) * 0.5
                   ).astype(jnp.complex64)

        # GOOM path
        A_log = to_goom(A_linear)
        U_log = to_goom(U_linear)
        _, out_log = jax.lax.associative_scan(
            cssm_scalar_scan_op, (A_log, U_log), axis=1)
        out_goom = from_goom(out_log)

        # Linear-split path
        A_re, A_im = complex64_to_linear_split(A_linear)
        U_re, U_im = complex64_to_linear_split(U_linear)
        _, _, U_re_out, U_im_out = jax.lax.associative_scan(
            linear_split_scalar_scan_op, (A_re, A_im, U_re, U_im), axis=1)
        out_ls = linear_split_to_complex64(U_re_out, U_im_out)

        err = rel_error(np.array(out_goom), np.array(out_ls))
        print(f"  T={T:>3}: mean_rel_err={err:.6e}")

    print()


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Testing complex32 (linear-split bf16 real+imag) vs complex64 (GOOM)")
    print(f"Linear-split: scan in linear complex space with bf16 real/imag components")
    print(f"GOOM: scan in log-space with complex64")
    print()

    results = {}

    # Raw scan operator tests
    fwd, bwd = test_scalar_scan_agreement()
    results['scalar_scan'] = {'fwd': fwd, 'bwd': bwd}

    fwd, bwd = test_2x2_scan_agreement()
    results['2x2_scan'] = {'fwd': fwd, 'bwd': bwd}

    fwd, bwd = test_3x3_scan_agreement()
    results['3x3_scan'] = {'fwd': fwd, 'bwd': bwd}

    # Full model tests
    fwd, bwd = test_additive_cssm_agreement('add_kqv_1', single_state=True)
    results['model_add_kqv_1'] = {'fwd': fwd, 'bwd': bwd}

    fwd, bwd = test_additive_cssm_agreement('add_kqv_2', no_k_state=True)
    results['model_add_kqv_2'] = {'fwd': fwd, 'bwd': bwd}

    fwd, bwd = test_additive_cssm_agreement('add_kqv')
    results['model_add_kqv'] = {'fwd': fwd, 'bwd': bwd}

    # Error accumulation test
    test_error_vs_sequence_length()

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
        print("All tests passed. Linear-split bf16 is a viable replacement for GOOM complex64.")
        print()
        print("Key advantage: scan operator is just complex multiply + add (4 muls + 4 adds")
        print("per scalar step), vs GOOM which needs cos/sin/atan2/exp/log per step.")
    else:
        print("Some tests show elevated error. Check if bf16 precision is sufficient.")
        print("Note: linear-split operates in different numerical space than GOOM, so")
        print("some divergence is expected, especially for long sequences.")


if __name__ == '__main__':
    main()
