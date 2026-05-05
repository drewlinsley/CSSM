"""
CSSM Speed Benchmarks: Sequential vs. Parallel Scan, across scan sizes and dtypes.

Measures wall-clock time, XLA compilation time, memory, and operation counts
for the temporal recurrence portion of a CSSM layer.

Scan types:
  - 3x3 matrix scan (add_kqv):  K=(B,T,C,H,W_freq,3,3), U=(B,T,C,H,W_freq,3)
  - Scalar scan (add_kqv_1):    K=(B,T,C,H,W_freq),      U=(B,T,C,H,W_freq)

Dtype representations:
  - complex64 GOOM:    float32 mag + float32 phase as native complex64 (8 bytes)
  - "complex32" split: bfloat16 mag + bfloat16 phase as separate arrays (4 bytes)
  - "complex16" split: float8 mag + float8 phase as separate arrays (2 bytes)

Usage:
    # Default: 3x3 matrix scan with complex64
    python benchmarks/cssm_speed.py --seq_lens 8,16,32,64

    # Scalar scan (add_kqv_1) — much smaller per-step tensor
    python benchmarks/cssm_speed.py --scalar --seq_lens 8,16,32,64,128,256

    # Custom dtype comparison (complex64 vs complex32 vs complex16)
    python benchmarks/cssm_speed.py --phase_split --seq_lens 8,16,32,64

    # Scalar + custom dtype (the sweet spot for parallel scan)
    python benchmarks/cssm_speed.py --scalar --phase_split --seq_lens 8,16,32,64,128,256
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# Add project root to path so we can import CSSM modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.math import cssm_3x3_matrix_scan_op, cssm_scalar_scan_op
from src.models.goom import to_goom


# =============================================================================
# Three Recurrence Modes
# =============================================================================

def parallel_scan(K_log, U_log):
    """Mode A: Current implementation — jax.lax.associative_scan (O(log T) depth)."""
    _, states = jax.lax.associative_scan(
        cssm_3x3_matrix_scan_op, (K_log, U_log), axis=1
    )
    return states


def sequential_scan(K_log, U_log):
    """Mode B: Sequential scan — jax.lax.scan with the same linear operator (O(T) depth).

    This applies the exact same associative operator step-by-step, equivalent to
    a linear recurrence computed sequentially. The operator is identical to the
    parallel version — only the execution order differs.
    """
    # Transpose time axis to be the leading (scan) axis
    # K_log: (B, T, C, H, W_freq, 3, 3) -> (T, B, C, H, W_freq, 3, 3)
    K_t = jnp.moveaxis(K_log, 1, 0)
    U_t = jnp.moveaxis(U_log, 1, 0)

    def step(carry, x):
        new_carry = cssm_3x3_matrix_scan_op(carry, x)
        return new_carry, new_carry[1]  # output the u state

    init = (K_t[0], U_t[0])
    _, u_states = jax.lax.scan(step, init, (K_t[1:], U_t[1:]))

    # Prepend the initial state and transpose back
    # u_states: (T-1, B, C, H, W_freq, 3) -> need (B, T, C, H, W_freq, 3)
    u_all = jnp.concatenate([U_t[0:1], u_states], axis=0)
    return jnp.moveaxis(u_all, 0, 1)


def sequential_nonlinear_scan(K_log, U_log):
    """Mode C: Sequential nonlinear RNN — jax.lax.scan with pointwise tanh (O(T) depth).

    Applies the same scan operator but adds a pointwise nonlinearity (tanh on real
    part of the state vector) between steps. This breaks associativity, representing
    a standard nonlinear RNN that CANNOT be parallelized.
    """
    K_t = jnp.moveaxis(K_log, 1, 0)
    U_t = jnp.moveaxis(U_log, 1, 0)

    def step(carry, x):
        new_carry = cssm_3x3_matrix_scan_op(carry, x)
        K_new, u_new = new_carry
        # Pointwise nonlinearity on the state vector — breaks associativity
        u_nonlinear = jnp.tanh(u_new.real).astype(jnp.complex64) + 1j * u_new.imag
        return (K_new, u_nonlinear), u_new

    init = (K_t[0], U_t[0])
    _, u_states = jax.lax.scan(step, init, (K_t[1:], U_t[1:]))

    u_all = jnp.concatenate([U_t[0:1], u_states], axis=0)
    return jnp.moveaxis(u_all, 0, 1)


# =============================================================================
# Scalar scan (add_kqv_1): per-step tensor is (B,C,H,W_freq) — 9x smaller
# =============================================================================

def parallel_scan_scalar(K_log, U_log):
    """Parallel scalar scan — jax.lax.associative_scan with cssm_scalar_scan_op."""
    _, states = jax.lax.associative_scan(
        cssm_scalar_scan_op, (K_log, U_log), axis=1
    )
    return states


def sequential_scan_scalar(K_log, U_log):
    """Sequential scalar scan — jax.lax.scan with cssm_scalar_scan_op."""
    K_t = jnp.moveaxis(K_log, 1, 0)
    U_t = jnp.moveaxis(U_log, 1, 0)

    def step(carry, x):
        new_carry = cssm_scalar_scan_op(carry, x)
        return new_carry, new_carry[1]

    init = (K_t[0], U_t[0])
    _, u_states = jax.lax.scan(step, init, (K_t[1:], U_t[1:]))

    u_all = jnp.concatenate([U_t[0:1], u_states], axis=0)
    return jnp.moveaxis(u_all, 0, 1)


def make_scalar_inputs(batch, seq_len, channels, h, w_freq, dtype=jnp.complex64):
    """Generate random scalar scan inputs in GOOM representation.

    K_log: (B, T, C, H, W_freq) — scalar transition in log-space
    U_log: (B, T, C, H, W_freq) — input in log-space
    """
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    if dtype == jnp.complex64:
        K_real = jax.random.normal(k1, (batch, seq_len, channels, h, w_freq)) * 0.1
        K_phase = jax.random.bernoulli(k1, 0.5, (batch, seq_len, channels, h, w_freq)).astype(jnp.float32) * jnp.pi
        K_log = K_real + 1j * K_phase

        U_real = jax.random.normal(k2, (batch, seq_len, channels, h, w_freq)) * 0.1
        U_phase = jax.random.bernoulli(k2, 0.5, (batch, seq_len, channels, h, w_freq)).astype(jnp.float32) * jnp.pi
        U_log = U_real + 1j * U_phase
    else:
        K_log = jax.random.normal(k1, (batch, seq_len, channels, h, w_freq)) * 0.1
        U_log = jax.random.normal(k2, (batch, seq_len, channels, h, w_freq)) * 0.1

    return K_log, U_log


# =============================================================================
# Phase-split representation: (magnitude, phase) as separate arrays
# =============================================================================
# "Custom complex32": bfloat16 mag + bfloat16 phase = 4 bytes (vs 8 for complex64)
# "Custom complex16": float8 mag + float8 phase = 2 bytes
#
# Unlike the boolean-sign split (which only works for real-valued GOOM where
# phase is 0 or pi), this stores the FULL phase as a float. Compatible with
# FFT spectral data where phases are arbitrary angles in (-pi, pi].
#
# log_add_exp requires cos/sin to convert back to linear complex space,
# but log_mul is just (mag_add, phase_add) — still very cheap.

# Map string names to JAX dtypes
PHASE_SPLIT_DTYPES = {
    'complex32': jnp.bfloat16,   # 2 bytes mag + 2 bytes phase = 4 bytes total
    'complex16': None,            # Will try float8 at runtime (1+1=2 bytes)
}


def _get_float8_dtype():
    """Get float8 dtype if available."""
    try:
        return jnp.float8_e4m3fn
    except AttributeError:
        try:
            import ml_dtypes
            return ml_dtypes.float8_e4m3fn
        except (ImportError, AttributeError):
            return None


def _phase_log_add_exp(mag_a, phase_a, mag_b, phase_b):
    """log(exp(a) + exp(b)) in (magnitude, phase) split representation.

    Converts to linear complex space, sums, converts back.
    Uses max-subtraction trick for numerical stability.

    All intermediate computation in float32 for stability, regardless of
    input dtype. Only the outputs are cast back to the input dtype.
    """
    out_dtype = mag_a.dtype

    # Promote to float32 for intermediate math
    mag_a = mag_a.astype(jnp.float32)
    mag_b = mag_b.astype(jnp.float32)
    phase_a = phase_a.astype(jnp.float32)
    phase_b = phase_b.astype(jnp.float32)

    c = jax.lax.stop_gradient(jnp.maximum(mag_a, mag_b))
    ea = jnp.exp(mag_a - c)
    eb = jnp.exp(mag_b - c)

    # Convert to linear complex: exp(mag) * (cos(phase) + i*sin(phase))
    re = ea * jnp.cos(phase_a) + eb * jnp.cos(phase_b)
    im = ea * jnp.sin(phase_a) + eb * jnp.sin(phase_b)

    # Convert back to (mag, phase)
    new_mag = c + 0.5 * jnp.log(jnp.maximum(re * re + im * im, 1e-30))
    new_phase = jnp.arctan2(im, re)

    return new_mag.astype(out_dtype), new_phase.astype(out_dtype)


def _phase_log_mul(mag_a, phase_a, mag_b, phase_b):
    """log(a * b) in (magnitude, phase) representation.
    Magnitudes add, phases add. Very cheap — no trig needed."""
    return mag_a + mag_b, phase_a + phase_b


# --- Scalar scan in phase-split form ---

def _phase_scalar_scan_op(carry_i, carry_j):
    """Scalar associative scan op in (mag, phase) split representation.

    Carry = (k_mag, k_phase, u_mag, u_phase), each (B, C, H, W_freq).
    Equivalent to cssm_scalar_scan_op but without complex64.
    """
    k_mag_i, k_phase_i, u_mag_i, u_phase_i = carry_i
    k_mag_j, k_phase_j, u_mag_j, u_phase_j = carry_j

    # k_new = k_j * k_i  →  mags add, phases add
    k_mag_new = k_mag_j + k_mag_i
    k_phase_new = k_phase_j + k_phase_i

    # k_j * u_i
    ku_mag = k_mag_j + u_mag_i
    ku_phase = k_phase_j + u_phase_i

    # u_new = log_add_exp(k_j * u_i, u_j)
    u_mag_new, u_phase_new = _phase_log_add_exp(
        ku_mag, ku_phase, u_mag_j, u_phase_j)

    return k_mag_new, k_phase_new, u_mag_new, u_phase_new


def parallel_scan_scalar_phase(k_mag, k_phase, u_mag, u_phase):
    """Parallel scalar scan with phase-split representation."""
    _, _, u_mag_out, u_phase_out = jax.lax.associative_scan(
        _phase_scalar_scan_op, (k_mag, k_phase, u_mag, u_phase), axis=1
    )
    return u_mag_out


def sequential_scan_scalar_phase(k_mag, k_phase, u_mag, u_phase):
    """Sequential scalar scan with phase-split representation."""
    km_t = jnp.moveaxis(k_mag, 1, 0)
    kp_t = jnp.moveaxis(k_phase, 1, 0)
    um_t = jnp.moveaxis(u_mag, 1, 0)
    up_t = jnp.moveaxis(u_phase, 1, 0)

    def step(carry, x):
        new_carry = _phase_scalar_scan_op(carry, x)
        return new_carry, new_carry[2]  # output u_mag

    init = (km_t[0], kp_t[0], um_t[0], up_t[0])
    _, u_states = jax.lax.scan(step, init, (km_t[1:], kp_t[1:], um_t[1:], up_t[1:]))

    u_all = jnp.concatenate([um_t[0:1], u_states], axis=0)
    return jnp.moveaxis(u_all, 0, 1)


# --- 3x3 matrix scan in phase-split form ---

def _phase_3x3_matmul(A_mag, A_phase, B_mag, B_phase):
    """3x3 matrix multiply in phase-split log-space."""
    C_mag_list = []
    C_phase_list = []
    for i in range(3):
        row_mag = []
        row_phase = []
        for j in range(3):
            # Accumulate over k=0,1,2: C[i,j] = sum_k A[i,k]*B[k,j]
            m0, p0 = _phase_log_mul(
                A_mag[..., i, 0], A_phase[..., i, 0],
                B_mag[..., 0, j], B_phase[..., 0, j])
            m1, p1 = _phase_log_mul(
                A_mag[..., i, 1], A_phase[..., i, 1],
                B_mag[..., 1, j], B_phase[..., 1, j])
            acc_m, acc_p = _phase_log_add_exp(m0, p0, m1, p1)
            m2, p2 = _phase_log_mul(
                A_mag[..., i, 2], A_phase[..., i, 2],
                B_mag[..., 2, j], B_phase[..., 2, j])
            acc_m, acc_p = _phase_log_add_exp(acc_m, acc_p, m2, p2)
            row_mag.append(acc_m)
            row_phase.append(acc_p)
        C_mag_list.append(jnp.stack(row_mag, axis=-1))
        C_phase_list.append(jnp.stack(row_phase, axis=-1))
    return jnp.stack(C_mag_list, axis=-2), jnp.stack(C_phase_list, axis=-2)


def _phase_3x3_matvec(A_mag, A_phase, v_mag, v_phase):
    """3x3 matrix-vector multiply in phase-split log-space."""
    ys_mag = []
    ys_phase = []
    for i in range(3):
        m0, p0 = _phase_log_mul(
            A_mag[..., i, 0], A_phase[..., i, 0], v_mag[..., 0], v_phase[..., 0])
        m1, p1 = _phase_log_mul(
            A_mag[..., i, 1], A_phase[..., i, 1], v_mag[..., 1], v_phase[..., 1])
        acc_m, acc_p = _phase_log_add_exp(m0, p0, m1, p1)
        m2, p2 = _phase_log_mul(
            A_mag[..., i, 2], A_phase[..., i, 2], v_mag[..., 2], v_phase[..., 2])
        acc_m, acc_p = _phase_log_add_exp(acc_m, acc_p, m2, p2)
        ys_mag.append(acc_m)
        ys_phase.append(acc_p)
    return jnp.stack(ys_mag, axis=-1), jnp.stack(ys_phase, axis=-1)


def _phase_3x3_scan_op(carry_i, carry_j):
    """3x3 matrix associative scan op in phase-split representation.

    Carry = (K_mag, K_phase, U_mag, U_phase)
    K: (..., 3, 3), U: (..., 3)
    """
    K_mag_i, K_phase_i, U_mag_i, U_phase_i = carry_i
    K_mag_j, K_phase_j, U_mag_j, U_phase_j = carry_j

    # K_new = K_j @ K_i
    K_mag_new, K_phase_new = _phase_3x3_matmul(
        K_mag_j, K_phase_j, K_mag_i, K_phase_i)

    # Ku_i = K_j @ u_i
    Ku_mag, Ku_phase = _phase_3x3_matvec(
        K_mag_j, K_phase_j, U_mag_i, U_phase_i)

    # u_new[k] = log_add_exp(Ku_i[k], u_j[k])
    u_mag_list = []
    u_phase_list = []
    for k in range(3):
        m, p = _phase_log_add_exp(
            Ku_mag[..., k], Ku_phase[..., k],
            U_mag_j[..., k], U_phase_j[..., k])
        u_mag_list.append(m)
        u_phase_list.append(p)

    U_mag_new = jnp.stack(u_mag_list, axis=-1)
    U_phase_new = jnp.stack(u_phase_list, axis=-1)

    return K_mag_new, K_phase_new, U_mag_new, U_phase_new


def parallel_scan_3x3_phase(K_mag, K_phase, U_mag, U_phase):
    """Parallel 3x3 scan with phase-split representation."""
    _, _, U_mag_out, U_phase_out = jax.lax.associative_scan(
        _phase_3x3_scan_op, (K_mag, K_phase, U_mag, U_phase), axis=1
    )
    return U_mag_out


def sequential_scan_3x3_phase(K_mag, K_phase, U_mag, U_phase):
    """Sequential 3x3 scan with phase-split representation."""
    Km_t = jnp.moveaxis(K_mag, 1, 0)
    Kp_t = jnp.moveaxis(K_phase, 1, 0)
    Um_t = jnp.moveaxis(U_mag, 1, 0)
    Up_t = jnp.moveaxis(U_phase, 1, 0)

    def step(carry, x):
        new_carry = _phase_3x3_scan_op(carry, x)
        return new_carry, new_carry[2]

    init = (Km_t[0], Kp_t[0], Um_t[0], Up_t[0])
    _, u_states = jax.lax.scan(step, init, (Km_t[1:], Kp_t[1:], Um_t[1:], Up_t[1:]))

    u_all = jnp.concatenate([Um_t[0:1], u_states], axis=0)
    return jnp.moveaxis(u_all, 0, 1)


# --- Phase-split input generators ---

def make_phase_split_inputs(batch, seq_len, channels, h, w_freq, shape='scalar',
                            dtype_name='complex32'):
    """Generate random scan inputs in phase-split representation.

    Args:
        shape: 'scalar' for (B,T,C,H,W_freq) or '3x3' for matrix+vector
        dtype_name: 'complex32' (bf16) or 'complex16' (float8)

    Returns:
        (K_mag, K_phase, U_mag, U_phase)
    """
    if dtype_name == 'complex16':
        mag_dtype = _get_float8_dtype()
        if mag_dtype is None:
            print("    WARNING: float8 not available, falling back to bfloat16")
            mag_dtype = jnp.bfloat16
    else:
        mag_dtype = jnp.bfloat16

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    if shape == 'scalar':
        K_mag = (jax.random.normal(k1, (batch, seq_len, channels, h, w_freq)) * 0.1).astype(mag_dtype)
        K_phase = (jax.random.uniform(k2, (batch, seq_len, channels, h, w_freq),
                                       minval=-jnp.pi, maxval=jnp.pi)).astype(mag_dtype)
        U_mag = (jax.random.normal(k3, (batch, seq_len, channels, h, w_freq)) * 0.1).astype(mag_dtype)
        U_phase = (jax.random.uniform(k4, (batch, seq_len, channels, h, w_freq),
                                       minval=-jnp.pi, maxval=jnp.pi)).astype(mag_dtype)
    else:  # '3x3'
        K_mag = (jax.random.normal(k1, (batch, seq_len, channels, h, w_freq, 3, 3)) * 0.1).astype(mag_dtype)
        K_phase = (jax.random.uniform(k2, (batch, seq_len, channels, h, w_freq, 3, 3),
                                       minval=-jnp.pi, maxval=jnp.pi)).astype(mag_dtype)
        U_mag = (jax.random.normal(k3, (batch, seq_len, channels, h, w_freq, 3)) * 0.1).astype(mag_dtype)
        U_phase = (jax.random.uniform(k4, (batch, seq_len, channels, h, w_freq, 3),
                                       minval=-jnp.pi, maxval=jnp.pi)).astype(mag_dtype)

    return K_mag, K_phase, U_mag, U_phase


# =============================================================================
# Float32 baseline (for dtype comparison)
# =============================================================================

def _float32_log_add_exp(log_x1, log_x2):
    """Simplified real-valued log-add-exp (no GOOM, no complex)."""
    c = jax.lax.stop_gradient(jnp.maximum(log_x1, log_x2))
    return c + jnp.log(jnp.exp(log_x1 - c) + jnp.exp(log_x2 - c))


def _float32_3x3_matmul(A, B):
    """3x3 matmul in real log-space."""
    result = jnp.zeros_like(A)
    for i in range(3):
        for j in range(3):
            acc = A[..., i, 0] + B[..., 0, j]
            for k in range(1, 3):
                acc = _float32_log_add_exp(acc, A[..., i, k] + B[..., k, j])
            result = result.at[..., i, j].set(acc)
    return result


def _float32_3x3_matvec(A, v):
    """3x3 matvec in real log-space."""
    ys = []
    for i in range(3):
        acc = A[..., i, 0] + v[..., 0]
        for k in range(1, 3):
            acc = _float32_log_add_exp(acc, A[..., i, k] + v[..., k])
        ys.append(acc)
    return jnp.stack(ys, axis=-1)


def _float32_scan_op(carry_i, carry_j):
    """3x3 scan operator using real float32 arithmetic."""
    K_i, u_i = carry_i
    K_j, u_j = carry_j
    K_new = _float32_3x3_matmul(K_j, K_i)
    Ku_i = _float32_3x3_matvec(K_j, u_i)
    u_new = _float32_log_add_exp(Ku_i, u_j)
    return K_new, u_new


def parallel_scan_float32(K_log, U_log):
    """Parallel scan with real float32 tensors (no GOOM complex overhead)."""
    _, states = jax.lax.associative_scan(
        _float32_scan_op, (K_log, U_log), axis=1
    )
    return states


# =============================================================================
# Split GOOM: (magnitude, sign) representation — eliminates complex64
# =============================================================================
# GOOM stores: real part = log(|x|), imag part = 0 (positive) or pi (negative)
# The imaginary part is BINARY. Storing it as float32 wastes bits.
#
# Split representation:
#   mag: log(|x|) as float32 or bfloat16
#   sign: True = negative, False = positive (boolean)
#
# Benefits:
#   - No complex arithmetic (each complex op = 2-4 real ops)
#   - No GOOM custom VJPs (goom_exp/goom_log disappear)
#   - Boolean sign = 1 bit vs float32 phase = 32 bits
#   - bfloat16 magnitude halves memory bandwidth
#   - XLA compiles MUCH faster without complex64 (JAX#18221)

def _split_log_add_exp(mag_a, sign_a, mag_b, sign_b):
    """log(exp(a) + exp(b)) in split (magnitude, sign) representation.

    Equivalent to GOOM log_add_exp but without complex numbers.
    Uses the same max-subtraction trick for numerical stability.
    """
    c = jax.lax.stop_gradient(jnp.maximum(mag_a, mag_b))
    # Convert to linear space with sign
    ea = jnp.exp(mag_a - c)
    eb = jnp.exp(mag_b - c)
    va = jnp.where(sign_a, -ea, ea)
    vb = jnp.where(sign_b, -eb, eb)
    result = va + vb
    new_mag = c + jnp.log(jnp.maximum(jnp.abs(result), 1e-30))
    new_sign = result < 0
    return new_mag, new_sign


def _split_log_mul(mag_a, sign_a, mag_b, sign_b):
    """log(a * b) in split representation. Magnitudes add, signs XOR."""
    return mag_a + mag_b, sign_a ^ sign_b


def _split_3x3_matmul(A_mag, A_sign, B_mag, B_sign):
    """3x3 matrix multiply in split log-space.

    C[i,j] = log_add_exp over k of (A[i,k] * B[k,j])
    where * in log-space = add magnitudes, XOR signs.
    """
    C_mag_list = []
    C_sign_list = []
    for i in range(3):
        row_mag = []
        row_sign = []
        for j in range(3):
            # Start with k=0
            m0, s0 = _split_log_mul(
                A_mag[..., i, 0], A_sign[..., i, 0],
                B_mag[..., 0, j], B_sign[..., 0, j])
            # Accumulate k=1
            m1, s1 = _split_log_mul(
                A_mag[..., i, 1], A_sign[..., i, 1],
                B_mag[..., 1, j], B_sign[..., 1, j])
            acc_m, acc_s = _split_log_add_exp(m0, s0, m1, s1)
            # Accumulate k=2
            m2, s2 = _split_log_mul(
                A_mag[..., i, 2], A_sign[..., i, 2],
                B_mag[..., 2, j], B_sign[..., 2, j])
            acc_m, acc_s = _split_log_add_exp(acc_m, acc_s, m2, s2)
            row_mag.append(acc_m)
            row_sign.append(acc_s)
        C_mag_list.append(jnp.stack(row_mag, axis=-1))
        C_sign_list.append(jnp.stack(row_sign, axis=-1))
    return jnp.stack(C_mag_list, axis=-2), jnp.stack(C_sign_list, axis=-2)


def _split_3x3_matvec(A_mag, A_sign, v_mag, v_sign):
    """3x3 matrix-vector multiply in split log-space."""
    ys_mag = []
    ys_sign = []
    for i in range(3):
        m0, s0 = _split_log_mul(
            A_mag[..., i, 0], A_sign[..., i, 0], v_mag[..., 0], v_sign[..., 0])
        m1, s1 = _split_log_mul(
            A_mag[..., i, 1], A_sign[..., i, 1], v_mag[..., 1], v_sign[..., 1])
        acc_m, acc_s = _split_log_add_exp(m0, s0, m1, s1)
        m2, s2 = _split_log_mul(
            A_mag[..., i, 2], A_sign[..., i, 2], v_mag[..., 2], v_sign[..., 2])
        acc_m, acc_s = _split_log_add_exp(acc_m, acc_s, m2, s2)
        ys_mag.append(acc_m)
        ys_sign.append(acc_s)
    return jnp.stack(ys_mag, axis=-1), jnp.stack(ys_sign, axis=-1)


def _split_scan_op(carry_i, carry_j):
    """3x3 associative scan operator on split (mag, sign) GOOM representation.

    Carry = (K_mag, K_sign, U_mag, U_sign)
    """
    K_mag_i, K_sign_i, U_mag_i, U_sign_i = carry_i
    K_mag_j, K_sign_j, U_mag_j, U_sign_j = carry_j

    # K_new = K_j @ K_i
    K_mag_new, K_sign_new = _split_3x3_matmul(
        K_mag_j, K_sign_j, K_mag_i, K_sign_i)

    # Ku_i = K_j @ u_i
    Ku_mag, Ku_sign = _split_3x3_matvec(
        K_mag_j, K_sign_j, U_mag_i, U_sign_i)

    # u_new[k] = log_add_exp(Ku_i[k], u_j[k]) for k in 0,1,2
    u_mag_list = []
    u_sign_list = []
    for k in range(3):
        m, s = _split_log_add_exp(
            Ku_mag[..., k], Ku_sign[..., k],
            U_mag_j[..., k], U_sign_j[..., k])
        u_mag_list.append(m)
        u_sign_list.append(s)

    U_mag_new = jnp.stack(u_mag_list, axis=-1)
    U_sign_new = jnp.stack(u_sign_list, axis=-1)

    return K_mag_new, K_sign_new, U_mag_new, U_sign_new


def parallel_scan_split(K_mag, K_sign, U_mag, U_sign):
    """Parallel associative scan with split GOOM representation."""
    _, _, U_mag_out, U_sign_out = jax.lax.associative_scan(
        _split_scan_op, (K_mag, K_sign, U_mag, U_sign), axis=1
    )
    return U_mag_out


def sequential_scan_split(K_mag, K_sign, U_mag, U_sign):
    """Sequential scan with split GOOM representation."""
    Km_t = jnp.moveaxis(K_mag, 1, 0)
    Ks_t = jnp.moveaxis(K_sign, 1, 0)
    Um_t = jnp.moveaxis(U_mag, 1, 0)
    Us_t = jnp.moveaxis(U_sign, 1, 0)

    def step(carry, x):
        new_carry = _split_scan_op(carry, x)
        return new_carry, new_carry[2]  # output U_mag

    init = (Km_t[0], Ks_t[0], Um_t[0], Us_t[0])
    _, u_states = jax.lax.scan(step, init, (Km_t[1:], Ks_t[1:], Um_t[1:], Us_t[1:]))

    u_all = jnp.concatenate([Um_t[0:1], u_states], axis=0)
    return jnp.moveaxis(u_all, 0, 1)


def make_split_inputs(batch, seq_len, channels, h, w_freq, mag_dtype=jnp.float32):
    """Generate random scan inputs in split GOOM representation.

    Returns (K_mag, K_sign, U_mag, U_sign) where:
      K_mag: (B, T, C, H, W_freq, 3, 3) mag_dtype — log magnitudes
      K_sign: (B, T, C, H, W_freq, 3, 3) bool — True = negative
      U_mag: (B, T, C, H, W_freq, 3) mag_dtype
      U_sign: (B, T, C, H, W_freq, 3) bool
    """
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    K_mag = (jax.random.normal(k1, (batch, seq_len, channels, h, w_freq, 3, 3)) * 0.1).astype(mag_dtype)
    K_sign = jax.random.bernoulli(k2, 0.5, (batch, seq_len, channels, h, w_freq, 3, 3))
    U_mag = (jax.random.normal(k3, (batch, seq_len, channels, h, w_freq, 3)) * 0.1).astype(mag_dtype)
    U_sign = jax.random.bernoulli(k4, 0.5, (batch, seq_len, channels, h, w_freq, 3))

    return K_mag, K_sign, U_mag, U_sign


# =============================================================================
# Benchmark Harness
# =============================================================================

def get_memory_usage():
    """Get current GPU memory usage in MB."""
    try:
        devices = jax.local_devices()
        for d in devices:
            if d.platform == 'gpu':
                stats = d.memory_stats()
                if stats:
                    return stats.get('peak_bytes_in_use', 0) / 1e6
    except Exception:
        pass
    return 0.0


def _make_grad_fn(fn, argnums=1):
    """Wrap a scan function so it returns a scalar loss and computes gradients.

    This measures forward + backward pass together, which is the realistic
    training cost. The loss is sum(real(output)) — simple but exercises
    all gradient paths.

    Args:
        fn: The scan function to differentiate.
        argnums: Which argument(s) to differentiate w.r.t. Default 1 (U_log).
                 For phase-split functions use (2,) for u_mag.
    """
    def loss_fn(*args):
        out = fn(*args)
        # Handle both complex and real outputs
        if jnp.iscomplexobj(out):
            return jnp.sum(out.real)
        return jnp.sum(out)

    return jax.grad(loss_fn, argnums=argnums)


def benchmark(fn, args, warmup=3, trials=10, label=""):
    """Time a JIT-compiled function, separating compilation from execution.

    Returns dict with compile_s, mean_ms, std_ms, min_ms, memory_mb.
    """
    fn_jit = jax.jit(fn)

    # Measure compilation (first call includes JIT compilation)
    print(f"    Compiling {label}...", end="", flush=True)
    t0 = time.time()
    result = fn_jit(*args)
    # Block until computation is done
    jax.block_until_ready(result)
    compile_time = time.time() - t0
    print(f" {compile_time:.1f}s")

    # Additional warmup
    for _ in range(warmup - 1):
        result = fn_jit(*args)
        jax.block_until_ready(result)

    # Timed trials
    times = []
    for _ in range(trials):
        t0 = time.time()
        result = fn_jit(*args)
        jax.block_until_ready(result)
        times.append(time.time() - t0)

    return {
        'compile_s': compile_time,
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'memory_mb': get_memory_usage(),
    }


# =============================================================================
# Data Generation
# =============================================================================

def make_scan_inputs(batch, seq_len, channels, h, w_freq, dtype=jnp.complex64):
    """Generate random scan inputs in GOOM representation.

    K_log: (B, T, C, H, W_freq, 3, 3) — transition matrices in log-space
    U_log: (B, T, C, H, W_freq, 3)    — input vectors in log-space
    """
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    if dtype == jnp.complex64:
        # Generate in GOOM space: real part = log magnitude, imag part = phase (0 or pi)
        K_real = jax.random.normal(k1, (batch, seq_len, channels, h, w_freq, 3, 3)) * 0.1
        K_phase = jax.random.bernoulli(k1, 0.5, (batch, seq_len, channels, h, w_freq, 3, 3)).astype(jnp.float32) * jnp.pi
        K_log = K_real + 1j * K_phase

        U_real = jax.random.normal(k2, (batch, seq_len, channels, h, w_freq, 3)) * 0.1
        U_phase = jax.random.bernoulli(k2, 0.5, (batch, seq_len, channels, h, w_freq, 3)).astype(jnp.float32) * jnp.pi
        U_log = U_real + 1j * U_phase
    else:
        # Float32: same shapes, real-valued
        K_log = jax.random.normal(k1, (batch, seq_len, channels, h, w_freq, 3, 3)) * 0.1
        U_log = jax.random.normal(k2, (batch, seq_len, channels, h, w_freq, 3)) * 0.1

    return K_log, U_log


# =============================================================================
# Profiling (--profile flag)
# =============================================================================

def profile_mode(fn, args, label, output_dir):
    """Save a JAX profiler trace for a single mode."""
    fn_jit = jax.jit(fn)
    # Warmup
    result = fn_jit(*args)
    jax.block_until_ready(result)

    trace_dir = os.path.join(output_dir, f"trace_{label}")
    os.makedirs(trace_dir, exist_ok=True)

    jax.profiler.start_trace(trace_dir)
    for _ in range(3):
        result = fn_jit(*args)
        jax.block_until_ready(result)
    jax.profiler.stop_trace()
    print(f"  Trace saved: {trace_dir}")


def count_ops(fn, args, label):
    """Count XLA operations using jax.make_jaxpr."""
    jaxpr = jax.make_jaxpr(fn)(*args)
    n_eqns = len(jaxpr.eqns)
    # Count unique primitives
    prim_counts = {}
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        prim_counts[name] = prim_counts.get(name, 0) + 1
    top5 = sorted(prim_counts.items(), key=lambda x: -x[1])[:5]
    top5_str = ", ".join(f"{name}:{count}" for name, count in top5)
    print(f"  {label}: {n_eqns} jaxpr ops (top: {top5_str})")
    return n_eqns


# =============================================================================
# Reporting
# =============================================================================

def print_table(results, batch, channels, h, w_freq):
    """Print benchmark results as a formatted table."""
    has_bwd = 'parallel_bwd' in results[0]

    print()
    print(f"Sequence Length Benchmark (B={batch}, C={channels}, H={h}, W_freq={w_freq}, 3x3 scan)")

    # Forward-only table
    print()
    print("FORWARD ONLY (execution time, post-JIT)")
    print("=" * 80)
    header = f"{'T':>5} | {'Parallel':>14} | {'Seq Linear':>14} | {'Seq Nonlinear':>14} | {'Speedup (P/SL)':>14}"
    print(header)
    print("-" * 80)

    for r in results:
        T = r['T']
        p = r['parallel']
        sl = r['seq_linear']
        sn = r['seq_nonlinear']
        speedup = sl['mean_ms'] / p['mean_ms'] if p['mean_ms'] > 0 else float('inf')
        print(
            f"{T:>5} | "
            f"{p['mean_ms']:>8.2f} ± {p['std_ms']:<4.1f}ms | "
            f"{sl['mean_ms']:>8.2f} ± {sl['std_ms']:<4.1f}ms | "
            f"{sn['mean_ms']:>8.2f} ± {sn['std_ms']:<4.1f}ms | "
            f"{speedup:>10.2f}x"
        )

    # Forward+Backward table
    if has_bwd:
        print()
        print("FORWARD + BACKWARD (execution time, post-JIT)")
        print("=" * 80)
        print(header)
        print("-" * 80)

        for r in results:
            T = r['T']
            p = r['parallel_bwd']
            sl = r['seq_linear_bwd']
            sn = r['seq_nonlinear_bwd']
            speedup = sl['mean_ms'] / p['mean_ms'] if p['mean_ms'] > 0 else float('inf')
            print(
                f"{T:>5} | "
                f"{p['mean_ms']:>8.2f} ± {p['std_ms']:<4.1f}ms | "
                f"{sl['mean_ms']:>8.2f} ± {sl['std_ms']:<4.1f}ms | "
                f"{sn['mean_ms']:>8.2f} ± {sn['std_ms']:<4.1f}ms | "
                f"{speedup:>10.2f}x"
            )

    print()
    print("XLA Compilation Times:")
    for r in results:
        T = r['T']
        line = (f"  T={T:<4}: Parallel={r['parallel']['compile_s']:.1f}s  "
                f"SeqLin={r['seq_linear']['compile_s']:.1f}s  "
                f"SeqNonlin={r['seq_nonlinear']['compile_s']:.1f}s")
        if has_bwd:
            line += (f"  |  +Bwd: P={r['parallel_bwd']['compile_s']:.1f}s  "
                     f"SL={r['seq_linear_bwd']['compile_s']:.1f}s  "
                     f"SN={r['seq_nonlinear_bwd']['compile_s']:.1f}s")
        print(line)


def _fit_theoretical(T_arr, measured, model_fn):
    """Fit a scalar constant c so that c * model_fn(T) best matches measured data (least-squares).

    Returns the fitted curve values at T_arr.
    """
    model_vals = np.array([model_fn(t) for t in T_arr])
    # c = argmin sum (c * model - measured)^2  =>  c = dot(model, measured) / dot(model, model)
    c = np.dot(model_vals, measured) / np.dot(model_vals, model_vals)
    return c * model_vals


def _add_theoretical_lines(ax, Ts, sl_means, p_means):
    """Add fitted theoretical O(T) and O(log T) reference lines to a plot axis.

    O(T):     sequential scan does T applications of the scan operator.
    O(T log T): parallel associative scan does T * ceil(log2(T)) operator
                applications total (but in O(log T) depth / wall-clock steps
                on a parallel machine with enough processors). On a GPU the
                actual time is between O(log T) and O(T log T) depending on
                how well the parallelism maps to hardware.

    We fit both O(T) and O(log T) to the measured data so the reference
    slopes are anchored to reality rather than a single arbitrary point.
    """
    if len(Ts) < 2:
        return

    T_arr = np.array(Ts, dtype=float)
    sl_arr = np.array(sl_means)
    p_arr = np.array(p_means)

    # Dense T range for smooth reference curves
    T_dense = np.geomspace(T_arr[0] * 0.8, T_arr[-1] * 1.2, 200)

    # --- O(T) fitted to sequential data ---
    fitted_linear = _fit_theoretical(T_arr, sl_arr, lambda t: t)
    curve_linear = (np.dot(T_arr, sl_arr) / np.dot(T_arr, T_arr)) * T_dense
    ax.plot(T_dense, curve_linear, '--', color='#F78C2A', alpha=0.35, linewidth=1.5,
            label=r'$\Theta(T)$ fit to sequential')

    # --- O(log T) fitted to parallel data ---
    fitted_log = _fit_theoretical(T_arr, p_arr, lambda t: np.log2(t))
    c_log = np.dot(np.log2(T_arr), p_arr) / np.dot(np.log2(T_arr), np.log2(T_arr))
    curve_log = c_log * np.log2(T_dense)
    ax.plot(T_dense, curve_log, '--', color='#2176AE', alpha=0.35, linewidth=1.5,
            label=r'$\Theta(\log T)$ fit to parallel')



def make_plot(results, output_path, subtitle=""):
    """Generate a log-log line plot comparing the three modes."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot generation.")
        return

    Ts = [r['T'] for r in results]
    p_means = [r['parallel']['mean_ms'] for r in results]
    p_stds = [r['parallel']['std_ms'] for r in results]
    sl_means = [r['seq_linear']['mean_ms'] for r in results]
    sl_stds = [r['seq_linear']['std_ms'] for r in results]
    sn_means = [r['seq_nonlinear']['mean_ms'] for r in results]
    sn_stds = [r['seq_nonlinear']['std_ms'] for r in results]

    has_bwd = 'parallel_bwd' in results[0]
    ncols = 2 if has_bwd else 1
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 6), squeeze=False)

    # --- Panel 1: Forward only ---
    ax = axes[0][0]
    ax.errorbar(Ts, p_means, yerr=p_stds, marker='o', color='#2176AE',
                linewidth=2, capsize=4, label='Parallel (jax.lax.associative_scan)')
    ax.errorbar(Ts, sl_means, yerr=sl_stds, marker='s', color='#F78C2A',
                linewidth=2, capsize=4, label='Sequential Linear (jax.lax.scan)')
    ax.errorbar(Ts, sn_means, yerr=sn_stds, marker='^', color='#D7263D',
                linewidth=2, capsize=4, label='Sequential Nonlinear RNN (jax.lax.scan + tanh)')
    _add_theoretical_lines(ax, Ts, sl_means, p_means)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Sequence Length (T)', fontsize=12)
    ax.set_ylabel('Execution Time (ms, post-JIT)', fontsize=12)
    ax.set_title('Forward Only', fontsize=12)
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(t) for t in Ts])
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    # --- Panel 2: Forward + Backward ---
    if has_bwd:
        ax2 = axes[0][1]
        pb_means = [r['parallel_bwd']['mean_ms'] for r in results]
        pb_stds = [r['parallel_bwd']['std_ms'] for r in results]
        slb_means = [r['seq_linear_bwd']['mean_ms'] for r in results]
        slb_stds = [r['seq_linear_bwd']['std_ms'] for r in results]
        snb_means = [r['seq_nonlinear_bwd']['mean_ms'] for r in results]
        snb_stds = [r['seq_nonlinear_bwd']['std_ms'] for r in results]

        ax2.errorbar(Ts, pb_means, yerr=pb_stds, marker='o', color='#2176AE',
                     linewidth=2, capsize=4, label='Parallel (jax.lax.associative_scan)')
        ax2.errorbar(Ts, slb_means, yerr=slb_stds, marker='s', color='#F78C2A',
                     linewidth=2, capsize=4, label='Sequential Linear (jax.lax.scan)')
        ax2.errorbar(Ts, snb_means, yerr=snb_stds, marker='^', color='#D7263D',
                     linewidth=2, capsize=4, label='Sequential Nonlinear RNN (jax.lax.scan + tanh)')
        _add_theoretical_lines(ax2, Ts, slb_means, pb_means)

        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        ax2.set_xlabel('Sequence Length (T)', fontsize=12)
        ax2.set_ylabel('Execution Time (ms, post-JIT)', fontsize=12)
        ax2.set_title('Forward + Backward', fontsize=12)
        ax2.set_xticks(Ts)
        ax2.set_xticklabels([str(t) for t in Ts])
        ax2.legend(fontsize=8, loc='upper left')
        ax2.grid(True, alpha=0.3, which='both')

    suptitle = 'CSSM Recurrence: Parallel Scan vs Sequential'
    if subtitle:
        suptitle += f'\n{subtitle}'
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {output_path}")


def make_sweep_plot(sweep_results, sweep_param_name, output_path):
    """Generate a multi-panel plot showing curves across a swept parameter.

    sweep_results: list of dicts with keys 'param_val', 'results' (list per T),
                   plus 'B', 'C', 'H', 'W_freq'.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not available — skipping plot generation.")
        return

    n = len(sweep_results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, entry in enumerate(sweep_results):
        ax = axes[idx // cols][idx % cols]
        val = entry['param_val']
        results = entry['results']

        Ts = [r['T'] for r in results]
        p_means = [r['parallel']['mean_ms'] for r in results]
        p_stds = [r['parallel']['std_ms'] for r in results]
        sl_means = [r['seq_linear']['mean_ms'] for r in results]
        sl_stds = [r['seq_linear']['std_ms'] for r in results]
        sn_means = [r['seq_nonlinear']['mean_ms'] for r in results]
        sn_stds = [r['seq_nonlinear']['std_ms'] for r in results]

        ax.errorbar(Ts, p_means, yerr=p_stds, marker='o', color='#2176AE',
                    linewidth=2, capsize=3, label='Parallel')
        ax.errorbar(Ts, sl_means, yerr=sl_stds, marker='s', color='#F78C2A',
                    linewidth=2, capsize=3, label='Seq Linear')
        ax.errorbar(Ts, sn_means, yerr=sn_stds, marker='^', color='#D7263D',
                    linewidth=2, capsize=3, label='Seq Nonlinear')

        # Fitted theoretical scaling lines
        _add_theoretical_lines(ax, Ts, sl_means, p_means)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(Ts)
        ax.set_xticklabels([str(t) for t in Ts], fontsize=8)
        ax.set_xlabel('Sequence Length (T)', fontsize=10)
        ax.set_ylabel('Execution Time (ms)', fontsize=10)

        # Build subtitle from params
        subtitle_parts = []
        if sweep_param_name == 'image_size':
            subtitle_parts.append(f"H={val}")
            subtitle_parts.append(f"C={entry['C']}")
        elif sweep_param_name == 'embed_dim':
            subtitle_parts.append(f"C={val}")
            subtitle_parts.append(f"H={entry['H']}")
        subtitle_parts.append(f"B={entry['B']}")
        ax.set_title(', '.join(subtitle_parts), fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(f'CSSM Scan Benchmark — Sweep over {sweep_param_name}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sweep plot saved: {output_path}")


def print_optimization_analysis():
    """Print a summary of optimization findings and recommendations."""
    print()
    print("=" * 80)
    print("OPTIMIZATION ANALYSIS")
    print("=" * 80)
    print("""
1. COMPLEX64 OVERHEAD
   Complex arithmetic costs ~2-4x vs float32. Each complex add = 2 real adds,
   each complex mul = 4 real muls + 2 real adds. GOOM log/exp on complex
   numbers further amplifies this.

   Opportunity: Split real/imaginary into separate float32 arrays ("struct of
   arrays"). GOOM phase is always 0 or pi, so imag part is binary — could use
   a separate boolean mask instead of float.

2. XLA COMPILATION (JAX#18221)
   associative_scan with complex64 triggers extremely slow XLA compilation
   (30-60s for T=64). This is a known JAX issue.

   Mitigation: Pre-compile with fixed shapes, use jax.jit(fn).lower().compile()
   ahead of time. Or split complex into 2x float32 before the scan.

3. FUSION BARRIERS
   log_add_exp uses jax.lax.stop_gradient on the max computation, which may
   inhibit XLA fusion of surrounding ops. Each log_add_exp call does:
   goom_real -> max -> stop_grad -> subtract -> from_goom -> add -> to_goom -> add
   That's 8+ ops per log_add_exp, and the 3x3 scan does ~30 of these per step.

   Opportunity: Write a fused log_add_exp as a single XLA custom call, or
   use jnp.logaddexp (real-valued) with manual phase tracking.

4. SCAN OPERATOR SIZE
   The 3x3 matrix scan does 9 matmul elements + 3 matvec elements, each
   requiring multiple log_add_exp calls. The operator itself is ~300+ jaxpr ops.

   Opportunity: For the triangular AdditiveCSSM matrix (6 of 9 entries
   are -inf / zero), exploit sparsity to skip ~half the computation.

5. MEMORY
   The scan materializes all intermediate states: (B, T, C, H, W_freq, 3, 3)
   complex64. For B=4, T=64, C=32, H=32, W_freq=17: ~600MB just for K_log.

   Opportunity: Gradient checkpointing on the scan (jax.checkpoint).
""")


# =============================================================================
# Main
# =============================================================================

def run_sweep(seq_lens, B, C, H, warmup, trials, include_backward=True):
    """Run the 3-mode benchmark for a single (B, C, H) configuration across seq_lens.

    Returns list of per-T result dicts. Each dict has keys:
        'parallel', 'seq_linear', 'seq_nonlinear' — forward-only timings
    and if include_backward:
        'parallel_bwd', 'seq_linear_bwd', 'seq_nonlinear_bwd' — forward+backward timings
    """
    W_freq = H // 2 + 1
    results = []
    for T in seq_lens:
        print(f"  T={T} (B={B}, C={C}, H={H}, W_freq={W_freq})")
        K_log, U_log = make_scan_inputs(B, T, C, H, W_freq, dtype=jnp.complex64)

        print(f"    [Forward only]")
        r_parallel = benchmark(parallel_scan, (K_log, U_log),
                               warmup=warmup, trials=trials,
                               label=f"parallel T={T}")

        r_seq_lin = benchmark(sequential_scan, (K_log, U_log),
                              warmup=warmup, trials=trials,
                              label=f"seq_linear T={T}")

        r_seq_nl = benchmark(sequential_nonlinear_scan, (K_log, U_log),
                             warmup=warmup, trials=trials,
                             label=f"seq_nonlinear T={T}")

        entry = {
            'T': T,
            'parallel': r_parallel,
            'seq_linear': r_seq_lin,
            'seq_nonlinear': r_seq_nl,
        }

        if include_backward:
            print(f"    [Forward + Backward]")
            grad_parallel = _make_grad_fn(parallel_scan)
            grad_seq_lin = _make_grad_fn(sequential_scan)
            grad_seq_nl = _make_grad_fn(sequential_nonlinear_scan)

            entry['parallel_bwd'] = benchmark(grad_parallel, (K_log, U_log),
                                              warmup=warmup, trials=trials,
                                              label=f"parallel+bwd T={T}")
            entry['seq_linear_bwd'] = benchmark(grad_seq_lin, (K_log, U_log),
                                                warmup=warmup, trials=trials,
                                                label=f"seq_linear+bwd T={T}")
            entry['seq_nonlinear_bwd'] = benchmark(grad_seq_nl, (K_log, U_log),
                                                   warmup=warmup, trials=trials,
                                                   label=f"seq_nonlinear+bwd T={T}")

        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(description='CSSM Speed Benchmarks')
    parser.add_argument('--seq_lens', type=str, default='8,16,32,64',
                        help='Comma-separated sequence lengths to benchmark')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--embed_dims', type=str, default='32',
                        help='Comma-separated embedding dimensions to sweep')
    parser.add_argument('--image_sizes', type=str, default='32',
                        help='Comma-separated image sizes (H) to sweep')
    # Keep singular forms as aliases for backwards compat
    parser.add_argument('--embed_dim', type=int, default=None,
                        help='(deprecated, use --embed_dims) Single embedding dim')
    parser.add_argument('--image_size', type=int, default=None,
                        help='(deprecated, use --image_sizes) Single image size')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup iterations')
    parser.add_argument('--trials', type=int, default=10,
                        help='Number of timed trials')
    parser.add_argument('--no_backward', action='store_true',
                        help='Skip backward pass benchmarks (forward only)')
    parser.add_argument('--scalar', action='store_true',
                        help='Benchmark scalar scan (add_kqv_1) alongside 3x3 matrix scan')
    parser.add_argument('--phase_split', action='store_true',
                        help='Benchmark phase-split custom dtypes (complex32/complex16) vs complex64')
    parser.add_argument('--split_goom', action='store_true',
                        help='Benchmark split GOOM (mag+sign) vs complex64, in float32 and bfloat16')
    parser.add_argument('--profile', action='store_true',
                        help='Save JAX profiler traces and count ops')
    parser.add_argument('--dtype_compare', action='store_true',
                        help='Compare complex64 vs float32 performance')
    parser.add_argument('--output_dir', type=str, default='/tmp/cssm_profiles',
                        help='Directory for profiler traces')
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(',')]
    B = args.batch_size

    # Resolve embed_dims
    if args.embed_dim is not None:
        embed_dims = [args.embed_dim]
    else:
        embed_dims = [int(x) for x in args.embed_dims.split(',')]

    # Resolve image_sizes
    if args.image_size is not None:
        image_sizes = [args.image_size]
    else:
        image_sizes = [int(x) for x in args.image_sizes.split(',')]

    bench_dir = os.path.dirname(__file__)

    include_backward = not args.no_backward

    print(f"CSSM Speed Benchmark")
    print(f"  Devices: {jax.devices()}")
    print(f"  B={B}")
    print(f"  Embed dims:  {embed_dims}")
    print(f"  Image sizes: {image_sizes}")
    print(f"  Sequence lengths: {seq_lens}")
    print(f"  Backward pass: {'YES' if include_backward else 'NO'}")
    print(f"  Warmup={args.warmup}, Trials={args.trials}")
    print(f"  (Times below are post-compilation execution only)")
    print()

    # =========================================================================
    # Main benchmark: sweep over image_sizes x embed_dims
    # =========================================================================
    all_sweep_results = []  # flat list of (C, H, results)
    image_size_sweep = []   # grouped by image_size (for sweep plot)
    embed_dim_sweep = []    # grouped by embed_dim (for sweep plot)

    for H in image_sizes:
        for C in embed_dims:
            W_freq = H // 2 + 1
            print(f"{'='*70}")
            print(f"Configuration: B={B}, C={C}, H={H}, W_freq={W_freq}")
            print(f"{'='*70}")

            results = run_sweep(seq_lens, B, C, H, args.warmup, args.trials,
                               include_backward=include_backward)
            print_table(results, B, C, H, W_freq)

            entry = {'B': B, 'C': C, 'H': H, 'W_freq': W_freq, 'results': results}
            all_sweep_results.append(entry)

            # Individual plot for this config
            subtitle = f"B={B}, C={C}, H={H}, W_freq={W_freq}  (execution time, post-JIT)"
            plot_name = f"cssm_speed_C{C}_H{H}.png"
            make_plot(results, os.path.join(bench_dir, plot_name), subtitle=subtitle)

    # =========================================================================
    # Sweep plots (multi-panel) when there are multiple values
    # =========================================================================
    if len(image_sizes) > 1:
        sweep_data = []
        for H in image_sizes:
            # Use the first embed_dim for this sweep
            C0 = embed_dims[0]
            for e in all_sweep_results:
                if e['H'] == H and e['C'] == C0:
                    sweep_data.append({'param_val': H, **e})
                    break
        make_sweep_plot(sweep_data, 'image_size',
                        os.path.join(bench_dir, 'cssm_speed_sweep_image_size.png'))

    if len(embed_dims) > 1:
        sweep_data = []
        for C in embed_dims:
            # Use the first image_size for this sweep
            H0 = image_sizes[0]
            for e in all_sweep_results:
                if e['C'] == C and e['H'] == H0:
                    sweep_data.append({'param_val': C, **e})
                    break
        make_sweep_plot(sweep_data, 'embed_dim',
                        os.path.join(bench_dir, 'cssm_speed_sweep_embed_dim.png'))

    # =========================================================================
    # Scalar scan benchmark (--scalar)
    # =========================================================================
    if args.scalar:
        print()
        print("=" * 80)
        print("SCALAR SCAN (add_kqv_1): per-step tensor (B,C,H,W_freq) — 9x smaller state")
        print("=" * 80)

        for H in image_sizes:
            for C in embed_dims:
                W_freq = H // 2 + 1
                print(f"\nConfiguration: B={B}, C={C}, H={H}, W_freq={W_freq}")
                print(f"  Per-step elements: {B*C*H*W_freq:,} (scalar) vs "
                      f"{B*C*H*W_freq*9:,} (3x3 matrix)")

                for T in seq_lens:
                    print(f"\n  --- T={T} ---")
                    K_s, U_s = make_scalar_inputs(B, T, C, H, W_freq, dtype=jnp.complex64)

                    r_par = benchmark(parallel_scan_scalar, (K_s, U_s),
                                      warmup=args.warmup, trials=args.trials,
                                      label=f"scalar parallel T={T}")
                    r_seq = benchmark(sequential_scan_scalar, (K_s, U_s),
                                      warmup=args.warmup, trials=args.trials,
                                      label=f"scalar sequential T={T}")

                    speedup = r_seq['mean_ms'] / r_par['mean_ms'] if r_par['mean_ms'] > 0 else float('inf')
                    print(f"  Parallel:   {r_par['mean_ms']:>8.2f} ms  (compile: {r_par['compile_s']:.1f}s)")
                    print(f"  Sequential: {r_seq['mean_ms']:>8.2f} ms  (compile: {r_seq['compile_s']:.1f}s)")
                    print(f"  Speedup (par/seq): {speedup:.2f}x")

                    if include_backward:
                        grad_par = _make_grad_fn(parallel_scan_scalar)
                        grad_seq = _make_grad_fn(sequential_scan_scalar)

                        r_par_bwd = benchmark(grad_par, (K_s, U_s),
                                              warmup=args.warmup, trials=args.trials,
                                              label=f"scalar parallel+bwd T={T}")
                        r_seq_bwd = benchmark(grad_seq, (K_s, U_s),
                                              warmup=args.warmup, trials=args.trials,
                                              label=f"scalar sequential+bwd T={T}")

                        speedup_bwd = r_seq_bwd['mean_ms'] / r_par_bwd['mean_ms'] if r_par_bwd['mean_ms'] > 0 else float('inf')
                        print(f"  Parallel+bwd:   {r_par_bwd['mean_ms']:>8.2f} ms  (compile: {r_par_bwd['compile_s']:.1f}s)")
                        print(f"  Sequential+bwd: {r_seq_bwd['mean_ms']:>8.2f} ms  (compile: {r_seq_bwd['compile_s']:.1f}s)")
                        print(f"  Speedup+bwd:    {speedup_bwd:.2f}x")

    # =========================================================================
    # Phase-split custom dtype benchmark (--phase_split)
    # =========================================================================
    if args.phase_split:
        print()
        print("=" * 80)
        print("PHASE-SPLIT DTYPE: complex64 vs complex32 (bf16) vs complex16 (float8)")
        print("=" * 80)
        print("Replaces native complex64 with separate (magnitude, phase) arrays.")
        print("Compatible with FFT spectral data (arbitrary complex phases).")
        print()

        # Check float8 availability
        f8_dtype = _get_float8_dtype()
        dtype_names = ['complex32']
        if f8_dtype is not None:
            dtype_names.append('complex16')
            print(f"  float8 dtype: {f8_dtype}")
        else:
            print("  WARNING: float8 not available — skipping complex16")
        print(f"  Dtypes to test: complex64 (baseline), {', '.join(dtype_names)}")

        scan_types = ['scalar']
        if not args.scalar:
            # If --scalar not already run, also benchmark scalar here
            pass
        scan_types.append('3x3')

        for H in image_sizes:
            for C in embed_dims:
                W_freq = H // 2 + 1

                for scan_type in scan_types:
                    print(f"\n{'='*60}")
                    print(f"  {scan_type.upper()} scan, B={B}, C={C}, H={H}, W_freq={W_freq}")
                    print(f"{'='*60}")

                    for T in seq_lens:
                        print(f"\n  --- T={T} ---")

                        # ---- complex64 GOOM baseline ----
                        if scan_type == 'scalar':
                            K_c, U_c = make_scalar_inputs(B, T, C, H, W_freq, dtype=jnp.complex64)
                            r_c64_par = benchmark(parallel_scan_scalar, (K_c, U_c),
                                                  warmup=args.warmup, trials=args.trials,
                                                  label="c64 scalar par")
                            r_c64_seq = benchmark(sequential_scan_scalar, (K_c, U_c),
                                                  warmup=args.warmup, trials=args.trials,
                                                  label="c64 scalar seq")
                        else:  # 3x3
                            K_c, U_c = make_scan_inputs(B, T, C, H, W_freq, dtype=jnp.complex64)
                            r_c64_par = benchmark(parallel_scan, (K_c, U_c),
                                                  warmup=args.warmup, trials=args.trials,
                                                  label="c64 3x3 par")
                            r_c64_seq = benchmark(sequential_scan, (K_c, U_c),
                                                  warmup=args.warmup, trials=args.trials,
                                                  label="c64 3x3 seq")

                        # ---- Phase-split dtypes ----
                        phase_results = {}  # dtype_name -> {'par': result, 'seq': result}
                        for dn in dtype_names:
                            inputs = make_phase_split_inputs(B, T, C, H, W_freq,
                                                             shape=scan_type, dtype_name=dn)

                            if scan_type == 'scalar':
                                r_par = benchmark(parallel_scan_scalar_phase, inputs,
                                                  warmup=args.warmup, trials=args.trials,
                                                  label=f"{dn} scalar par")
                                r_seq = benchmark(sequential_scan_scalar_phase, inputs,
                                                  warmup=args.warmup, trials=args.trials,
                                                  label=f"{dn} scalar seq")
                            else:  # 3x3
                                r_par = benchmark(parallel_scan_3x3_phase, inputs,
                                                  warmup=args.warmup, trials=args.trials,
                                                  label=f"{dn} 3x3 par")
                                r_seq = benchmark(sequential_scan_3x3_phase, inputs,
                                                  warmup=args.warmup, trials=args.trials,
                                                  label=f"{dn} 3x3 seq")

                            phase_results[dn] = {'par': r_par, 'seq': r_seq}

                        # ---- Print comparison table ----
                        print(f"\n  {'Mode':<32} | {'Fwd (ms)':>10} | {'Compile':>8} | {'Bytes/elem':>10}")
                        print(f"  {'-'*32}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")

                        bytes_map = {'complex64': 8, 'complex32': 4, 'complex16': 2}
                        all_rows = [
                            ("complex64 parallel",   r_c64_par,  8),
                            ("complex64 sequential", r_c64_seq,  8),
                        ]
                        for dn in dtype_names:
                            all_rows.append((f"{dn} parallel",   phase_results[dn]['par'], bytes_map.get(dn, 4)))
                            all_rows.append((f"{dn} sequential", phase_results[dn]['seq'], bytes_map.get(dn, 4)))

                        for name, r, bpe in all_rows:
                            print(f"  {name:<32} | {r['mean_ms']:>7.2f} ms | {r['compile_s']:>6.1f}s | {bpe:>6}")

                        # Speedup vs complex64 parallel
                        print(f"\n  Speedup vs complex64 parallel:")
                        for dn in dtype_names:
                            p_sp = r_c64_par['mean_ms'] / phase_results[dn]['par']['mean_ms']
                            s_sp = r_c64_par['mean_ms'] / phase_results[dn]['seq']['mean_ms']
                            c_sp = r_c64_par['compile_s'] / max(phase_results[dn]['par']['compile_s'], 0.01)
                            print(f"    {dn} parallel:    {p_sp:.2f}x exec, {c_sp:.0f}x compile")
                            print(f"    {dn} sequential:  {s_sp:.2f}x exec")

                        # Forward+Backward for phase-split
                        if include_backward:
                            print(f"\n  Forward + Backward:")
                            if scan_type == 'scalar':
                                grad_c64 = _make_grad_fn(parallel_scan_scalar)
                                r_c64_bwd = benchmark(grad_c64, (K_c, U_c),
                                                      warmup=args.warmup, trials=args.trials,
                                                      label="c64 scalar par+bwd")
                            else:
                                grad_c64 = _make_grad_fn(parallel_scan)
                                r_c64_bwd = benchmark(grad_c64, (K_c, U_c),
                                                      warmup=args.warmup, trials=args.trials,
                                                      label="c64 3x3 par+bwd")

                            print(f"    complex64 parallel+bwd: {r_c64_bwd['mean_ms']:.2f} ms")

                            for dn in dtype_names:
                                inputs = make_phase_split_inputs(B, T, C, H, W_freq,
                                                                 shape=scan_type, dtype_name=dn)
                                if scan_type == 'scalar':
                                    grad_fn = _make_grad_fn(parallel_scan_scalar_phase, argnums=2)
                                else:
                                    grad_fn = _make_grad_fn(parallel_scan_3x3_phase, argnums=2)

                                r_bwd = benchmark(grad_fn, inputs,
                                                  warmup=args.warmup, trials=args.trials,
                                                  label=f"{dn} par+bwd")
                                sp = r_c64_bwd['mean_ms'] / r_bwd['mean_ms']
                                print(f"    {dn} parallel+bwd:  {r_bwd['mean_ms']:.2f} ms ({sp:.2f}x)")

    # =========================================================================
    # Profiling (--profile)
    # =========================================================================
    if args.profile:
        print()
        print("=" * 80)
        print("PROFILING")
        print("=" * 80)

        # Use the first config for profiling
        C_prof, H_prof = embed_dims[0], image_sizes[0]
        W_freq_prof = H_prof // 2 + 1
        T_profile = seq_lens[-1]
        K_log, U_log = make_scan_inputs(B, T_profile, C_prof, H_prof, W_freq_prof,
                                        dtype=jnp.complex64)

        print(f"\nJAX Profiler Traces (T={T_profile}, C={C_prof}, H={H_prof}):")
        profile_mode(parallel_scan, (K_log, U_log), "parallel", args.output_dir)
        profile_mode(sequential_scan, (K_log, U_log), "seq_linear", args.output_dir)
        profile_mode(sequential_nonlinear_scan, (K_log, U_log), "seq_nonlinear", args.output_dir)

        print(f"\nOperation Counts (T={T_profile}):")
        count_ops(parallel_scan, (K_log, U_log), "Parallel")
        count_ops(sequential_scan, (K_log, U_log), "Seq Linear")
        count_ops(sequential_nonlinear_scan, (K_log, U_log), "Seq Nonlinear")

        # Memory stats
        print(f"\nMemory:")
        mem = get_memory_usage()
        if mem > 0:
            print(f"  Peak GPU memory: {mem:.1f} MB")
        else:
            print("  GPU memory stats not available (CPU backend or stats disabled)")

    # =========================================================================
    # Dtype comparison (--dtype_compare)
    # =========================================================================
    if args.dtype_compare:
        print()
        print("=" * 80)
        print("DTYPE COMPARISON: complex64 vs float32")
        print("=" * 80)

        C_dt, H_dt = embed_dims[0], image_sizes[0]
        W_freq_dt = H_dt // 2 + 1
        T_dtype = seq_lens[-1]
        print(f"\nUsing T={T_dtype}, B={B}, C={C_dt}, H={H_dt}, W_freq={W_freq_dt}")

        # Complex64 (GOOM)
        K_c, U_c = make_scan_inputs(B, T_dtype, C_dt, H_dt, W_freq_dt, dtype=jnp.complex64)
        r_complex = benchmark(parallel_scan, (K_c, U_c),
                              warmup=args.warmup, trials=args.trials,
                              label="complex64")

        # Float32 (same shapes, real-valued, no GOOM)
        K_f, U_f = make_scan_inputs(B, T_dtype, C_dt, H_dt, W_freq_dt, dtype=jnp.float32)
        r_float = benchmark(parallel_scan_float32, (K_f, U_f),
                            warmup=args.warmup, trials=args.trials,
                            label="float32")

        overhead = r_complex['mean_ms'] / r_float['mean_ms'] if r_float['mean_ms'] > 0 else float('inf')
        compile_ratio = r_complex['compile_s'] / r_float['compile_s'] if r_float['compile_s'] > 0 else float('inf')
        print(f"\n  complex64: {r_complex['mean_ms']:.2f} ms  (compile: {r_complex['compile_s']:.1f}s)")
        print(f"  float32:   {r_float['mean_ms']:.2f} ms  (compile: {r_float['compile_s']:.1f}s)")
        print(f"  Execution overhead:  {overhead:.2f}x from complex arithmetic + GOOM")
        print(f"  Compile overhead:    {compile_ratio:.1f}x")

    # =========================================================================
    # Split GOOM benchmark (--split_goom)
    # =========================================================================
    if args.split_goom:
        print()
        print("=" * 80)
        print("SPLIT GOOM: (magnitude + sign) vs complex64")
        print("=" * 80)
        print("Eliminates complex64 entirely. Magnitude as float32 or bfloat16,")
        print("sign as boolean. Same math, no complex arithmetic or GOOM custom VJPs.")

        C_sg, H_sg = embed_dims[0], image_sizes[0]
        W_freq_sg = H_sg // 2 + 1

        for T_sg in seq_lens:
            print(f"\n--- T={T_sg}, B={B}, C={C_sg}, H={H_sg}, W_freq={W_freq_sg} ---")

            # Complex64 GOOM baseline (parallel + sequential)
            K_c, U_c = make_scan_inputs(B, T_sg, C_sg, H_sg, W_freq_sg, dtype=jnp.complex64)
            r_c64_par = benchmark(parallel_scan, (K_c, U_c),
                                  warmup=args.warmup, trials=args.trials,
                                  label="complex64 parallel")
            r_c64_seq = benchmark(sequential_scan, (K_c, U_c),
                                  warmup=args.warmup, trials=args.trials,
                                  label="complex64 sequential")

            # Split GOOM float32
            Km32, Ks32, Um32, Us32 = make_split_inputs(
                B, T_sg, C_sg, H_sg, W_freq_sg, mag_dtype=jnp.float32)
            r_f32_par = benchmark(parallel_scan_split, (Km32, Ks32, Um32, Us32),
                                  warmup=args.warmup, trials=args.trials,
                                  label="split-f32 parallel")
            r_f32_seq = benchmark(sequential_scan_split, (Km32, Ks32, Um32, Us32),
                                  warmup=args.warmup, trials=args.trials,
                                  label="split-f32 sequential")

            # Split GOOM bfloat16
            Km16, Ks16, Um16, Us16 = make_split_inputs(
                B, T_sg, C_sg, H_sg, W_freq_sg, mag_dtype=jnp.bfloat16)
            r_bf16_par = benchmark(parallel_scan_split, (Km16, Ks16, Um16, Us16),
                                   warmup=args.warmup, trials=args.trials,
                                   label="split-bf16 parallel")
            r_bf16_seq = benchmark(sequential_scan_split, (Km16, Ks16, Um16, Us16),
                                   warmup=args.warmup, trials=args.trials,
                                   label="split-bf16 sequential")

            # Forward + backward for split
            if include_backward:
                grad_c64_par = _make_grad_fn(parallel_scan)
                grad_f32_par = _make_grad_fn(parallel_scan_split)
                grad_f32_seq = _make_grad_fn(sequential_scan_split)

                r_c64_par_bwd = benchmark(grad_c64_par, (K_c, U_c),
                                          warmup=args.warmup, trials=args.trials,
                                          label="complex64 parallel+bwd")
                r_f32_par_bwd = benchmark(grad_f32_par, (Km32, Ks32, Um32, Us32),
                                          warmup=args.warmup, trials=args.trials,
                                          label="split-f32 parallel+bwd")
                r_f32_seq_bwd = benchmark(grad_f32_seq, (Km32, Ks32, Um32, Us32),
                                          warmup=args.warmup, trials=args.trials,
                                          label="split-f32 sequential+bwd")

            # Print comparison table
            print(f"\n  {'Mode':<28} | {'Fwd (ms)':>10} | {'Compile':>8} |", end="")
            if include_backward:
                print(f" {'Fwd+Bwd (ms)':>12} | {'Compile':>8} |", end="")
            print()
            print(f"  {'-'*28}-+-{'-'*10}-+-{'-'*8}-+", end="")
            if include_backward:
                print(f"-{'-'*12}-+-{'-'*8}-+", end="")
            print()

            rows = [
                ("complex64 parallel",   r_c64_par,  r_c64_par_bwd if include_backward else None),
                ("complex64 sequential", r_c64_seq,  None),
                ("split-f32 parallel",   r_f32_par,  r_f32_par_bwd if include_backward else None),
                ("split-f32 sequential", r_f32_seq,  r_f32_seq_bwd if include_backward else None),
                ("split-bf16 parallel",  r_bf16_par, None),
                ("split-bf16 sequential",r_bf16_seq, None),
            ]
            for name, r_fwd, r_bwd in rows:
                line = f"  {name:<28} | {r_fwd['mean_ms']:>7.2f} ms | {r_fwd['compile_s']:>6.1f}s |"
                if include_backward:
                    if r_bwd:
                        line += f" {r_bwd['mean_ms']:>9.2f} ms | {r_bwd['compile_s']:>6.1f}s |"
                    else:
                        line += f" {'—':>9}    | {'—':>6}  |"
                print(line)

            # Speedup summary
            print(f"\n  Speedup vs complex64 parallel (forward):")
            print(f"    split-f32 parallel:    {r_c64_par['mean_ms']/r_f32_par['mean_ms']:.2f}x")
            print(f"    split-f32 sequential:  {r_c64_par['mean_ms']/r_f32_seq['mean_ms']:.2f}x")
            print(f"    split-bf16 parallel:   {r_c64_par['mean_ms']/r_bf16_par['mean_ms']:.2f}x")
            print(f"    split-bf16 sequential: {r_c64_par['mean_ms']/r_bf16_seq['mean_ms']:.2f}x")
            print(f"    Compile: {r_c64_par['compile_s']:.1f}s → {r_f32_par['compile_s']:.1f}s "
                  f"({r_c64_par['compile_s']/max(r_f32_par['compile_s'], 0.01):.0f}x faster)")

    # =========================================================================
    # Optimization analysis (always)
    # =========================================================================
    print_optimization_analysis()


if __name__ == '__main__':
    main()
