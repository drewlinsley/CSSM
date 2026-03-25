"""
Pallas GPU kernel for fused complex scalar scan.

Replaces jax.lax.associative_scan for the recurrence h_t = A_t * h_{t-1} + X_t
(complex-valued, per channel per frequency bin) with a single fused kernel that
keeps the carry in GPU registers.

For T=8-64, the sequential fused kernel is faster than log2(T) separate XLA
kernel launches with intermediate HBM round-trips.

Reference: RecurrentGemma's Pallas kernel (sequential fori_loop over T,
parallelism via grid over batch/spatial dims).
"""

import functools

import jax
import jax.numpy as jnp
from jax import core
from jax.experimental import pallas as pl

# Tile size for the inner (D) dimension. Each program instance handles TILE_D
# elements. 128 is a good default for GPU (fits in registers, good occupancy).
TILE_D = 128


# =============================================================================
# Forward kernel
# =============================================================================

def _fwd_kernel(
    # Input refs
    a_re_ref, a_im_ref, x_re_ref, x_im_ref,
    h0_re_ref, h0_im_ref,
    # Output refs
    y_re_ref, y_im_ref, h_last_re_ref, h_last_im_ref,
    # Static args
    T: int, has_h0: bool,
):
    """Sequential scan kernel: h_t = a_t * h_{t-1} + x_t, complex via (re,im) split.

    Grid: (B, D // TILE_D). Each program processes one batch element and one
    tile of the flattened spatial dimension.

    Inputs are bf16, carry is f32, outputs are bf16.
    """
    # Initialize carry in f32
    if has_h0:
        h_re = h0_re_ref[0, :].astype(jnp.float32)
        h_im = h0_im_ref[0, :].astype(jnp.float32)
    else:
        h_re = jnp.zeros(TILE_D, dtype=jnp.float32)
        h_im = jnp.zeros(TILE_D, dtype=jnp.float32)

    # Sequential scan over T
    def body(t, carry):
        h_re, h_im = carry

        # Load a_t and x_t, promote to f32
        ar = a_re_ref[0, t, :].astype(jnp.float32)
        ai = a_im_ref[0, t, :].astype(jnp.float32)
        xr = x_re_ref[0, t, :].astype(jnp.float32)
        xi = x_im_ref[0, t, :].astype(jnp.float32)

        # Complex multiply-add: h = a * h_prev + x
        new_h_re = ar * h_re - ai * h_im + xr
        new_h_im = ar * h_im + ai * h_re + xi

        # Store output as bf16
        y_re_ref[0, t, :] = new_h_re.astype(jnp.bfloat16)
        y_im_ref[0, t, :] = new_h_im.astype(jnp.bfloat16)

        return new_h_re, new_h_im

    h_re, h_im = jax.lax.fori_loop(0, T, body, (h_re, h_im))

    # Store final hidden state (f32)
    h_last_re_ref[0, :] = h_re
    h_last_im_ref[0, :] = h_im


# =============================================================================
# Backward kernel
# =============================================================================

def _bwd_kernel(
    # Input refs
    dy_re_ref, dy_im_ref, a_re_ref, a_im_ref,
    dh_last_re_ref, dh_last_im_ref,
    # Output refs
    dx_re_ref, dx_im_ref, dh0_re_ref, dh0_im_ref,
    # Static args
    T: int, has_h0: bool,
):
    """Reverse scan for backward pass.

    The gradient of h_t = A_t * h_{t-1} + X_t w.r.t. X is:
        dX_t = dh_t  (accumulated gradient at time t)

    The gradient propagates backward: dh_{t-1} += conj(A_t) * dh_t

    Grid: same as forward (B, D // TILE_D).
    """
    # Initialize reverse carry with dh_last (gradient from loss w.r.t. final state)
    h_re = dh_last_re_ref[0, :].astype(jnp.float32)
    h_im = dh_last_im_ref[0, :].astype(jnp.float32)

    # Reverse scan: t = T-1, T-2, ..., 0
    def body(t_fwd, carry):
        h_re, h_im = carry
        t = T - 1 - t_fwd  # reverse index

        # Accumulate output gradient
        dyr = dy_re_ref[0, t, :].astype(jnp.float32)
        dyi = dy_im_ref[0, t, :].astype(jnp.float32)
        h_re = h_re + dyr
        h_im = h_im + dyi

        # Store dx_t = accumulated gradient
        dx_re_ref[0, t, :] = h_re.astype(jnp.bfloat16)
        dx_im_ref[0, t, :] = h_im.astype(jnp.bfloat16)

        # Propagate backward via conj(A_t): (ar, -ai) * (hr, hi)
        # = (ar*hr + ai*hi, ar*hi - ai*hr)
        ar = a_re_ref[0, t, :].astype(jnp.float32)
        ai = a_im_ref[0, t, :].astype(jnp.float32)
        new_h_re = ar * h_re + ai * h_im
        new_h_im = ar * h_im - ai * h_re

        return new_h_re, new_h_im

    h_re, h_im = jax.lax.fori_loop(0, T, body, (h_re, h_im))

    # Final carry = dh0
    if has_h0:
        dh0_re_ref[0, :] = h_re
        dh0_im_ref[0, :] = h_im


# =============================================================================
# Pallas call wrappers
# =============================================================================

def _run_fwd_kernel(a_re, a_im, x_re, x_im, h0_re, h0_im, has_h0):
    """Launch the forward Pallas kernel."""
    B, T, D = a_re.shape
    assert D % TILE_D == 0, f"D={D} must be multiple of TILE_D={TILE_D}"
    n_tiles = D // TILE_D

    grid = (B, n_tiles)

    # BlockSpecs: each program gets (1, T, TILE_D) for temporal arrays,
    # (1, TILE_D) for state arrays.
    def temporal_index(b, d_tile):
        return (b, 0, d_tile * TILE_D)

    def state_index(b, d_tile):
        return (b, d_tile * TILE_D)

    temporal_spec = pl.BlockSpec(block_shape=(1, T, TILE_D), index_map=temporal_index)
    state_spec = pl.BlockSpec(block_shape=(1, TILE_D), index_map=state_index)

    kernel = functools.partial(_fwd_kernel, T=T, has_h0=has_h0)

    y_re, y_im, h_last_re, h_last_im = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            temporal_spec,  # a_re
            temporal_spec,  # a_im
            temporal_spec,  # x_re
            temporal_spec,  # x_im
            state_spec,     # h0_re
            state_spec,     # h0_im
        ],
        out_specs=[
            temporal_spec,  # y_re
            temporal_spec,  # y_im
            state_spec,     # h_last_re
            state_spec,     # h_last_im
        ],
        out_shape=[
            jax.ShapeDtypeStruct((B, T, D), jnp.bfloat16),   # y_re
            jax.ShapeDtypeStruct((B, T, D), jnp.bfloat16),   # y_im
            jax.ShapeDtypeStruct((B, D), jnp.float32),       # h_last_re
            jax.ShapeDtypeStruct((B, D), jnp.float32),       # h_last_im
        ],
        interpret=False,
        name='pallas_scan_fwd',
    )(a_re, a_im, x_re, x_im, h0_re, h0_im)

    return y_re, y_im, h_last_re, h_last_im


def _run_bwd_kernel(dy_re, dy_im, a_re, a_im, dh_last_re, dh_last_im, has_h0):
    """Launch the backward Pallas kernel."""
    B, T, D = dy_re.shape
    n_tiles = D // TILE_D
    grid = (B, n_tiles)

    def temporal_index(b, d_tile):
        return (b, 0, d_tile * TILE_D)

    def state_index(b, d_tile):
        return (b, d_tile * TILE_D)

    temporal_spec = pl.BlockSpec(block_shape=(1, T, TILE_D), index_map=temporal_index)
    state_spec = pl.BlockSpec(block_shape=(1, TILE_D), index_map=state_index)

    kernel = functools.partial(_bwd_kernel, T=T, has_h0=has_h0)

    out_shapes = [
        jax.ShapeDtypeStruct((B, T, D), jnp.bfloat16),  # dx_re
        jax.ShapeDtypeStruct((B, T, D), jnp.bfloat16),  # dx_im
    ]
    out_specs_list = [temporal_spec, temporal_spec]

    if has_h0:
        out_shapes.append(jax.ShapeDtypeStruct((B, D), jnp.float32))  # dh0_re
        out_shapes.append(jax.ShapeDtypeStruct((B, D), jnp.float32))  # dh0_im
        out_specs_list.append(state_spec)
        out_specs_list.append(state_spec)

    results = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            temporal_spec,  # dy_re
            temporal_spec,  # dy_im
            temporal_spec,  # a_re
            temporal_spec,  # a_im
            state_spec,     # dh_last_re
            state_spec,     # dh_last_im
        ],
        out_specs=out_specs_list,
        out_shape=out_shapes,
        interpret=False,
        name='pallas_scan_bwd',
    )(dy_re, dy_im, a_re, a_im, dh_last_re, dh_last_im)

    if has_h0:
        dx_re, dx_im, dh0_re, dh0_im = results
    else:
        dx_re, dx_im = results
        dh0_re = None
        dh0_im = None

    return dx_re, dx_im, dh0_re, dh0_im


# =============================================================================
# custom_vjp wrapper
# =============================================================================

@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def pallas_complex_scan(a_re, a_im, x_re, x_im, has_h0, h0_re=None, h0_im=None):
    """Complex scalar scan with Pallas kernels and custom backward.

    Args:
        a_re, a_im: (B, T, D) bf16 real/imag parts of transition A
        x_re, x_im: (B, T, D) bf16 real/imag parts of input X
        has_h0: bool (static) — whether h0 is provided
        h0_re, h0_im: (B, D) f32 initial state (or None/dummy)

    Returns:
        y_re, y_im: (B, T, D) bf16 real/imag parts of output
    """
    B, D = a_re.shape[0], a_re.shape[2]
    if not has_h0:
        h0_re = jnp.zeros((B, D), dtype=jnp.float32)
        h0_im = jnp.zeros((B, D), dtype=jnp.float32)

    y_re, y_im, _, _ = _run_fwd_kernel(a_re, a_im, x_re, x_im, h0_re, h0_im, has_h0)
    return y_re, y_im


def _pallas_scan_fwd(a_re, a_im, x_re, x_im, has_h0, h0_re=None, h0_im=None):
    """Forward pass: run kernel, save residuals."""
    B, D = a_re.shape[0], a_re.shape[2]
    if not has_h0:
        h0_re = jnp.zeros((B, D), dtype=jnp.float32)
        h0_im = jnp.zeros((B, D), dtype=jnp.float32)

    y_re, y_im, h_last_re, h_last_im = _run_fwd_kernel(
        a_re, a_im, x_re, x_im, h0_re, h0_im, has_h0)

    residuals = (a_re, a_im, y_re, y_im, h0_re, h0_im)
    return (y_re, y_im), residuals


def _pallas_scan_bwd(has_h0, residuals, g):
    """Backward pass: reverse scan for dx, element-wise for da."""
    dy_re, dy_im = g
    a_re, a_im, y_re, y_im, h0_re, h0_im = residuals

    B, T, D = a_re.shape

    # dh_last = 0 (no gradient flows from final hidden state in this interface)
    dh_last_re = jnp.zeros((B, D), dtype=jnp.float32)
    dh_last_im = jnp.zeros((B, D), dtype=jnp.float32)

    # Run backward kernel to get dx
    dx_re, dx_im, dh0_re, dh0_im = _run_bwd_kernel(
        dy_re, dy_im, a_re, a_im, dh_last_re, dh_last_im, has_h0)

    # Gradient for A: da_t = dx_t * conj(y_{t-1})
    # y_shifted = [h0, y[:, :-1]]  (previous hidden state)
    if has_h0:
        y_prev_re = jnp.concatenate([h0_re[:, None, :].astype(jnp.bfloat16),
                                      y_re[:, :-1, :]], axis=1)
        y_prev_im = jnp.concatenate([h0_im[:, None, :].astype(jnp.bfloat16),
                                      y_im[:, :-1, :]], axis=1)
    else:
        zeros_t = jnp.zeros((B, 1, D), dtype=jnp.bfloat16)
        y_prev_re = jnp.concatenate([zeros_t, y_re[:, :-1, :]], axis=1)
        y_prev_im = jnp.concatenate([zeros_t, y_im[:, :-1, :]], axis=1)

    # da = dx * conj(y_prev) = (dx_re + j*dx_im) * (y_prev_re - j*y_prev_im)
    # da_re = dx_re * y_prev_re + dx_im * y_prev_im
    # da_im = dx_im * y_prev_re - dx_re * y_prev_im
    dxr = dx_re.astype(jnp.float32)
    dxi = dx_im.astype(jnp.float32)
    ypr = y_prev_re.astype(jnp.float32)
    ypi = y_prev_im.astype(jnp.float32)
    da_re = (dxr * ypr + dxi * ypi).astype(jnp.bfloat16)
    da_im = (dxi * ypr - dxr * ypi).astype(jnp.bfloat16)

    if has_h0:
        return da_re, da_im, dx_re, dx_im, dh0_re, dh0_im
    else:
        return da_re, da_im, dx_re, dx_im, None, None


pallas_complex_scan.defvjp(_pallas_scan_fwd, _pallas_scan_bwd)


# =============================================================================
# Public API
# =============================================================================

def pallas_scalar_scan(A, X, h0=None):
    """Drop-in replacement for associative_scan scalar recurrence.

    Computes h_t = A_t * h_{t-1} + X_t using a fused Pallas GPU kernel.

    Args:
        A: complex64 (B, T, C, H, W_freq) — transition coefficients
        X: complex64 (B, T, C, H, W_freq) — inputs
        h0: complex64 (B, C, H, W_freq) — optional initial state

    Returns:
        Y: complex64 (B, T, C, H, W_freq) — all hidden states
    """
    shape_5d = A.shape  # (B, T, C, H, W_freq)
    B, T = shape_5d[0], shape_5d[1]
    D = 1
    for s in shape_5d[2:]:
        D *= s

    # Flatten spatial dims: (B, T, C, H, W_freq) -> (B, T, D)
    A_flat = A.reshape(B, T, D).astype(jnp.complex64)
    X_flat = X.reshape(B, T, D).astype(jnp.complex64)

    # Pad D to multiple of TILE_D
    pad_d = (TILE_D - D % TILE_D) % TILE_D
    if pad_d > 0:
        # Pad A with 1+0j (identity transition), X with 0 (zero input)
        A_flat = jnp.pad(A_flat, ((0, 0), (0, 0), (0, pad_d)),
                         constant_values=1.0 + 0j)
        X_flat = jnp.pad(X_flat, ((0, 0), (0, 0), (0, pad_d)),
                         constant_values=0.0 + 0j)
    D_padded = D + pad_d

    # Split complex -> (re_bf16, im_bf16)
    a_re = A_flat.real.astype(jnp.bfloat16)
    a_im = A_flat.imag.astype(jnp.bfloat16)
    x_re = X_flat.real.astype(jnp.bfloat16)
    x_im = X_flat.imag.astype(jnp.bfloat16)

    has_h0 = h0 is not None
    if has_h0:
        h0_flat = h0.reshape(B, D).astype(jnp.complex64)
        if pad_d > 0:
            h0_flat = jnp.pad(h0_flat, ((0, 0), (0, pad_d)),
                              constant_values=0.0 + 0j)
        h0_re = h0_flat.real.astype(jnp.float32)
        h0_im = h0_flat.imag.astype(jnp.float32)
    else:
        h0_re = None
        h0_im = None

    # Run fused kernel
    y_re, y_im = pallas_complex_scan(a_re, a_im, x_re, x_im, has_h0, h0_re, h0_im)

    # Convert back to complex64
    Y_flat = (y_re.astype(jnp.float32) + 1j * y_im.astype(jnp.float32)).astype(jnp.complex64)

    # Strip padding and reshape back to 5D
    if pad_d > 0:
        Y_flat = Y_flat[:, :, :D]
    return Y_flat.reshape(shape_5d)


# =============================================================================
# 2×2 Matrix Scan: h_t = M_t @ h_{t-1} + u_t  (jax.lax.scan)
# =============================================================================
# Uses jax.lax.scan for the sequential recurrence, which XLA compiles into an
# efficient fused GPU loop. This avoids Pallas Triton limitations (sub-range
# ref access, lax.slice, power-of-2 constraints) while providing the same
# O(T) sequential scan advantage over O(T log T) associative scan.
#
# JAX autodiff handles the backward pass automatically.

def pallas_matrix_2x2_scan(M, U, h0=None):
    """Sequential 2×2 matrix scan via jax.lax.scan.

    Computes h_t = M_t @ h_{t-1} + u_t, returning all hidden states.
    XLA JIT-compiles this into an efficient fused GPU loop.

    Args:
        M: complex64 (B, T, ..., 2, 2) — transition matrices
        U: complex64 (B, T, ..., 2) — input vectors
        h0: complex64 (B, ..., 2) — optional initial state

    Returns:
        Y: complex64 (B, T, ..., 2) — all hidden states
    """
    shape_U = U.shape    # (B, T, ..., 2)
    B, T = M.shape[0], M.shape[1]
    spatial = M.shape[2:-2]
    D = 1
    for s in spatial:
        D *= s

    # Flatten spatial: (B, T, D, 2, 2) and (B, T, D, 2)
    M_flat = M.reshape(B, T, D, 2, 2).astype(jnp.complex64)
    U_flat = U.reshape(B, T, D, 2).astype(jnp.complex64)

    # Transpose to (T, B, D, ...) for scan along axis 0
    M_scan = jnp.moveaxis(M_flat, 1, 0)  # (T, B, D, 2, 2)
    U_scan = jnp.moveaxis(U_flat, 1, 0)  # (T, B, D, 2)

    if h0 is not None:
        h_init = h0.reshape(B, D, 2).astype(jnp.complex64)
    else:
        h_init = jnp.zeros((B, D, 2), dtype=jnp.complex64)

    def scan_body(h, inputs):
        M_t, U_t = inputs  # M_t: (B, D, 2, 2), U_t: (B, D, 2)
        # h_new = M_t @ h + U_t  via einsum (complex matvec per element)
        h_new = jnp.einsum('...ij,...j->...i', M_t, h) + U_t
        return h_new, h_new

    _, Y_scan = jax.lax.scan(scan_body, h_init, (M_scan, U_scan))
    # Y_scan: (T, B, D, 2)

    Y_flat = jnp.moveaxis(Y_scan, 0, 1)  # (B, T, D, 2)
    return Y_flat.reshape(shape_U)
