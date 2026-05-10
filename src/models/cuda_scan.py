"""
CUDA parallel scan kernel for complex scalar SSM recurrence.

Replaces jax.lax.associative_scan for the recurrence h_t = A_t * h_{t-1} + U_t
(complex-valued, per channel per frequency bin) with a fused multi-block
Blelloch scan kernel written in CUDA.

Forward pass: CUDA kernel (Blelloch parallel scan)
Backward pass: JAX-native associative_scan (reverse adjoint scan)

Requires JAX >= 0.7.0 (uses jax.ffi API).

Build:
    cd src/models/cuda
    nvcc -shared -o libssm_scan.so ssm_fwd_large_complex.cu \
         -I$CUDA_HOME/include --compiler-options '-fPIC'
"""

import os
import ctypes
import struct
from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp
import jax.ffi

# ============================================================================
# Constants (must match the .cu file)
# ============================================================================
CHUNK_SIZE = 1024

# ============================================================================
# Library loading
# ============================================================================
_LIB_PATH = os.path.join(os.path.dirname(__file__), "cuda", "libssm_scan.so")
_lib_loaded = False


def _ensure_lib():
    """Load the shared library and register FFI targets (once)."""
    global _lib_loaded
    if _lib_loaded:
        return
    if not os.path.exists(_LIB_PATH):
        raise RuntimeError(
            f"CUDA scan library not found at {_LIB_PATH}. "
            "Build it with:\n"
            "  cd src/models/cuda && nvcc -shared -o libssm_scan.so "
            "ssm_fwd_large_complex.cu -I$CUDA_HOME/include "
            "--compiler-options '-fPIC'"
        )
    lib = ctypes.cdll.LoadLibrary(_LIB_PATH)
    jax.ffi.register_ffi_target(
        "cuda_scan_complex_fwd",
        jax.ffi.pycapsule(lib.cuda_scan_complex_fwd),
        platform="CUDA",
        api_version=0,
    )
    _lib_loaded = True


# ============================================================================
# Helpers
# ============================================================================
def _scan_geometry(T: int):
    """Compute M (next power-of-2 >= T), num_chunks, and num_levels."""
    M = 1
    while M < T:
        M *= 2
    num_chunks = (M + CHUNK_SIZE - 1) // CHUNK_SIZE
    num_levels = 0
    n = M
    while n > CHUNK_SIZE:
        n = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
        num_levels += 1
    if num_levels == 0 and M > 1:
        num_levels = 1
    return M, num_chunks, num_levels


def _pack_descriptor(B_size, T, D, M, num_chunks, num_levels):
    """Pack scan parameters matching the C ScanDescriptor struct."""
    return struct.pack("llllli", B_size, T, D, M, num_chunks, num_levels)


def _scan_op(carry_i, carry_j):
    """Associative operator for h_t = A_t * h_{t-1} + U_t in complex space."""
    A_i, u_i = carry_i
    A_j, u_j = carry_j
    return A_j * A_i, A_j * u_i + u_j


# ============================================================================
# Raw CUDA forward (no autodiff)
# ============================================================================
def _cuda_fwd_raw(A_re, A_im, U_re, U_im):
    """Call the CUDA kernel. Inputs/outputs are split float32 (B, T, D)."""
    _ensure_lib()

    B_size, T, D = A_re.shape
    M, num_chunks, num_levels = _scan_geometry(T)
    opaque = _pack_descriptor(B_size, T, D, M, num_chunks, num_levels)

    h_re, h_im = jax.ffi.ffi_call(
        "cuda_scan_complex_fwd",
        (
            jax.ShapeDtypeStruct((B_size, T, D), jnp.float32),
            jax.ShapeDtypeStruct((B_size, T, D), jnp.float32),
        ),
        custom_call_api_version=1,
        legacy_backend_config=opaque,
    )(A_re, A_im, U_re, U_im)

    return h_re, h_im


# ============================================================================
# Differentiable public API
# ============================================================================
@jax.custom_vjp
def cuda_complex_scan(A, U):
    """
    CUDA-accelerated parallel scan for complex scalar SSM.

    Computes h_t = A_t * h_{t-1} + U_t for all t in parallel.

    Args:
        A: complex64 (B, T, ...) — per-timestep decay
        U: complex64 (B, T, ...) — per-timestep input

    Returns:
        h: complex64 (B, T, ...) — output states at all timesteps
    """
    orig_shape = A.shape
    B_dim, T = orig_shape[0], orig_shape[1]
    spatial = orig_shape[2:]

    D = 1
    for s in spatial:
        D *= s
    A_flat = A.reshape(B_dim, T, D)
    U_flat = U.reshape(B_dim, T, D)

    A_re = A_flat.real.astype(jnp.float32)
    A_im = A_flat.imag.astype(jnp.float32)
    U_re = U_flat.real.astype(jnp.float32)
    U_im = U_flat.imag.astype(jnp.float32)

    h_re, h_im = _cuda_fwd_raw(A_re, A_im, U_re, U_im)

    # h = (h_re + 1j * h_im).astype(jnp.complex64)

    # Handle inclusive conversion
    h_excl = (h_re + 1j * h_im).astype(jnp.complex64)
    h = A_flat * h_excl.reshape(B_dim, T, D) + U_flat
    
    return h.reshape(orig_shape)


def _cuda_scan_fwd(A, U):
    """Forward pass: run CUDA kernel, save residuals for backward."""
    h = cuda_complex_scan(A, U)
    return h, (A, h)


def _cuda_scan_bwd(res, g):
    """
    Backward pass: reverse adjoint scan using JAX associative_scan.

    The adjoint satisfies:
        lambda[T-1] = g[T-1]
        lambda[t]   = g[t] + conj(A[t+1]) * lambda[t+1]

    This is itself a linear scan running in reverse, computed by:
    flipping the sequence, scanning forward, then flipping back.

    Gradients:
        dU[t] = lambda[t]
        dA[t] = lambda[t] * conj(h[t-1])    (with h[-1] = 0)
    """
    A, h = res

    # Build shifted A: A_shift[t] = A[t+1], with identity padding at end
    A_shift = jnp.concatenate(
        [A[:, 1:], jnp.ones_like(A[:, :1])], axis=1
    )

    # Reverse scan: flip, scan forward, flip back
    A_bwd = jnp.conj(A_shift)[:, ::-1]
    g_rev = g[:, ::-1]

    _, adjoint_rev = jax.lax.associative_scan(
        _scan_op, (A_bwd, g_rev), axis=1
    )
    adjoint = adjoint_rev[:, ::-1]

    # dL/dU = adjoint
    dU = adjoint

    # dL/dA[t] = adjoint[t] * conj(h[t-1]), with h[-1] = 0
    h_prev = jnp.concatenate(
        [jnp.zeros_like(h[:, :1]), h[:, :-1]], axis=1
    )
    dA = adjoint * jnp.conj(h_prev)

    return dA, dU


cuda_complex_scan.defvjp(_cuda_scan_fwd, _cuda_scan_bwd)


# ============================================================================
# Real-valued convenience wrapper
# ============================================================================
def cuda_real_scan(A, U):
    """
    CUDA-accelerated parallel scan for real-valued scalar SSM.

    Args:
        A: float32 (B, T, ...) — per-timestep decay
        U: float32 (B, T, ...) — per-timestep input

    Returns:
        h: float32 (B, T, ...) — output states
    """
    A_c = A.astype(jnp.complex64)
    U_c = U.astype(jnp.complex64)
    return cuda_complex_scan(A_c, U_c).real