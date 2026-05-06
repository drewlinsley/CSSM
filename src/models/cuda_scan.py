"""
CUDA parallel scan kernel for complex scalar SSM recurrence.

Replaces jax.lax.associative_scan for the recurrence h_t = A_t * h_{t-1} + U_t
(complex-valued, per channel per frequency bin) with a fused multi-block
Blelloch scan kernel written in CUDA.

Uses the same split real/imaginary convention as linear_split_scalar_scan_op:
complex64 inputs are split into (float32 real, float32 imag) pairs.

Integration:
    1. Build the shared library:
       nvcc -shared -o libssm_scan.so ssm_fwd_large_complex.cu \
            -I$CUDA_HOME/include --compiler-options '-fPIC'

    2. Set scan_mode='cuda' in the CSSM model config.

Requires JAX >= 0.7.0 (uses jax.ffi API).
"""

import os
import ctypes
import struct
from typing import Optional

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
    # Load the .so and register the entry point for XLA custom call
    lib = ctypes.cdll.LoadLibrary(_LIB_PATH)
    jax.ffi.register_ffi_target(
        "cuda_scan_complex_fwd",
        jax.ffi.pycapsule(lib.cuda_scan_complex_fwd),
        platform="CUDA",
        api_version=0,
    )
    _lib_loaded = True


# ============================================================================
# Helper: compute scan geometry
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
    """Pack scan parameters into an opaque byte string matching ScanDescriptor.

    Must match the C struct layout:
        struct ScanDescriptor {
            long B_size, T, D, M, num_chunks;
            int  num_levels;
        };
    """
    # 'l' = long (8 bytes on 64-bit), 'i' = int (4 bytes)
    return struct.pack("llllli", B_size, T, D, M, num_chunks, num_levels)


# ============================================================================
# Low-level JAX custom call wrapper
# ============================================================================
def _cuda_scan_complex_fwd_impl(A_re, A_im, U_re, U_im):
    """
    Call the CUDA kernel via XLA custom call.

    Args:
        A_re, A_im: (B, T, D) float32 — split complex decay
        U_re, U_im: (B, T, D) float32 — split complex input

    Returns:
        h_re, h_im: (B, T, D) float32 — split complex output states
    """
    _ensure_lib()

    B_size, T, D = A_re.shape
    M, num_chunks, num_levels = _scan_geometry(T)

    opaque = _pack_descriptor(B_size, T, D, M, num_chunks, num_levels)

    # JAX 0.7+ API: ffi_call returns a callable
    # api_version=0 for legacy custom call (void **buffers, const char *opaque)
    h_re, h_im = jax.ffi.ffi_call(
        "cuda_scan_complex_fwd",
        (
            jax.ShapeDtypeStruct((B_size, T, D), jnp.float32),  # h_re
            jax.ShapeDtypeStruct((B_size, T, D), jnp.float32),  # h_im
        ),
        custom_call_api_version=1,
        legacy_backend_config=opaque,
    )(A_re, A_im, U_re, U_im)

    return h_re, h_im


# ============================================================================
# Public API
# ============================================================================
def cuda_complex_scan(A: jnp.ndarray, U: jnp.ndarray,
                      h0: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    CUDA-accelerated parallel scan for complex scalar SSM.

    Computes h_t = A_t * h_{t-1} + U_t for all t in parallel using the
    Blelloch algorithm implemented in CUDA.

    Args:
        A: complex64 (B, T, ...) — per-timestep decay
        U: complex64 (B, T, ...) — per-timestep input (already gated by B_proj)
        h0: optional complex64 (B, ...) — initial state (not yet supported,
            use learned_init prepend trick if needed)

    Returns:
        h: complex64 (B, T, ...) — output states at all timesteps
    """
    if h0 is not None:
        raise NotImplementedError(
            "Initial state h0 not yet supported in CUDA scan. "
            "Use the learned_init prepend approach instead."
        )

    orig_shape = A.shape
    B_dim, T = orig_shape[0], orig_shape[1]
    spatial = orig_shape[2:]  # e.g. (C, H, W_freq)

    # Flatten spatial dims: (B, T, C, H, W_freq) -> (B, T, D)
    D = 1
    for s in spatial:
        D *= s
    A_flat = A.reshape(B_dim, T, D)
    U_flat = U.reshape(B_dim, T, D)

    # Split complex -> real/imag float32
    A_re = A_flat.real.astype(jnp.float32)
    A_im = A_flat.imag.astype(jnp.float32)
    U_re = U_flat.real.astype(jnp.float32)
    U_im = U_flat.imag.astype(jnp.float32)

    # Call kernel
    h_re, h_im = _cuda_scan_complex_fwd_impl(A_re, A_im, U_re, U_im)

    # Recombine to complex and unflatten
    h = (h_re + 1j * h_im).astype(jnp.complex64)
    h = h.reshape(orig_shape)

    return h


def cuda_real_scan(A: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
    """
    CUDA-accelerated parallel scan for real-valued scalar SSM.

    Convenience wrapper: passes real data as complex with zero imaginary part,
    returns real output.

    Args:
        A: float32 (B, T, ...) — per-timestep decay
        U: float32 (B, T, ...) — per-timestep input

    Returns:
        h: float32 (B, T, ...) — output states
    """
    A_c = A.astype(jnp.complex64)
    U_c = U.astype(jnp.complex64)
    return cuda_complex_scan(A_c, U_c).real