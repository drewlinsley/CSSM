"""
Log-space operations using GOOM primitives.

Direct port from https://github.com/hkozachkov/goom/tree/main/src/goom

These operations perform arithmetic in log-space for numerical stability,
using the max-subtraction trick for log-sum-exp style computations.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Union

from .goom import to_goom, from_goom, goom_real


def log_add_exp(log_x1: jnp.ndarray, log_x2: jnp.ndarray) -> jnp.ndarray:
    """
    Add two values in log-space: log(exp(log_x1) + exp(log_x2)).

    Uses the max-subtraction trick for numerical stability:
    1. Subtract the maximum (with stop_gradient to prevent gradient discontinuities)
    2. Convert from GOOM and add
    3. Convert back to GOOM
    4. Add the maximum back

    Args:
        log_x1: First value in GOOM representation
        log_x2: Second value in GOOM representation

    Returns:
        Sum in GOOM representation
    """
    # CRITICAL: stop_gradient prevents gradients from flowing through the max
    # selection, which avoids gradient discontinuities (per paper/GOOM reference)
    c = jax.lax.stop_gradient(jnp.maximum(goom_real(log_x1), goom_real(log_x2)))

    # Subtract max, convert to linear space, add, convert back
    x = from_goom(log_x1 - c) + from_goom(log_x2 - c)

    return to_goom(x) + c


def log_sum_exp(log_x: jnp.ndarray, axis: Optional[int] = None) -> jnp.ndarray:
    """
    Sum values in log-space along an axis: log(sum(exp(log_x), axis)).

    Args:
        log_x: Values in GOOM representation
        axis: Axis to sum along (None for all elements)

    Returns:
        Sum in GOOM representation
    """
    # CRITICAL: stop_gradient prevents gradient discontinuities from max
    c = jax.lax.stop_gradient(jnp.max(goom_real(log_x), axis=axis, keepdims=True))

    # Subtract max, convert to linear space, sum, convert back
    x = from_goom(log_x - c).sum(axis=axis)

    return to_goom(x) + jnp.squeeze(c, axis=axis)


def log_sub_exp(log_x1: jnp.ndarray, log_x2: jnp.ndarray) -> jnp.ndarray:
    """
    Subtract two values in log-space: log(exp(log_x1) - exp(log_x2)).

    Note: Result is only valid when exp(log_x1) > exp(log_x2).

    Args:
        log_x1: First value in GOOM representation (larger)
        log_x2: Second value in GOOM representation (smaller)

    Returns:
        Difference in GOOM representation
    """
    # CRITICAL: stop_gradient prevents gradient discontinuities from max
    c = jax.lax.stop_gradient(jnp.maximum(goom_real(log_x1), goom_real(log_x2)))
    x = from_goom(log_x1 - c) - from_goom(log_x2 - c)
    return to_goom(x) + c


def log_mul(log_x1: jnp.ndarray, log_x2: jnp.ndarray) -> jnp.ndarray:
    """
    Multiply two values in log-space: log(exp(log_x1) * exp(log_x2)).

    In log-space, multiplication is just addition.

    Args:
        log_x1: First value in GOOM representation
        log_x2: Second value in GOOM representation

    Returns:
        Product in GOOM representation
    """
    return log_x1 + log_x2


def log_div(log_x1: jnp.ndarray, log_x2: jnp.ndarray) -> jnp.ndarray:
    """
    Divide two values in log-space: log(exp(log_x1) / exp(log_x2)).

    In log-space, division is just subtraction.

    Args:
        log_x1: Numerator in GOOM representation
        log_x2: Denominator in GOOM representation

    Returns:
        Quotient in GOOM representation
    """
    return log_x1 - log_x2


def log_matmul_2x2(
    log_A: jnp.ndarray,
    log_B: jnp.ndarray
) -> jnp.ndarray:
    """
    2x2 matrix multiplication in log-space.

    Computes C = A @ B where all matrices are in GOOM representation.
    C[i,j] = log_sum_exp(A[i,:] + B[:,j])

    Args:
        log_A: First matrix in GOOM representation, shape (..., 2, 2)
        log_B: Second matrix in GOOM representation, shape (..., 2, 2)

    Returns:
        Product matrix in GOOM representation, shape (..., 2, 2)
    """
    # Extract elements
    a00, a01 = log_A[..., 0, 0], log_A[..., 0, 1]
    a10, a11 = log_A[..., 1, 0], log_A[..., 1, 1]
    b00, b01 = log_B[..., 0, 0], log_B[..., 0, 1]
    b10, b11 = log_B[..., 1, 0], log_B[..., 1, 1]

    # C[i,j] = LSE(A[i,0] + B[0,j], A[i,1] + B[1,j])
    c00 = log_add_exp(a00 + b00, a01 + b10)
    c01 = log_add_exp(a00 + b01, a01 + b11)
    c10 = log_add_exp(a10 + b00, a11 + b10)
    c11 = log_add_exp(a10 + b01, a11 + b11)

    # Stack back to (..., 2, 2)
    row0 = jnp.stack([c00, c01], axis=-1)
    row1 = jnp.stack([c10, c11], axis=-1)
    return jnp.stack([row0, row1], axis=-2)


def log_matvec_2x2(
    log_A: jnp.ndarray,
    log_v: jnp.ndarray
) -> jnp.ndarray:
    """
    2x2 matrix-vector multiplication in log-space.

    Computes y = A @ v where matrix and vector are in GOOM representation.
    y[i] = log_sum_exp(A[i,:] + v[:])

    Args:
        log_A: Matrix in GOOM representation, shape (..., 2, 2)
        log_v: Vector in GOOM representation, shape (..., 2)

    Returns:
        Result vector in GOOM representation, shape (..., 2)
    """
    a00, a01 = log_A[..., 0, 0], log_A[..., 0, 1]
    a10, a11 = log_A[..., 1, 0], log_A[..., 1, 1]
    v0, v1 = log_v[..., 0], log_v[..., 1]

    y0 = log_add_exp(a00 + v0, a01 + v1)
    y1 = log_add_exp(a10 + v0, a11 + v1)

    return jnp.stack([y0, y1], axis=-1)


def log_matmul_3x3(
    log_A: jnp.ndarray,
    log_B: jnp.ndarray
) -> jnp.ndarray:
    """
    3x3 matrix multiplication in log-space.

    Computes C = A @ B where all matrices are in GOOM representation.
    C[i,j] = log_sum_exp(A[i,:] + B[:,j])

    Used for BilinearOpponentCSSM with 3 states: [X, Y, Z].

    Args:
        log_A: First matrix in GOOM representation, shape (..., 3, 3)
        log_B: Second matrix in GOOM representation, shape (..., 3, 3)

    Returns:
        Product matrix in GOOM representation, shape (..., 3, 3)
    """
    # Extract elements
    a00, a01, a02 = log_A[..., 0, 0], log_A[..., 0, 1], log_A[..., 0, 2]
    a10, a11, a12 = log_A[..., 1, 0], log_A[..., 1, 1], log_A[..., 1, 2]
    a20, a21, a22 = log_A[..., 2, 0], log_A[..., 2, 1], log_A[..., 2, 2]

    b00, b01, b02 = log_B[..., 0, 0], log_B[..., 0, 1], log_B[..., 0, 2]
    b10, b11, b12 = log_B[..., 1, 0], log_B[..., 1, 1], log_B[..., 1, 2]
    b20, b21, b22 = log_B[..., 2, 0], log_B[..., 2, 1], log_B[..., 2, 2]

    # C[i,j] = LSE(A[i,0] + B[0,j], A[i,1] + B[1,j], A[i,2] + B[2,j])
    c00 = log_add_exp(log_add_exp(a00 + b00, a01 + b10), a02 + b20)
    c01 = log_add_exp(log_add_exp(a00 + b01, a01 + b11), a02 + b21)
    c02 = log_add_exp(log_add_exp(a00 + b02, a01 + b12), a02 + b22)

    c10 = log_add_exp(log_add_exp(a10 + b00, a11 + b10), a12 + b20)
    c11 = log_add_exp(log_add_exp(a10 + b01, a11 + b11), a12 + b21)
    c12 = log_add_exp(log_add_exp(a10 + b02, a11 + b12), a12 + b22)

    c20 = log_add_exp(log_add_exp(a20 + b00, a21 + b10), a22 + b20)
    c21 = log_add_exp(log_add_exp(a20 + b01, a21 + b11), a22 + b21)
    c22 = log_add_exp(log_add_exp(a20 + b02, a21 + b12), a22 + b22)

    # Stack back to (..., 3, 3)
    row0 = jnp.stack([c00, c01, c02], axis=-1)
    row1 = jnp.stack([c10, c11, c12], axis=-1)
    row2 = jnp.stack([c20, c21, c22], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def log_matvec_3x3(
    log_A: jnp.ndarray,
    log_v: jnp.ndarray
) -> jnp.ndarray:
    """
    3x3 matrix-vector multiplication in log-space.

    Computes y = A @ v where matrix and vector are in GOOM representation.
    y[i] = log_sum_exp(A[i,:] + v[:])

    Used for BilinearOpponentCSSM with 3 states: [X, Y, Z].

    Args:
        log_A: Matrix in GOOM representation, shape (..., 3, 3)
        log_v: Vector in GOOM representation, shape (..., 3)

    Returns:
        Result vector in GOOM representation, shape (..., 3)
    """
    a00, a01, a02 = log_A[..., 0, 0], log_A[..., 0, 1], log_A[..., 0, 2]
    a10, a11, a12 = log_A[..., 1, 0], log_A[..., 1, 1], log_A[..., 1, 2]
    a20, a21, a22 = log_A[..., 2, 0], log_A[..., 2, 1], log_A[..., 2, 2]
    v0, v1, v2 = log_v[..., 0], log_v[..., 1], log_v[..., 2]

    y0 = log_add_exp(log_add_exp(a00 + v0, a01 + v1), a02 + v2)
    y1 = log_add_exp(log_add_exp(a10 + v0, a11 + v1), a12 + v2)
    y2 = log_add_exp(log_add_exp(a20 + v0, a21 + v1), a22 + v2)

    return jnp.stack([y0, y1, y2], axis=-1)


def log_cum_sum_exp(
    log_x: jnp.ndarray,
    axis: int = 0
) -> jnp.ndarray:
    """
    Cumulative sum in log-space along an axis.

    Args:
        log_x: Values in GOOM representation
        axis: Axis to compute cumulative sum along

    Returns:
        Cumulative sum in GOOM representation
    """
    # Get max for stability
    c = jnp.max(goom_real(log_x), axis=axis, keepdims=True)

    # Cumsum in linear space
    x_cumsum = from_goom(log_x - c).cumsum(axis=axis)

    return to_goom(x_cumsum) + c


def log_mean_exp(
    log_x: jnp.ndarray,
    axis: Optional[int] = None
) -> jnp.ndarray:
    """
    Mean in log-space along an axis: log(mean(exp(log_x), axis)).

    Args:
        log_x: Values in GOOM representation
        axis: Axis to compute mean along (None for all elements)

    Returns:
        Mean in GOOM representation
    """
    # Get max for stability
    c = jnp.max(goom_real(log_x), axis=axis, keepdims=True)

    # Mean in linear space
    x_mean = from_goom(log_x - c).mean(axis=axis)

    return to_goom(x_mean) + jnp.squeeze(c, axis=axis)


# =============================================================================
# Block-Diagonal LMME (Log-Matrix-Matrix-Exp) for Channel Mixing
# =============================================================================
# Following the CSSM paper: LMME(A, B)_ij = LSE_k(A_ik + B_kj)
# This is matrix multiplication in log-space, enabling channel mixing
# while maintaining numerical stability through the log-semiring.
#
# We use block-diagonal structure for efficiency:
# - Full C×C mixing: O(C³) complexity (intractable for large C)
# - Block-diagonal (block_size=d): O(C × d²) complexity
# - Depthwise (block_size=1): O(C) complexity (current default)

def log_matmul_block(
    log_A: jnp.ndarray,
    log_B: jnp.ndarray,
    block_size: int
) -> jnp.ndarray:
    """
    Block-diagonal matrix multiplication in log-space (LMME).

    Computes C = A @ B where all matrices are block-diagonal with
    blocks of size (block_size × block_size), in GOOM representation.

    LMME(A, B)_ij = LSE_k(A_ik + B_kj) for each block

    Args:
        log_A: First block-diagonal matrix in GOOM representation
               Shape: (..., num_blocks, block_size, block_size)
        log_B: Second block-diagonal matrix in GOOM representation
               Shape: (..., num_blocks, block_size, block_size)
        block_size: Size of each block (for documentation, already in shape)

    Returns:
        Product matrix in GOOM representation
        Shape: (..., num_blocks, block_size, block_size)
    """
    # General case: use einsum-like computation
    # C[..., b, i, j] = LSE_k(A[..., b, i, k] + B[..., b, k, j])
    # where b is the block index

    # Expand for broadcasting: A[..., i, k, 1] + B[..., 1, k, j]
    log_A_exp = log_A[..., :, :, None]  # (..., num_blocks, d, d, 1)
    log_B_exp = log_B[..., None, :, :]  # (..., num_blocks, 1, d, d)

    # Sum in log-space (this is A_ik + B_kj for all k)
    log_sum = log_A_exp + log_B_exp  # (..., num_blocks, d, d, d)

    # LSE over k dimension (axis=-2)
    result = log_sum_exp(log_sum, axis=-2)  # (..., num_blocks, d, d)

    return result


def log_matvec_block(
    log_A: jnp.ndarray,
    log_v: jnp.ndarray,
    block_size: int
) -> jnp.ndarray:
    """
    Block-diagonal matrix-vector multiplication in log-space.

    Computes y = A @ v where matrix is block-diagonal and vector is
    blocked, all in GOOM representation.

    y[i] = LSE_k(A_ik + v_k) for each block

    Args:
        log_A: Block-diagonal matrix in GOOM representation
               Shape: (..., num_blocks, block_size, block_size)
        log_v: Blocked vector in GOOM representation
               Shape: (..., num_blocks, block_size)
        block_size: Size of each block

    Returns:
        Result vector in GOOM representation
        Shape: (..., num_blocks, block_size)
    """
    # y[..., b, i] = LSE_k(A[..., b, i, k] + v[..., b, k])

    # Expand v for broadcasting: v[..., b, 1, k]
    log_v_exp = log_v[..., None, :]  # (..., num_blocks, 1, block_size)

    # Sum in log-space: A_ik + v_k
    log_sum = log_A + log_v_exp  # (..., num_blocks, block_size, block_size)

    # LSE over k dimension (last axis)
    result = log_sum_exp(log_sum, axis=-1)  # (..., num_blocks, block_size)

    return result


def reshape_to_blocks(x: jnp.ndarray, block_size: int, axis: int = -1) -> jnp.ndarray:
    """
    Reshape a tensor to have block structure along specified axis.

    Converts (..., C, ...) to (..., num_blocks, block_size, ...)

    Args:
        x: Input tensor with C elements along axis
        block_size: Size of each block (must divide C evenly)
        axis: Axis to reshape (default -1)

    Returns:
        Reshaped tensor with block structure
    """
    # Move axis to end for easier manipulation
    x = jnp.moveaxis(x, axis, -1)
    shape = x.shape
    C = shape[-1]

    if C % block_size != 0:
        raise ValueError(f"Channel dim {C} must be divisible by block_size {block_size}")

    num_blocks = C // block_size
    new_shape = shape[:-1] + (num_blocks, block_size)
    x = x.reshape(new_shape)

    return x


def reshape_from_blocks(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Reshape a tensor from block structure back to flat.

    Converts (..., num_blocks, block_size, ...) to (..., C, ...)

    Args:
        x: Input tensor with block structure at last two dims
        axis: Target axis for the flattened dimension (default -1)

    Returns:
        Flattened tensor
    """
    shape = x.shape
    # Last two dims are (num_blocks, block_size)
    new_shape = shape[:-2] + (shape[-2] * shape[-1],)
    x = x.reshape(new_shape)

    return x
