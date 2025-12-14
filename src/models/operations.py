"""
Log-space operations using GOOM primitives.

Direct port from https://github.com/hkozachkov/goom/tree/main/src/goom

These operations perform arithmetic in log-space for numerical stability,
using the max-subtraction trick for log-sum-exp style computations.
"""

import jax.numpy as jnp
from typing import Optional, Union

from .goom import to_goom, from_goom, goom_real


def log_add_exp(log_x1: jnp.ndarray, log_x2: jnp.ndarray) -> jnp.ndarray:
    """
    Add two values in log-space: log(exp(log_x1) + exp(log_x2)).

    Uses the max-subtraction trick for numerical stability:
    1. Subtract the maximum
    2. Convert from GOOM and add
    3. Convert back to GOOM
    4. Add the maximum back

    Args:
        log_x1: First value in GOOM representation
        log_x2: Second value in GOOM representation

    Returns:
        Sum in GOOM representation
    """
    # Find maximum for numerical stability
    c = jnp.maximum(goom_real(log_x1), goom_real(log_x2))

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
    # Find maximum for numerical stability
    c = jnp.max(goom_real(log_x), axis=axis, keepdims=True)

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
    c = jnp.maximum(goom_real(log_x1), goom_real(log_x2))
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
