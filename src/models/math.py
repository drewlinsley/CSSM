"""
Mathematical primitives for CSSM (Cepstral State Space Models).

Implements associative scan operators for log-semiring computations
using proper GOOM (Generalized Order of Magnitude) primitives.
"""

import jax.numpy as jnp
from typing import Tuple

from .operations import log_add_exp, log_matmul_2x2, log_matvec_2x2


def cssm_scalar_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for Standard CSSM (Scalar/Diagonal).

    Computes: (k_j, u_j) o (k_i, u_i) = (k_j + k_i, LSE(k_j + u_i, u_j))

    This implements the recurrence h_t = k * h_{t-1} + u_t in log-space,
    where multiplication becomes addition and addition becomes log-sum-exp.

    All values are expected to be in GOOM representation (complex log-space).

    Args:
        carry_i: Tuple of (kernel_log, input_log) for position i
        carry_j: Tuple of (kernel_log, input_log) for position j

    Returns:
        Combined carry for the associative scan
    """
    k_i, u_i = carry_i
    k_j, u_j = carry_j

    # In log-space: multiplication -> addition
    k_new = k_j + k_i

    # In log-space: addition -> log-sum-exp (using GOOM)
    u_new = log_add_exp(k_j + u_i, u_j)

    return k_new, u_new


def cssm_matrix_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for Gated Opponent CSSM (2x2 Matrix).

    Implements matrix multiplication in the log-semiring for the
    coupled oscillator system with X and Y channels.

    The 2x2 matrix represents:
        [A_xx  A_xy]   where A_xx, A_yy are decay terms
        [A_yx  A_yy]   and A_xy (inhibition), A_yx (excitation) are coupling

    All values are expected to be in GOOM representation.

    Args:
        carry_i: Tuple of (K_matrix_log, u_vector_log) for position i
                 K has shape (..., 2, 2), u has shape (..., 2)
        carry_j: Tuple of (K_matrix_log, u_vector_log) for position j

    Returns:
        Combined carry: (K_new, u_new) where
            K_new = K_j @ K_i (log-matrix multiplication)
            u_new = K_j @ u_i + u_j (log-matrix-vector + log-sum-exp)
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    # K_new = K_j @ K_i in log-space
    K_new = log_matmul_2x2(K_j, K_i)

    # u_new = K_j @ u_i + u_j in log-space
    # First: matrix-vector multiplication
    Ku_i = log_matvec_2x2(K_j, u_i)

    # Then: element-wise log-sum-exp with u_j
    u_new_0 = log_add_exp(Ku_i[..., 0], u_j[..., 0])
    u_new_1 = log_add_exp(Ku_i[..., 1], u_j[..., 1])
    u_new = jnp.stack([u_new_0, u_new_1], axis=-1)

    return K_new, u_new
