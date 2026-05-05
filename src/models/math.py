"""
Mathematical primitives for CSSM (Cepstral State Space Models).

Implements associative scan operators for log-semiring computations
using proper GOOM (Generalized Order of Magnitude) primitives.

Supports three modes:
1. Scalar (depthwise): Each channel independent - O(C) complexity
2. 2x2 Matrix (opponent): X/Y coupled oscillator - O(C) complexity
3. Block-diagonal (dense mixing): LMME channel mixing - O(C × block_size²) complexity
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from .operations import (
    log_add_exp, log_matmul_2x2, log_matvec_2x2,
    log_matmul_3x3, log_matvec_3x3,
    log_matmul_block, log_matvec_block
)


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
    Associative operator for 2x2 Linear Opponent CSSM (hgru mode).

    ============================================================================
    WHAT THIS COMPUTES (Linear Recurrence)
    ============================================================================

    Original dynamics in LINEAR space:
        [X_t]   [decay_x    -μ_I·K_I] [X_{t-1}]   [U_X]
        [Y_t] = [μ_E·K_E     decay_y] [Y_{t-1}] + [U_Y]

    Or written out:
        X_t = decay_x·X_{t-1} - μ_I·K_I·Y_{t-1} + U_X   (Y inhibits X via K_I)
        Y_t = μ_E·K_E·X_{t-1} + decay_y·Y_{t-1} + U_Y   (X excites Y via K_E)

    The K_I and K_E are FFT'd convolution kernels, so in spectral domain
    the matrix multiplication IS the convolution!

    ============================================================================
    DOMAIN TRANSFORMATIONS
    ============================================================================

    SPATIAL         SPECTRAL (FFT)       LOG-SPECTRAL (GOOM)
    -------         --------------       -------------------
    conv(A,B)  -->  A_hat * B_hat   -->  log(A) + log(B)
    a + b      -->  a + b           -->  log_add_exp(log(a), log(b))
    a * b      -->  a * b           -->  log(a) + log(b)

    So in GOOM:
    - Matrix multiplication → log_matmul (uses log_add_exp internally)
    - Matrix-vector mult → log_matvec (uses log_add_exp internally)
    - Addition → log_add_exp

    ============================================================================
    ASSOCIATIVE SCAN SEMANTICS
    ============================================================================

    The scan combines results from position i (earlier) and j (later).

    carry_i = (K_i, u_i) represents accumulated computation from positions 0..i
    carry_j = (K_j, u_j) represents accumulated computation from positions i+1..j

    To combine them:
    1. K_new = K_j @ K_i  (compose transition matrices)
       - In log-space: log_matmul_2x2
       - This accumulates the "decay and coupling" over time
       - Receptive field GROWS as K's accumulate (conv of conv of conv...)

    2. u_new = K_j @ u_i + u_j  (propagate inputs through transitions)
       - K_j @ u_i: Apply later transitions to earlier accumulated inputs
       - + u_j: Add the later inputs
       - In log-space: log_matvec then log_add_exp

    ============================================================================
    STEP-BY-STEP EXAMPLE
    ============================================================================

    Say we have 4 timesteps: t=0,1,2,3

    Sequential would compute:
        state_1 = K_1 @ state_0 + u_1
        state_2 = K_2 @ state_1 + u_2
        state_3 = K_3 @ state_2 + u_3

    Parallel scan does:
        Level 0: (K_0,u_0), (K_1,u_1), (K_2,u_2), (K_3,u_3)

        Level 1: Combine pairs
            (K_0,u_0) ⊕ (K_1,u_1) = (K_1@K_0, K_1@u_0 + u_1)
            (K_2,u_2) ⊕ (K_3,u_3) = (K_3@K_2, K_3@u_2 + u_3)

        Level 2: Combine results
            Result gives us state_3 = K_3@K_2@K_1@(K_0@init + u_0) + K_3@K_2@u_1 + K_3@u_2 + u_3

    This is O(log T) depth instead of O(T)!

    ============================================================================

    Args:
        carry_i: Tuple of (K_matrix_log, u_vector_log) for earlier positions
                 K has shape (..., 2, 2), u has shape (..., 2)
        carry_j: Tuple of (K_matrix_log, u_vector_log) for later positions

    Returns:
        Combined carry: (K_new, u_new)
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    # =========================================================================
    # STEP 1: Compose transition matrices
    # =========================================================================
    # K_new = K_j @ K_i in log-space
    #
    # This computes the combined transition from positions 0..j
    # In log-space, matrix multiply uses log_add_exp for the inner sums:
    #   C[i,j] = log_add_exp(A[i,0] + B[0,j], A[i,1] + B[1,j])
    #          = log(exp(A[i,0] + B[0,j]) + exp(A[i,1] + B[1,j]))
    #          = log(A[i,0]*B[0,j] + A[i,1]*B[1,j])  in linear space
    #
    # The spatial convolution kernels K_I, K_E are baked into K.
    # Accumulating K matrices = accumulating convolutions = growing receptive field!
    K_new = log_matmul_2x2(K_j, K_i)

    # =========================================================================
    # STEP 2: Propagate accumulated inputs through later transitions
    # =========================================================================
    # Ku_i = K_j @ u_i in log-space
    #
    # u_i contains accumulated weighted inputs from positions 0..i
    # K_j contains the transition matrix from position j
    # This applies the later transition to the earlier accumulated state
    #
    # In log-space: y[k] = log_add_exp(K[k,0] + u[0], K[k,1] + u[1])
    # Which is: y[k] = log(K[k,0]*u[0] + K[k,1]*u[1]) in linear space
    Ku_i = log_matvec_2x2(K_j, u_i)

    # =========================================================================
    # STEP 3: Add later inputs
    # =========================================================================
    # u_new = Ku_i + u_j in log-space (element-wise)
    #
    # log_add_exp computes log(exp(a) + exp(b)) = log(a_linear + b_linear)
    # This adds the propagated earlier inputs to the later inputs
    #
    # u_new[0] = total accumulated X input
    # u_new[1] = total accumulated Y input
    u_new_0 = log_add_exp(Ku_i[..., 0], u_j[..., 0])
    u_new_1 = log_add_exp(Ku_i[..., 1], u_j[..., 1])
    u_new = jnp.stack([u_new_0, u_new_1], axis=-1)

    return K_new, u_new


def cssm_block_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray],
    block_size: int = 32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for Block-Diagonal CSSM (LMME Channel Mixing).

    Implements LMME (Log-Matrix-Matrix-Exp) for channel mixing following
    the CSSM paper. Channels are grouped into blocks that mix internally.

    This enables channel mixing with O(C × block_size²) complexity instead
    of O(C³) for full mixing or O(C) for depthwise.

    The LMME operation:
        LMME(A, B)_ij = LSE_k(A_ik + B_kj)

    Args:
        carry_i: Tuple of (K_block_log, u_block_log) for position i
                 K has shape (..., num_blocks, block_size, block_size)
                 u has shape (..., num_blocks, block_size)
        carry_j: Tuple of (K_block_log, u_block_log) for position j
        block_size: Size of each channel mixing block

    Returns:
        Combined carry: (K_new, u_new) where
            K_new = LMME(K_j, K_i) (log-matrix multiplication per block)
            u_new = LMME(K_j, u_i) + u_j (log-matrix-vector + log-sum-exp)
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    # K_new = K_j @ K_i in log-space (per block)
    K_new = log_matmul_block(K_j, K_i, block_size)

    # u_new = K_j @ u_i + u_j in log-space
    # First: block matrix-vector multiplication
    Ku_i = log_matvec_block(K_j, u_i, block_size)

    # Then: element-wise log-sum-exp with u_j
    u_new = log_add_exp(Ku_i, u_j)

    return K_new, u_new


def make_block_scan_op(block_size: int):
    """
    Create a block scan operator with fixed block_size.

    JAX's associative_scan requires a binary operator, but cssm_block_scan_op
    has an extra block_size parameter. This factory creates a closure with
    the block_size baked in.

    Args:
        block_size: Size of each channel mixing block

    Returns:
        Binary scan operator suitable for jax.lax.associative_scan
    """
    def scan_op(carry_i, carry_j):
        return cssm_block_scan_op(carry_i, carry_j, block_size)
    return scan_op


# =============================================================================
# 3x3 CSSM with Interaction Channel (hgru_bi mode)
# =============================================================================
# State: [X, Y, Z] where:
#   X = Excitatory state (receives input, inhibited by Y and Z)
#   Y = Inhibitory state (excited by X)
#   Z = Interaction channel (learns to track X-Y interaction for "bilinear-like" effect)
#
# Key insight: Z can learn to approximate X*Y interaction through linear dynamics!
# The 3x3 matrix lets Z mix X and Y contributions, then feed back into X and Y.

def cssm_3x3_matrix_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for 3x3 CSSM with Interaction Channel (hgru_bi mode).

    ============================================================================
    WHAT THIS COMPUTES (3-State Linear Recurrence with Interaction)
    ============================================================================

    Original dynamics in LINEAR space:
        [X_t]   [decay_x    -μ_I·K_I    -α_I·K_I] [X_{t-1}]   [U_X]
        [Y_t] = [μ_E·K_E     decay_y    +α_E·K_E] [Y_{t-1}] + [U_Y]
        [Z_t]   [γ           δ           ε      ] [Z_{t-1}]   [U_Z]

    Or written out:
        X_t = decay_x·X - μ_I·K_I·Y - α_I·K_I·Z + U_X   (inhibited by Y AND Z)
        Y_t = μ_E·K_E·X + decay_y·Y + α_E·K_E·Z + U_Y   (excited by X AND Z)
        Z_t = γ·X + δ·Y + ε·Z + U_Z                     (learns X-Y interaction)

    ============================================================================
    WHY THIS APPROXIMATES BILINEAR
    ============================================================================

    True hGRU bilinear: X_t = ... - α·K_I·X_{t-1}·Y_{t-1}

    We can't do X*Y directly in associative scan (breaks associativity).
    But Z can LEARN to track an interaction:

    If Z learns: Z ≈ β·X + γ·Y + accumulated_interaction
    Then: -α_I·K_I·Z ≈ -α_I·K_I·(β·X + γ·Y + ...)

    This isn't exactly X*Y, but it's a learnable proxy that:
    1. Depends on both X and Y history
    2. Feeds back into X (inhibition) and Y (excitation)
    3. Can capture correlation/interaction patterns

    ============================================================================
    THE 3x3 TRANSITION MATRIX
    ============================================================================

    [decay_x    -μ_I·K_I    -α_I·K_I]    Row 0: X update
    [μ_E·K_E     decay_y    +α_E·K_E]    Row 1: Y update
    [γ           δ           ε      ]    Row 2: Z update (interaction tracker)

    Key positions:
    - (0,2) = -α_I·K_I: Z inhibits X (like X*Y would)
    - (1,2) = +α_E·K_E: Z excites Y (like X*Y would)
    - (2,0) = γ: X contributes to Z
    - (2,1) = δ: Y contributes to Z
    - (2,2) = ε: Z self-decay

    ============================================================================
    SCAN MECHANICS (same as 2x2, just bigger matrices)
    ============================================================================

    carry_i = (K_i, u_i): Accumulated from positions 0..i
    carry_j = (K_j, u_j): Accumulated from positions i+1..j

    Combine:
    1. K_new = K_j @ K_i  (3x3 log-matmul)
    2. u_new = K_j @ u_i + u_j  (3x3 log-matvec + log-add-exp)

    ============================================================================

    Args:
        carry_i: Tuple of (K_matrix_log, u_vector_log) for earlier positions
                 K has shape (..., 3, 3), u has shape (..., 3)
        carry_j: Tuple of (K_matrix_log, u_vector_log) for later positions

    Returns:
        Combined carry: (K_new, u_new)
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    # =========================================================================
    # STEP 1: Compose 3x3 transition matrices
    # =========================================================================
    # K_new = K_j @ K_i in log-space
    #
    # For 3x3: C[i,j] = log_add_exp(A[i,0]+B[0,j], A[i,1]+B[1,j], A[i,2]+B[2,j])
    #
    # This accumulates transitions including the Z interaction channel.
    # Over time, Z's contribution to X and Y grows through the off-diagonal terms.
    K_new = log_matmul_3x3(K_j, K_i)

    # =========================================================================
    # STEP 2: Propagate accumulated inputs through later transitions
    # =========================================================================
    # Ku_i = K_j @ u_i in log-space (3x3 matrix × 3-vector)
    #
    # For each output k: y[k] = log_add_exp(K[k,0]+u[0], K[k,1]+u[1], K[k,2]+u[2])
    #
    # This mixes X, Y, Z contributions according to the transition matrix
    Ku_i = log_matvec_3x3(K_j, u_i)

    # =========================================================================
    # STEP 3: Add later inputs
    # =========================================================================
    # u_new = Ku_i + u_j in log-space (element-wise log-sum-exp)
    #
    # u_new[0] = accumulated X state
    # u_new[1] = accumulated Y state
    # u_new[2] = accumulated Z state (interaction memory)
    u_new_0 = log_add_exp(Ku_i[..., 0], u_j[..., 0])
    u_new_1 = log_add_exp(Ku_i[..., 1], u_j[..., 1])
    u_new_2 = log_add_exp(Ku_i[..., 2], u_j[..., 2])
    u_new = jnp.stack([u_new_0, u_new_1, u_new_2], axis=-1)

    return K_new, u_new


def cssm_general_matrix_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for general d_k x d_k CSSM in log-space.

    Uses log_matmul_block / log_matvec_block from operations.py for
    arbitrary d_k (not just 2 or 3).

    K: (..., d_k, d_k), u: (..., d_k) in GOOM representation.
    Scan: K_new = K_j @ K_i, u_new = K_j @ u_i + u_j
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    d_k = K_i.shape[-1]

    # K_new = K_j @ K_i in log-space
    K_new = log_matmul_block(K_j, K_i, d_k)

    # Ku_i = K_j @ u_i in log-space
    Ku_i = log_matvec_block(K_j, u_i, d_k)

    # u_new = Ku_i + u_j in log-space (element-wise)
    u_new = log_add_exp(Ku_i, u_j)

    return K_new, u_new


def cssm_3x3_bilinear_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray],
    K_bilinear: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for 3x3 CSSM with bilinear X*Z term.

    Computes: X_t = linear_terms - K_I*ν*X*Z (hGRU-style X² inhibition)

    The bilinear term log(X*Z) = log(X) + log(Z) is computed in log-space
    and added to the X update.

    State vector: [X, Y, Z] where Z = X_{t-1}

    Args:
        carry_i: Tuple of (K_matrix_log, u_vector_log) for position i
                 K has shape (..., 3, 3), u has shape (..., 3)
        carry_j: Tuple of (K_matrix_log, u_vector_log) for position j
        K_bilinear: Bilinear coefficient log(K_I * ν) with phase π for subtraction
                    Shape: (...) matching spatial dimensions

    Returns:
        Combined carry: (K_new, u_new) where u_new[0] includes bilinear term
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    # Standard 3x3 matrix composition
    K_new = log_matmul_3x3(K_j, K_i)
    Ku_i = log_matvec_3x3(K_j, u_i)

    # === BILINEAR TERM: X * Z ===
    # In log-space: log(X * Z) = log(X) + log(Z)
    log_X = u_i[..., 0]  # log(X_{t-1})
    log_Z = u_i[..., 2]  # log(Z_{t-1}) = log(X_{t-2})
    log_XZ = log_X + log_Z  # log(X * Z) - product becomes sum!

    # Bilinear inhibition: -K_I * ν * X * Z
    # K_bilinear includes phase π for the negative sign
    bilinear_inhib = K_bilinear + log_XZ

    # Combine: X_new = LSE(linear_terms, bilinear_term)
    u_new_X = log_add_exp(
        log_add_exp(Ku_i[..., 0], u_j[..., 0]),
        bilinear_inhib
    )
    u_new_Y = log_add_exp(Ku_i[..., 1], u_j[..., 1])
    u_new_Z = log_add_exp(Ku_i[..., 2], u_j[..., 2])

    u_new = jnp.stack([u_new_X, u_new_Y, u_new_Z], axis=-1)
    return K_new, u_new


def make_3x3_bilinear_scan_op(K_bilinear: jnp.ndarray):
    """
    Create a 3x3 bilinear scan operator with fixed K_bilinear.

    JAX's associative_scan requires a binary operator, but cssm_3x3_bilinear_scan_op
    has an extra K_bilinear parameter. This factory creates a closure with
    K_bilinear baked in.

    Args:
        K_bilinear: Bilinear coefficient log(K_I * ν) in GOOM representation
                    Should include phase π for subtraction (negative inhibition)

    Returns:
        Binary scan operator suitable for jax.lax.associative_scan
    """
    def scan_op(carry_i, carry_j):
        return cssm_3x3_bilinear_scan_op(carry_i, carry_j, K_bilinear)
    return scan_op


def cssm_hgru_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray],
    K_inhib_bilinear: jnp.ndarray,
    K_excit_bilinear: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for hGRU-style CSSM with X*Y bilinear terms.

    Implements true hGRU equations:
        X_t = linear_X - α·K_I·X·Y + U   (X*Y inhibits X)
        Y_t = linear_Y + α·K_E·X·Y       (X*Y excites Y)

    In log-space: log(X*Y) = log(X) + log(Y) — just addition!

    State vector: [X, Y] (2-element)

    Args:
        carry_i: Tuple of (K_matrix_log, u_vector_log) for position i
                 K has shape (..., 2, 2), u has shape (..., 2)
        carry_j: Tuple of (K_matrix_log, u_vector_log) for position j
        K_inhib_bilinear: log(α·K_I) with phase π for subtraction (inhibits X)
        K_excit_bilinear: log(α·K_E) with phase 0 for addition (excites Y)

    Returns:
        Combined carry: (K_new, u_new) with bilinear X*Y terms included
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    # Standard 2x2 matrix composition (linear part)
    K_new = log_matmul_2x2(K_j, K_i)
    Ku_i = log_matvec_2x2(K_j, u_i)

    # === BILINEAR TERM: X * Y ===
    # In log-space: log(X * Y) = log(X) + log(Y)
    log_X = u_i[..., 0]  # log(X accumulated)
    log_Y = u_i[..., 1]  # log(Y accumulated)
    # Clip to prevent unbounded accumulation over many epochs (causes NaN around epoch 40+)
    log_XY = jnp.clip(log_X + log_Y, -50.0, 50.0)  # log(X * Y) - bounded!

    # X update: add -α·K_I·X·Y (inhibition from Y, gated by X)
    # K_inhib_bilinear includes phase π for the negative sign
    bilinear_inhib = K_inhib_bilinear + log_XY
    u_new_X = log_add_exp(
        log_add_exp(Ku_i[..., 0], u_j[..., 0]),
        bilinear_inhib
    )

    # Y update: add +α·K_E·X·Y (excitation from X, gated by Y)
    # K_excit_bilinear has phase 0 for positive addition
    bilinear_excit = K_excit_bilinear + log_XY
    u_new_Y = log_add_exp(
        log_add_exp(Ku_i[..., 1], u_j[..., 1]),
        bilinear_excit
    )

    u_new = jnp.stack([u_new_X, u_new_Y], axis=-1)
    return K_new, u_new


def make_hgru_scan_op(K_inhib_bilinear: jnp.ndarray, K_excit_bilinear: jnp.ndarray):
    """
    Create an hGRU-style scan operator with fixed bilinear coefficients.

    JAX's associative_scan requires a binary operator, but cssm_hgru_scan_op
    has extra parameters. This factory creates a closure with them baked in.

    Args:
        K_inhib_bilinear: log(α·K_I) in GOOM with phase π (for X inhibition)
        K_excit_bilinear: log(α·K_E) in GOOM with phase 0 (for Y excitation)

    Returns:
        Binary scan operator suitable for jax.lax.associative_scan
    """
    def scan_op(carry_i, carry_j):
        return cssm_hgru_scan_op(carry_i, carry_j, K_inhib_bilinear, K_excit_bilinear)
    return scan_op


# =============================================================================
# KQV-CSSM: Transformer-inspired K*Q bilinear gating
# =============================================================================
# State: [K, Q, V] where:
#   K = Key state (accumulates spatial features via conv)
#   Q = Query state (accumulates spatial features via conv)
#   V = Value state (receives input GATED by K*Q product)
#
# Key insight: K*Q (Hadamard product) gates input flow to V, similar to
# how attention = softmax(Q @ K^T) @ V gates value contribution.
# The bilinear K*Q term is added to the INPUT accumulation (u), not the
# transition matrix, which preserves associativity.

def cssm_kqv_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for KQV-CSSM with K*Q bilinear gating.

    ============================================================================
    WHAT THIS COMPUTES (3-State with Bilinear Gating)
    ============================================================================

    State: [K, Q, V] where V receives input gated by K*Q.

    Dynamics in linear space:
        K_t = decay_K * conv(K_{t-1}, W_K) + B_K * U
        Q_t = decay_Q * conv(Q_{t-1}, W_Q) + B_Q * U
        V_t = decay_V * conv(V_{t-1}, W_V) + (K_{t-1} * Q_{t-1}) * B_V * U

    K and Q evolve independently via diagonal transitions.
    V receives bilinear K*Q gating on its input term.

    ============================================================================
    WHY THIS IS ASSOCIATIVE
    ============================================================================

    The bilinear term K*Q is added to the INPUT accumulation (u_j), not the
    transition matrix (K_j). Since u_i contains previously accumulated states,
    the operation:
        u_new = K_j @ u_i + u_j + bilinear_term_from_u_i
    is associative.

    ============================================================================
    LOG-SPACE COMPUTATION
    ============================================================================

    In GOOM log-space:
        log(K * Q) = log(K) + log(Q)  (no clipping - let gradients flow)

    The gated V input becomes:
        gated_input = log_K + log_Q + log_B_V + log_U_V

    ============================================================================

    Args:
        carry_i: Tuple of (K_matrix_log, u_vector_log) for earlier positions
                 K has shape (..., 3, 3), u has shape (..., 3)
        carry_j: Tuple of (K_matrix_log, u_vector_log) for later positions

    Returns:
        Combined carry: (K_new, u_new)
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    # =========================================================================
    # STEP 1: Compose 3x3 transition matrices (diagonal for K/Q/V)
    # =========================================================================
    K_new = log_matmul_3x3(K_j, K_i)

    # =========================================================================
    # STEP 2: Propagate accumulated inputs through later transitions
    # =========================================================================
    Ku_i = log_matvec_3x3(K_j, u_i)

    # =========================================================================
    # STEP 3: K and Q - standard linear accumulation
    # =========================================================================
    u_new_K = log_add_exp(Ku_i[..., 0], u_j[..., 0])
    u_new_Q = log_add_exp(Ku_i[..., 1], u_j[..., 1])

    # =========================================================================
    # STEP 4: BILINEAR GATING - V input gated by K*Q
    # =========================================================================
    # In log-space: log(K * Q) = log(K) + log(Q)
    log_K = u_i[..., 0]  # log(K accumulated)
    log_Q = u_i[..., 1]  # log(Q accumulated)
    log_KQ = log_K + log_Q  # K*Q in log-space (no clipping for gradient flow)

    # V accumulates: linear_V + (K*Q) * U_V
    # The K*Q term gates the input to V (u_j[..., 2])
    gated_input = log_KQ + u_j[..., 2]
    u_new_V = log_add_exp(Ku_i[..., 2], gated_input)

    u_new = jnp.stack([u_new_K, u_new_Q, u_new_V], axis=-1)
    return K_new, u_new


def cssm_kqv_block_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray],
    block_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for KQV-CSSM with block-diagonal channel mixing.

    ============================================================================
    BLOCK-DIAGONAL CHANNEL MIXING (Multi-Head Analogy)
    ============================================================================

    Channels are grouped into blocks of size `block_size`:
    - num_heads = C / block_size (like attention heads)
    - Within each block, channels can mix during recurrence
    - Blocks operate independently (like parallel attention heads)

    State vector per block: [K_1, ..., K_d, Q_1, ..., Q_d, V_1, ..., V_d]
    where d = block_size.

    Transition matrix per block: (3d x 3d) with structure:
        [K_block    0          0        ]
        [0          Q_block    0        ]
        [0          0          V_block  ]

    Each *_block is (d x d), allowing channel mixing within K, Q, V separately.

    ============================================================================
    BILINEAR GATING WITH BLOCKS
    ============================================================================

    The K*Q gating is computed per-channel within each block:
        gate[c] = K[c] * Q[c]  for c in block

    This gates the corresponding V[c] input. No cross-channel gating.

    ============================================================================

    Args:
        carry_i: Tuple of (K_matrix_log, u_vector_log) for earlier positions
                 K has shape (..., num_blocks, 3*block_size, 3*block_size)
                 u has shape (..., num_blocks, 3*block_size)
        carry_j: Tuple of (K_matrix_log, u_vector_log) for later positions
        block_size: Number of channels per block (d)

    Returns:
        Combined carry: (K_new, u_new)
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j
    d = block_size

    # =========================================================================
    # STEP 1: Block matrix multiplication for transition composition
    # =========================================================================
    # K_new = K_j @ K_i per block (using log-space matmul)
    K_new = log_matmul_block(K_j, K_i, 3 * d)

    # =========================================================================
    # STEP 2: Block matrix-vector for state propagation
    # =========================================================================
    Ku_i = log_matvec_block(K_j, u_i, 3 * d)

    # =========================================================================
    # STEP 3: Standard accumulation for K and Q portions
    # =========================================================================
    # K channels: indices 0 to d-1
    # Q channels: indices d to 2d-1
    # V channels: indices 2d to 3d-1
    u_new_KQ = log_add_exp(Ku_i[..., :2*d], u_j[..., :2*d])

    # =========================================================================
    # STEP 4: Bilinear gating for V portion
    # =========================================================================
    # Per-channel K*Q gating within each block
    log_K = u_i[..., :d]       # K channels from earlier accumulation
    log_Q = u_i[..., d:2*d]    # Q channels from earlier accumulation
    log_KQ = log_K + log_Q     # Per-channel K*Q gate

    # Gate the V input and accumulate
    gated_V_input = log_KQ + u_j[..., 2*d:]
    u_new_V = log_add_exp(Ku_i[..., 2*d:], gated_V_input)

    u_new = jnp.concatenate([u_new_KQ, u_new_V], axis=-1)
    return K_new, u_new


def make_kqv_block_scan_op(block_size: int):
    """
    Create a KQV block scan operator with fixed block_size.

    JAX's associative_scan requires a binary operator, but cssm_kqv_block_scan_op
    has an extra block_size parameter. This factory creates a closure with
    block_size baked in.

    Args:
        block_size: Number of channels per block

    Returns:
        Binary scan operator suitable for jax.lax.associative_scan
    """
    def scan_op(carry_i, carry_j):
        return cssm_kqv_block_scan_op(carry_i, carry_j, block_size)
    return scan_op


# =============================================================================
# Linear-split scan operators ("complex32": bfloat16 real + bfloat16 imag)
# =============================================================================
#
# Works in LINEAR complex space (not GOOM log-space). The scan operator is
# just complex multiply + add — no trig, no log, no exp, no GOOM.
#
# Representation: split complex64 into (re_bf16, im_bf16).
#   re = x.real.astype(bfloat16)
#   im = x.imag.astype(bfloat16)
#
# Complex multiply: (a_re + i*a_im) * (b_re + i*b_im)
#   = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
#
# Complex add: trivial component-wise addition.
#
# All intermediate computation promotes to float32 for numerical stability.
# Only the stored scan state uses bfloat16.

def complex64_to_linear_split(x: jnp.ndarray, dtype=jnp.bfloat16) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert complex64 to (re, im) in the specified dtype (bf16, fp8, etc.)."""
    return x.real.astype(dtype), x.imag.astype(dtype)


def linear_split_to_complex64(re: jnp.ndarray, im: jnp.ndarray) -> jnp.ndarray:
    """Convert (re, im) back to complex64. Works with any input dtype."""
    return (re.astype(jnp.float32) + 1j * im.astype(jnp.float32)).astype(jnp.complex64)


def _ls_complex_mul(a_re, a_im, b_re, b_im):
    """Complex multiply: promote to float32 for math, cast back to input dtype."""
    store_dtype = a_re.dtype
    ar = a_re.astype(jnp.float32)
    ai = a_im.astype(jnp.float32)
    br = b_re.astype(jnp.float32)
    bi = b_im.astype(jnp.float32)
    out_re = ar * br - ai * bi
    out_im = ar * bi + ai * br
    return out_re.astype(store_dtype), out_im.astype(store_dtype)


def _ls_complex_add(a_re, a_im, b_re, b_im):
    """Complex add: promote to float32 for narrow types, cast back to input dtype."""
    store_dtype = a_re.dtype
    sum_re = a_re.astype(jnp.float32) + b_re.astype(jnp.float32)
    sum_im = a_im.astype(jnp.float32) + b_im.astype(jnp.float32)
    return sum_re.astype(store_dtype), sum_im.astype(store_dtype)


# --- Scalar (add_kqv_1) ---

def linear_split_scalar_scan_op(carry_i, carry_j):
    """Scalar associative scan op in linear-split representation.

    Carry = (k_re, k_im, u_re, u_im).
    Scan recurrence: k_new = k_j * k_i, u_new = k_j * u_i + u_j
    Just complex multiply + add — no trig, no log/exp.
    """
    k_re_i, k_im_i, u_re_i, u_im_i = carry_i
    k_re_j, k_im_j, u_re_j, u_im_j = carry_j

    # k_new = k_j * k_i (complex multiply)
    k_re_new, k_im_new = _ls_complex_mul(k_re_j, k_im_j, k_re_i, k_im_i)

    # k_j * u_i
    ku_re, ku_im = _ls_complex_mul(k_re_j, k_im_j, u_re_i, u_im_i)

    # u_new = k_j * u_i + u_j
    u_re_new, u_im_new = _ls_complex_add(ku_re, ku_im, u_re_j, u_im_j)

    return k_re_new, k_im_new, u_re_new, u_im_new


# --- 2x2 matrix (add_kqv_2) ---

def _ls_matmul_2x2(A_re, A_im, B_re, B_im):
    """2x2 complex matrix multiply in linear-split representation."""
    rows_re, rows_im = [], []
    for i in range(2):
        cols_re, cols_im = [], []
        for j in range(2):
            # C[i,j] = sum_k A[i,k] * B[k,j]
            # k=0
            p0_re, p0_im = _ls_complex_mul(
                A_re[..., i, 0], A_im[..., i, 0],
                B_re[..., 0, j], B_im[..., 0, j])
            # k=1
            p1_re, p1_im = _ls_complex_mul(
                A_re[..., i, 1], A_im[..., i, 1],
                B_re[..., 1, j], B_im[..., 1, j])
            # sum
            c_re, c_im = _ls_complex_add(p0_re, p0_im, p1_re, p1_im)
            cols_re.append(c_re)
            cols_im.append(c_im)
        rows_re.append(jnp.stack(cols_re, axis=-1))
        rows_im.append(jnp.stack(cols_im, axis=-1))
    return jnp.stack(rows_re, axis=-2), jnp.stack(rows_im, axis=-2)


def _ls_matvec_2x2(A_re, A_im, v_re, v_im):
    """2x2 complex matrix-vector multiply in linear-split representation."""
    ys_re, ys_im = [], []
    for i in range(2):
        # y[i] = sum_k A[i,k] * v[k]
        p0_re, p0_im = _ls_complex_mul(
            A_re[..., i, 0], A_im[..., i, 0],
            v_re[..., 0], v_im[..., 0])
        p1_re, p1_im = _ls_complex_mul(
            A_re[..., i, 1], A_im[..., i, 1],
            v_re[..., 1], v_im[..., 1])
        c_re, c_im = _ls_complex_add(p0_re, p0_im, p1_re, p1_im)
        ys_re.append(c_re)
        ys_im.append(c_im)
    return jnp.stack(ys_re, axis=-1), jnp.stack(ys_im, axis=-1)


def linear_split_2x2_scan_op(carry_i, carry_j):
    """2x2 matrix associative scan op in linear-split representation.

    Carry = (K_re, K_im, U_re, U_im).
    K: (..., 2, 2), U: (..., 2).
    Scan: K_new = K_j @ K_i, U_new = K_j @ U_i + U_j
    """
    K_re_i, K_im_i, U_re_i, U_im_i = carry_i
    K_re_j, K_im_j, U_re_j, U_im_j = carry_j

    # K_new = K_j @ K_i
    K_re_new, K_im_new = _ls_matmul_2x2(K_re_j, K_im_j, K_re_i, K_im_i)

    # K_j @ U_i
    Ku_re, Ku_im = _ls_matvec_2x2(K_re_j, K_im_j, U_re_i, U_im_i)

    # U_new = K_j @ U_i + U_j
    U_re_new, U_im_new = _ls_complex_add(Ku_re, Ku_im, U_re_j, U_im_j)

    return K_re_new, K_im_new, U_re_new, U_im_new


# --- 3x3 matrix (add_kqv) ---

def _ls_matmul_3x3(A_re, A_im, B_re, B_im):
    """3x3 complex matrix multiply in linear-split representation."""
    rows_re, rows_im = [], []
    for i in range(3):
        cols_re, cols_im = [], []
        for j in range(3):
            # C[i,j] = sum_k A[i,k] * B[k,j]
            p0_re, p0_im = _ls_complex_mul(
                A_re[..., i, 0], A_im[..., i, 0],
                B_re[..., 0, j], B_im[..., 0, j])
            p1_re, p1_im = _ls_complex_mul(
                A_re[..., i, 1], A_im[..., i, 1],
                B_re[..., 1, j], B_im[..., 1, j])
            acc_re, acc_im = _ls_complex_add(p0_re, p0_im, p1_re, p1_im)
            p2_re, p2_im = _ls_complex_mul(
                A_re[..., i, 2], A_im[..., i, 2],
                B_re[..., 2, j], B_im[..., 2, j])
            acc_re, acc_im = _ls_complex_add(acc_re, acc_im, p2_re, p2_im)
            cols_re.append(acc_re)
            cols_im.append(acc_im)
        rows_re.append(jnp.stack(cols_re, axis=-1))
        rows_im.append(jnp.stack(cols_im, axis=-1))
    return jnp.stack(rows_re, axis=-2), jnp.stack(rows_im, axis=-2)


def _ls_matvec_3x3(A_re, A_im, v_re, v_im):
    """3x3 complex matrix-vector multiply in linear-split representation."""
    ys_re, ys_im = [], []
    for i in range(3):
        p0_re, p0_im = _ls_complex_mul(
            A_re[..., i, 0], A_im[..., i, 0],
            v_re[..., 0], v_im[..., 0])
        p1_re, p1_im = _ls_complex_mul(
            A_re[..., i, 1], A_im[..., i, 1],
            v_re[..., 1], v_im[..., 1])
        acc_re, acc_im = _ls_complex_add(p0_re, p0_im, p1_re, p1_im)
        p2_re, p2_im = _ls_complex_mul(
            A_re[..., i, 2], A_im[..., i, 2],
            v_re[..., 2], v_im[..., 2])
        acc_re, acc_im = _ls_complex_add(acc_re, acc_im, p2_re, p2_im)
        ys_re.append(acc_re)
        ys_im.append(acc_im)
    return jnp.stack(ys_re, axis=-1), jnp.stack(ys_im, axis=-1)


def linear_split_3x3_scan_op(carry_i, carry_j):
    """3x3 matrix associative scan op in linear-split representation.

    Carry = (K_re, K_im, U_re, U_im).
    K: (..., 3, 3), U: (..., 3).
    Scan: K_new = K_j @ K_i, U_new = K_j @ U_i + U_j
    """
    K_re_i, K_im_i, U_re_i, U_im_i = carry_i
    K_re_j, K_im_j, U_re_j, U_im_j = carry_j

    # K_new = K_j @ K_i
    K_re_new, K_im_new = _ls_matmul_3x3(K_re_j, K_im_j, K_re_i, K_im_i)

    # K_j @ U_i
    Ku_re, Ku_im = _ls_matvec_3x3(K_re_j, K_im_j, U_re_i, U_im_i)

    # U_new = K_j @ U_i + U_j
    U_re_new, U_im_new = _ls_complex_add(Ku_re, Ku_im, U_re_j, U_im_j)

    return K_re_new, K_im_new, U_re_new, U_im_new


# --- General NxN matrix (arbitrary d_k) ---

def _ls_matmul_nxn(A_re, A_im, B_re, B_im):
    """General NxN complex matrix multiply in linear-split representation.
    A, B: (..., N, N). Uses einsum for the sum over k."""
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # complex mul: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    C_re = jnp.einsum('...ik,...kj->...ij', A_re, B_re) - \
           jnp.einsum('...ik,...kj->...ij', A_im, B_im)
    C_im = jnp.einsum('...ik,...kj->...ij', A_re, B_im) + \
           jnp.einsum('...ik,...kj->...ij', A_im, B_re)
    return C_re, C_im


def _ls_matvec_nxn(A_re, A_im, v_re, v_im):
    """General NxN complex matrix-vector multiply in linear-split representation.
    A: (..., N, N), v: (..., N)."""
    # y[i] = sum_k A[i,k] * v[k]
    y_re = jnp.einsum('...ik,...k->...i', A_re, v_re) - \
           jnp.einsum('...ik,...k->...i', A_im, v_im)
    y_im = jnp.einsum('...ik,...k->...i', A_re, v_im) + \
           jnp.einsum('...ik,...k->...i', A_im, v_re)
    return y_re, y_im


def linear_split_general_scan_op(carry_i, carry_j):
    """General NxN matrix associative scan op in linear-split representation.

    Carry = (K_re, K_im, U_re, U_im).
    K: (..., N, N), U: (..., N).
    Scan: K_new = K_j @ K_i, U_new = K_j @ U_i + U_j
    """
    K_re_i, K_im_i, U_re_i, U_im_i = carry_i
    K_re_j, K_im_j, U_re_j, U_im_j = carry_j

    K_re_new, K_im_new = _ls_matmul_nxn(K_re_j, K_im_j, K_re_i, K_im_i)
    Ku_re, Ku_im = _ls_matvec_nxn(K_re_j, K_im_j, U_re_i, U_im_i)
    U_re_new, U_im_new = _ls_complex_add(Ku_re, Ku_im, U_re_j, U_im_j)

    return K_re_new, K_im_new, U_re_new, U_im_new


# =============================================================================
# Direct complex64 scan ops (no log-space, no bf16 split — full precision)
# =============================================================================

def direct_scalar_scan_op(carry_i, carry_j):
    """Scalar associative scan in direct complex64. No log transform."""
    K_i, U_i = carry_i
    K_j, U_j = carry_j
    return K_j * K_i, K_j * U_i + U_j

def direct_2x2_scan_op(carry_i, carry_j):
    """2x2 matrix scan in direct complex64."""
    K_i, U_i = carry_i  # K: (...,2,2), U: (...,2)
    K_j, U_j = carry_j
    K_new = jnp.einsum('...ik,...kj->...ij', K_j, K_i)
    U_new = jnp.einsum('...ik,...k->...i', K_j, U_i) + U_j
    return K_new, U_new

def direct_3x3_scan_op(carry_i, carry_j):
    """3x3 matrix scan in direct complex64."""
    K_i, U_i = carry_i  # K: (...,3,3), U: (...,3)
    K_j, U_j = carry_j
    K_new = jnp.einsum('...ik,...kj->...ij', K_j, K_i)
    U_new = jnp.einsum('...ik,...k->...i', K_j, U_i) + U_j
    return K_new, U_new

def direct_general_scan_op(carry_i, carry_j):
    """General NxN matrix scan in direct complex64."""
    K_i, U_i = carry_i  # K: (...,N,N), U: (...,N)
    K_j, U_j = carry_j
    K_new = jnp.einsum('...ik,...kj->...ij', K_j, K_i)
    U_new = jnp.einsum('...ik,...k->...i', K_j, U_i) + U_j
    return K_new, U_new


# =============================================================================
# SSD (State Space Duality) chunked scan — Mamba-2 algorithm for scalar SSMs
# =============================================================================
#
# Ported from Mamba-2 (arXiv 2405.21060, Listing 1) to JAX.
# Simplified for B=C=1 (scalar SSM per spatial position).
#
# Key idea: instead of scanning over T timesteps element-wise (memory-bound),
# split into chunks of size L, compute intra-chunk via batched matmul
# (tensor-core friendly), and scan over only T/L chunk boundaries.
#
# Works in linear complex space (same as linear_split operators above).

def _ssd_scalar_scan_op(carry_i, carry_j):
    """Scalar associative scan op for inter-chunk SSD propagation.

    Same recurrence as linear_split_scalar_scan_op but operates on
    complex64 directly (used for the tiny T/L inter-chunk scan).
    """
    a_i, x_i = carry_i
    a_j, x_j = carry_j
    return a_j * a_i, a_j * x_i + x_j


def ssd_scan(A, X, chunk_size=8, initial_state=None):
    """SSD chunked scan for scalar SSM: h_t = A_t * h_{t-1} + X_t

    Ported from Mamba-2 Listing 1 with B=C=identity (d_state=1).
    Works with complex A, X for Fourier-domain CSSM.

    Args:
        A: (B, T, ...) scalar decay in LINEAR space (complex64)
        X: (B, T, ...) input in LINEAR space (complex64)
        chunk_size: chunk size L (default 8). T must be divisible by L.
        initial_state: optional (B, 1, ...) initial state (complex64)

    Returns:
        Y: (B, T, ...) accumulated states at all timesteps (complex64)
    """
    B_dim, T = A.shape[0], A.shape[1]
    spatial = A.shape[2:]
    L = chunk_size
    assert T % L == 0, f"T={T} must be divisible by chunk_size={L}"
    n_chunks = T // L

    # Reshape into chunks: (B, T, ...) → (B, n_chunks, L, ...)
    A = A.reshape(B_dim, n_chunks, L, *spatial)
    X = X.reshape(B_dim, n_chunks, L, *spatial)

    # --- Step 1: Intra-chunk via causal matmul ---
    # Cumulative product of A within each chunk (in log-space for numerical stability)
    log_A = jnp.log(A + 0j)  # ensure complex
    log_A_cumsum = jnp.cumsum(log_A, axis=2)  # (B, n_chunks, L, ...)

    # Build L×L lower-triangular decay matrix per spatial position
    # M[i,j] = prod(A[j+1..i]) = exp(cumsum[i] - cumsum[j]) for i >= j
    # We use einsum-friendly indexing: move L to last dims
    # log_A_cumsum shape: (B, n_chunks, L, *spatial)
    # We need: (B, n_chunks, L_i, L_j, *spatial)
    log_A_i = log_A_cumsum[:, :, :, None]  # (..., L, 1, *spatial) — but spatial is after
    log_A_j = log_A_cumsum[:, :, None, :]  # (..., 1, L, *spatial)

    # For arbitrary spatial dims, we need to insert the new axis in the right place
    # log_A_cumsum: (B, nc, L, s0, s1, ...)
    # We want: diff[b,c,i,j,s0,s1,...] = cumsum[b,c,i,s0,...] - cumsum[b,c,j,s0,...]
    n_spatial = len(spatial)
    # Expand dims for broadcasting: i-axis at position 2, j-axis at position 3
    expand_i = log_A_cumsum[:, :, :, None]  # (B, nc, L, 1, *spatial)
    expand_j = log_A_cumsum[:, :, None, :]  # (B, nc, 1, L, *spatial)
    log_decay = expand_i - expand_j  # (B, nc, L, L, *spatial)

    # Causal mask: only i >= j
    mask = jnp.tril(jnp.ones((L, L), dtype=bool))
    # Broadcast mask to match: (L, L) → (1, 1, L, L, 1, 1, ...)
    mask_shape = (1, 1, L, L) + (1,) * n_spatial
    mask = mask.reshape(mask_shape)

    # Build causal decay matrix M
    # Zero upper triangle BEFORE exp to prevent overflow → NaN in backward pass
    log_decay = jnp.where(mask, log_decay, jnp.zeros_like(log_decay))
    M = jnp.where(mask, jnp.exp(log_decay), jnp.zeros_like(log_decay))
    # M: (B, nc, L, L, *spatial)

    # Intra-chunk output: Y_diag[b,c,i,...] = sum_j M[b,c,i,j,...] * X[b,c,j,...]
    # X: (B, nc, L, *spatial) — we need X at the j-axis
    X_expand = X[:, :, None, :]  # (B, nc, 1, L, *spatial)
    Y_diag = jnp.sum(M * X_expand, axis=3)  # sum over j → (B, nc, L, *spatial)

    # --- Step 2: Chunk boundary states ---
    # How much each position decays to the end of its chunk
    # decay_to_end[j] = exp(cumsum[L-1] - cumsum[j])
    last_cumsum = log_A_cumsum[:, :, -1:]  # (B, nc, 1, *spatial)
    decay_to_end = jnp.exp(last_cumsum - log_A_cumsum)  # (B, nc, L, *spatial)
    # State at chunk boundary = sum of decayed inputs
    chunk_states = jnp.sum(decay_to_end * X, axis=2)  # (B, nc, *spatial)
    # Total decay across each chunk
    chunk_decay = jnp.exp(log_A_cumsum[:, :, -1])  # (B, nc, *spatial)

    # --- Step 3: Inter-chunk scan (tiny: only n_chunks elements) ---
    if initial_state is not None:
        # Prepend initial state as chunk 0
        chunk_states = jnp.concatenate([initial_state, chunk_states], axis=1)
        # Prepend identity decay for initial state
        ones_decay = jnp.ones_like(chunk_decay[:, :1])
        chunk_decay = jnp.concatenate([ones_decay, chunk_decay], axis=1)

        _, propagated = jax.lax.associative_scan(
            _ssd_scalar_scan_op, (chunk_decay, chunk_states), axis=1
        )
        # propagated: (B, n_chunks+1, *spatial)
        # We want the state ENTERING each chunk (i.e., output of previous chunk)
        # propagated[0] = initial_state, propagated[1] = state after chunk 0, etc.
        boundary_states = propagated[:, :-1]  # (B, nc, *spatial) — state entering each chunk
    else:
        # No initial state — state entering chunk 0 is zero, so we shift
        _, propagated = jax.lax.associative_scan(
            _ssd_scalar_scan_op, (chunk_decay, chunk_states), axis=1
        )
        # propagated[c] = accumulated state at END of chunk c
        # State entering chunk c = propagated[c-1], with zeros for chunk 0
        zeros = jnp.zeros_like(propagated[:, :1])
        boundary_states = jnp.concatenate([zeros, propagated[:, :-1]], axis=1)
        # boundary_states: (B, nc, *spatial)

    # --- Step 4: Correct intra-chunk outputs with boundary states ---
    # decay_from_start[i] = exp(cumsum[i]) = cumulative decay from chunk start to position i
    # But we need cumsum starting from 0 (position before first element of chunk)
    # cumsum[i] = sum(log_A[0..i]), so exp(cumsum[0]) = A[0], exp(cumsum[i]) = prod(A[0..i])
    # We want: decay from "entering state" to position i = prod(A[0..i]) = exp(cumsum[i])
    # But actually log_A_cumsum[0] = log_A[0], so exp gives A[0]. We need to include
    # the first element too. exp(cumsum) is correct: state * A[0] * A[1] * ... * A[i].
    decay_from_start = jnp.exp(log_A_cumsum)  # (B, nc, L, *spatial)
    # Broadcast boundary state into chunk positions
    Y_off = boundary_states[:, :, None] * decay_from_start  # (B, nc, L, *spatial)

    Y = (Y_diag + Y_off).reshape(B_dim, T, *spatial)
    return Y


# =============================================================================
# Quadratic (attention) scan — O(T²) but fully parallel
# =============================================================================

def quadratic_scan(A, X):
    """SSM via quadratic (attention) form: Y = M @ X.

    Builds the T×T causal decay matrix and contracts over source time.
    O(T²) per position but fully parallel — no sequential dependency.
    For T=8 this is 64 multiply-adds per position, batched across all
    (B, C, H, W_freq) positions.

    Args:
        A: (B, T, ...) complex decay per position per timestep (linear space)
        X: (B, T, ...) complex input (linear space)
    Returns:
        Y: (B, T, ...) scan output, identical to sequential h_t = A_t * h_{t-1} + X_t
    """
    B, T = A.shape[0], A.shape[1]
    spatial = A.shape[2:]

    # Cumulative product in log-space for numerical stability
    log_A = jnp.log(A + 0j)
    log_A_cumsum = jnp.cumsum(log_A, axis=1)  # (B, T, ...)

    # M[t,s] = prod_{k=s+1}^{t} A_k = exp(cumsum[t] - cumsum[s])  for t >= s
    # Expand for broadcasting: (B, T_target, 1, ...) - (B, 1, T_source, ...)
    n_spatial = len(spatial)
    expand_t = log_A_cumsum[:, :, None]  # (B, T, 1, ...)
    expand_s = log_A_cumsum[:, None, :]  # (B, 1, T, ...)
    log_decay = expand_t - expand_s  # (B, T, T, ...)

    # Causal mask: only t >= s
    mask_shape = (1, T, T) + (1,) * n_spatial
    mask = jnp.tril(jnp.ones((T, T), dtype=bool)).reshape(mask_shape)
    # Zero upper triangle BEFORE exp to prevent overflow → NaN in backward pass
    # (exp of large positive in upper triangle → inf, and 0 * inf = NaN in gradient)
    log_decay = jnp.where(mask, log_decay, jnp.zeros_like(log_decay))
    M = jnp.where(mask, jnp.exp(log_decay), jnp.zeros_like(log_decay))

    # Y[t] = sum_s M[t,s] * X[s]  (contraction over source time axis=2)
    X_expand = X[:, None, :]  # (B, 1, T, ...)
    Y = jnp.sum(M * X_expand, axis=2)
    return Y


def chunked_quadratic_scan(A, X, chunk_size=8):
    """Chunked quadratic scan for longer sequences.

    Uses L×L quadratic attention within chunks plus inter-chunk recurrence.
    Identical to ssd_scan mathematically but expressed as chunked attention.

    Steps:
        1. L×L quadratic form within each chunk (parallel)
        2. Compute chunk boundary states (parallel)
        3. Inter-chunk scan (T/L elements, short)
        4. Correct intra-chunk outputs with boundary states (parallel)

    Args:
        A: (B, T, ...) complex decay (linear space)
        X: (B, T, ...) complex input (linear space)
        chunk_size: chunk size L (default 8). T must be divisible by L.
    Returns:
        Y: (B, T, ...) scan output
    """
    B_dim, T = A.shape[0], A.shape[1]
    spatial = A.shape[2:]
    L = chunk_size

    if T <= L:
        return quadratic_scan(A, X)

    assert T % L == 0, f"T={T} must be divisible by chunk_size={L}"
    n_chunks = T // L

    # Reshape into chunks: (B, T, ...) -> (B, n_chunks, L, ...)
    A_c = A.reshape(B_dim, n_chunks, L, *spatial)
    X_c = X.reshape(B_dim, n_chunks, L, *spatial)

    # --- Step 1: Intra-chunk via quadratic form ---
    # Build L×L causal decay matrix per chunk
    log_A = jnp.log(A_c + 0j)
    log_A_cumsum = jnp.cumsum(log_A, axis=2)  # (B, nc, L, ...)

    n_spatial = len(spatial)
    expand_i = log_A_cumsum[:, :, :, None]  # (B, nc, L, 1, ...)
    expand_j = log_A_cumsum[:, :, None, :]  # (B, nc, 1, L, ...)
    log_decay = expand_i - expand_j  # (B, nc, L, L, ...)

    mask_shape = (1, 1, L, L) + (1,) * n_spatial
    mask = jnp.tril(jnp.ones((L, L), dtype=bool)).reshape(mask_shape)
    # Zero upper triangle BEFORE exp to prevent overflow → NaN in backward pass
    log_decay = jnp.where(mask, log_decay, jnp.zeros_like(log_decay))
    M = jnp.where(mask, jnp.exp(log_decay), jnp.zeros_like(log_decay))

    # Intra-chunk output: Y_diag[b,c,i,...] = sum_j M[b,c,i,j,...] * X[b,c,j,...]
    X_expand = X_c[:, :, None, :]  # (B, nc, 1, L, ...)
    Y_diag = jnp.sum(M * X_expand, axis=3)  # (B, nc, L, ...)

    # --- Step 2: Chunk boundary states ---
    last_cumsum = log_A_cumsum[:, :, -1:]  # (B, nc, 1, ...)
    decay_to_end = jnp.exp(last_cumsum - log_A_cumsum)  # (B, nc, L, ...)
    chunk_states = jnp.sum(decay_to_end * X_c, axis=2)  # (B, nc, ...)
    chunk_decay = jnp.exp(log_A_cumsum[:, :, -1])  # (B, nc, ...)

    # --- Step 3: Inter-chunk scan ---
    _, propagated = jax.lax.associative_scan(
        _ssd_scalar_scan_op, (chunk_decay, chunk_states), axis=1
    )
    # State entering chunk c = propagated[c-1], zeros for chunk 0
    zeros = jnp.zeros_like(propagated[:, :1])
    boundary_states = jnp.concatenate([zeros, propagated[:, :-1]], axis=1)

    # --- Step 4: Correct intra-chunk outputs with boundary states ---
    decay_from_start = jnp.exp(log_A_cumsum)  # (B, nc, L, ...)
    Y_off = boundary_states[:, :, None] * decay_from_start  # (B, nc, L, ...)

    Y = (Y_diag + Y_off).reshape(B_dim, T, *spatial)
    return Y


# =============================================================================
# Legacy phase-split scan operators (DEPRECATED — kept for benchmark comparison)
# =============================================================================
# These use (mag_bf16, phase_bf16) in GOOM log-space, requiring expensive
# trig (cos/sin/atan2) in _ps_log_add_exp. The linear-split operators above
# are much faster because they use only multiply + add.

def complex64_to_phase_split(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert GOOM complex64 to (mag_bf16, phase_bf16). DEPRECATED."""
    return x.real.astype(jnp.bfloat16), x.imag.astype(jnp.bfloat16)


def phase_split_to_complex64(mag: jnp.ndarray, phase: jnp.ndarray) -> jnp.ndarray:
    """Convert (mag, phase) back to GOOM complex64. DEPRECATED."""
    return (mag.astype(jnp.float32) + 1j * phase.astype(jnp.float32)).astype(jnp.complex64)


def _ps_log_add_exp(
    mag_a: jnp.ndarray, phase_a: jnp.ndarray,
    mag_b: jnp.ndarray, phase_b: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """log(exp(a) + exp(b)) in phase-split representation. DEPRECATED."""
    out_dtype = mag_a.dtype
    ma = mag_a.astype(jnp.float32)
    mb = mag_b.astype(jnp.float32)
    pa = phase_a.astype(jnp.float32)
    pb = phase_b.astype(jnp.float32)

    c = jax.lax.stop_gradient(jnp.maximum(ma, mb))
    ea = jnp.exp(ma - c)
    eb = jnp.exp(mb - c)

    re = ea * jnp.cos(pa) + eb * jnp.cos(pb)
    im = ea * jnp.sin(pa) + eb * jnp.sin(pb)

    new_mag = c + 0.5 * jnp.log(jnp.maximum(re * re + im * im, 1e-30))
    new_phase = jnp.arctan2(im, re)

    return new_mag.astype(out_dtype), new_phase.astype(out_dtype)


def phase_split_scalar_scan_op(carry_i, carry_j):
    """Scalar scan op in phase-split GOOM. DEPRECATED — use linear_split_scalar_scan_op."""
    k_mag_i, k_phase_i, u_mag_i, u_phase_i = carry_i
    k_mag_j, k_phase_j, u_mag_j, u_phase_j = carry_j
    k_mag_new = k_mag_j + k_mag_i
    k_phase_new = k_phase_j + k_phase_i
    ku_mag = k_mag_j + u_mag_i
    ku_phase = k_phase_j + u_phase_i
    u_mag_new, u_phase_new = _ps_log_add_exp(ku_mag, ku_phase, u_mag_j, u_phase_j)
    return k_mag_new, k_phase_new, u_mag_new, u_phase_new


def phase_split_2x2_scan_op(carry_i, carry_j):
    """2x2 scan op in phase-split GOOM. DEPRECATED — use linear_split_2x2_scan_op."""
    K_mag_i, K_phase_i, U_mag_i, U_phase_i = carry_i
    K_mag_j, K_phase_j, U_mag_j, U_phase_j = carry_j

    def _ps_matmul_2x2(Am, Ap, Bm, Bp):
        rows_m, rows_p = [], []
        for i in range(2):
            cols_m, cols_p = [], []
            for j in range(2):
                m0 = Am[..., i, 0] + Bm[..., 0, j]
                p0 = Ap[..., i, 0] + Bp[..., 0, j]
                m1 = Am[..., i, 1] + Bm[..., 1, j]
                p1 = Ap[..., i, 1] + Bp[..., 1, j]
                cm, cp = _ps_log_add_exp(m0, p0, m1, p1)
                cols_m.append(cm)
                cols_p.append(cp)
            rows_m.append(jnp.stack(cols_m, axis=-1))
            rows_p.append(jnp.stack(cols_p, axis=-1))
        return jnp.stack(rows_m, axis=-2), jnp.stack(rows_p, axis=-2)

    def _ps_matvec_2x2(Am, Ap, vm, vp):
        ys_m, ys_p = [], []
        for i in range(2):
            m0 = Am[..., i, 0] + vm[..., 0]
            p0 = Ap[..., i, 0] + vp[..., 0]
            m1 = Am[..., i, 1] + vm[..., 1]
            p1 = Ap[..., i, 1] + vp[..., 1]
            cm, cp = _ps_log_add_exp(m0, p0, m1, p1)
            ys_m.append(cm)
            ys_p.append(cp)
        return jnp.stack(ys_m, axis=-1), jnp.stack(ys_p, axis=-1)

    K_mag_new, K_phase_new = _ps_matmul_2x2(K_mag_j, K_phase_j, K_mag_i, K_phase_i)
    Ku_mag, Ku_phase = _ps_matvec_2x2(K_mag_j, K_phase_j, U_mag_i, U_phase_i)
    u_mag_list, u_phase_list = [], []
    for k in range(2):
        m, p = _ps_log_add_exp(Ku_mag[..., k], Ku_phase[..., k], U_mag_j[..., k], U_phase_j[..., k])
        u_mag_list.append(m)
        u_phase_list.append(p)
    return (K_mag_new, K_phase_new, jnp.stack(u_mag_list, axis=-1), jnp.stack(u_phase_list, axis=-1))


def phase_split_3x3_scan_op(carry_i, carry_j):
    """3x3 scan op in phase-split GOOM. DEPRECATED — use linear_split_3x3_scan_op."""
    K_mag_i, K_phase_i, U_mag_i, U_phase_i = carry_i
    K_mag_j, K_phase_j, U_mag_j, U_phase_j = carry_j

    def _ps_matmul_3x3(Am, Ap, Bm, Bp):
        rows_m, rows_p = [], []
        for i in range(3):
            cols_m, cols_p = [], []
            for j in range(3):
                m0 = Am[..., i, 0] + Bm[..., 0, j]
                p0 = Ap[..., i, 0] + Bp[..., 0, j]
                m1 = Am[..., i, 1] + Bm[..., 1, j]
                p1 = Ap[..., i, 1] + Bp[..., 1, j]
                acc_m, acc_p = _ps_log_add_exp(m0, p0, m1, p1)
                m2 = Am[..., i, 2] + Bm[..., 2, j]
                p2 = Ap[..., i, 2] + Bp[..., 2, j]
                acc_m, acc_p = _ps_log_add_exp(acc_m, acc_p, m2, p2)
                cols_m.append(acc_m)
                cols_p.append(acc_p)
            rows_m.append(jnp.stack(cols_m, axis=-1))
            rows_p.append(jnp.stack(cols_p, axis=-1))
        return jnp.stack(rows_m, axis=-2), jnp.stack(rows_p, axis=-2)

    def _ps_matvec_3x3(Am, Ap, vm, vp):
        ys_m, ys_p = [], []
        for i in range(3):
            m0 = Am[..., i, 0] + vm[..., 0]
            p0 = Ap[..., i, 0] + vp[..., 0]
            m1 = Am[..., i, 1] + vm[..., 1]
            p1 = Ap[..., i, 1] + vp[..., 1]
            acc_m, acc_p = _ps_log_add_exp(m0, p0, m1, p1)
            m2 = Am[..., i, 2] + vm[..., 2]
            p2 = Ap[..., i, 2] + vp[..., 2]
            acc_m, acc_p = _ps_log_add_exp(acc_m, acc_p, m2, p2)
            ys_m.append(acc_m)
            ys_p.append(acc_p)
        return jnp.stack(ys_m, axis=-1), jnp.stack(ys_p, axis=-1)

    K_mag_new, K_phase_new = _ps_matmul_3x3(K_mag_j, K_phase_j, K_mag_i, K_phase_i)
    Ku_mag, Ku_phase = _ps_matvec_3x3(K_mag_j, K_phase_j, U_mag_i, U_phase_i)
    u_mag_list, u_phase_list = [], []
    for k in range(3):
        m, p = _ps_log_add_exp(Ku_mag[..., k], Ku_phase[..., k], U_mag_j[..., k], U_phase_j[..., k])
        u_mag_list.append(m)
        u_phase_list.append(p)
    return (K_mag_new, K_phase_new, jnp.stack(u_mag_list, axis=-1), jnp.stack(u_phase_list, axis=-1))
