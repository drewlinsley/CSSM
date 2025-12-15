"""
GOOM (Generalized Order of Magnitude) primitives for numerically stable log-space computation.

Direct port from https://github.com/hkozachkov/goom/tree/main/src/goom

GOOM represents values as complex numbers where:
- Real part = log(|x|) (log magnitude)
- Imaginary part = encodes sign information (via pi for negative)

This enables stable gradient flow through log/exp operations using custom VJPs.
"""

import math
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp


@dataclass
class GOOMConfig:
    """Configuration for GOOM operations."""
    keep_logs_finite: bool = True
    cast_all_logs_to_complex: bool = True


# Global config instance
config = GOOMConfig()


# =============================================================================
# Custom VJP Functions
# =============================================================================

@jax.custom_vjp
def goom_abs(x: jax.Array) -> jax.Array:
    """
    Absolute value with custom derivative that equals 1 at zero.

    Standard |x| has undefined gradient at x=0. This version returns
    gradient of 1 at zero, which is useful for GOOM computations.
    """
    return jnp.abs(x)


def goom_abs_fwd(x: jax.Array):
    """Forward pass for goom_abs."""
    return goom_abs(x), x


def goom_abs_bwd(x: jax.Array, g: jax.Array):
    """Backward pass for goom_abs - gradient is sign(x), but 1 at zero."""
    # Use sign, but replace 0 with 1
    grad = jnp.where(x == 0, 1.0, jnp.sign(x))
    return (g * grad,)


goom_abs.defvjp(goom_abs_fwd, goom_abs_bwd)


@jax.custom_vjp
def goom_exp(x: jax.Array) -> jax.Array:
    """
    Exponential with perturbed gradients for stability.

    Adds a signed epsilon in the backward pass to prevent gradient
    issues when x is very negative (exp(x) ≈ 0).
    """
    return jnp.exp(x)


def goom_exp_fwd(x: jax.Array):
    """Forward pass for goom_exp."""
    return goom_exp(x), x


def goom_exp_bwd(x: jax.Array, g: jax.Array):
    """Backward pass for goom_exp with signed epsilon perturbation."""
    eps = jnp.finfo(x.real.dtype).eps
    # Add signed epsilon based on sign of x for stability
    sign_eps = jnp.where(x.real < 0, -eps, eps)
    return (g * (jnp.exp(x) + sign_eps),)


goom_exp.defvjp(goom_exp_fwd, goom_exp_bwd)


@jax.custom_vjp
def goom_log(x: jax.Array) -> jax.Array:
    """
    Logarithm with finite-value clamping for stability.

    When keep_logs_finite=True, clamps outputs to prevent -inf values
    that would cause NaN gradients.
    """
    log_result = jnp.log(x)

    if config.keep_logs_finite:
        # Compute finite floor based on dtype
        snn = jnp.finfo(x.real.dtype).smallest_normal
        finite_floor = math.log(snn) * 2

        # Clamp to finite floor while PRESERVING the imaginary part (phase)
        # This is critical for opponent inhibition: log(-1) = log(1) + iπ
        # If we lose the iπ phase, destructive interference won't work
        keep_finite_idx = log_result.real < finite_floor
        if jnp.iscomplexobj(log_result):
            # Preserve the imaginary part (phase) when clamping the real part
            clamped_value = finite_floor + 1j * log_result.imag
        else:
            clamped_value = finite_floor
        log_result = jnp.where(keep_finite_idx, clamped_value, log_result)

    return log_result


def goom_log_fwd(x: jax.Array):
    """Forward pass for goom_log."""
    return goom_log(x), x


def goom_log_bwd(x: jax.Array, g: jax.Array):
    """Backward pass for goom_log with epsilon for stability."""
    eps = jnp.finfo(x.real.dtype).eps
    # Add epsilon to prevent division by zero
    return (g / (x + eps),)


goom_log.defvjp(goom_log_fwd, goom_log_bwd)


# =============================================================================
# Conversion Functions
# =============================================================================

def to_goom(x: jax.Array) -> jax.Array:
    """
    Convert float values to GOOM (complex log) representation.

    The GOOM representation encodes:
    - Real part: log(|x|)
    - Imaginary part: 0 for positive, pi for negative (via log of negative)

    Args:
        x: Input tensor (real or complex)

    Returns:
        Complex tensor in GOOM representation
    """
    if config.cast_all_logs_to_complex:
        # Cast to complex to handle negative values properly
        x_complex = x.astype(jnp.complex64) if not jnp.iscomplexobj(x) else x
        return goom_log(x_complex)
    else:
        return goom_log(x)


def from_goom(log_x: jax.Array) -> jax.Array:
    """
    Convert GOOM representation back to original values.

    For real-valued inputs that went through to_goom, the result will be
    essentially real (with negligible imaginary parts from floating point).
    For complex inputs (e.g., FFT coefficients), the full complex value
    is preserved.

    Args:
        log_x: Tensor in GOOM (complex log) representation

    Returns:
        Tensor with original values recovered (complex if input was complex)
    """
    return goom_exp(log_x)


# =============================================================================
# Utility Functions
# =============================================================================

def safe_to_goom(x: jax.Array, min_val: Optional[float] = None) -> jax.Array:
    """
    Safely convert to GOOM with optional minimum value clamping.

    Useful when input might contain zeros or very small values.

    Args:
        x: Input tensor
        min_val: Optional minimum value to clamp to before conversion

    Returns:
        GOOM representation
    """
    if min_val is not None:
        x = jnp.maximum(goom_abs(x), min_val)
    return to_goom(x)


def goom_real(log_x: jax.Array) -> jax.Array:
    """Extract real (log magnitude) part of GOOM representation."""
    return log_x.real if jnp.iscomplexobj(log_x) else log_x


def goom_sign(log_x: jax.Array) -> jax.Array:
    """
    Extract sign from GOOM representation.

    In GOOM, negative values have imaginary part ≈ pi.
    """
    if not jnp.iscomplexobj(log_x):
        return jnp.ones_like(log_x)

    # Sign is encoded in imaginary part: 0 = positive, pi = negative
    return jnp.where(jnp.abs(log_x.imag) > 1.0, -1.0, 1.0)
