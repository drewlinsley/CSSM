"""
Dual loss functions for CSSM-JEPA.

Implements both spectral (FFT) and feature space prediction losses,
with configurable weights for ablation studies.
"""

import jax.numpy as jnp
from typing import Dict, Tuple


def smooth_l1_loss(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    beta: float = 1.0,
) -> jnp.ndarray:
    """
    Smooth L1 loss (Huber loss) for feature space prediction.

    Less sensitive to outliers than MSE, smoother gradients than L1.

    Args:
        pred: Predicted features
        target: Target features
        beta: Threshold for switching between L1 and L2

    Returns:
        Scalar loss
    """
    diff = jnp.abs(pred - target)
    loss = jnp.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta
    )
    return loss.mean()


def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error loss."""
    return ((pred - target) ** 2).mean()


def cosine_loss(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """
    Cosine similarity loss.

    Args:
        pred: Predicted features (..., D)
        target: Target features (..., D)
        eps: Small epsilon for numerical stability

    Returns:
        Scalar loss (1 - cosine_similarity)
    """
    pred_norm = pred / (jnp.linalg.norm(pred, axis=-1, keepdims=True) + eps)
    target_norm = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + eps)
    cosine_sim = (pred_norm * target_norm).sum(axis=-1)
    return (1 - cosine_sim).mean()


def phase_loss(
    pred_phase: jnp.ndarray,
    target_phase: jnp.ndarray,
) -> jnp.ndarray:
    """
    Angular loss for phase prediction.

    Phase wraps around at 2*pi, so we use the circular distance.

    Args:
        pred_phase: Predicted phase angles
        target_phase: Target phase angles

    Returns:
        Scalar loss
    """
    # Circular distance: min(|a - b|, 2*pi - |a - b|)
    diff = jnp.abs(pred_phase - target_phase)
    circular_diff = jnp.minimum(diff, 2 * jnp.pi - diff)
    return circular_diff.mean()


def spectral_loss(
    pred_spectral: jnp.ndarray,
    target_spectral: jnp.ndarray,
    magnitude_weight: float = 1.0,
    phase_weight: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Spectral domain loss on FFT coefficients.

    Predicts both magnitude and phase of the Fourier transform,
    which captures both frequency content and spatial structure.

    Args:
        pred_spectral: Predicted FFT (complex)
        target_spectral: Target FFT (complex)
        magnitude_weight: Weight for magnitude loss
        phase_weight: Weight for phase loss
        eps: Small epsilon for numerical stability

    Returns:
        total_loss: Weighted sum of magnitude and phase losses
        metrics: Dict with individual loss components
    """
    # Magnitude (amplitude spectrum) with numerical stability
    pred_mag = jnp.abs(pred_spectral) + eps
    target_mag = jnp.abs(target_spectral) + eps

    # Clamp magnitudes to prevent extreme values
    pred_mag = jnp.clip(pred_mag, eps, 1e6)
    target_mag = jnp.clip(target_mag, eps, 1e6)

    # Log magnitude for scale invariance
    pred_log_mag = jnp.log1p(pred_mag)
    target_log_mag = jnp.log1p(target_mag)

    # Clamp log magnitudes
    pred_log_mag = jnp.clip(pred_log_mag, -20, 20)
    target_log_mag = jnp.clip(target_log_mag, -20, 20)

    mag_loss = mse_loss(pred_log_mag, target_log_mag)

    # Phase
    pred_phase = jnp.angle(pred_spectral)
    target_phase = jnp.angle(target_spectral)

    ph_loss = phase_loss(pred_phase, target_phase)

    # Combined loss with NaN protection
    total = magnitude_weight * mag_loss + phase_weight * ph_loss
    total = jnp.where(jnp.isnan(total), 0.0, total)

    metrics = {
        'spectral_magnitude_loss': mag_loss,
        'spectral_phase_loss': ph_loss,
        'spectral_total_loss': total,
    }

    return total, metrics


def feature_loss(
    pred_features: jnp.ndarray,
    target_features: jnp.ndarray,
    loss_type: str = 'smooth_l1',
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Feature space prediction loss.

    Args:
        pred_features: Predicted features from predictor
        target_features: Target features from EMA encoder
        loss_type: 'smooth_l1', 'mse', or 'cosine'

    Returns:
        loss: Scalar loss
        metrics: Dict with loss value
    """
    # Clamp features to prevent extreme values
    pred_features = jnp.clip(pred_features, -1e6, 1e6)
    target_features = jnp.clip(target_features, -1e6, 1e6)

    # Replace NaNs with zeros
    pred_features = jnp.where(jnp.isnan(pred_features), 0.0, pred_features)
    target_features = jnp.where(jnp.isnan(target_features), 0.0, target_features)

    if loss_type == 'smooth_l1':
        loss = smooth_l1_loss(pred_features, target_features)
    elif loss_type == 'mse':
        loss = mse_loss(pred_features, target_features)
    elif loss_type == 'cosine':
        loss = cosine_loss(pred_features, target_features)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # NaN protection
    loss = jnp.where(jnp.isnan(loss), 0.0, loss)

    metrics = {
        'feature_loss': loss,
    }

    return loss, metrics


def jepa_dual_loss(
    pred_features: jnp.ndarray,
    target_features: jnp.ndarray,
    pred_spectral: jnp.ndarray,
    target_spectral: jnp.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    feature_loss_type: str = 'smooth_l1',
    spectral_magnitude_weight: float = 1.0,
    spectral_phase_weight: float = 0.5,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Dual prediction loss for CSSM-JEPA.

    L = alpha * L_spectral + beta * L_feature

    This allows ablating between:
    - Spectral-only (alpha=1, beta=0)
    - Feature-only (alpha=0, beta=1)
    - Combined (alpha=1, beta=1)

    Args:
        pred_features: Predicted features from predictor
        target_features: Target features from EMA encoder (stop gradient)
        pred_spectral: FFT of predicted features (complex)
        target_spectral: FFT of target features (complex)
        alpha: Weight for spectral loss (default 1.0)
        beta: Weight for feature loss (default 1.0)
        feature_loss_type: Type of feature loss ('smooth_l1', 'mse', 'cosine')
        spectral_magnitude_weight: Weight for spectral magnitude component
        spectral_phase_weight: Weight for spectral phase component

    Returns:
        total_loss: Combined loss scalar
        metrics: Dict with all loss components for logging
    """
    metrics = {}

    # Feature space loss
    if beta > 0:
        feat_loss, feat_metrics = feature_loss(
            pred_features,
            target_features,
            loss_type=feature_loss_type,
        )
        metrics.update(feat_metrics)
    else:
        feat_loss = 0.0

    # Spectral loss
    if alpha > 0:
        spec_loss, spec_metrics = spectral_loss(
            pred_spectral,
            target_spectral,
            magnitude_weight=spectral_magnitude_weight,
            phase_weight=spectral_phase_weight,
        )
        metrics.update(spec_metrics)
    else:
        spec_loss = 0.0

    # Combined loss
    total_loss = alpha * spec_loss + beta * feat_loss

    metrics['loss_total'] = total_loss
    metrics['loss_spectral_weighted'] = alpha * spec_loss if alpha > 0 else 0.0
    metrics['loss_feature_weighted'] = beta * feat_loss if beta > 0 else 0.0
    metrics['alpha'] = alpha
    metrics['beta'] = beta

    return total_loss, metrics


def variance_covariance_loss(
    features: jnp.ndarray,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    VICReg-style variance and covariance regularization.

    Prevents representation collapse by:
    1. Variance: Ensuring each feature dimension has sufficient variance
    2. Covariance: Decorrelating feature dimensions

    Args:
        features: (B, D) batch of feature vectors
        variance_weight: Weight for variance loss
        covariance_weight: Weight for covariance loss
        eps: Small constant for numerical stability

    Returns:
        loss: Combined variance + covariance loss
        metrics: Dict with individual components
    """
    B, D = features.shape

    # Center features
    features_centered = features - features.mean(axis=0, keepdims=True)

    # Variance loss: penalize if std < 1
    std = jnp.sqrt(features_centered.var(axis=0) + eps)
    var_loss = jnp.maximum(0, 1 - std).mean()

    # Covariance loss: penalize off-diagonal correlations
    cov = (features_centered.T @ features_centered) / (B - 1)
    # Zero out diagonal
    cov_off_diag = cov - jnp.diag(jnp.diag(cov))
    cov_loss = (cov_off_diag ** 2).sum() / D

    total = variance_weight * var_loss + covariance_weight * cov_loss

    metrics = {
        'variance_loss': var_loss,
        'covariance_loss': cov_loss,
        'vicreg_total': total,
    }

    return total, metrics
