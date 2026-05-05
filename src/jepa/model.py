"""
CSSM-JEPA: Full model combining encoder, predictor, masking, and loss.

Joint Embedding Predictive Architecture for video self-supervised learning
with dual spectral and feature space prediction.

Two variants:
1. CSSMJEPA: Spatial masking (original) - masks spatial regions across all frames
2. CSSMVJEPA: V-JEPA style causal masking - predicts future from past (recommended)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Tuple, Optional, Any

from .encoder import VideoEncoder, update_ema_params
from .predictor import JEPAPredictor, PositionEmbedding
from .masking import (
    TubeMasking, MaskInfo, extract_at_indices,
    VJEPAMasking, VJEPAMaskInfo, extract_context_target, extract_masked_target_tokens
)
from .loss import jepa_dual_loss, variance_covariance_loss


class CSSMJEPA(nn.Module):
    """
    CSSM-JEPA: Video self-supervised learning model.

    Architecture:
        1. Online Encoder: CSSM-SHViT processing visible patches
        2. Target Encoder: EMA copy providing prediction targets
        3. Predictor: Lightweight transformer predicting masked regions
        4. Dual Loss: Spectral + feature space prediction

    Args:
        encoder_model: Base encoder model name ('cssm_shvit_s4', etc.)
        embed_dim: Encoder output dimension
        predictor_depth: Number of predictor transformer blocks
        predictor_heads: Number of predictor attention heads
        mask_ratio: Fraction of patches to mask
        tube_size: Spatial size of masking tubes
        alpha: Spectral loss weight
        beta: Feature loss weight
        ema_decay: EMA decay rate for target encoder
        vicreg_weight: Weight for VICReg regularization (0 to disable)
    """
    encoder_model: str = 'cssm_shvit_s4'
    embed_dim: int = 512  # Should match encoder output dim
    predictor_depth: int = 4
    predictor_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    mask_ratio: float = 0.75
    tube_size: Tuple[int, int] = (2, 2)
    alpha: float = 1.0  # Spectral loss weight
    beta: float = 1.0   # Feature loss weight
    ema_decay: float = 0.996
    vicreg_weight: float = 0.0  # Optional VICReg regularization
    rope_mode: str = 'spatiotemporal'

    def setup(self):
        """Initialize model components."""
        # Online encoder
        self.online_encoder = VideoEncoder(
            base_model=self.encoder_model,
            output_spectral=True,
            remove_head=True,
            rope_mode=self.rope_mode,
        )

        # Predictor
        self.predictor = JEPAPredictor(
            embed_dim=self.embed_dim,
            depth=self.predictor_depth,
            num_heads=self.predictor_heads,
            mlp_ratio=self.predictor_mlp_ratio,
        )

        # Masking
        self.masker = TubeMasking(
            mask_ratio=self.mask_ratio,
            tube_size=self.tube_size,
            temporal_consistent=True,
        )

        # Learnable mask token
        self.mask_token = self.param(
            'mask_token',
            nn.initializers.normal(0.02),
            (1, 1, self.embed_dim)
        )

        # Position embeddings
        self.pos_embed = PositionEmbedding(
            embed_dim=self.embed_dim,
            max_h=14,  # For 224 input with patch_size=16 and 4 stages
            max_w=14,
            max_t=16,
        )

    @nn.compact
    def __call__(
        self,
        video: jnp.ndarray,
        rng: jax.random.PRNGKey,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Forward pass for training.

        Args:
            video: Input video (B, T, H, W, 3)
            rng: Random key for masking
            training: Training mode flag

        Returns:
            loss: Scalar total loss
            metrics: Dict with all loss components
        """
        B, T, H, W, C = video.shape

        # Split RNG
        rng, mask_rng, drop_rng = jax.random.split(rng, 3)

        # === Online Encoder ===
        # Encode full video to get features
        online_features, online_spectral, _ = self.online_encoder(
            video, training=training
        )  # (B, T, H', W', D)

        _, _, H_feat, W_feat, D = online_features.shape

        # === Target Encoder ===
        # Note: Target encoder uses same architecture but different params (EMA)
        # In practice, the target params are maintained separately and passed in
        # For now, we use stop_gradient on online encoder output as placeholder
        # The actual EMA update happens in the training loop
        target_features = jax.lax.stop_gradient(online_features)
        target_spectral = jax.lax.stop_gradient(online_spectral)

        # === Masking ===
        # Generate tube mask
        mask_info = self.masker(
            (B, T, H_feat, W_feat),
            mask_rng
        )

        # === Prepare for Predictor ===
        # Flatten spatial dimensions: (B, T, H', W', D) -> (B, T, N, D)
        N = H_feat * W_feat
        online_flat = online_features.reshape(B, T, N, D)

        # Get visible and masked tokens
        num_visible = mask_info.num_visible
        num_masked = mask_info.num_masked

        # Gather visible tokens
        visible_indices = mask_info.visible_indices  # (B, num_visible)
        visible_tokens = self._gather_tokens(online_flat, visible_indices)  # (B, T, num_visible, D)

        # Pool over time for predictor context
        visible_context = visible_tokens.mean(axis=1)  # (B, num_visible, D)

        # Create mask tokens
        mask_tokens = jnp.broadcast_to(
            self.mask_token,
            (B, num_masked, D)
        )

        # Get position embeddings
        visible_pos = self.pos_embed(visible_indices)  # (B, num_visible, D)
        masked_pos = self.pos_embed(mask_info.masked_indices)  # (B, num_masked, D)

        # === Predictor ===
        predictions = self.predictor(
            context_tokens=visible_context,
            mask_tokens=mask_tokens,
            context_positions=visible_pos,
            mask_positions=masked_pos,
            deterministic=not training,
        )  # (B, num_masked, D)

        # === Extract Targets at Masked Positions ===
        # Average target features over time
        target_flat = target_features.reshape(B, T, N, D).mean(axis=1)  # (B, N, D)
        target_at_mask = self._gather_tokens_2d(target_flat, mask_info.masked_indices)  # (B, num_masked, D)

        # Spectral targets
        target_spectral_flat = target_spectral.reshape(B, T, N, -1).mean(axis=1)  # (B, N, D_freq)
        target_spectral_at_mask = self._gather_tokens_2d(
            target_spectral_flat,
            mask_info.masked_indices
        )  # (B, num_masked, D_freq)

        # Compute spectral of predictions
        pred_spectral = jnp.fft.rfft(predictions, axis=-1)

        # === Dual Loss ===
        loss, metrics = jepa_dual_loss(
            pred_features=predictions,
            target_features=target_at_mask,
            pred_spectral=pred_spectral,
            target_spectral=target_spectral_at_mask,
            alpha=self.alpha,
            beta=self.beta,
        )

        # === Optional VICReg Regularization ===
        if self.vicreg_weight > 0:
            # Apply to predictions to prevent collapse
            pred_flat = predictions.reshape(B * num_masked, D)
            vicreg_loss, vicreg_metrics = variance_covariance_loss(pred_flat)
            loss = loss + self.vicreg_weight * vicreg_loss
            metrics.update({f'vicreg_{k}': v for k, v in vicreg_metrics.items()})

        return loss, metrics

    def _gather_tokens(
        self,
        tokens: jnp.ndarray,
        indices: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Gather tokens at specified indices.

        Args:
            tokens: (B, T, N, D)
            indices: (B, num_indices)

        Returns:
            gathered: (B, T, num_indices, D)
        """
        B, T, N, D = tokens.shape
        num_indices = indices.shape[1]

        # Expand indices for gathering
        indices_exp = indices[:, jnp.newaxis, :, jnp.newaxis]
        indices_exp = jnp.broadcast_to(indices_exp, (B, T, num_indices, D))

        return jnp.take_along_axis(tokens, indices_exp, axis=2)

    def _gather_tokens_2d(
        self,
        tokens: jnp.ndarray,
        indices: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Gather tokens at specified indices (2D version).

        Args:
            tokens: (B, N, D)
            indices: (B, num_indices)

        Returns:
            gathered: (B, num_indices, D)
        """
        B, N, D = tokens.shape
        num_indices = indices.shape[1]

        indices_exp = indices[:, :, jnp.newaxis]
        indices_exp = jnp.broadcast_to(indices_exp, (B, num_indices, D))

        return jnp.take_along_axis(tokens, indices_exp, axis=1)

    def get_encoder_params(self) -> Dict[str, Any]:
        """Get online encoder parameters for EMA update."""
        # This would be called in the training loop
        # In Flax, you'd access params['online_encoder']
        pass

    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.embed_dim


def create_cssm_jepa(
    encoder: str = 'cssm_shvit_s4',
    embed_dim: int = 512,
    predictor_depth: int = 4,
    predictor_heads: int = 8,
    mask_ratio: float = 0.75,
    alpha: float = 1.0,
    beta: float = 1.0,
    ema_decay: float = 0.996,
    rope_mode: str = 'spatiotemporal',
    **kwargs,
) -> CSSMJEPA:
    """
    Factory function for CSSM-JEPA.

    Args:
        encoder: Encoder model name
        embed_dim: Feature dimension
        predictor_depth: Predictor depth
        predictor_heads: Predictor heads
        mask_ratio: Masking ratio
        alpha: Spectral loss weight
        beta: Feature loss weight
        ema_decay: EMA decay
        rope_mode: Position encoding mode

    Returns:
        CSSMJEPA model
    """
    return CSSMJEPA(
        encoder_model=encoder,
        embed_dim=embed_dim,
        predictor_depth=predictor_depth,
        predictor_heads=predictor_heads,
        mask_ratio=mask_ratio,
        alpha=alpha,
        beta=beta,
        ema_decay=ema_decay,
        rope_mode=rope_mode,
        **kwargs,
    )


# =============================================================================
# V-JEPA: Causal Future Prediction
# =============================================================================

class CSSMVJEPA(nn.Module):
    """
    CSSM-V-JEPA: Video self-supervised learning with causal future prediction.

    Key difference from CSSMJEPA:
    - CAUSAL: Uses past frames (context) to predict future frames (target)
    - Masks spatiotemporal BLOCKS in the future region
    - Better for learning temporal dynamics and motion

    Architecture:
        Video: [Context Frames | Target Frames]
                    ↓               ↓
              Online Encoder   Target Encoder (EMA)
                    ↓               ↓
              Context Features  Target Features (stop_grad)
                    ↓               ↓
                Predictor  →  Predict masked target blocks
                    ↓               ↓
                        Dual Loss (spectral + feature)

    Args:
        encoder_model: Base encoder model name ('cssm_shvit_s4', etc.)
        embed_dim: Encoder output dimension
        predictor_depth: Number of predictor transformer blocks
        predictor_heads: Number of predictor attention heads
        context_ratio: Fraction of frames used as context (default 0.5)
        mask_ratio: Fraction of target region to mask (default 0.75)
        num_mask_blocks: Number of spatiotemporal blocks to mask
        block_size: (t, h, w) size of each masked block
        alpha: Spectral loss weight
        beta: Feature loss weight
        ema_decay: EMA decay rate for target encoder
        vicreg_weight: Weight for VICReg regularization (0 to disable)
    """
    encoder_model: str = 'cssm_shvit_s4'
    embed_dim: int = 512
    predictor_depth: int = 4
    predictor_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    context_ratio: float = 0.5  # Fraction of frames as context
    mask_ratio: float = 0.75   # Fraction of target to mask
    num_mask_blocks: int = 4   # Number of masked blocks
    block_size: Tuple[int, int, int] = (2, 2, 2)  # (t, h, w)
    alpha: float = 1.0  # Spectral loss weight
    beta: float = 1.0   # Feature loss weight
    ema_decay: float = 0.996
    vicreg_weight: float = 0.0
    rope_mode: str = 'spatiotemporal'

    def setup(self):
        """Initialize model components."""
        # Online encoder
        self.online_encoder = VideoEncoder(
            base_model=self.encoder_model,
            output_spectral=True,
            remove_head=True,
            rope_mode=self.rope_mode,
        )

        # V-JEPA masking (causal)
        self.masker = VJEPAMasking(
            context_ratio=self.context_ratio,
            mask_ratio=self.mask_ratio,
            num_blocks=self.num_mask_blocks,
            block_size=self.block_size,
        )

        # NOTE: Predictor, mask_token, and position embeddings are created
        # dynamically in __call__ using the encoder's actual output dimension,
        # since embed_dim may not match the encoder output.

    @nn.compact
    def __call__(
        self,
        video: jnp.ndarray,
        rng: jax.random.PRNGKey,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Forward pass for V-JEPA training.

        Args:
            video: Input video (B, T, H, W, 3)
            rng: Random key for masking
            training: Training mode flag

        Returns:
            loss: Scalar total loss
            metrics: Dict with all loss components
        """
        B, T, H, W, C = video.shape

        # Split RNG
        rng, mask_rng, drop_rng = jax.random.split(rng, 3)

        # === Encode FULL video with online encoder ===
        online_features, online_spectral, _ = self.online_encoder(
            video, training=training
        )  # (B, T, H', W', D)

        _, _, H_feat, W_feat, D = online_features.shape

        # === Create predictor dynamically with encoder's output dimension ===
        predictor = JEPAPredictor(
            embed_dim=D,  # Use encoder's actual output dimension
            depth=self.predictor_depth,
            num_heads=self.predictor_heads,
            mlp_ratio=self.predictor_mlp_ratio,
            name='predictor',
        )

        # === Learnable mask token with encoder's output dimension ===
        mask_token = self.param(
            'mask_token',
            nn.initializers.normal(0.02),
            (1, 1, D)  # Use encoder's actual output dimension
        )

        # === Generate V-JEPA mask (causal) ===
        mask_info = self.masker(
            (B, T, H_feat, W_feat),
            mask_rng
        )

        # === Split into context and target ===
        T_c = mask_info.context_frames
        T_t = mask_info.target_frames

        # Context features (from online encoder)
        context_features = online_features[:, :T_c]  # (B, T_c, H', W', D)

        # Target features (from online encoder, but we use stop_gradient for EMA target)
        # Note: We don't use encoder's spectral output - we compute FFT on extracted features instead
        target_features = jax.lax.stop_gradient(online_features[:, T_c:])  # (B, T_t, H', W', D)

        # Pool context over spatial dimensions, keep time
        context_pooled = context_features.mean(axis=(2, 3))  # (B, T_c, D)

        # === Prepare mask tokens for target predictions ===
        num_masked = mask_info.num_target_masked
        mask_tokens = jnp.broadcast_to(
            mask_token,
            (B, num_masked, D)
        )

        # === Generate position embeddings with encoder's output dimension ===
        # Use FIXED maximum sizes for position embeddings (Flax params need fixed shapes)
        MAX_TEMPORAL = 32   # Maximum temporal positions
        MAX_SPATIAL = 196   # Maximum spatial positions (14x14)
        MAX_TARGET = MAX_TEMPORAL * MAX_SPATIAL  # Maximum target positions

        # Temporal position embeddings for context
        temporal_pos_embed = self.param(
            'temporal_pos_embed',
            nn.initializers.normal(0.02),
            (1, MAX_TEMPORAL, D)
        )
        # Context positions: first T_c frames (slice from fixed-size embedding)
        context_positions = temporal_pos_embed[:, :T_c, :]
        context_positions = jnp.broadcast_to(context_positions, (B, T_c, D))

        # Target position embeddings for masked tokens
        # Use factorized temporal + spatial position embeddings
        target_temporal_pos = self.param(
            'target_temporal_pos',
            nn.initializers.normal(0.02),
            (1, MAX_TEMPORAL, D)
        )
        target_spatial_pos = self.param(
            'target_spatial_pos',
            nn.initializers.normal(0.02),
            (1, MAX_SPATIAL, D)
        )

        # Compute positions for masked indices
        masked_indices = mask_info.target_masked_indices  # (B, num_masked)
        # Convert flat index to (t, h*w) for factorized embeddings
        num_spatial = H_feat * W_feat
        # Clamp indices to valid range
        masked_indices_safe = jnp.clip(masked_indices, 0, T_t * num_spatial - 1)
        t_idx = masked_indices_safe // num_spatial  # Temporal index
        s_idx = masked_indices_safe % num_spatial   # Spatial index

        # Clamp to MAX sizes
        t_idx = jnp.clip(t_idx, 0, MAX_TEMPORAL - 1)
        s_idx = jnp.clip(s_idx, 0, MAX_SPATIAL - 1)

        # Gather factorized position embeddings and sum
        t_pos = target_temporal_pos[0, t_idx, :]  # (B, num_masked, D)
        s_pos = target_spatial_pos[0, s_idx, :]   # (B, num_masked, D)
        mask_positions = t_pos + s_pos  # Factorized: temporal + spatial

        # === Predictor: context → predict masked target ===
        # Use context as keys/values, mask tokens as queries
        predictions = predictor(
            context_tokens=context_pooled,  # (B, T_c, D)
            mask_tokens=mask_tokens,        # (B, num_masked, D)
            context_positions=context_positions,
            mask_positions=mask_positions,
            deterministic=not training,
        )  # (B, num_masked, D)

        # === Extract targets at masked positions ===
        target_at_mask = extract_masked_target_tokens(target_features, mask_info)  # (B, num_masked, D)

        # === Numerical stability: clip and replace NaN ===
        predictions = jnp.clip(predictions, -1e4, 1e4)
        predictions = jnp.where(jnp.isnan(predictions), 0.0, predictions)
        target_at_mask = jnp.clip(target_at_mask, -1e4, 1e4)
        target_at_mask = jnp.where(jnp.isnan(target_at_mask), 0.0, target_at_mask)

        # Compute spectral features CONSISTENTLY for both prediction and target
        # Apply 1D FFT over feature dimension for both
        # (Don't use encoder's 2D spatial FFT as it's incompatible with per-position predictions)
        pred_spectral = jnp.fft.rfft(predictions, axis=-1)  # (B, num_masked, D//2+1)
        target_spectral_at_mask = jnp.fft.rfft(target_at_mask, axis=-1)  # (B, num_masked, D//2+1)

        # === Dual Loss ===
        loss, metrics = jepa_dual_loss(
            pred_features=predictions,
            target_features=target_at_mask,
            pred_spectral=pred_spectral,
            target_spectral=target_spectral_at_mask,
            alpha=self.alpha,
            beta=self.beta,
        )

        # Add V-JEPA specific metrics
        metrics['context_frames'] = T_c
        metrics['target_frames'] = T_t
        metrics['num_masked_blocks'] = mask_info.num_blocks

        # === Optional VICReg Regularization ===
        if self.vicreg_weight > 0:
            pred_flat = predictions.reshape(B * num_masked, D)
            vicreg_loss, vicreg_metrics = variance_covariance_loss(pred_flat)
            loss = loss + self.vicreg_weight * vicreg_loss
            metrics.update({f'vicreg_{k}': v for k, v in vicreg_metrics.items()})

        # Final NaN protection
        loss = jnp.where(jnp.isnan(loss), 0.0, loss)
        loss = jnp.where(jnp.isinf(loss), 1e4, loss)

        return loss, metrics

    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.embed_dim


def create_cssm_vjepa(
    encoder: str = 'cssm_shvit_s4',
    embed_dim: int = 512,
    predictor_depth: int = 4,
    predictor_heads: int = 8,
    context_ratio: float = 0.5,
    mask_ratio: float = 0.75,
    num_mask_blocks: int = 4,
    block_size: Tuple[int, int, int] = (2, 2, 2),
    alpha: float = 1.0,
    beta: float = 1.0,
    ema_decay: float = 0.996,
    rope_mode: str = 'spatiotemporal',
    **kwargs,
) -> CSSMVJEPA:
    """
    Factory function for CSSM-V-JEPA (causal future prediction).

    Args:
        encoder: Encoder model name
        embed_dim: Feature dimension
        predictor_depth: Predictor depth
        predictor_heads: Predictor heads
        context_ratio: Fraction of frames as context (past)
        mask_ratio: Fraction of target to mask
        num_mask_blocks: Number of spatiotemporal blocks to mask
        block_size: (t, h, w) size of each block
        alpha: Spectral loss weight
        beta: Feature loss weight
        ema_decay: EMA decay
        rope_mode: Position encoding mode

    Returns:
        CSSMVJEPA model
    """
    return CSSMVJEPA(
        encoder_model=encoder,
        embed_dim=embed_dim,
        predictor_depth=predictor_depth,
        predictor_heads=predictor_heads,
        context_ratio=context_ratio,
        mask_ratio=mask_ratio,
        num_mask_blocks=num_mask_blocks,
        block_size=block_size,
        alpha=alpha,
        beta=beta,
        ema_decay=ema_decay,
        rope_mode=rope_mode,
        **kwargs,
    )
