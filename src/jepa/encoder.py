"""
Video Encoder for CSSM-JEPA.

Wraps CSSM-SHViT to process video input with optional spectral output.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional, Dict, Any

from ..models.cssm_shvit import (
    CSSMSHViT,
    cssm_shvit_s1, cssm_shvit_s2, cssm_shvit_s3, cssm_shvit_s4,
    CSSMSHViTBlock, PatchEmbed, Downsample, ConvBlock
)


# Factory functions registry
ENCODER_REGISTRY = {
    'cssm_shvit_s1': cssm_shvit_s1,
    'cssm_shvit_s2': cssm_shvit_s2,
    'cssm_shvit_s3': cssm_shvit_s3,
    'cssm_shvit_s4': cssm_shvit_s4,
}


class VideoEncoder(nn.Module):
    """
    Video encoder for CSSM-JEPA.

    Wraps CSSM-SHViT and optionally returns spectral features
    for the dual-loss objective.

    Args:
        base_model: Which CSSM-SHViT variant ('cssm_shvit_s1', ..., 'cssm_shvit_s4')
        output_spectral: If True, also return FFT of features
        remove_head: If True, remove classification head (for feature extraction)
        embed_dims: Override embed dims from base model
        depths: Override depths from base model
        cssm_type: CSSM type ('gated' or 'opponent')
        dense_mixing: Use dense LMME channel mixing
        rope_mode: Position encoding ('spatiotemporal', 'temporal', 'none')
    """
    base_model: str = 'cssm_shvit_s4'
    output_spectral: bool = True
    remove_head: bool = True
    num_classes: int = 1000
    embed_dims: Optional[Tuple[int, ...]] = None
    depths: Optional[Tuple[int, ...]] = None
    cssm_type: str = 'gated'
    dense_mixing: bool = False
    rope_mode: str = 'spatiotemporal'

    def setup(self):
        """Initialize the base model."""
        # Get factory function
        if self.base_model not in ENCODER_REGISTRY:
            raise ValueError(f"Unknown model: {self.base_model}. "
                           f"Available: {list(ENCODER_REGISTRY.keys())}")

        factory = ENCODER_REGISTRY[self.base_model]

        # Build model with overrides
        kwargs = {
            'num_classes': self.num_classes,
            'cssm_type': self.cssm_type,
            'dense_mixing': self.dense_mixing,
            'rope_mode': self.rope_mode,
        }
        if self.embed_dims is not None:
            kwargs['embed_dims'] = self.embed_dims
        if self.depths is not None:
            kwargs['depths'] = self.depths

        self.encoder = factory(**kwargs)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
        return_intermediate: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[Dict[str, jnp.ndarray]]]:
        """
        Encode video to features.

        Args:
            x: Video tensor (B, T, H, W, 3) or image (B, H, W, 3)
            training: Training mode flag
            return_intermediate: If True, return intermediate features

        Returns:
            features: (B, T, H', W', D) or (B, H', W', D) encoder features
            spectral: (B, T, H', W', D_freq) FFT of features (if output_spectral)
            intermediates: Dict of intermediate features (if return_intermediate)
        """
        deterministic = not training

        # Handle image input by adding time dimension
        if x.ndim == 4:
            # (B, H, W, C) -> (B, 1, H, W, C)
            x = x[:, jnp.newaxis, :, :, :]

        B, T, H, W, C = x.shape

        # We need to modify forward pass to get features before pooling
        # For now, we'll implement a custom forward

        intermediates = {} if return_intermediate else None

        # Process each frame through patch embedding
        # CSSM-SHViT expects (B, H, W, C) but we'll process video properly
        features = self._encode_video(x, deterministic, intermediates)

        # Compute spectral features if requested
        if self.output_spectral:
            # FFT over spatial dimensions
            spectral = jnp.fft.rfft2(features, axes=(-3, -2))
        else:
            spectral = None

        return features, spectral, intermediates

    def _encode_video(
        self,
        x: jnp.ndarray,
        deterministic: bool,
        intermediates: Optional[Dict],
    ) -> jnp.ndarray:
        """
        Custom video encoding that returns spatial features (not pooled).

        Args:
            x: (B, T, H, W, C) video input
            deterministic: If True, disable dropout

        Returns:
            features: (B, T, H', W', D) spatial features
        """
        B, T, H, W, C = x.shape

        # Reshape for batch processing: (B*T, H, W, C)
        x_flat = x.reshape(B * T, H, W, C)

        # Get model config
        embed_dims = self.encoder.embed_dims
        depths = self.encoder.depths
        use_cssm_stages = self.encoder.use_cssm_stages

        # Patch embedding
        x_flat = PatchEmbed(embed_dims[0], name='patch_embed')(x_flat, deterministic)

        # Reshape back to video: (B, T, H', W', D)
        _, H_p, W_p, D = x_flat.shape
        x_video = x_flat.reshape(B, T, H_p, W_p, D)

        # Store intermediate
        if intermediates is not None:
            intermediates['after_patch_embed'] = x_video

        # Process through stages
        # For CSSM stages, we need to handle the temporal dimension specially
        import numpy as np
        total_depth = sum(depths)
        dp_rates = np.linspace(0, self.encoder.drop_path_rate, total_depth)
        dp_idx = 0

        for stage_idx in range(4):
            # Downsample between stages (process as batch)
            if stage_idx > 0:
                B, T, H_curr, W_curr, D_curr = x_video.shape
                x_flat = x_video.reshape(B * T, H_curr, W_curr, D_curr)
                x_flat = Downsample(embed_dims[stage_idx],
                                   name=f'downsample{stage_idx}')(x_flat, deterministic)
                _, H_new, W_new, D_new = x_flat.shape
                x_video = x_flat.reshape(B, T, H_new, W_new, D_new)

            # Process blocks
            for block_idx in range(depths[stage_idx]):
                if use_cssm_stages[stage_idx]:
                    # CSSM block expects (B, T, H, W, C) - perfect for video!
                    x_video = CSSMSHViTBlock(
                        dim=embed_dims[stage_idx],
                        mlp_ratio=self.encoder.mlp_ratio,
                        drop_path=float(dp_rates[dp_idx]),
                        cssm_type=self.encoder.cssm_type,
                        dense_mixing=self.encoder.dense_mixing,
                        gate_activation=self.encoder.gate_activation,
                        num_timesteps=T,  # Use actual video length
                        kernel_size=self.encoder.kernel_sizes[stage_idx],
                        spectral_rho=self.encoder.spectral_rho,
                        rope_mode=self.encoder.rope_mode,
                        name=f'stage{stage_idx}_block{block_idx}'
                    )(x_video, deterministic)
                else:
                    # Conv block - process as batch
                    B, T, H_curr, W_curr, D_curr = x_video.shape
                    x_flat = x_video.reshape(B * T, H_curr, W_curr, D_curr)
                    x_flat = ConvBlock(
                        dim=embed_dims[stage_idx],
                        mlp_ratio=self.encoder.mlp_ratio,
                        drop_path=float(dp_rates[dp_idx]),
                        name=f'stage{stage_idx}_block{block_idx}'
                    )(x_flat, deterministic)
                    x_video = x_flat.reshape(B, T, H_curr, W_curr, D_curr)

                dp_idx += 1

            # Store intermediate
            if intermediates is not None:
                intermediates[f'after_stage{stage_idx}'] = x_video

        # Final norm (process as batch)
        B, T, H_final, W_final, D_final = x_video.shape
        x_flat = x_video.reshape(B * T, H_final, W_final, D_final)
        x_flat = nn.LayerNorm(name='norm')(x_flat)
        x_video = x_flat.reshape(B, T, H_final, W_final, D_final)

        return x_video

    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.encoder.embed_dims[-1]


class EMAEncoder(nn.Module):
    """
    EMA (Exponential Moving Average) target encoder.

    This is a wrapper that maintains EMA of the online encoder's parameters.
    The EMA update is done externally in the training loop.

    Note: In practice, we just use the same VideoEncoder class and
    manually update its parameters with EMA in the training loop.
    """
    encoder: VideoEncoder

    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,  # Always False for target encoder
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Forward pass (always in eval mode)."""
        features, spectral, _ = self.encoder(x, training=False, return_intermediate=False)
        return features, spectral


def update_ema_params(
    online_params: Dict[str, Any],
    target_params: Dict[str, Any],
    decay: float,
) -> Dict[str, Any]:
    """
    Update target parameters with exponential moving average.

    target = decay * target + (1 - decay) * online

    Args:
        online_params: Parameters from online encoder
        target_params: Parameters from target encoder
        decay: EMA decay rate (e.g., 0.996)

    Returns:
        Updated target parameters
    """
    return jax.tree_util.tree_map(
        lambda o, t: decay * t + (1 - decay) * o,
        online_params, target_params
    )
