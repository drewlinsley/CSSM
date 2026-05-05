"""
CSSM-DeiT3: DeiT3-Large with CSSM replacing self-attention.

Key design:
1. CSSM operates on 2D spatial structure with temporal recurrence
2. Preserves DeiT3's layer scale, pre-norm, and DropPath
3. Uses spatial max pool for classification (no CLS token)
4. Configurable timesteps for temporal recurrence
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Optional

from .cssm import GatedCSSM
from .deit3 import DropPath, PatchEmbed, Mlp


class CSSMDeiT3Block(nn.Module):
    """
    DeiT3 block with CSSM replacing self-attention.

    CSSM requires temporal dimension, so we:
    1. Add temporal dimension by repeating (B, H', W', C) -> (B, T, H', W', C)
    2. Apply CSSM with temporal recurrence
    3. Take last timestep for output

    Structure (pre-norm with layer scale):
        x -> Norm -> CSSM -> LayerScale -> DropPath -> + x
        x -> Norm -> MLP  -> LayerScale -> DropPath -> + x
    """
    dim: int
    num_heads: int = 16  # Kept for API compatibility
    mlp_ratio: float = 4.0
    drop: float = 0.0
    drop_path: float = 0.0
    init_values: float = 1e-6
    cssm_type: str = 'opponent'
    dense_mixing: bool = False
    concat_xy: bool = True
    gate_activation: str = 'sigmoid'
    num_timesteps: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, H', W', C)

        Returns:
            (B, H', W', C)
        """
        B, H, W, C = x.shape

        # Select CSSM class
        CSSM = GatedCSSM

        # --- CSSM Path ---
        residual = x
        x = nn.LayerNorm(name='norm1')(x)

        # Add temporal dimension: (B, H', W', C) -> (B, T, H', W', C)
        x_video = jnp.repeat(x[:, jnp.newaxis, :, :, :], self.num_timesteps, axis=1)

        # Apply CSSM
        x_video = CSSM(
            channels=self.dim,
            dense_mixing=self.dense_mixing,
            concat_xy=self.concat_xy,
            gate_activation=self.gate_activation,
            name='cssm'
        )(x_video)  # (B, T, H', W', C)

        # Take last timestep
        x = x_video[:, -1]  # (B, H', W', C)

        # Layer Scale for CSSM
        gamma1 = self.param(
            'gamma1',
            nn.initializers.constant(self.init_values),
            (self.dim,)
        )
        x = x * gamma1

        x = DropPath(self.drop_path)(x, deterministic=deterministic)
        x = residual + x

        # --- MLP Path ---
        residual = x
        x = nn.LayerNorm(name='norm2')(x)
        x = Mlp(
            hidden_dim=int(self.dim * self.mlp_ratio),
            out_dim=self.dim,
            drop=self.drop,
            name='mlp'
        )(x, deterministic=deterministic)

        # Layer Scale for MLP
        gamma2 = self.param(
            'gamma2',
            nn.initializers.constant(self.init_values),
            (self.dim,)
        )
        x = x * gamma2

        x = DropPath(self.drop_path)(x, deterministic=deterministic)
        x = residual + x

        return x


class CSSMDeiT3Large(nn.Module):
    """
    CSSM-DeiT3-Large: DeiT3-Large with CSSM replacing attention.

    Maintains exact same architecture as DeiT3-Large except:
    - Self-attention replaced with CSSM in all blocks
    - Configurable number of CSSM timesteps
    - Uses spatial max pool for classification
    """
    num_classes: int = 1000
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16  # Kept for API compatibility
    mlp_ratio: float = 4.0
    patch_size: int = 16
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    init_values: float = 1e-6
    cssm_type: str = 'opponent'
    dense_mixing: bool = False
    concat_xy: bool = True
    gate_activation: str = 'sigmoid'
    num_timesteps: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, H, W, 3) or (B, T, H, W, 3)

        Returns:
            (B, num_classes) logits
        """
        deterministic = not training

        # Handle video input - use last timestep for patch embedding
        if x.ndim == 5:
            x = x[:, -1]

        B, H, W, _ = x.shape

        # Patch embedding - maintains 2D structure
        x = PatchEmbed(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            name='patch_embed'
        )(x)  # (B, H', W', embed_dim)

        H_p, W_p = x.shape[1], x.shape[2]

        # 2D Position embeddings
        pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(0.02),
            (1, H_p, W_p, self.embed_dim)
        )
        x = x + pos_embed

        # Dropout after embedding
        if not deterministic and self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x, deterministic=False)

        # CSSM Transformer blocks with linear drop path increase
        dp_rates = np.linspace(0, self.drop_path_rate, self.depth)

        for i in range(self.depth):
            x = CSSMDeiT3Block(
                dim=self.embed_dim,
                mlp_ratio=self.mlp_ratio,
                drop=self.drop_rate,
                drop_path=float(dp_rates[i]),
                init_values=self.init_values,
                cssm_type=self.cssm_type,
                dense_mixing=self.dense_mixing,
                concat_xy=self.concat_xy,
                gate_activation=self.gate_activation,
                num_timesteps=self.num_timesteps,
                name=f'block{i}'
            )(x, deterministic=deterministic)

        # Final norm
        x = nn.LayerNorm(name='norm')(x)

        # Spatial max pool
        x = jnp.max(x, axis=(1, 2))  # (B, embed_dim)

        # Classification head
        x = nn.Dense(self.num_classes, name='head')(x)

        return x


# Model factory functions
def cssm_deit3_large_patch16_384(
    num_classes: int = 1000,
    num_timesteps: int = 8,
    cssm_type: str = 'opponent',
    **kwargs
):
    """CSSM-DeiT3-Large with 384x384 input."""
    return CSSMDeiT3Large(
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_size=16,
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )


def cssm_deit3_large_patch16_224(
    num_classes: int = 1000,
    num_timesteps: int = 8,
    cssm_type: str = 'opponent',
    **kwargs
):
    """CSSM-DeiT3-Large with 224x224 input."""
    return CSSMDeiT3Large(
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_size=16,
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )


def cssm_deit3_base_patch16_224(
    num_classes: int = 1000,
    num_timesteps: int = 8,
    cssm_type: str = 'opponent',
    **kwargs
):
    """CSSM-DeiT3-Base with 224x224 input."""
    return CSSMDeiT3Large(
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        patch_size=16,
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )


def cssm_deit3_small_patch16_224(
    num_classes: int = 1000,
    num_timesteps: int = 8,
    cssm_type: str = 'opponent',
    **kwargs
):
    """CSSM-DeiT3-Small with 224x224 input."""
    return CSSMDeiT3Large(
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        patch_size=16,
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )
