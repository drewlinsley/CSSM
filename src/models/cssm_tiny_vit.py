"""
CSSM-TinyViT: TinyViT with CSSM replacing self-attention.

Much faster and more memory-efficient than CSSM-DeiT3-Large.
~21M params with 224x224 input.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Optional

from .cssm import GatedCSSM
from .tiny_vit import DropPath, PatchEmbed, Mlp


class CSSMTinyViTBlock(nn.Module):
    """
    TinyViT block with CSSM replacing self-attention.

    CSSM operates on 2D spatial structure with temporal recurrence.
    """
    dim: int
    num_heads: int = 6  # For API compatibility
    mlp_ratio: float = 4.0
    drop: float = 0.0
    drop_path: float = 0.0
    init_values: Optional[float] = None
    cssm_type: str = 'opponent'
    dense_mixing: bool = False
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

        CSSM = GatedCSSM

        # CSSM path
        residual = x
        x = nn.LayerNorm(name='norm1')(x)

        # Add temporal dimension: (B, H', W', C) -> (B, T, H', W', C)
        x_video = jnp.repeat(x[:, jnp.newaxis, :, :, :], self.num_timesteps, axis=1)

        # Apply CSSM
        x_video = CSSM(
            channels=self.dim,
            dense_mixing=self.dense_mixing,
            gate_activation=self.gate_activation,
            name='cssm'
        )(x_video)

        # Take last timestep
        x = x_video[:, -1]

        if self.init_values is not None:
            gamma1 = self.param('gamma1', nn.initializers.constant(self.init_values), (self.dim,))
            x = x * gamma1

        x = DropPath(self.drop_path)(x, deterministic=deterministic)
        x = residual + x

        # MLP path
        residual = x
        x = nn.LayerNorm(name='norm2')(x)

        # Flatten for MLP: (B, H', W', C) -> (B, H'*W', C)
        x_flat = x.reshape(B, H * W, C)
        x_flat = Mlp(
            hidden_dim=int(self.dim * self.mlp_ratio),
            out_dim=self.dim,
            drop=self.drop,
            name='mlp'
        )(x_flat, deterministic=deterministic)
        x = x_flat.reshape(B, H, W, C)

        if self.init_values is not None:
            gamma2 = self.param('gamma2', nn.initializers.constant(self.init_values), (self.dim,))
            x = x * gamma2

        x = DropPath(self.drop_path)(x, deterministic=deterministic)
        x = residual + x

        return x


class CSSMTinyViT(nn.Module):
    """
    CSSM-TinyViT: Efficient CSSM vision transformer.

    Default config (~21M params + CSSM overhead):
    - embed_dim: 384
    - depth: 12
    - num_heads: 6 (unused, for API compat)
    """
    num_classes: int = 1000
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    patch_size: int = 16
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    init_values: Optional[float] = None
    cssm_type: str = 'opponent'
    dense_mixing: bool = False
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

        # Handle video input
        if x.ndim == 5:
            x = x[:, -1]

        B, H, W, _ = x.shape

        # Patch embedding - keeps 2D structure
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

        # Dropout
        if not deterministic and self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x, deterministic=False)

        # CSSM Transformer blocks
        dp_rates = np.linspace(0, self.drop_path_rate, self.depth)

        for i in range(self.depth):
            x = CSSMTinyViTBlock(
                dim=self.embed_dim,
                mlp_ratio=self.mlp_ratio,
                drop=self.drop_rate,
                drop_path=float(dp_rates[i]),
                init_values=self.init_values,
                cssm_type=self.cssm_type,
                dense_mixing=self.dense_mixing,
                gate_activation=self.gate_activation,
                num_timesteps=self.num_timesteps,
                name=f'block{i}'
            )(x, deterministic=deterministic)

        # Final norm
        x = nn.LayerNorm(name='norm')(x)

        # Spatial max pool (no CLS token)
        x = jnp.max(x, axis=(1, 2))  # (B, embed_dim)

        # Classification head
        x = nn.Dense(self.num_classes, name='head')(x)

        return x


# Factory functions

def cssm_tiny_vit_21m_224(
    num_classes: int = 1000,
    num_timesteps: int = 8,
    cssm_type: str = 'opponent',
    **kwargs
):
    """CSSM-TinyViT-21M with 224x224 input."""
    return CSSMTinyViT(
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        patch_size=16,
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )


def cssm_tiny_vit_5m_224(
    num_classes: int = 1000,
    num_timesteps: int = 8,
    cssm_type: str = 'opponent',
    **kwargs
):
    """CSSM-TinyViT-5M with 224x224 input (~5M params)."""
    return CSSMTinyViT(
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        patch_size=16,
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )


def cssm_tiny_vit_11m_224(
    num_classes: int = 1000,
    num_timesteps: int = 8,
    cssm_type: str = 'opponent',
    **kwargs
):
    """CSSM-TinyViT-11M with 224x224 input (~11M params)."""
    return CSSMTinyViT(
        num_classes=num_classes,
        embed_dim=256,
        depth=12,
        num_heads=4,
        mlp_ratio=4.0,
        patch_size=16,
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )
