"""
TinyViT / ViT-Small: Efficient vision transformer (~21M params).

Based on ViT-Small configuration:
- embed_dim: 384
- depth: 12
- num_heads: 6
- mlp_ratio: 4
- patch_size: 16
- image_size: 224

This is much faster and more memory-efficient than DeiT3-Large.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Optional


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""
    rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        if self.rate == 0.0 or deterministic:
            return x
        keep_prob = 1.0 - self.rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rng = self.make_rng('dropout')
        mask = jax.random.bernoulli(rng, keep_prob, shape)
        return x * mask / keep_prob


class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""
    embed_dim: int = 384
    patch_size: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, H, W, 3)
        Returns:
            (B, H', W', embed_dim) where H' = H // patch_size
        """
        x = nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            name='proj'
        )(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    dim: int
    num_heads: int = 6
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        B, N, C = x.shape
        head_dim = C // self.num_heads
        scale = head_dim ** -0.5

        qkv = nn.Dense(3 * C, use_bias=self.qkv_bias, name='qkv')(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = jax.nn.softmax(attn, axis=-1)

        if not deterministic and self.attn_drop > 0:
            attn = nn.Dropout(self.attn_drop)(attn, deterministic=False)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(C, name='proj')(x)

        if not deterministic and self.proj_drop > 0:
            x = nn.Dropout(self.proj_drop)(x, deterministic=False)

        return x


class Mlp(nn.Module):
    """MLP block."""
    hidden_dim: int
    out_dim: int
    drop: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, name='fc1')(x)
        x = jax.nn.gelu(x)
        if not deterministic and self.drop > 0:
            x = nn.Dropout(self.drop)(x, deterministic=False)
        x = nn.Dense(self.out_dim, name='fc2')(x)
        if not deterministic and self.drop > 0:
            x = nn.Dropout(self.drop)(x, deterministic=False)
        return x


class TinyViTBlock(nn.Module):
    """Transformer block with pre-norm and optional layer scale."""
    dim: int
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop: float = 0.0
    drop_path: float = 0.0
    init_values: Optional[float] = None  # Layer scale init (None = no layer scale)

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # x: (B, N, C) where N = H' * W'

        # Attention path
        residual = x
        x = nn.LayerNorm(name='norm1')(x)
        x = Attention(
            dim=self.dim,
            num_heads=self.num_heads,
            name='attn'
        )(x, deterministic=deterministic)

        if self.init_values is not None:
            gamma1 = self.param('gamma1', nn.initializers.constant(self.init_values), (self.dim,))
            x = x * gamma1

        x = DropPath(self.drop_path)(x, deterministic=deterministic)
        x = residual + x

        # MLP path
        residual = x
        x = nn.LayerNorm(name='norm2')(x)
        x = Mlp(
            hidden_dim=int(self.dim * self.mlp_ratio),
            out_dim=self.dim,
            drop=self.drop,
            name='mlp'
        )(x, deterministic=deterministic)

        if self.init_values is not None:
            gamma2 = self.param('gamma2', nn.initializers.constant(self.init_values), (self.dim,))
            x = x * gamma2

        x = DropPath(self.drop_path)(x, deterministic=deterministic)
        x = residual + x

        return x


class TinyViT(nn.Module):
    """
    TinyViT / ViT-Small: Efficient vision transformer.

    Default config (~21M params):
    - embed_dim: 384
    - depth: 12
    - num_heads: 6
    """
    num_classes: int = 1000
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    patch_size: int = 16
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    init_values: Optional[float] = None  # Layer scale (None = disabled)

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, H, W, 3) or (B, T, H, W, 3)
        Returns:
            (B, num_classes) logits
        """
        deterministic = not training

        # Handle video input - take last frame
        if x.ndim == 5:
            x = x[:, -1]

        B, H, W, _ = x.shape

        # Patch embedding
        x = PatchEmbed(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            name='patch_embed'
        )(x)  # (B, H', W', embed_dim)

        H_p, W_p = x.shape[1], x.shape[2]

        # Flatten spatial dims: (B, H', W', C) -> (B, N, C)
        x = x.reshape(B, H_p * W_p, self.embed_dim)

        # Add CLS token
        cls_token = self.param(
            'cls_token',
            nn.initializers.zeros,
            (1, 1, self.embed_dim)
        )
        cls_tokens = jnp.broadcast_to(cls_token, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (B, 1+N, C)

        # Position embeddings
        pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(0.02),
            (1, 1 + H_p * W_p, self.embed_dim)
        )
        x = x + pos_embed

        # Dropout
        if not deterministic and self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x, deterministic=False)

        # Transformer blocks
        dp_rates = np.linspace(0, self.drop_path_rate, self.depth)

        for i in range(self.depth):
            x = TinyViTBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                drop=self.drop_rate,
                drop_path=float(dp_rates[i]),
                init_values=self.init_values,
                name=f'block{i}'
            )(x, deterministic=deterministic)

        # Final norm
        x = nn.LayerNorm(name='norm')(x)

        # CLS token output
        x = x[:, 0]  # (B, embed_dim)

        # Classification head
        x = nn.Dense(self.num_classes, name='head')(x)

        return x


# Factory functions

def tiny_vit_21m_224(num_classes: int = 1000, **kwargs):
    """TinyViT-21M with 224x224 input (~21M params)."""
    return TinyViT(
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        patch_size=16,
        **kwargs
    )


def tiny_vit_5m_224(num_classes: int = 1000, **kwargs):
    """TinyViT-5M with 224x224 input (~5M params)."""
    return TinyViT(
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        patch_size=16,
        **kwargs
    )


def tiny_vit_11m_224(num_classes: int = 1000, **kwargs):
    """TinyViT-11M with 224x224 input (~11M params)."""
    return TinyViT(
        num_classes=num_classes,
        embed_dim=256,
        depth=12,
        num_heads=4,
        mlp_ratio=4.0,
        patch_size=16,
        **kwargs
    )
