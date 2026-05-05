"""
DeiT3-Large implementation in JAX/Flax.

Matches timm's deit3_large_patch16_384 architecture with modifications:
- Uses spatial max pool instead of CLS token (for fair comparison with CSSM)
- Supports video input (B, T, H, W, C) - uses last timestep

Key DeiT3 features:
1. Layer Scale (learnable per-channel scaling initialized to 1e-6)
2. Pre-norm transformer blocks
3. Stochastic depth (DropPath)
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Optional


class DropPath(nn.Module):
    """
    Stochastic depth regularization (drop entire residual branch).
    """
    drop_prob: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        if self.drop_prob == 0.0 or deterministic:
            return x

        keep_prob = 1 - self.drop_prob
        # Drop entire samples, not individual elements
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rng = self.make_rng('dropout')
        mask = jax.random.bernoulli(rng, keep_prob, shape)
        return x * mask / keep_prob


class PatchEmbed(nn.Module):
    """
    Patch embedding that maintains 2D spatial structure.

    For DeiT3-Large with 384x384 input and patch_size=16:
    Output shape: (B, 24, 24, embed_dim)
    """
    embed_dim: int = 1024
    patch_size: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, H, W, 3) or (B, T, H, W, 3) for video

        Returns:
            (B, H', W', embed_dim) where H' = H/patch_size, W' = W/patch_size
        """
        # Handle video input - use last timestep
        if x.ndim == 5:
            x = x[:, -1]  # (B, H, W, 3)

        # Patchify with conv - maintains 2D structure
        x = nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            name='proj'
        )(x)  # (B, H', W', embed_dim)

        return x


class Attention(nn.Module):
    """
    Multi-head self-attention for 2D spatial input.

    Flattens spatial dims for attention, then reshapes back.
    """
    dim: int
    num_heads: int = 16
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, H', W', C)

        Returns:
            (B, H', W', C)
        """
        B, H, W, C = x.shape
        N = H * W
        head_dim = C // self.num_heads
        scale = head_dim ** -0.5

        # Flatten spatial dims for attention
        x_flat = x.reshape(B, N, C)  # (B, N, C)

        # QKV projection
        qkv = nn.Dense(3 * C, use_bias=self.qkv_bias, name='qkv')(x_flat)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = nn.softmax(attn, axis=-1)

        if not deterministic and self.attn_drop > 0:
            attn = nn.Dropout(self.attn_drop)(attn, deterministic=False)

        # Attention output
        x_flat = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x_flat = nn.Dense(C, name='proj')(x_flat)

        if not deterministic and self.proj_drop > 0:
            x_flat = nn.Dropout(self.proj_drop)(x_flat, deterministic=False)

        # Reshape back to 2D
        return x_flat.reshape(B, H, W, C)


class Mlp(nn.Module):
    """MLP block with GELU activation (1x1 conv equivalent)."""
    hidden_dim: int
    out_dim: int
    drop: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, name='fc1')(x)
        x = nn.gelu(x)
        if not deterministic and self.drop > 0:
            x = nn.Dropout(self.drop)(x, deterministic=False)
        x = nn.Dense(self.out_dim, name='fc2')(x)
        if not deterministic and self.drop > 0:
            x = nn.Dropout(self.drop)(x, deterministic=False)
        return x


class DeiT3Block(nn.Module):
    """
    DeiT3 Transformer block with Layer Scale.

    Pre-norm structure:
        x -> Norm -> Attn -> LayerScale -> DropPath -> + x
        x -> Norm -> MLP  -> LayerScale -> DropPath -> + x

    Works with 2D spatial input (B, H', W', C).
    """
    dim: int
    num_heads: int = 16
    mlp_ratio: float = 4.0
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.0
    init_values: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, H', W', C)

        Returns:
            (B, H', W', C)
        """
        # Attention path
        residual = x
        x = nn.LayerNorm(name='norm1')(x)
        x = Attention(
            dim=self.dim,
            num_heads=self.num_heads,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
            name='attn'
        )(x, deterministic=deterministic)

        # Layer Scale for attention
        gamma1 = self.param(
            'gamma1',
            nn.initializers.constant(self.init_values),
            (self.dim,)
        )
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


class DeiT3Large(nn.Module):
    """
    DeiT3-Large Vision Transformer.

    Architecture matches timm deit3_large_patch16_384:
    - embed_dim=1024, depth=24, num_heads=16
    - Layer scale with init_values=1e-6
    - Uses spatial max pool (not CLS token) for classification

    Maintains 2D spatial structure throughout for fair comparison with CSSM.
    """
    num_classes: int = 1000
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    patch_size: int = 16
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    init_values: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            x: (B, H, W, 3) or (B, T, H, W, 3)

        Returns:
            (B, num_classes) logits
        """
        deterministic = not training

        # Handle video input - use last timestep
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
        N = H_p * W_p

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

        # Transformer blocks with linear drop path increase
        dp_rates = np.linspace(0, self.drop_path_rate, self.depth)

        for i in range(self.depth):
            x = DeiT3Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=float(dp_rates[i]),
                init_values=self.init_values,
                name=f'block{i}'
            )(x, deterministic=deterministic)

        # Final norm
        x = nn.LayerNorm(name='norm')(x)

        # Spatial max pool (not CLS token)
        x = jnp.max(x, axis=(1, 2))  # (B, embed_dim)

        # Classification head
        x = nn.Dense(self.num_classes, name='head')(x)

        return x


# Model factory functions
def deit3_large_patch16_224(num_classes: int = 1000, **kwargs):
    """DeiT3-Large with 224x224 input."""
    return DeiT3Large(
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_size=16,
        **kwargs
    )


def deit3_large_patch16_384(num_classes: int = 1000, **kwargs):
    """DeiT3-Large with 384x384 input."""
    return DeiT3Large(
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_size=16,
        **kwargs
    )


def deit3_base_patch16_224(num_classes: int = 1000, **kwargs):
    """DeiT3-Base with 224x224 input."""
    return DeiT3Large(
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        patch_size=16,
        **kwargs
    )


def deit3_small_patch16_224(num_classes: int = 1000, **kwargs):
    """DeiT3-Small with 224x224 input."""
    return DeiT3Large(
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        patch_size=16,
        **kwargs
    )
