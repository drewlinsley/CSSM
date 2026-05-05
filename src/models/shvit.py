"""
SHViT: Single-Head Vision Transformer.

Efficient hierarchical vision transformer using single-head attention.
Reference: https://github.com/ysj9909/SHViT

SHViT-S4 config (~22M params):
- embed_dims: [128, 256, 384, 512]
- depths: [1, 2, 4, 1]
- Single-head attention throughout
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Sequence, Optional


class DropPath(nn.Module):
    """Stochastic depth."""
    rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        if self.rate == 0.0 or deterministic:
            return x
        keep_prob = 1.0 - self.rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = jax.random.bernoulli(self.make_rng('dropout'), keep_prob, shape)
        return x * mask / keep_prob


class ConvBN(nn.Module):
    """Conv + BatchNorm (original SHViT)."""
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: str = 'SAME'
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = nn.Conv(
            self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding=self.padding,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='conv'
        )(x)
        x = nn.BatchNorm(
            use_running_average=deterministic,
            momentum=0.9,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='bn'
        )(x)
        return x


class PatchEmbed(nn.Module):
    """Overlapping patch embedding with conv layers."""
    embed_dim: int
    patch_size: int = 4
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Cast float inputs to the compute dtype at the network entry.
        x = x.astype(self.dtype)
        x = ConvBN(self.embed_dim // 2, kernel_size=3, stride=2,
                   dtype=self.dtype, param_dtype=self.param_dtype)(x, deterministic)
        x = jax.nn.gelu(x)
        x = ConvBN(self.embed_dim, kernel_size=3, stride=2,
                   dtype=self.dtype, param_dtype=self.param_dtype)(x, deterministic)
        return x


class Downsample(nn.Module):
    """Spatial downsampling between stages."""
    out_dim: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = ConvBN(self.out_dim, kernel_size=3, stride=2,
                   dtype=self.dtype, param_dtype=self.param_dtype)(x, deterministic)
        return x


class DWConv(nn.Module):
    """Depthwise convolution for local mixing."""
    dim: int
    kernel_size: int = 3
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W, C)
        x = nn.Conv(
            self.dim,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding='SAME',
            feature_group_count=self.dim,  # Depthwise
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='dwconv'
        )(x)
        return x


class SingleHeadAttention(nn.Module):
    """
    Single-Head Self-Attention.

    More efficient than multi-head attention for vision tasks.
    Uses depthwise conv for positional encoding.
    """
    dim: int
    qkv_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        B, H, W, C = x.shape
        N = H * W

        # QKV projection
        qkv = nn.Dense(3 * C, use_bias=self.qkv_bias, name='qkv')(x)
        qkv = qkv.reshape(B, N, 3, C).transpose(2, 0, 1, 3)  # (3, B, N, C)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Single-head attention
        scale = C ** -0.5
        attn = (q @ k.transpose(0, 2, 1)) * scale  # (B, N, N)
        attn = jax.nn.softmax(attn, axis=-1)

        # Apply attention
        x = (attn @ v).reshape(B, H, W, C)

        # Local positional encoding via depthwise conv
        x = x + DWConv(C, kernel_size=3, name='pos_conv')(x)

        # Output projection
        x = nn.Dense(C, name='proj')(x)

        return x


class Mlp(nn.Module):
    """MLP with GELU and optional DWConv."""
    hidden_dim: int
    out_dim: int
    use_dwconv: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        B, H, W, C = x.shape

        x = nn.Dense(self.hidden_dim,
                     dtype=self.dtype, param_dtype=self.param_dtype,
                     name='fc1')(x)
        x = jax.nn.gelu(x)

        if self.use_dwconv:
            x = x + DWConv(self.hidden_dim,
                           dtype=self.dtype, param_dtype=self.param_dtype,
                           name='dwconv')(x)

        x = nn.Dense(self.out_dim,
                     dtype=self.dtype, param_dtype=self.param_dtype,
                     name='fc2')(x)

        return x


class SHViTBlock(nn.Module):
    """
    SHViT block with single-head attention.

    Structure:
        x -> Norm -> SingleHeadAttn -> DropPath -> + x
        x -> Norm -> MLP -> DropPath -> + x
    """
    dim: int
    mlp_ratio: float = 4.0
    drop_path: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Attention path
        residual = x
        x = nn.LayerNorm(name='norm1')(x)
        x = SingleHeadAttention(self.dim, name='attn')(x, deterministic)
        x = DropPath(self.drop_path)(x, deterministic)
        x = residual + x

        # MLP path
        residual = x
        x = nn.LayerNorm(name='norm2')(x)
        x = Mlp(
            hidden_dim=int(self.dim * self.mlp_ratio),
            out_dim=self.dim,
            name='mlp'
        )(x, deterministic)
        x = DropPath(self.drop_path)(x, deterministic)
        x = residual + x

        return x


class ConvBlock(nn.Module):
    """
    Convolutional block for early stages.

    Uses depthwise separable convolution instead of attention.
    """
    dim: int
    mlp_ratio: float = 4.0
    drop_path: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Conv path (replaces attention)
        residual = x
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype,
                         name='norm1')(x)
        x = DWConv(self.dim, kernel_size=7,
                   dtype=self.dtype, param_dtype=self.param_dtype,
                   name='dwconv')(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(self.dim,
                     dtype=self.dtype, param_dtype=self.param_dtype,
                     name='pwconv')(x)
        x = DropPath(self.drop_path)(x, deterministic)
        x = residual + x

        # MLP path
        residual = x
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype,
                         name='norm2')(x)
        x = Mlp(
            hidden_dim=int(self.dim * self.mlp_ratio),
            out_dim=self.dim,
            use_dwconv=False,
            dtype=self.dtype, param_dtype=self.param_dtype,
            name='mlp'
        )(x, deterministic)
        x = DropPath(self.drop_path)(x, deterministic)
        x = residual + x

        return x


class SHViT(nn.Module):
    """
    Single-Head Vision Transformer.

    Hierarchical 4-stage architecture with:
    - ConvBlocks in early stages (efficient local processing)
    - SHViTBlocks with single-head attention in later stages
    """
    num_classes: int = 1000
    embed_dims: Sequence[int] = (128, 256, 384, 512)
    depths: Sequence[int] = (1, 2, 4, 1)
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_attn_stages: Sequence[bool] = (False, False, True, True)

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        deterministic = not training

        # Handle video input
        if x.ndim == 5:
            x = x[:, -1]

        B = x.shape[0]

        # Patch embedding (4x downsample)
        x = PatchEmbed(self.embed_dims[0], name='patch_embed')(x, deterministic)

        # Stochastic depth
        total_depth = sum(self.depths)
        dp_rates = np.linspace(0, self.drop_path_rate, total_depth)
        dp_idx = 0

        # 4 stages
        for stage_idx in range(4):
            # Downsample between stages
            if stage_idx > 0:
                x = Downsample(self.embed_dims[stage_idx], name=f'downsample{stage_idx}')(x, deterministic)

            # Blocks
            for block_idx in range(self.depths[stage_idx]):
                if self.use_attn_stages[stage_idx]:
                    x = SHViTBlock(
                        dim=self.embed_dims[stage_idx],
                        mlp_ratio=self.mlp_ratio,
                        drop_path=float(dp_rates[dp_idx]),
                        name=f'stage{stage_idx}_block{block_idx}'
                    )(x, deterministic)
                else:
                    x = ConvBlock(
                        dim=self.embed_dims[stage_idx],
                        mlp_ratio=self.mlp_ratio,
                        drop_path=float(dp_rates[dp_idx]),
                        name=f'stage{stage_idx}_block{block_idx}'
                    )(x, deterministic)
                dp_idx += 1

        # Final norm
        x = nn.LayerNorm(name='norm')(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, C)

        # Classification head
        if not deterministic and self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x, deterministic=False)
        x = nn.Dense(self.num_classes, name='head')(x)

        return x


# Factory functions

def shvit_s1(num_classes: int = 1000, **kwargs):
    """SHViT-S1 (~6M params)."""
    return SHViT(
        num_classes=num_classes,
        embed_dims=(64, 128, 256, 384),
        depths=(1, 2, 2, 1),
        use_attn_stages=(False, False, True, True),
        **kwargs
    )


def shvit_s2(num_classes: int = 1000, **kwargs):
    """SHViT-S2 (~11M params)."""
    return SHViT(
        num_classes=num_classes,
        embed_dims=(96, 192, 320, 448),
        depths=(1, 2, 3, 1),
        use_attn_stages=(False, False, True, True),
        **kwargs
    )


def shvit_s3(num_classes: int = 1000, **kwargs):
    """SHViT-S3 (~16M params)."""
    return SHViT(
        num_classes=num_classes,
        embed_dims=(112, 224, 352, 480),
        depths=(1, 2, 4, 1),
        use_attn_stages=(False, False, True, True),
        **kwargs
    )


def shvit_s4(num_classes: int = 1000, **kwargs):
    """SHViT-S4 (~22M params)."""
    return SHViT(
        num_classes=num_classes,
        embed_dims=(128, 256, 384, 512),
        depths=(1, 2, 4, 1),
        use_attn_stages=(False, False, True, True),
        **kwargs
    )
