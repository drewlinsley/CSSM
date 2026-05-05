"""
JAX/Flax implementation of CoTracker3 building blocks.

Faithful port of attention and MLP blocks from the original PyTorch implementation.
Reference: https://github.com/facebookresearch/co-tracker
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Callable
from functools import partial


class Mlp(nn.Module):
    """MLP as used in Vision Transformer and CoTracker.

    Attributes:
        in_features: Input dimension
        hidden_features: Hidden dimension (default: in_features)
        out_features: Output dimension (default: in_features)
        drop: Dropout rate
    """
    in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    drop: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features

        x = nn.Dense(hidden_features, name='fc1')(x)
        x = nn.gelu(x, approximate=True)  # tanh approximation
        x = nn.Dropout(rate=self.drop, deterministic=not training)(x)
        x = nn.Dense(out_features, name='fc2')(x)
        x = nn.Dropout(rate=self.drop, deterministic=not training)(x)
        return x


class Attention(nn.Module):
    """Multi-head attention with optional cross-attention.

    Supports both self-attention (context=None) and cross-attention (context provided).

    Attributes:
        query_dim: Query dimension
        context_dim: Context dimension for cross-attention (default: query_dim)
        num_heads: Number of attention heads
        dim_head: Dimension per head
        qkv_bias: Whether to use bias in QKV projections
    """
    query_dim: int
    context_dim: Optional[int] = None
    num_heads: int = 8
    dim_head: int = 48
    qkv_bias: bool = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        attn_bias: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Args:
            x: Query tensor (B, N1, D)
            context: Context for cross-attention (B, N2, D_ctx), None for self-attention
            attn_bias: Optional attention bias/mask (B, heads, N1, N2)

        Returns:
            Output tensor (B, N1, D)
        """
        context_dim = self.context_dim if self.context_dim is not None else self.query_dim
        inner_dim = self.dim_head * self.num_heads
        scale = self.dim_head ** -0.5

        B, N1, _ = x.shape

        # Query projection
        q = nn.Dense(inner_dim, use_bias=self.qkv_bias, name='to_q')(x)
        q = q.reshape(B, N1, self.num_heads, self.dim_head).transpose(0, 2, 1, 3)

        # Key-Value projection from context (or x for self-attention)
        ctx = context if context is not None else x
        N2 = ctx.shape[1]

        kv = nn.Dense(inner_dim * 2, use_bias=self.qkv_bias, name='to_kv')(ctx)
        kv = kv.reshape(B, N2, 2, self.num_heads, self.dim_head)
        k, v = kv[:, :, 0], kv[:, :, 1]
        k = k.transpose(0, 2, 1, 3)  # (B, heads, N2, dim_head)
        v = v.transpose(0, 2, 1, 3)

        # Attention scores
        sim = jnp.einsum('bhid,bhjd->bhij', q, k) * scale

        if attn_bias is not None:
            sim = sim + attn_bias

        attn = jax.nn.softmax(sim, axis=-1)

        # Attend to values
        out = jnp.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, N1, inner_dim)

        # Output projection
        out = nn.Dense(self.query_dim, name='to_out')(out)
        return out


class AttnBlock(nn.Module):
    """Attention block with pre-norm and MLP.

    Structure: LN -> Attn -> Residual -> LN -> MLP -> Residual

    Attributes:
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tensor (B, N, D)
            mask: Optional boolean mask (B, N), True = valid
            training: Training mode

        Returns:
            Output tensor (B, N, D)
        """
        # Compute attention bias from mask if provided
        attn_bias = None
        if mask is not None:
            # mask: (B, N) -> attention mask (B, 1, N, N)
            mask_2d = mask[:, None, :] * mask[:, :, None]  # (B, N, N)
            mask_2d = mask_2d[:, None, :, :]  # (B, 1, N, N)
            # Expand to all heads
            mask_2d = jnp.broadcast_to(mask_2d, (mask_2d.shape[0], self.num_heads, mask_2d.shape[2], mask_2d.shape[3]))
            # Convert to additive bias: False -> -inf
            attn_bias = jnp.where(mask_2d, 0.0, -1e9)

        # Pre-norm attention
        norm1 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name='norm1')
        attn = Attention(
            query_dim=self.hidden_size,
            num_heads=self.num_heads,
            qkv_bias=True,
            name='attn',
        )
        x = x + attn(norm1(x), attn_bias=attn_bias)

        # Pre-norm MLP
        norm2 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name='norm2')
        mlp_hidden = int(self.hidden_size * self.mlp_ratio)
        mlp = Mlp(
            in_features=self.hidden_size,
            hidden_features=mlp_hidden,
            drop=0.0,
            name='mlp',
        )
        x = x + mlp(norm2(x), training=training)

        return x


class CrossAttnBlock(nn.Module):
    """Cross-attention block for point-to-virtual and virtual-to-point attention.

    Structure: LN -> CrossAttn -> Residual -> LN -> MLP -> Residual

    Attributes:
        query_dim: Query dimension
        context_dim: Context dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
    """
    query_dim: int
    context_dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        context: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            x: Query tensor (B, N1, D)
            context: Context tensor (B, N2, D_ctx)
            mask: Optional mask
            training: Training mode

        Returns:
            Output tensor (B, N1, D)
        """
        # Pre-norm cross-attention
        norm1 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name='norm1')
        norm_ctx = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name='norm_context')
        attn = Attention(
            query_dim=self.query_dim,
            context_dim=self.context_dim,
            num_heads=self.num_heads,
            qkv_bias=True,
            name='attn',
        )
        x = x + attn(norm1(x), context=norm_ctx(context))

        # Pre-norm MLP
        norm2 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name='norm2')
        mlp_hidden = int(self.query_dim * self.mlp_ratio)
        mlp = Mlp(
            in_features=self.query_dim,
            hidden_features=mlp_hidden,
            drop=0.0,
            name='mlp',
        )
        x = x + mlp(norm2(x), training=training)

        return x
