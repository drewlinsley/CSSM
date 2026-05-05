"""
JEPA Predictor: Lightweight transformer for masked region prediction.

The predictor should be intentionally shallow and weak to prevent
the model from finding trivial shortcuts. A strong predictor can
lead to representation collapse.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Optional


class MLP(nn.Module):
    """Simple MLP block with GELU activation."""
    hidden_dim: int
    out_dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(self.out_dim)(x)
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        return x


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""
    num_heads: int
    head_dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        B, N, D = x.shape

        # QKV projection
        qkv = nn.Dense(3 * self.num_heads * self.head_dim, use_bias=False)(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, num_heads, N, N)
        attn = nn.softmax(attn, axis=-1)

        if self.dropout > 0:
            attn = nn.Dropout(self.dropout)(attn, deterministic=deterministic)

        # Output
        x = (attn @ v).transpose(0, 2, 1, 3)  # (B, N, num_heads, head_dim)
        x = x.reshape(B, N, self.num_heads * self.head_dim)

        # Output projection
        x = nn.Dense(D)(x)

        return x


class PredictorBlock(nn.Module):
    """
    Transformer block for the predictor.

    Pre-norm style with attention and MLP.
    """
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        head_dim = self.embed_dim // self.num_heads

        # Attention path
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            dropout=self.dropout,
        )(x, deterministic=deterministic)
        x = residual + x

        # MLP path
        residual = x
        x = nn.LayerNorm()(x)
        x = MLP(
            hidden_dim=int(self.embed_dim * self.mlp_ratio),
            out_dim=self.embed_dim,
            dropout=self.dropout,
        )(x, deterministic=deterministic)
        x = residual + x

        return x


class JEPAPredictor(nn.Module):
    """
    Lightweight transformer predictor for JEPA.

    Takes visible tokens + mask tokens with positions and predicts
    features for masked regions.

    The predictor should be intentionally shallow (4 layers)
    to prevent finding trivial solutions.

    Args:
        embed_dim: Embedding dimension (should match encoder)
        depth: Number of transformer blocks (default 4, keep shallow!)
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        dropout: Dropout rate
    """
    embed_dim: int = 384
    depth: int = 4
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        context_tokens: jnp.ndarray,
        mask_tokens: jnp.ndarray,
        context_positions: jnp.ndarray,
        mask_positions: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Predict features for masked regions.

        Args:
            context_tokens: (B, num_visible, D) encoded visible tokens
            mask_tokens: (B, num_masked, D) learnable mask tokens
            context_positions: (B, num_visible, D) position embeddings for visible
            mask_positions: (B, num_masked, D) position embeddings for masked

        Returns:
            predictions: (B, num_masked, D) predicted features for masked positions
        """
        B = context_tokens.shape[0]
        num_visible = context_tokens.shape[1]
        num_masked = mask_tokens.shape[1]

        # Add position embeddings
        context = context_tokens + context_positions
        masked = mask_tokens + mask_positions

        # Concatenate visible and masked tokens
        # Masked tokens come first so we can easily extract predictions
        x = jnp.concatenate([masked, context], axis=1)  # (B, num_masked + num_visible, D)

        # Transformer blocks
        for i in range(self.depth):
            x = PredictorBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                name=f'block_{i}',
            )(x, deterministic=deterministic)

        # Final norm
        x = nn.LayerNorm(name='norm')(x)

        # Extract predictions for masked positions (first num_masked tokens)
        predictions = x[:, :num_masked, :]

        return predictions


class PositionEmbedding(nn.Module):
    """
    Learnable position embeddings for spatial and temporal positions.

    Supports both 2D spatial and 3D spatiotemporal positions.
    """
    embed_dim: int
    max_h: int = 14
    max_w: int = 14
    max_t: int = 16

    @nn.compact
    def __call__(
        self,
        indices: jnp.ndarray,
        temporal: bool = True,
    ) -> jnp.ndarray:
        """
        Get position embeddings for given indices.

        Args:
            indices: (B, N) spatial indices (h * W + w)
            temporal: If True, add temporal position embedding

        Returns:
            pos_embed: (B, N, D) position embeddings
        """
        B, N = indices.shape

        # Learnable 2D position embeddings
        spatial_embed = self.param(
            'spatial_embed',
            nn.initializers.normal(0.02),
            (self.max_h * self.max_w, self.embed_dim)
        )

        # Gather embeddings at indices
        pos_embed = spatial_embed[indices]  # (B, N, D)

        return pos_embed


def create_mask_tokens(
    num_tokens: int,
    embed_dim: int,
    batch_size: int,
) -> jnp.ndarray:
    """
    Create learnable mask tokens.

    In practice this should be done as a module parameter,
    but this helper shows the expected shape.

    Returns:
        mask_tokens: (B, num_tokens, embed_dim)
    """
    # Single learnable token, broadcast to batch
    mask_token = jnp.zeros((1, 1, embed_dim))
    mask_tokens = jnp.broadcast_to(mask_token, (batch_size, num_tokens, embed_dim))
    return mask_tokens
