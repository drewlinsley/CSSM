"""
JAX/Flax implementation of CoTracker3 EfficientUpdateFormer.

Faithful port of the transformer-based update block from the original PyTorch implementation.
Reference: https://github.com/facebookresearch/co-tracker
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple

from .blocks import AttnBlock, CrossAttnBlock


class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.

    Faithful port of CoTracker's EfficientUpdateFormer. Uses interleaved
    temporal self-attention and spatial cross-attention with virtual tokens.

    Architecture:
        1. Input projection: input_dim -> hidden_size
        2. Add virtual tokens for spatial reasoning
        3. Interleaved temporal and spatial attention:
           - Temporal: Self-attention over time dimension
           - Spatial: Cross-attention between points and virtual tokens
        4. Output heads for flow prediction

    Attributes:
        space_depth: Number of spatial attention layers
        time_depth: Number of temporal attention layers
        input_dim: Input feature dimension
        hidden_size: Transformer hidden dimension
        num_heads: Number of attention heads
        output_dim: Output dimension (4 for dx, dy, vis, conf)
        mlp_ratio: MLP expansion ratio
        num_virtual_tracks: Number of virtual tokens
        add_space_attn: Whether to use spatial attention
    """
    space_depth: int = 3
    time_depth: int = 3
    input_dim: int = 1110
    hidden_size: int = 384
    num_heads: int = 8
    output_dim: int = 4
    mlp_ratio: float = 4.0
    num_virtual_tracks: int = 64
    add_space_attn: bool = True

    def setup(self):
        # Input projection
        self.input_transform = nn.Dense(self.hidden_size, name='input_transform')

        # Virtual tokens for spatial reasoning
        self.virtual_tracks = self.param(
            'virtual_tracks',
            nn.initializers.normal(stddev=0.02),
            (1, self.num_virtual_tracks, 1, self.hidden_size),
        )

        # Temporal attention blocks
        self.time_blocks = [
            AttnBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                name=f'time_block_{i}',
            )
            for i in range(self.time_depth)
        ]

        # Spatial attention blocks (if enabled)
        if self.add_space_attn:
            # Virtual token self-attention
            self.space_virtual_blocks = [
                AttnBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    name=f'space_virtual_block_{i}',
                )
                for i in range(self.space_depth)
            ]

            # Cross-attention: virtual tokens attend to point tokens
            self.space_virtual2point_blocks = [
                CrossAttnBlock(
                    query_dim=self.hidden_size,
                    context_dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    name=f'space_v2p_block_{i}',
                )
                for i in range(self.space_depth)
            ]

            # Cross-attention: point tokens attend to virtual tokens
            self.space_point2virtual_blocks = [
                CrossAttnBlock(
                    query_dim=self.hidden_size,
                    context_dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    name=f'space_p2v_block_{i}',
                )
                for i in range(self.space_depth)
            ]

        # Output head
        self.flow_head = nn.Dense(self.output_dim, name='flow_head')

    def __call__(
        self,
        input_tensor: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        add_space_attn: bool = True,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Update track predictions.

        Args:
            input_tensor: Input features (B, N, T, input_dim)
            mask: Optional attention mask
            add_space_attn: Whether to apply spatial attention
            training: Training mode

        Returns:
            Flow predictions (B, N, T, output_dim) containing (dx, dy, vis, conf)
        """
        # Input projection
        tokens = self.input_transform(input_tensor)  # (B, N, T, hidden_size)

        B, N_points, T, _ = tokens.shape

        # Add virtual tokens
        # virtual_tracks: (1, V, 1, hidden_size) -> (B, V, T, hidden_size)
        virtual_tokens = jnp.broadcast_to(
            self.virtual_tracks,
            (B, self.num_virtual_tracks, T, self.hidden_size),
        )
        tokens = jnp.concatenate([tokens, virtual_tokens], axis=1)  # (B, N+V, T, hidden_size)

        N = N_points + self.num_virtual_tracks

        # Interleaved temporal and spatial attention
        j = 0  # Spatial block index

        for i in range(self.time_depth):
            # Temporal attention: attend over time dimension
            # Reshape: (B, N, T, D) -> (B*N, T, D)
            time_tokens = tokens.reshape(B * N, T, -1)
            time_tokens = self.time_blocks[i](time_tokens, training=training)
            tokens = time_tokens.reshape(B, N, T, -1)

            # Spatial attention (interleaved)
            should_apply_space = (
                add_space_attn
                and self.add_space_attn
                and (i % (self.time_depth // self.space_depth) == 0)
                and j < self.space_depth
            )

            if should_apply_space:
                # Reshape for spatial attention: (B, N, T, D) -> (B*T, N, D)
                space_tokens = tokens.transpose(0, 2, 1, 3)  # (B, T, N, D)
                space_tokens = space_tokens.reshape(B * T, N, -1)

                # Split point and virtual tokens
                point_tokens = space_tokens[:, :N_points]  # (B*T, N_points, D)
                virtual_tokens_space = space_tokens[:, N_points:]  # (B*T, V, D)

                # Virtual tokens attend to points
                virtual_tokens_space = self.space_virtual2point_blocks[j](
                    virtual_tokens_space,
                    point_tokens,
                    mask=mask,
                    training=training,
                )

                # Virtual self-attention
                virtual_tokens_space = self.space_virtual_blocks[j](
                    virtual_tokens_space,
                    training=training,
                )

                # Points attend to virtual tokens
                point_tokens = self.space_point2virtual_blocks[j](
                    point_tokens,
                    virtual_tokens_space,
                    mask=mask,
                    training=training,
                )

                # Recombine
                space_tokens = jnp.concatenate([point_tokens, virtual_tokens_space], axis=1)

                # Reshape back: (B*T, N, D) -> (B, N, T, D)
                tokens = space_tokens.reshape(B, T, N, -1).transpose(0, 2, 1, 3)

                j += 1

        # Extract point tokens (remove virtual)
        tokens = tokens[:, :N_points]  # (B, N_points, T, hidden_size)

        # Output head
        flow = self.flow_head(tokens)  # (B, N_points, T, output_dim)

        return flow

    def initialize_weights(self):
        """Weight initialization following original implementation."""
        # This is handled by Flax's default initialization
        # The flow_head uses truncated normal with std=0.001 in original
        pass


class CSSMUpdateFormer(nn.Module):
    """
    CSSM-based update block replacing transformer attention.

    Replaces temporal attention with CSSM processing while optionally
    keeping spatial cross-attention for point-virtual token interactions.

    Attributes:
        space_depth: Number of spatial attention layers
        time_depth: Number of CSSM temporal layers
        input_dim: Input feature dimension
        hidden_size: Hidden dimension
        output_dim: Output dimension
        mlp_ratio: MLP expansion ratio
        num_virtual_tracks: Number of virtual tokens
        add_space_attn: Whether to use spatial attention
        cssm_type: Type of CSSM ('standard', 'opponent', 'hgru_bi')
        kernel_size: CSSM spatial kernel size
    """
    space_depth: int = 3
    time_depth: int = 3
    input_dim: int = 1110
    hidden_size: int = 384
    output_dim: int = 4
    mlp_ratio: float = 4.0
    num_virtual_tracks: int = 64
    add_space_attn: bool = True
    cssm_type: str = 'opponent'
    kernel_size: int = 11

    def setup(self):
        # Input projection
        self.input_transform = nn.Dense(self.hidden_size, name='input_transform')

        # Virtual tokens
        self.virtual_tracks = self.param(
            'virtual_tracks',
            nn.initializers.normal(stddev=0.02),
            (1, self.num_virtual_tracks, 1, self.hidden_size),
        )

        # Spatial attention blocks (keep these from transformer version)
        if self.add_space_attn:
            self.space_virtual_blocks = [
                AttnBlock(
                    hidden_size=self.hidden_size,
                    num_heads=8,
                    mlp_ratio=self.mlp_ratio,
                    name=f'space_virtual_block_{i}',
                )
                for i in range(self.space_depth)
            ]
            self.space_virtual2point_blocks = [
                CrossAttnBlock(
                    query_dim=self.hidden_size,
                    context_dim=self.hidden_size,
                    num_heads=8,
                    mlp_ratio=self.mlp_ratio,
                    name=f'space_v2p_block_{i}',
                )
                for i in range(self.space_depth)
            ]
            self.space_point2virtual_blocks = [
                CrossAttnBlock(
                    query_dim=self.hidden_size,
                    context_dim=self.hidden_size,
                    num_heads=8,
                    mlp_ratio=self.mlp_ratio,
                    name=f'space_p2v_block_{i}',
                )
                for i in range(self.space_depth)
            ]

        # Output head
        self.flow_head = nn.Dense(self.output_dim, name='flow_head')

    def _get_cssm_layer(self):
        """Get CSSM layer based on type."""
        # Import here to avoid circular dependency
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from models.cssm import GatedOpponentCSSM, StandardCSSM, HGRUBilinearCSSM

        if self.cssm_type == 'opponent':
            return GatedOpponentCSSM(
                channels=self.hidden_size,
                kernel_size=self.kernel_size,
            )
        elif self.cssm_type == 'hgru_bi':
            return HGRUBilinearCSSM(
                channels=self.hidden_size,
                kernel_size=self.kernel_size,
            )
        else:
            return StandardCSSM(
                channels=self.hidden_size,
                kernel_size=self.kernel_size,
            )

    @nn.compact
    def __call__(
        self,
        input_tensor: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        add_space_attn: bool = True,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Update track predictions using CSSM.

        Args:
            input_tensor: Input features (B, N, T, input_dim)
            mask: Optional attention mask
            add_space_attn: Whether to apply spatial attention
            training: Training mode

        Returns:
            Flow predictions (B, N, T, output_dim)
        """
        # Input projection
        tokens = self.input_transform(input_tensor)

        B, N_points, T, _ = tokens.shape

        # Add virtual tokens
        virtual_tokens = jnp.broadcast_to(
            self.virtual_tracks,
            (B, self.num_virtual_tracks, T, self.hidden_size),
        )
        tokens = jnp.concatenate([tokens, virtual_tokens], axis=1)

        N = N_points + self.num_virtual_tracks

        j = 0  # Spatial block index

        for i in range(self.time_depth):
            # CSSM temporal processing (replaces temporal attention)
            # Reshape: (B, N, T, D) -> (B*N, T, 1, 1, D) for CSSM
            cssm_input = tokens.reshape(B * N, T, 1, 1, self.hidden_size)

            cssm_layer = self._get_cssm_layer()
            cssm_output = cssm_layer(cssm_input)

            # Reshape back and residual
            cssm_output = cssm_output.reshape(B, N, T, self.hidden_size)
            tokens = nn.LayerNorm(epsilon=1e-6, name=f'temporal_norm_{i}')(tokens + cssm_output)

            # Spatial attention (same as transformer version)
            should_apply_space = (
                add_space_attn
                and self.add_space_attn
                and (i % max(1, self.time_depth // self.space_depth) == 0)
                and j < self.space_depth
            )

            if should_apply_space:
                space_tokens = tokens.transpose(0, 2, 1, 3).reshape(B * T, N, -1)

                point_tokens = space_tokens[:, :N_points]
                virtual_tokens_space = space_tokens[:, N_points:]

                virtual_tokens_space = self.space_virtual2point_blocks[j](
                    virtual_tokens_space, point_tokens, mask=mask, training=training
                )
                virtual_tokens_space = self.space_virtual_blocks[j](
                    virtual_tokens_space, training=training
                )
                point_tokens = self.space_point2virtual_blocks[j](
                    point_tokens, virtual_tokens_space, mask=mask, training=training
                )

                space_tokens = jnp.concatenate([point_tokens, virtual_tokens_space], axis=1)
                tokens = space_tokens.reshape(B, T, N, -1).transpose(0, 2, 1, 3)

                j += 1

        # Extract point tokens
        tokens = tokens[:, :N_points]

        # Output head
        flow = self.flow_head(tokens)

        return flow
