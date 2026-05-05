"""
ConvNeXt architecture with CSSM integration.

Provides three block types:
- ConvNextBlock: Standard ConvNeXt block
- CSSMNextBlock: Pure CSSM variant (replaces depthwise conv with CSSM)
- HybridBlock: ConvNeXt + CSSM combined

And a ModelFactory to build all 8 model variants.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Type, Optional

from .cssm import GatedCSSM, HGRUBilinearCSSM, TransformerCSSM


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) regularization.

    Randomly drops entire residual branches during training,
    which helps regularization in deep networks.

    Attributes:
        drop_prob: Probability of dropping the path (0 = no drop, 1 = always drop)
    """
    drop_prob: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        if self.drop_prob == 0.0 or deterministic:
            return x

        keep_prob = 1.0 - self.drop_prob
        # Generate random mask with shape (batch_size, 1, 1, 1, 1) for broadcasting
        rng = self.make_rng('dropout')
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = jax.random.bernoulli(rng, keep_prob, shape)

        # Scale during training to maintain expected values
        return x * random_tensor / keep_prob


class ConvNextBlock(nn.Module):
    """
    Standard ConvNeXt Block.

    Structure: DWConv(7x7) -> LN -> Dense(4x) -> GELU -> Dense -> LayerScale -> Residual

    Attributes:
        dim: Number of channels
        drop_path: Drop path probability for stochastic depth
        layer_scale_init_value: Initial value for learnable layer scale
    """
    dim: int
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        shortcut = x

        # 1. Depthwise Conv (Spatial mixing)
        # For video input (B, T, H, W, C), we apply conv over H, W
        x = nn.Conv(
            self.dim,
            kernel_size=(7, 7),
            feature_group_count=self.dim,
            padding='SAME',
            name='dwconv'
        )(x)
        x = nn.LayerNorm(name='norm')(x)

        # 2. Pointwise Expansion (4x channels)
        x = nn.Dense(4 * self.dim, name='pwconv1')(x)
        x = nn.gelu(x)

        # 3. Pointwise Projection (back to dim channels)
        x = nn.Dense(self.dim, name='pwconv2')(x)

        # 4. Layer Scale (learnable per-channel scaling)
        gamma = self.param(
            'gamma',
            lambda key, shape: self.layer_scale_init_value * jnp.ones(shape),
            (self.dim,)
        )
        x = x * gamma

        # 5. DropPath + Residual
        x = DropPath(self.drop_path)(x, deterministic=not training)
        return shortcut + x


class CSSMNextBlock(nn.Module):
    """
    Pure CSSM Block - replaces depthwise conv with CSSM.

    Structure: CSSM -> LN -> Dense(4x) -> GELU -> Dense -> LayerScale -> Residual

    This variant fully replaces spatial convolution with the temporal-aware
    CSSM layer, making it a purely recurrent spatial mixer.

    Attributes:
        dim: Number of channels
        cssm_cls: CSSM class to use (StandardCSSM or GatedOpponentCSSM)
        dense_mixing: Whether CSSM should use dense (multi-head) mixing
        concat_xy: Whether to concat [X,Y] and project (GatedOpponentCSSM only)
        gate_activation: Gate activation type for GatedOpponentCSSM
        drop_path: Drop path probability
        layer_scale_init_value: Initial value for layer scale
    """
    dim: int
    cssm_cls: Type[nn.Module]
    dense_mixing: bool = False
    concat_xy: bool = True
    gate_activation: str = 'sigmoid'
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        shortcut = x

        # 1. Pre-norm for CSSM stability (CSSM has unbounded log-space ops)
        x = nn.LayerNorm(name='pre_norm')(x)

        # 2. CSSM (Time-Varying Spatial Mixing)
        x = self.cssm_cls(
            channels=self.dim,
            dense_mixing=self.dense_mixing,
            concat_xy=self.concat_xy,
            gate_activation=self.gate_activation,
            name='cssm'
        )(x)

        # 3. Pointwise Expansion
        x = nn.Dense(4 * self.dim, name='pwconv1')(x)
        x = nn.gelu(x)

        # 4. Pointwise Projection
        x = nn.Dense(self.dim, name='pwconv2')(x)

        # 5. Layer Scale
        gamma = self.param(
            'gamma',
            lambda key, shape: self.layer_scale_init_value * jnp.ones(shape),
            (self.dim,)
        )
        x = x * gamma

        # 6. DropPath + Residual
        x = DropPath(self.drop_path)(x, deterministic=not training)
        return shortcut + x


class HybridBlock(nn.Module):
    """
    Hybrid Block - Standard ConvNeXt + CSSM combined.

    Structure: ConvNextBlock -> Pre-Norm -> CSSM -> Post-Norm -> Residual

    This variant keeps the standard ConvNeXt spatial processing and adds
    CSSM as an additional temporal mixing mechanism.

    Attributes:
        dim: Number of channels
        cssm_cls: CSSM class to use
        dense_mixing: Whether CSSM should use dense mixing
        concat_xy: Whether to concat [X,Y] and project (GatedOpponentCSSM only)
        gate_activation: Gate activation type for GatedOpponentCSSM
        drop_path: Drop path probability
    """
    dim: int
    cssm_cls: Type[nn.Module]
    dense_mixing: bool = False
    concat_xy: bool = True
    gate_activation: str = 'sigmoid'
    drop_path: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # 1. Standard Spatial Processing (ConvNeXt)
        x = ConvNextBlock(
            self.dim,
            drop_path=self.drop_path,
            name='convnext'
        )(x, training=training)

        # 2. CSSM Processing (Additional Residual)
        shortcut = x

        # Pre-Norm for stability entering recurrent block
        y = nn.LayerNorm(name='pre_norm')(x)
        y = self.cssm_cls(
            channels=self.dim,
            dense_mixing=self.dense_mixing,
            concat_xy=self.concat_xy,
            gate_activation=self.gate_activation,
            name='cssm'
        )(y)
        y = nn.LayerNorm(name='post_norm')(y)  # Post-Norm

        # DropPath on CSSM branch
        y = DropPath(self.drop_path)(y, deterministic=not training)
        return x + y


class ModelFactory(nn.Module):
    """
    Main Model Builder for CSSM-ConvNeXt architectures.

    Creates one of 8 model variants based on configuration:
    - mode: 'pure' (CSSM replaces conv) or 'hybrid' (ConvNeXt + CSSM)
    - cssm_type: 'standard' or 'opponent' (gated opponent)
    - mixing: 'dense' (multi-head) or 'depthwise' (independent)

    Architecture follows ConvNeXt-Tiny:
    - Stem: 4x4 patchify to 96 channels
    - 4 Stages with dims [96, 192, 384, 768]
    - 1 block per stage (configurable)
    - Global pooling -> Classification head

    Attributes:
        mode: 'hybrid' or 'pure'
        cssm_type: 'standard' or 'opponent'
        mixing: 'dense' or 'depthwise'
        num_classes: Number of output classes
        depths: Number of blocks per stage (default: [1, 1, 1, 1])
        drop_path_rate: Maximum drop path rate (linearly increases)
        concat_xy: Whether to concat [X,Y] and project (GatedOpponentCSSM only)
        gate_activation: Gate activation type for GatedOpponentCSSM
    """
    mode: str
    cssm_type: str
    mixing: str
    num_classes: int = 10
    depths: tuple = (1, 1, 1, 1)
    drop_path_rate: float = 0.1
    concat_xy: bool = True
    gate_activation: str = 'sigmoid'

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input video tensor of shape (B, T, H, W, 3)
            training: Whether in training mode (affects dropout, etc.)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Select CSSM class based on config
        if self.cssm_type == 'gated':
            CSSM = GatedCSSM
        elif self.cssm_type == 'transformer':
            CSSM = TransformerCSSM
        else:  # hgru_bi (default)
            CSSM = HGRUBilinearCSSM

        dense = (self.mixing == 'dense')

        # Stage dimensions (ConvNeXt-Tiny style)
        dims = [96, 192, 384, 768]

        # Calculate drop path rates for each block (use numpy for concrete values)
        total_blocks = sum(self.depths)
        dp_rates = np.linspace(0, self.drop_path_rate, total_blocks)
        block_idx = 0

        # --- Stem (Patchify) ---
        # (B, T, H, W, 3) -> (B, T, H/4, W/4, 96)
        x = nn.Conv(
            dims[0],
            kernel_size=(4, 4),
            strides=(4, 4),
            padding='VALID',
            name='stem_conv'
        )(x)
        x = nn.LayerNorm(name='stem_norm')(x)

        # --- Stages ---
        for stage_idx in range(4):
            dim = dims[stage_idx]

            # Downsample between stages (not before first)
            if stage_idx > 0:
                x = nn.LayerNorm(name=f'downsample_norm_{stage_idx}')(x)
                x = nn.Conv(
                    dim,
                    kernel_size=(2, 2),
                    strides=(2, 2),
                    padding='VALID',
                    name=f'downsample_conv_{stage_idx}'
                )(x)

            # Apply blocks for this stage
            for block_idx_in_stage in range(self.depths[stage_idx]):
                dp_rate = float(dp_rates[block_idx])

                if self.mode == 'pure':
                    x = CSSMNextBlock(
                        dim=dim,
                        cssm_cls=CSSM,
                        dense_mixing=dense,
                        concat_xy=self.concat_xy,
                        gate_activation=self.gate_activation,
                        drop_path=dp_rate,
                        name=f'stage{stage_idx}_block{block_idx_in_stage}'
                    )(x, training=training)
                else:  # hybrid
                    x = HybridBlock(
                        dim=dim,
                        cssm_cls=CSSM,
                        dense_mixing=dense,
                        concat_xy=self.concat_xy,
                        gate_activation=self.gate_activation,
                        drop_path=dp_rate,
                        name=f'stage{stage_idx}_block{block_idx_in_stage}'
                    )(x, training=training)

                block_idx += 1

        # --- Head ---
        # Global average pooling over Time, Height, Width
        x = jnp.mean(x, axis=(1, 2, 3))  # (B, dim)

        x = nn.LayerNorm(name='head_norm')(x)
        x = nn.Dense(self.num_classes, name='head')(x)

        return x
