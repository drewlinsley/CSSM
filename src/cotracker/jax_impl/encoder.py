"""
JAX/Flax implementation of CoTracker3 BasicEncoder.

Faithful port of the ResNet-style CNN encoder from the original PyTorch implementation.
Reference: https://github.com/facebookresearch/co-tracker
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional


class InstanceNorm2d(nn.Module):
    """Instance Normalization for 2D inputs (NHWC format)."""
    num_features: int
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W, C)
        mean = x.mean(axis=(1, 2), keepdims=True)
        var = x.var(axis=(1, 2), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.epsilon)

        # Learnable scale and shift
        scale = self.param('scale', nn.initializers.ones, (self.num_features,))
        bias = self.param('bias', nn.initializers.zeros, (self.num_features,))
        return x * scale + bias


class ResidualBlock(nn.Module):
    """Residual block with configurable normalization.

    Faithful port of CoTracker's ResidualBlock.

    Attributes:
        in_planes: Input channels
        planes: Output channels
        norm_fn: Normalization type ('instance', 'batch', 'group', 'none')
        stride: Convolution stride
    """
    in_planes: int
    planes: int
    norm_fn: str = 'instance'
    stride: int = 1

    def setup(self):
        self.conv1 = nn.Conv(
            self.planes,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding='SAME',
        )
        self.conv2 = nn.Conv(
            self.planes,
            kernel_size=(3, 3),
            padding='SAME',
        )

        # Normalization layers
        self.norm1 = self._make_norm(self.planes)
        self.norm2 = self._make_norm(self.planes)

        # Downsample if needed
        if self.stride != 1 or self.in_planes != self.planes:
            self.downsample = nn.Conv(
                self.planes,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
            )
            self.norm3 = self._make_norm(self.planes)
        else:
            self.downsample = None
            self.norm3 = None

    def _make_norm(self, features: int):
        if self.norm_fn == 'instance':
            return InstanceNorm2d(features)
        elif self.norm_fn == 'batch':
            return nn.BatchNorm()
        elif self.norm_fn == 'group':
            return nn.GroupNorm(num_groups=features // 8)
        else:
            return lambda x: x

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        y = x

        y = self.conv1(y)
        if self.norm_fn == 'batch':
            y = self.norm1(y, use_running_average=not training)
        else:
            y = self.norm1(y)
        y = nn.relu(y)

        y = self.conv2(y)
        if self.norm_fn == 'batch':
            y = self.norm2(y, use_running_average=not training)
        else:
            y = self.norm2(y)

        if self.downsample is not None:
            x = self.downsample(x)
            if self.norm_fn == 'batch':
                x = self.norm3(x, use_running_average=not training)
            else:
                x = self.norm3(x)

        return nn.relu(x + y)


class BasicEncoder(nn.Module):
    """Basic CNN encoder for video feature extraction.

    Faithful port of CoTracker's BasicEncoder with multi-scale feature fusion.

    Architecture:
        conv1 (7x7, stride=2) -> layer1 (stride=1) -> layer2 (stride=2)
        -> layer3 (stride=2) -> layer4 (stride=2)
        -> upsample all to H/4 -> concat -> conv2 -> conv3

    Attributes:
        input_dim: Input channels (default: 3 for RGB)
        output_dim: Output feature dimension (default: 128)
        stride: Output stride (default: 4)
    """
    input_dim: int = 3
    output_dim: int = 128
    stride: int = 4

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Extract features from input images or video.

        Args:
            x: Input images (B, H, W, C) or video (B, T, H, W, C)
            training: Training mode

        Returns:
            Features (B, H/stride, W/stride, output_dim) or
                     (B, T, H/stride, W/stride, output_dim)
        """
        # Handle video input
        is_video = x.ndim == 5
        if is_video:
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, H, W, C)
        else:
            B, H, W, C = x.shape

        norm_fn = 'instance'
        in_planes = self.output_dim // 2  # 64

        # Initial convolution
        x = nn.Conv(
            in_planes,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='SAME',
            name='conv1',
        )(x)
        x = InstanceNorm2d(in_planes, name='norm1')(x)
        x = nn.relu(x)

        # Residual layers
        # layer1: stride=1, out=64
        x, in_planes = self._make_layer(x, in_planes, self.output_dim // 2, 2, 1, norm_fn, training, 'layer1')
        a = x  # H/2

        # layer2: stride=2, out=96
        x, in_planes = self._make_layer(x, in_planes, self.output_dim // 4 * 3, 2, 2, norm_fn, training, 'layer2')
        b = x  # H/4

        # layer3: stride=2, out=128
        x, in_planes = self._make_layer(x, in_planes, self.output_dim, 2, 2, norm_fn, training, 'layer3')
        c = x  # H/8

        # layer4: stride=2, out=128
        x, in_planes = self._make_layer(x, in_planes, self.output_dim, 2, 2, norm_fn, training, 'layer4')
        d = x  # H/16

        # Target size for fusion
        target_h = H // self.stride
        target_w = W // self.stride

        # Upsample all to target size and concatenate
        def _bilinear_interpolate(feat):
            return jax.image.resize(
                feat,
                (feat.shape[0], target_h, target_w, feat.shape[-1]),
                method='bilinear',
            )

        a = _bilinear_interpolate(a)  # 64 channels
        b = _bilinear_interpolate(b)  # 96 channels
        c = _bilinear_interpolate(c)  # 128 channels
        d = _bilinear_interpolate(d)  # 128 channels

        # Concatenate: 64 + 96 + 128 + 128 = 416 channels
        x = jnp.concatenate([a, b, c, d], axis=-1)

        # Final convolutions
        x = nn.Conv(
            self.output_dim * 2,
            kernel_size=(3, 3),
            padding='SAME',
            name='conv2',
        )(x)
        x = InstanceNorm2d(self.output_dim * 2, name='norm2')(x)
        x = nn.relu(x)

        x = nn.Conv(
            self.output_dim,
            kernel_size=(1, 1),
            name='conv3',
        )(x)

        # Reshape back to video if needed
        if is_video:
            x = x.reshape(B, T, target_h, target_w, self.output_dim)

        return x

    def _make_layer(
        self,
        x: jnp.ndarray,
        in_planes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        norm_fn: str,
        training: bool,
        name: str,
    ) -> Tuple[jnp.ndarray, int]:
        """Create a layer of residual blocks."""
        x = ResidualBlock(
            in_planes=in_planes,
            planes=planes,
            norm_fn=norm_fn,
            stride=stride,
            name=f'{name}_block0',
        )(x, training)

        for i in range(1, num_blocks):
            x = ResidualBlock(
                in_planes=planes,
                planes=planes,
                norm_fn=norm_fn,
                stride=1,
                name=f'{name}_block{i}',
            )(x, training)

        return x, planes
