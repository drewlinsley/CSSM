"""
JAX/Flax implementation of CoTracker3 correlation computation.

Faithful port of the correlation blocks from the original PyTorch implementation.
Reference: https://github.com/facebookresearch/co-tracker
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, List, Optional


def bilinear_sampler(
    img: jnp.ndarray,
    coords: jnp.ndarray,
    padding_mode: str = 'zeros',
) -> jnp.ndarray:
    """
    Bilinear sampling of image features at specified coordinates.

    Faithful port of PyTorch's grid_sample with bilinear interpolation.

    Args:
        img: Feature map (B, H, W, C) or (B, C, 1, H, W) for 3D
        coords: Normalized coordinates in [-1, 1] range
                For 2D: (B, N, 1, 1, 2) or (B, N, H_out, W_out, 2)
                The last dim is (x, y) where x is width, y is height

    Returns:
        Sampled features
    """
    # Handle 5D input (B, C, 1, H, W) - reshape to 4D
    is_5d = img.ndim == 5
    if is_5d:
        B, C, _, H, W = img.shape
        img = img.squeeze(2).transpose(0, 2, 3, 1)  # (B, H, W, C)
    else:
        B, H, W, C = img.shape

    # Coords: last dim is (x, y) in [-1, 1]
    # Convert to pixel coordinates
    x = coords[..., 0]  # width direction
    y = coords[..., 1]  # height direction

    # Unnormalize: [-1, 1] -> [0, W-1] and [0, H-1]
    x = (x + 1) / 2 * (W - 1)
    y = (y + 1) / 2 * (H - 1)

    # Get integer and fractional parts
    x0 = jnp.floor(x).astype(jnp.int32)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(jnp.int32)
    y1 = y0 + 1

    # Clip to valid range
    x0_clip = jnp.clip(x0, 0, W - 1)
    x1_clip = jnp.clip(x1, 0, W - 1)
    y0_clip = jnp.clip(y0, 0, H - 1)
    y1_clip = jnp.clip(y1, 0, H - 1)

    # Compute interpolation weights
    wa = (x1.astype(jnp.float32) - x) * (y1.astype(jnp.float32) - y)
    wb = (x1.astype(jnp.float32) - x) * (y - y0.astype(jnp.float32))
    wc = (x - x0.astype(jnp.float32)) * (y1.astype(jnp.float32) - y)
    wd = (x - x0.astype(jnp.float32)) * (y - y0.astype(jnp.float32))

    # Sample at four corners
    # Use advanced indexing: img[batch_idx, y, x, :]
    batch_shape = x.shape[:-1] if x.ndim > 1 else ()

    # Flatten for indexing
    img_flat = img.reshape(B, H * W, C)

    def sample_at(y_idx, x_idx):
        # Combine y and x into linear index
        lin_idx = y_idx * W + x_idx
        # Gather
        batch_idx = jnp.arange(B).reshape((B,) + (1,) * (lin_idx.ndim - 1))
        batch_idx = jnp.broadcast_to(batch_idx, lin_idx.shape)
        return img_flat[batch_idx.flatten(), lin_idx.flatten()].reshape(lin_idx.shape + (C,))

    Ia = sample_at(y0_clip, x0_clip)
    Ib = sample_at(y1_clip, x0_clip)
    Ic = sample_at(y0_clip, x1_clip)
    Id = sample_at(y1_clip, x1_clip)

    # Handle padding mode for out-of-bounds
    if padding_mode == 'zeros':
        # Zero out contributions from out-of-bounds samples
        valid_a = ((x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H))[..., None]
        valid_b = ((x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H))[..., None]
        valid_c = ((x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H))[..., None]
        valid_d = ((x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H))[..., None]

        Ia = jnp.where(valid_a, Ia, 0.0)
        Ib = jnp.where(valid_b, Ib, 0.0)
        Ic = jnp.where(valid_c, Ic, 0.0)
        Id = jnp.where(valid_d, Id, 0.0)

    # Weighted sum
    out = wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id

    return out


def bilinear_sample_simple(
    feat: jnp.ndarray,
    coords: jnp.ndarray,
) -> jnp.ndarray:
    """
    Simple bilinear sampling for (B, H, W, C) features at (B, N, 2) coords.

    Args:
        feat: Feature map (B, H, W, C)
        coords: Pixel coordinates (B, N, 2) in [0, W-1] x [0, H-1]

    Returns:
        Sampled features (B, N, C)
    """
    B, H, W, C = feat.shape
    N = coords.shape[1]

    x = coords[..., 0]  # (B, N)
    y = coords[..., 1]  # (B, N)

    x0 = jnp.floor(x).astype(jnp.int32)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(jnp.int32)
    y1 = y0 + 1

    x0 = jnp.clip(x0, 0, W - 1)
    x1 = jnp.clip(x1, 0, W - 1)
    y0 = jnp.clip(y0, 0, H - 1)
    y1 = jnp.clip(y1, 0, H - 1)

    wa = (x1.astype(jnp.float32) - x) * (y1.astype(jnp.float32) - y)
    wb = (x1.astype(jnp.float32) - x) * (y - y0.astype(jnp.float32))
    wc = (x - x0.astype(jnp.float32)) * (y1.astype(jnp.float32) - y)
    wd = (x - x0.astype(jnp.float32)) * (y - y0.astype(jnp.float32))

    batch_idx = jnp.arange(B)[:, None]  # (B, 1)

    Ia = feat[batch_idx, y0, x0, :]
    Ib = feat[batch_idx, y1, x0, :]
    Ic = feat[batch_idx, y0, x1, :]
    Id = feat[batch_idx, y1, x1, :]

    out = (wa[..., None] * Ia +
           wb[..., None] * Ib +
           wc[..., None] * Ic +
           wd[..., None] * Id)

    return out


class EfficientCorrBlock:
    """
    Memory-efficient correlation block for point tracking.

    Faithful port of CoTracker's EfficientCorrBlock. Computes correlation
    between query features and a multi-scale feature pyramid without
    materializing the full correlation volume.

    Attributes:
        fmaps: Feature maps (B, S, C, H, W) - note: channel-first for compat
        num_levels: Number of pyramid levels
        radius: Correlation sampling radius
    """

    def __init__(
        self,
        fmaps: jnp.ndarray,
        num_levels: int = 4,
        radius: int = 4,
    ):
        """
        Initialize with feature maps.

        Args:
            fmaps: Feature maps (B, S, H, W, C) in JAX format
            num_levels: Number of pyramid levels
            radius: Sampling radius
        """
        self.num_levels = num_levels
        self.radius = radius

        # Convert to (B, S, C, H, W) for compatibility with original code
        B, S, H, W, C = fmaps.shape
        fmaps_chw = fmaps.transpose(0, 1, 4, 2, 3)  # (B, S, C, H, W)

        # Build feature pyramid
        self.fmaps_pyramid = [fmaps_chw]
        curr = fmaps_chw

        for i in range(self.num_levels - 1):
            # Average pool with stride 2
            curr_reshaped = curr.reshape(B * S, C, H // (2 ** i), W // (2 ** i))
            # Pool in spatial dims
            curr_pooled = jax.image.resize(
                curr_reshaped.transpose(0, 2, 3, 1),  # to NHWC
                (B * S, H // (2 ** (i + 1)), W // (2 ** (i + 1)), C),
                method='bilinear',
            ).transpose(0, 3, 1, 2)  # back to NCHW
            curr = curr_pooled.reshape(B, S, C, H // (2 ** (i + 1)), W // (2 ** (i + 1)))
            self.fmaps_pyramid.append(curr)

    def sample(
        self,
        coords: jnp.ndarray,
        target: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Sample correlations at specified coordinates.

        Args:
            coords: Query coordinates (B, S, N, 2) in pixel space
            target: Target features (B, N, C) to correlate against

        Returns:
            Correlation features (B*N, S, num_levels * (2*r+1)^2)
        """
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        # Reshape target for correlation: (B, S, N, C) with broadcast
        target = target[:, None, :, :]  # (B, 1, N, C)
        target = jnp.broadcast_to(target, (B, S, N, target.shape[-1]))
        target = target.transpose(0, 1, 3, 2)[:, :, :, :, None]  # (B, S, C, N, 1)

        out_pyramid = []

        for i in range(self.num_levels):
            pyramid = self.fmaps_pyramid[i]  # (B, S, C, H, W)
            C, H, W = pyramid.shape[2], pyramid.shape[3], pyramid.shape[4]

            # Scale coordinates for this level
            scale = 2 ** i
            coords_lvl = coords / scale  # (B, S, N, 2)

            # Create sampling grid: centroid + delta offsets
            # coords_lvl: (B, S, N, 2) -> need (B*S, N, 1, 1, 3) for 3D sampling
            # Add zero for temporal dim
            centroid_lvl = jnp.concatenate([
                jnp.zeros_like(coords_lvl[..., :1]),  # t=0
                coords_lvl,
            ], axis=-1)  # (B, S, N, 3)
            centroid_lvl = centroid_lvl.reshape(B * S, N, 1, 1, 3)

            # Delta grid
            dx = jnp.linspace(-r, r, 2 * r + 1)
            dy = jnp.linspace(-r, r, 2 * r + 1)
            xgrid, ygrid = jnp.meshgrid(dy, dx, indexing='ij')
            zgrid = jnp.zeros_like(xgrid)
            delta = jnp.stack([zgrid, xgrid, ygrid], axis=-1)  # (2r+1, 2r+1, 3)
            delta_lvl = delta.reshape(1, 1, 2 * r + 1, 2 * r + 1, 3)

            # Combine: (B*S, N, 2r+1, 2r+1, 3)
            coords_sample = centroid_lvl + delta_lvl

            # Normalize to [-1, 1]
            coords_norm = coords_sample.copy()
            coords_norm = coords_norm.at[..., 1].set(coords_sample[..., 1] / (H - 1) * 2 - 1)
            coords_norm = coords_norm.at[..., 2].set(coords_sample[..., 2] / (W - 1) * 2 - 1)

            # Sample from pyramid
            # pyramid: (B, S, C, H, W) -> (B*S, C, 1, H, W)
            pyramid_5d = pyramid.reshape(B * S, C, 1, H, W)

            # Use grid sample
            pyramid_sample = bilinear_sampler(pyramid_5d, coords_norm)  # (B*S, N, 2r+1, 2r+1, C)
            pyramid_sample = pyramid_sample.reshape(B, S, N, (2 * r + 1) ** 2, C)

            # Compute correlation with target
            target_expand = target.transpose(0, 1, 3, 2, 4)  # (B, S, N, C, 1)
            target_expand = target_expand.reshape(B, S, N, C)
            target_expand = target_expand[:, :, :, None, :]  # (B, S, N, 1, C)

            corr = jnp.sum(target_expand * pyramid_sample, axis=-1)  # (B, S, N, (2r+1)^2)
            corr = corr / jnp.sqrt(jnp.array(C, dtype=jnp.float32))

            out_pyramid.append(corr)

        # Concatenate all levels
        out = jnp.concatenate(out_pyramid, axis=-1)  # (B, S, N, L*(2r+1)^2)

        # Reshape to (B*N, S, D)
        out = out.transpose(0, 2, 1, 3)  # (B, N, S, D)
        out = out.reshape(B * N, S, -1)

        return out


def get_support_points(
    coords: jnp.ndarray,
    support_radius: int = 5,
    support_samples: int = 16,
) -> jnp.ndarray:
    """
    Get support point coordinates around query points.

    Args:
        coords: Query coordinates (B, N, 2)
        support_radius: Radius for support sampling
        support_samples: Number of support points

    Returns:
        Support coordinates (B, N, support_samples, 2)
    """
    B, N, _ = coords.shape

    # Create grid of offsets
    side = int(jnp.sqrt(support_samples))
    x_offsets = jnp.linspace(-support_radius, support_radius, side)
    y_offsets = jnp.linspace(-support_radius, support_radius, side)
    xx, yy = jnp.meshgrid(x_offsets, y_offsets, indexing='ij')
    offsets = jnp.stack([xx.flatten(), yy.flatten()], axis=-1)  # (support_samples, 2)

    # Add offsets to coords
    support = coords[:, :, None, :] + offsets[None, None, :, :]  # (B, N, S, 2)

    return support
