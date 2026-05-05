"""
JAX/Flax implementation of CoTracker3.

Faithful port of the complete point tracking model from the original PyTorch implementation.
Reference: https://github.com/facebookresearch/co-tracker

This module provides:
- CoTrackerThree: Full CoTracker3 with transformer attention
- CoTrackerThreeCSSM: Variant with CSSM replacing transformer
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Optional, Tuple, Any
from functools import partial

from .encoder import BasicEncoder
from .correlation import EfficientCorrBlock, bilinear_sample_simple
from .updateformer import EfficientUpdateFormer, CSSMUpdateFormer
from .blocks import Mlp


def get_sinusoidal_encoding(
    coords: jnp.ndarray,
    num_bands: int = 11,
) -> jnp.ndarray:
    """
    Sinusoidal positional encoding for coordinates.

    Args:
        coords: Coordinates (..., 2) containing (x, y)
        num_bands: Number of frequency bands (0 to num_bands-1)

    Returns:
        Encoded coordinates (..., 2 * 2 * num_bands)
    """
    # Frequency bands: 2^0, 2^1, ..., 2^(num_bands-1)
    freqs = 2.0 ** jnp.arange(num_bands)  # (num_bands,)

    # Encode x and y separately
    x = coords[..., 0:1]  # (..., 1)
    y = coords[..., 1:2]  # (..., 1)

    # sin and cos for each frequency
    x_enc = jnp.concatenate([
        jnp.sin(x * freqs),
        jnp.cos(x * freqs),
    ], axis=-1)  # (..., 2*num_bands)

    y_enc = jnp.concatenate([
        jnp.sin(y * freqs),
        jnp.cos(y * freqs),
    ], axis=-1)  # (..., 2*num_bands)

    return jnp.concatenate([x_enc, y_enc], axis=-1)  # (..., 4*num_bands)


class CoTrackerThree(nn.Module):
    """
    CoTracker3: Transformer-based point tracking model.

    Faithful JAX port of the original PyTorch implementation.

    Attributes:
        stride: Feature extraction stride
        window_len: Temporal window length
        hidden_size: Transformer hidden dimension
        latent_dim: Feature encoder output dimension
        corr_levels: Number of correlation pyramid levels
        corr_radius: Correlation sampling radius
        num_iters: Number of refinement iterations
        space_depth: Transformer spatial attention depth
        time_depth: Transformer temporal attention depth
        num_virtual_tracks: Number of virtual tokens
        add_space_attn: Whether to use spatial attention
    """
    stride: int = 4
    window_len: int = 8
    hidden_size: int = 384
    latent_dim: int = 128
    corr_levels: int = 4
    corr_radius: int = 3
    num_iters: int = 4
    space_depth: int = 3
    time_depth: int = 3
    num_virtual_tracks: int = 64
    add_space_attn: bool = True

    def setup(self):
        # Feature encoder
        self.fnet = BasicEncoder(
            output_dim=self.latent_dim,
            stride=self.stride,
        )

        # Correlation MLP: process correlation features
        corr_dim = self.corr_levels * (2 * self.corr_radius + 1) ** 2
        self.corr_mlp = Mlp(
            in_features=corr_dim,
            hidden_features=corr_dim * 2,
            out_features=corr_dim,
        )

        # Compute input dimension for updateformer
        # Correlation features + flow encoding + vis + conf
        flow_enc_dim = 4 * 11  # sinusoidal encoding: 4 * num_bands
        self.input_dim = corr_dim + flow_enc_dim + 1 + 1  # corr + flow + vis + conf

        # Update transformer
        self.updateformer = EfficientUpdateFormer(
            space_depth=self.space_depth,
            time_depth=self.time_depth,
            input_dim=self.input_dim,
            hidden_size=self.hidden_size,
            output_dim=4,  # dx, dy, vis, conf
            num_virtual_tracks=self.num_virtual_tracks,
            add_space_attn=self.add_space_attn,
        )

    def __call__(
        self,
        video: jnp.ndarray,
        queries: jnp.ndarray,
        training: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Track points through video.

        Args:
            video: Input video (B, T, H, W, 3), values in [0, 1]
            queries: Query points (B, N, 3) where each row is [frame_idx, x, y]
                     x, y are in pixel coordinates [0, W-1] x [0, H-1]
            training: Training mode

        Returns:
            Dictionary with:
            - coords: Predicted coordinates (B, T, N, 2) in pixel space
            - vis: Visibility predictions (B, T, N, 1)
            - conf: Confidence predictions (B, T, N, 1)
        """
        B, T, H, W, C = video.shape
        N = queries.shape[1]

        # Normalize video to [-1, 1]
        video_norm = video * 2 - 1

        # Extract features
        fmaps = self.fnet(video_norm, training=training)  # (B, T, H/s, W/s, latent_dim)
        _, _, H_f, W_f, _ = fmaps.shape

        # Scale factor from image to feature space
        scale = jnp.array([W_f / W, H_f / H])

        # Initialize tracks from query positions
        query_frames = queries[:, :, 0].astype(jnp.int32)  # (B, N)
        query_coords = queries[:, :, 1:3]  # (B, N, 2) - x, y in pixel space

        # Initialize coordinates at all timesteps
        coords = jnp.broadcast_to(
            query_coords[:, :, None, :],
            (B, N, T, 2),
        ).transpose(0, 2, 1, 3)  # (B, T, N, 2)
        coords = jnp.array(coords)  # Make mutable

        # Initialize visibility and confidence
        vis = jnp.ones((B, T, N, 1)) * 0.5
        conf = jnp.ones((B, T, N, 1)) * 0.5

        # Sample initial track features at query locations
        track_feats = self._sample_features(fmaps, query_frames, query_coords * scale)
        # track_feats: (B, N, latent_dim)

        # Build correlation block
        corr_block = EfficientCorrBlock(
            fmaps=fmaps,
            num_levels=self.corr_levels,
            radius=self.corr_radius,
        )

        # Collect predictions from all iterations
        all_coords = []
        all_vis = []
        all_conf = []

        # Iterative refinement
        for iter_idx in range(self.num_iters):
            # Stop gradient on coords for stability
            coords_detached = jax.lax.stop_gradient(coords)

            # Scale coords to feature space
            coords_feat = coords_detached * scale

            # Compute correlation features
            # coords_feat: (B, T, N, 2), track_feats: (B, N, latent_dim)
            corr_feats = corr_block.sample(
                coords_feat,  # (B, T, N, 2) - S=T, N=N
                track_feats,
            )  # (B*N, T, corr_dim)
            corr_feats = corr_feats.reshape(B, N, T, -1)

            # Process through correlation MLP
            corr_feats = self.corr_mlp(corr_feats, training=training)

            # Compute flow encoding (relative coords from query)
            query_coords_expanded = jnp.broadcast_to(
                query_coords[:, :, None, :],
                (B, N, T, 2),
            )
            flow = coords_detached.transpose(0, 2, 1, 3) - query_coords_expanded  # (B, N, T, 2)
            flow_enc = get_sinusoidal_encoding(flow)  # (B, N, T, flow_enc_dim)

            # Concatenate all features
            transformer_input = jnp.concatenate([
                corr_feats,
                flow_enc,
                vis.transpose(0, 2, 1, 3),  # (B, N, T, 1)
                conf.transpose(0, 2, 1, 3),  # (B, N, T, 1)
            ], axis=-1)  # (B, N, T, input_dim)

            # Run transformer
            delta = self.updateformer(
                transformer_input,
                add_space_attn=self.add_space_attn,
                training=training,
            )  # (B, N, T, 4)

            # Update predictions
            delta_coords = delta[..., :2]  # (B, N, T, 2)
            delta_vis = delta[..., 2:3]  # (B, N, T, 1)
            delta_conf = delta[..., 3:4]  # (B, N, T, 1)

            coords = coords + delta_coords.transpose(0, 2, 1, 3)  # (B, T, N, 2)
            vis = jax.nn.sigmoid(vis + delta_vis.transpose(0, 2, 1, 3))
            conf = jax.nn.sigmoid(conf + delta_conf.transpose(0, 2, 1, 3))

            all_coords.append(coords)
            all_vis.append(vis)
            all_conf.append(conf)

        return {
            'coords': coords,  # (B, T, N, 2) in pixel space
            'vis': vis,  # (B, T, N, 1)
            'conf': conf,  # (B, T, N, 1)
            'all_coords': jnp.stack(all_coords, axis=0),  # (num_iters, B, T, N, 2)
            'all_vis': jnp.stack(all_vis, axis=0),  # (num_iters, B, T, N, 1)
            'all_conf': jnp.stack(all_conf, axis=0),  # (num_iters, B, T, N, 1)
        }

    def _sample_features(
        self,
        fmaps: jnp.ndarray,
        query_frames: jnp.ndarray,
        query_coords: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Sample features at query locations.

        Args:
            fmaps: Feature maps (B, T, H, W, C)
            query_frames: Frame indices (B, N)
            query_coords: Coordinates in feature space (B, N, 2)

        Returns:
            Sampled features (B, N, C)
        """
        B, T, H, W, C = fmaps.shape
        N = query_frames.shape[1]

        # Sample from each batch element
        features = []
        for b in range(B):
            batch_features = []
            for n in range(N):
                frame_idx = query_frames[b, n]
                fmap = fmaps[b, frame_idx]  # (H, W, C)
                coord = query_coords[b, n:n+1]  # (1, 2)
                feat = bilinear_sample_simple(fmap[None], coord[None])  # (1, 1, C)
                batch_features.append(feat[0, 0])
            features.append(jnp.stack(batch_features, axis=0))

        return jnp.stack(features, axis=0)  # (B, N, C)


class CoTrackerThreeCSSM(nn.Module):
    """
    CoTracker3 with CSSM replacing transformer attention.

    Same architecture as CoTrackerThree but uses CSSMUpdateFormer
    for temporal processing instead of transformer attention.

    Attributes:
        (same as CoTrackerThree)
        cssm_type: Type of CSSM ('standard', 'opponent', 'hgru_bi')
        kernel_size: CSSM kernel size
    """
    stride: int = 4
    window_len: int = 8
    hidden_size: int = 384
    latent_dim: int = 128
    corr_levels: int = 4
    corr_radius: int = 3
    num_iters: int = 4
    space_depth: int = 3
    time_depth: int = 3
    num_virtual_tracks: int = 64
    add_space_attn: bool = True
    cssm_type: str = 'opponent'
    kernel_size: int = 11

    def setup(self):
        # Feature encoder (same as transformer version)
        self.fnet = BasicEncoder(
            output_dim=self.latent_dim,
            stride=self.stride,
        )

        # Correlation MLP
        corr_dim = self.corr_levels * (2 * self.corr_radius + 1) ** 2
        self.corr_mlp = Mlp(
            in_features=corr_dim,
            hidden_features=corr_dim * 2,
            out_features=corr_dim,
        )

        # Input dimension
        flow_enc_dim = 4 * 11
        self.input_dim = corr_dim + flow_enc_dim + 1 + 1

        # CSSM Update block (replaces transformer)
        self.updateformer = CSSMUpdateFormer(
            space_depth=self.space_depth,
            time_depth=self.time_depth,
            input_dim=self.input_dim,
            hidden_size=self.hidden_size,
            output_dim=4,
            num_virtual_tracks=self.num_virtual_tracks,
            add_space_attn=self.add_space_attn,
            cssm_type=self.cssm_type,
            kernel_size=self.kernel_size,
        )

    def __call__(
        self,
        video: jnp.ndarray,
        queries: jnp.ndarray,
        training: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """Same interface as CoTrackerThree."""
        B, T, H, W, C = video.shape
        N = queries.shape[1]

        video_norm = video * 2 - 1
        fmaps = self.fnet(video_norm, training=training)
        _, _, H_f, W_f, _ = fmaps.shape

        scale = jnp.array([W_f / W, H_f / H])

        query_frames = queries[:, :, 0].astype(jnp.int32)
        query_coords = queries[:, :, 1:3]

        coords = jnp.broadcast_to(
            query_coords[:, :, None, :],
            (B, N, T, 2),
        ).transpose(0, 2, 1, 3)
        coords = jnp.array(coords)

        vis = jnp.ones((B, T, N, 1)) * 0.5
        conf = jnp.ones((B, T, N, 1)) * 0.5

        track_feats = self._sample_features(fmaps, query_frames, query_coords * scale)

        corr_block = EfficientCorrBlock(
            fmaps=fmaps,
            num_levels=self.corr_levels,
            radius=self.corr_radius,
        )

        all_coords = []
        all_vis = []
        all_conf = []

        for iter_idx in range(self.num_iters):
            coords_detached = jax.lax.stop_gradient(coords)
            coords_feat = coords_detached * scale

            # coords_feat: (B, T, N, 2), track_feats: (B, N, latent_dim)
            corr_feats = corr_block.sample(
                coords_feat,  # (B, T, N, 2) - S=T, N=N
                track_feats,
            )  # (B*N, T, corr_dim)
            corr_feats = corr_feats.reshape(B, N, T, -1)
            corr_feats = self.corr_mlp(corr_feats, training=training)

            query_coords_expanded = jnp.broadcast_to(
                query_coords[:, :, None, :],
                (B, N, T, 2),
            )
            flow = coords_detached.transpose(0, 2, 1, 3) - query_coords_expanded
            flow_enc = get_sinusoidal_encoding(flow)

            transformer_input = jnp.concatenate([
                corr_feats,
                flow_enc,
                vis.transpose(0, 2, 1, 3),
                conf.transpose(0, 2, 1, 3),
            ], axis=-1)

            delta = self.updateformer(
                transformer_input,
                add_space_attn=self.add_space_attn,
                training=training,
            )

            delta_coords = delta[..., :2]
            delta_vis = delta[..., 2:3]
            delta_conf = delta[..., 3:4]

            coords = coords + delta_coords.transpose(0, 2, 1, 3)
            vis = jax.nn.sigmoid(vis + delta_vis.transpose(0, 2, 1, 3))
            conf = jax.nn.sigmoid(conf + delta_conf.transpose(0, 2, 1, 3))

            all_coords.append(coords)
            all_vis.append(vis)
            all_conf.append(conf)

        return {
            'coords': coords,
            'vis': vis,
            'conf': conf,
            'all_coords': jnp.stack(all_coords, axis=0),
            'all_vis': jnp.stack(all_vis, axis=0),
            'all_conf': jnp.stack(all_conf, axis=0),
        }

    def _sample_features(
        self,
        fmaps: jnp.ndarray,
        query_frames: jnp.ndarray,
        query_coords: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample features at query locations."""
        B, T, H, W, C = fmaps.shape
        N = query_frames.shape[1]

        features = []
        for b in range(B):
            batch_features = []
            for n in range(N):
                frame_idx = query_frames[b, n]
                fmap = fmaps[b, frame_idx]
                coord = query_coords[b, n:n+1]
                feat = bilinear_sample_simple(fmap[None], coord[None])
                batch_features.append(feat[0, 0])
            features.append(jnp.stack(batch_features, axis=0))

        return jnp.stack(features, axis=0)


def create_cotracker3(
    use_cssm: bool = False,
    cssm_type: str = 'opponent',
    **kwargs,
) -> nn.Module:
    """
    Factory function to create CoTracker3 model.

    Args:
        use_cssm: Whether to use CSSM variant
        cssm_type: Type of CSSM if use_cssm=True
        **kwargs: Additional model configuration

    Returns:
        CoTrackerThree or CoTrackerThreeCSSM model
    """
    if use_cssm:
        return CoTrackerThreeCSSM(cssm_type=cssm_type, **kwargs)
    else:
        return CoTrackerThree(**kwargs)
