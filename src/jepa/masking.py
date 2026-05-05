"""
Tube masking strategies for video JEPA.

Implements spatiotemporal masking strategies:
1. TubeMasking: Same spatial positions masked across all frames
2. VJEPAMasking: Causal future prediction with multi-block masking (V-JEPA style)

V-JEPA Reference: https://arxiv.org/abs/2312.00821
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class MaskInfo:
    """Container for masking information."""
    mask: jnp.ndarray  # Binary mask (B, T, H, W) where 1=visible, 0=masked
    visible_indices: jnp.ndarray  # Indices of visible positions
    masked_indices: jnp.ndarray  # Indices of masked positions
    num_visible: int  # Number of visible positions per sample
    num_masked: int  # Number of masked positions per sample


@dataclass
class VJEPAMaskInfo:
    """Container for V-JEPA causal masking information."""
    # Context (past) - all visible
    context_frames: int  # Number of context frames

    # Target (future) - partially masked
    target_frames: int  # Number of target frames
    target_mask: jnp.ndarray  # Binary mask for target region (B, T_target, H, W)

    # Indices for target region only
    target_visible_indices: jnp.ndarray  # (B, num_target_visible)
    target_masked_indices: jnp.ndarray   # (B, num_target_masked)
    num_target_visible: int
    num_target_masked: int

    # Block info
    num_blocks: int  # Number of masked blocks
    block_positions: jnp.ndarray  # (B, num_blocks, 4) - (t_start, t_end, h_start, w_start)


class TubeMasking:
    """
    Spatiotemporal tube masking for video JEPA.

    Same spatial positions are masked across all temporal frames,
    creating "tubes" through time. This forces the model to learn
    temporal consistency and spatiotemporal representations.

    Args:
        mask_ratio: Fraction of spatial positions to mask (default 0.75)
        tube_size: Size of each masking unit (h, w) for block masking
        temporal_consistent: If True, same spatial mask across all frames
    """

    def __init__(
        self,
        mask_ratio: float = 0.75,
        tube_size: Tuple[int, int] = (1, 1),
        temporal_consistent: bool = True,
    ):
        self.mask_ratio = mask_ratio
        self.tube_size = tube_size
        self.temporal_consistent = temporal_consistent

    def __call__(
        self,
        shape: Tuple[int, int, int, int],
        rng: jax.random.PRNGKey,
    ) -> MaskInfo:
        """
        Generate tube mask for video.

        Args:
            shape: (B, T, H, W) feature map shape
            rng: JAX random key

        Returns:
            MaskInfo containing mask and indices
        """
        B, T, H, W = shape
        tube_h, tube_w = self.tube_size

        # Number of masking units in each dimension
        num_h = H // tube_h
        num_w = W // tube_w
        num_patches = num_h * num_w

        # Number of patches to mask
        num_masked = int(num_patches * self.mask_ratio)
        num_visible = num_patches - num_masked

        # Generate mask for each sample in batch
        masks = []
        visible_indices_list = []
        masked_indices_list = []

        for b in range(B):
            rng, key = jax.random.split(rng)

            # Random permutation of patch indices
            perm = jax.random.permutation(key, num_patches)

            # Visible patches are first (1 - mask_ratio) positions
            visible_idx = jnp.sort(perm[:num_visible])
            masked_idx = jnp.sort(perm[num_visible:])

            # Create spatial mask (num_h, num_w)
            spatial_mask = jnp.zeros((num_h, num_w), dtype=jnp.float32)

            # Set visible positions to 1
            visible_h = visible_idx // num_w
            visible_w = visible_idx % num_w
            spatial_mask = spatial_mask.at[visible_h, visible_w].set(1.0)

            # Expand mask to full resolution with tube_size
            if tube_h > 1 or tube_w > 1:
                spatial_mask = jnp.repeat(spatial_mask, tube_h, axis=0)
                spatial_mask = jnp.repeat(spatial_mask, tube_w, axis=1)

            # Handle edge cases where H, W are not perfectly divisible
            spatial_mask = spatial_mask[:H, :W]

            # Extend across time dimension (tube masking)
            if self.temporal_consistent:
                # Same mask for all frames
                frame_mask = jnp.broadcast_to(
                    spatial_mask[jnp.newaxis, :, :],
                    (T, H, W)
                )
            else:
                # Different mask per frame (not tube masking)
                frame_masks = []
                for t in range(T):
                    rng, key = jax.random.split(rng)
                    perm = jax.random.permutation(key, num_patches)
                    vis_idx = jnp.sort(perm[:num_visible])
                    sm = jnp.zeros((num_h, num_w), dtype=jnp.float32)
                    sm = sm.at[vis_idx // num_w, vis_idx % num_w].set(1.0)
                    if tube_h > 1 or tube_w > 1:
                        sm = jnp.repeat(sm, tube_h, axis=0)
                        sm = jnp.repeat(sm, tube_w, axis=1)
                    frame_masks.append(sm[:H, :W])
                frame_mask = jnp.stack(frame_masks, axis=0)

            masks.append(frame_mask)
            visible_indices_list.append(visible_idx)
            masked_indices_list.append(masked_idx)

        # Stack batch
        mask = jnp.stack(masks, axis=0)  # (B, T, H, W)
        visible_indices = jnp.stack(visible_indices_list, axis=0)  # (B, num_visible)
        masked_indices = jnp.stack(masked_indices_list, axis=0)  # (B, num_masked)

        return MaskInfo(
            mask=mask,
            visible_indices=visible_indices,
            masked_indices=masked_indices,
            num_visible=num_visible,
            num_masked=num_masked,
        )


def apply_mask_to_tokens(
    tokens: jnp.ndarray,
    mask_info: MaskInfo,
    mask_token: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply mask to tokenized features.

    Args:
        tokens: (B, T, N, D) tokenized features where N = H' * W'
        mask_info: MaskInfo from TubeMasking
        mask_token: Optional learnable mask token (D,)

    Returns:
        visible_tokens: (B, T, num_visible, D) visible tokens only
        masked_positions: (B, num_masked) indices of masked tokens
    """
    B, T, N, D = tokens.shape

    # Gather visible tokens using indices
    # For tube masking, same indices across time
    visible_indices = mask_info.visible_indices  # (B, num_visible)

    # Expand indices for gathering
    visible_indices_expanded = visible_indices[:, jnp.newaxis, :, jnp.newaxis]
    visible_indices_expanded = jnp.broadcast_to(
        visible_indices_expanded,
        (B, T, mask_info.num_visible, D)
    )

    # Gather visible tokens
    tokens_flat = tokens.reshape(B, T, N, D)
    visible_tokens = jnp.take_along_axis(
        tokens_flat,
        visible_indices_expanded,
        axis=2
    )

    return visible_tokens, mask_info.masked_indices


def extract_at_indices(
    features: jnp.ndarray,
    indices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Extract features at specific spatial indices.

    Args:
        features: (B, T, H, W, D) or (B, T, N, D) features
        indices: (B, num_indices) spatial indices to extract

    Returns:
        extracted: (B, T, num_indices, D) features at indices
    """
    if features.ndim == 5:
        B, T, H, W, D = features.shape
        # Flatten spatial dimensions
        features = features.reshape(B, T, H * W, D)

    B, T, N, D = features.shape
    num_indices = indices.shape[1]

    # Expand indices for gathering
    indices_expanded = indices[:, jnp.newaxis, :, jnp.newaxis]
    indices_expanded = jnp.broadcast_to(
        indices_expanded,
        (B, T, num_indices, D)
    )

    # Gather
    extracted = jnp.take_along_axis(features, indices_expanded, axis=2)

    return extracted


# =============================================================================
# V-JEPA Style Causal Masking
# =============================================================================

class VJEPAMasking:
    """
    V-JEPA style causal masking for video future prediction.

    Splits video into:
    - Context (past): First T_c frames, fully visible
    - Target (future): Last T_t frames, with masked spatiotemporal blocks

    Multiple random spatiotemporal blocks are masked in the target region.
    The predictor must predict these masked regions using only past context.

    This is CAUSAL: only past information is used to predict future.

    Reference: "V-JEPA: Video Joint Embedding Predictive Architecture"
               https://arxiv.org/abs/2312.00821

    Args:
        context_ratio: Fraction of frames used as context (default 0.5)
        mask_ratio: Fraction of target region to mask (default 0.75)
        num_blocks: Number of masked blocks to sample (default 4)
        block_size: (temporal, height, width) size of each block
        min_context_frames: Minimum context frames (default 4)
    """

    def __init__(
        self,
        context_ratio: float = 0.5,
        mask_ratio: float = 0.75,
        num_blocks: int = 4,
        block_size: Tuple[int, int, int] = (2, 2, 2),  # (t, h, w)
        min_context_frames: int = 4,
    ):
        self.context_ratio = context_ratio
        self.mask_ratio = mask_ratio
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.min_context_frames = min_context_frames

    def __call__(
        self,
        shape: Tuple[int, int, int, int],
        rng: jax.random.PRNGKey,
    ) -> VJEPAMaskInfo:
        """
        Generate V-JEPA causal mask (JIT-compatible).

        Args:
            shape: (B, T, H, W) feature map shape
            rng: JAX random key

        Returns:
            VJEPAMaskInfo with context/target split and masked block info
        """
        B, T, H, W = shape
        block_t, block_h, block_w = self.block_size

        # Split into context (past) and target (future)
        # Use Python ints for static values (these are known at trace time)
        context_frames = max(
            self.min_context_frames,
            int(T * self.context_ratio)
        )
        target_frames = T - context_frames

        if target_frames < block_t:
            target_frames = min(block_t, T - self.min_context_frames)
            context_frames = T - target_frames

        # Number of blocks in target region (all static Python ints)
        num_t_blocks = max(1, target_frames // block_t)
        num_h_blocks = max(1, H // block_h)
        num_w_blocks = max(1, W // block_w)
        total_blocks = num_t_blocks * num_h_blocks * num_w_blocks

        # Number of blocks to mask
        num_masked_blocks = min(self.num_blocks, total_blocks)

        # Total positions in target
        target_positions = target_frames * H * W

        # Estimate number of masked positions based on block coverage
        # Each block masks block_t * block_h * block_w positions
        block_volume = block_t * block_h * block_w
        estimated_masked = min(num_masked_blocks * block_volume, target_positions)
        # Add buffer for overlapping blocks
        max_masked_size = min(int(estimated_masked * 1.5) + 10, target_positions)

        # Generate mask using vectorized JAX operations
        # Create block masks for all batches at once
        rng_keys = jax.random.split(rng, B)

        def generate_single_mask(key):
            """Generate mask for a single batch element."""
            # Sample random block indices
            block_indices = jax.random.permutation(key, total_blocks)[:num_masked_blocks]

            # Create coordinate grids for the target region
            t_coords = jnp.arange(target_frames)[:, None, None]  # (T_t, 1, 1)
            h_coords = jnp.arange(H)[None, :, None]              # (1, H, 1)
            w_coords = jnp.arange(W)[None, None, :]              # (1, 1, W)

            # Initialize mask as all visible
            target_mask = jnp.ones((target_frames, H, W), dtype=jnp.float32)

            # For each block, compute which positions it masks
            def mask_block(carry, block_idx):
                current_mask = carry
                # Convert block index to block coordinates
                t_block = (block_idx // (num_h_blocks * num_w_blocks)) % num_t_blocks
                h_block = (block_idx // num_w_blocks) % num_h_blocks
                w_block = block_idx % num_w_blocks

                # Block boundaries
                t_start = t_block * block_t
                h_start = h_block * block_h
                w_start = w_block * block_w

                # Create mask for this block (1 where block is, 0 elsewhere)
                in_t = (t_coords >= t_start) & (t_coords < t_start + block_t)
                in_h = (h_coords >= h_start) & (h_coords < h_start + block_h)
                in_w = (w_coords >= w_start) & (w_coords < w_start + block_w)
                block_mask = in_t & in_h & in_w

                # Apply mask (set to 0 where block is)
                new_mask = jnp.where(block_mask, 0.0, current_mask)
                return new_mask, None

            # Apply all blocks using scan
            target_mask, _ = jax.lax.scan(mask_block, target_mask, block_indices)

            return target_mask

        # Vectorize over batch
        target_masks = jax.vmap(generate_single_mask)(rng_keys)  # (B, T_t, H, W)

        # Get masked indices from the mask
        # Flatten each mask and find masked positions
        target_masks_flat = target_masks.reshape(B, -1)  # (B, T_t * H * W)

        # Use argsort to get indices of masked (0) vs visible (1) positions
        # Masked positions (0) will come first after sorting
        sorted_indices = jnp.argsort(target_masks_flat, axis=1)

        # Count masked positions per batch element
        num_masked_per_batch = (target_masks_flat < 0.5).sum(axis=1)  # (B,)

        # Get maximum number of masked positions across batch
        # Use a fixed size based on expected masking
        max_num_masked = max_masked_size

        # Extract masked indices (first max_num_masked of sorted indices)
        target_masked_indices = sorted_indices[:, :max_num_masked]

        # Create block positions array (simplified - just store block indices)
        # For full position info, would need more complex tracking
        block_positions = jnp.zeros((B, num_masked_blocks, 4), dtype=jnp.int32)

        return VJEPAMaskInfo(
            context_frames=context_frames,
            target_frames=target_frames,
            target_mask=target_masks,
            target_visible_indices=sorted_indices[:, max_num_masked:],  # Visible come after masked
            target_masked_indices=target_masked_indices,
            num_target_visible=target_positions - max_num_masked,
            num_target_masked=max_num_masked,
            num_blocks=num_masked_blocks,
            block_positions=block_positions,
        )


def extract_context_target(
    video: jnp.ndarray,
    mask_info: VJEPAMaskInfo,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Split video into context and target portions.

    Args:
        video: (B, T, H, W, C) or (B, T, H, W, D) video/features
        mask_info: VJEPAMaskInfo from VJEPAMasking

    Returns:
        context: (B, T_context, H, W, C/D) past frames (fully visible)
        target: (B, T_target, H, W, C/D) future frames (to be masked)
    """
    T_c = mask_info.context_frames
    context = video[:, :T_c]
    target = video[:, T_c:]
    return context, target


def extract_masked_target_tokens(
    target_features: jnp.ndarray,
    mask_info: VJEPAMaskInfo,
) -> jnp.ndarray:
    """
    Extract features at masked positions in target region.

    Args:
        target_features: (B, T_target, H, W, D) target region features
        mask_info: VJEPAMaskInfo

    Returns:
        masked_features: (B, num_masked, D) features at masked positions
    """
    B, T_t, H, W, D = target_features.shape

    # Flatten spatiotemporal dimensions
    target_flat = target_features.reshape(B, T_t * H * W, D)

    # Gather at masked indices
    indices = mask_info.target_masked_indices  # (B, num_masked)
    indices_exp = indices[:, :, jnp.newaxis]
    indices_exp = jnp.broadcast_to(indices_exp, (B, mask_info.num_target_masked, D))

    # Handle -1 padding by clamping
    indices_exp = jnp.clip(indices_exp, 0, T_t * H * W - 1)

    masked_features = jnp.take_along_axis(target_flat, indices_exp, axis=1)

    return masked_features
