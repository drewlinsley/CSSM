"""
CSSM-SHViT: SHViT with CSSM replacing single-head attention.

Maintains the hierarchical structure with:
- ConvBlocks in early stages (unchanged)
- CSSM blocks replacing attention in later stages
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Sequence, Optional

from .cssm import GatedCSSM, GatedDeltaNetCSSM
from .shvit import DropPath, ConvBN, PatchEmbed, Downsample, Mlp, ConvBlock


class CSSMSHViTBlock(nn.Module):
    """
    SHViT block with CSSM replacing single-head attention.

    Supports two CSSM types:
    - 'gated': GatedCSSM (Spectral Mamba) — Mamba-style scalar recurrence with FFT spatial mixing
    - 'gdn': GatedDeltaNetCSSM — Matrix delta rule with QKV, RMSNorm output, SiLU gating
    """
    dim: int
    mlp_ratio: float = 4.0
    drop_path: float = 0.0
    cssm_type: str = 'gated'  # 'gated' (Spectral Mamba) or 'gdn' (GatedDeltaNet)
    dense_mixing: bool = False
    block_size: int = 32  # Block size for LMME channel mixing (only with gated + dense_mixing)
    mixing_rank: int = 0  # If > 0, use low-rank mixing (recommended: 4-16)
    gate_activation: str = 'softplus'
    num_timesteps: int = 8
    kernel_size: int = 15  # CSSM spectral kernel size
    spectral_rho: float = 0.999  # Maximum spectral magnitude for stability
    rope_mode: str = 'none'  # 'spatiotemporal', 'temporal', or 'none'
    use_dwconv: bool = True  # DWConv in MLP (matches SHViT)
    output_act: str = 'none'  # Output activation after CSSM: 'gelu', 'silu', or 'none'
    short_conv_size: int = 4       # Temporal causal conv kernel size (0=disabled)
    short_conv_spatial_size: int = 3  # Spatial depthwise conv kernel size (0=disabled, set 0 for 1x1)
    # GDN-specific
    delta_key_dim: int = 2         # Key dimension for GDN (2 or 3)
    output_norm: str = 'rms'       # Output norm for GDN ('rms', 'layer', 'none')
    gate_type: str = 'factored'    # Gate parameterization for GDN
    # Block norm
    block_norm: str = 'global_layer'  # 'layer' (per-frame C), 'global_layer' (T,H,W,C), 'temporal_layer' (T,C)
    use_pos_conv: bool = True         # Matches tog9av97 75% run. Helps under the default
                                      # 5-epoch warmup schedule; can hurt under shorter warmup.
    gate_proj_bias_init: float = 0.0  # Bisect 2026-04-16: 1.0 regresses epoch-1 val by ~2× on ImageNet
    # Static-image fast path (ImageNet speedup)
    static_image_fast_path: bool = False
    use_input_gates: bool = True  # B_k/B_v spatial input gates in GDN CSSM
    # Temporal SSL projection (per-block sown features for contrastive loss)
    ssl_proj_dim: int = 0  # 0=disabled. Forwarded to GatedDeltaNetCSSM only (gdn).
    # Timestep gate: data-dependent softmax pooling over the CSSM's T hidden
    # states, replacing the hard selection of x[:, -1]. Bias-initialized so
    # the softmax starts concentrated on t=T-1 (matches un-gated baseline).
    timestep_gate: bool = False
    # Mixed precision
    dtype: jnp.dtype = jnp.float32       # compute dtype for Dense/Conv/Norm (bf16 for mixed precision)
    param_dtype: jnp.dtype = jnp.float32 # storage dtype for weights (keep fp32 for optimizer stability)

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Forward pass supporting both image and video input.

        Args:
            x: Input tensor - either (B, H, W, C) for images or (B, T, H, W, C) for video
            deterministic: If True, disable dropout

        Returns:
            Output tensor with same dimensionality as input
        """
        is_image = x.ndim == 4
        fast_path = (
            self.static_image_fast_path
            and is_image
            and self.cssm_type == 'gdn'
        )

        ln_kwargs = dict(dtype=self.dtype, param_dtype=self.param_dtype)

        if fast_path:
            # --- 4D path: skip T-replication entirely ---
            B, H, W, C = x.shape
            residual = x  # 4D

            if self.block_norm == 'global_layer':
                x = nn.LayerNorm(reduction_axes=(1, 2, 3), feature_axes=-1,
                                 name='norm1', **ln_kwargs)(x)
            elif self.block_norm == 'temporal_layer':
                # No T dim in fast path — equivalent to per-C LayerNorm
                x = nn.LayerNorm(reduction_axes=(3,), feature_axes=-1,
                                 name='norm1', **ln_kwargs)(x)
            else:  # 'layer'
                x = nn.LayerNorm(name='norm1', **ln_kwargs)(x)
        else:
            # --- Existing 5D path ---
            if is_image:
                B, H, W, C = x.shape
                T = self.num_timesteps
                x = jnp.repeat(x[:, jnp.newaxis, :, :, :], T, axis=1)  # (B, T, H, W, C)
            else:
                B, T, H, W, C = x.shape

            residual = x  # (B, T, H, W, C)

            if self.block_norm == 'global_layer':
                x = nn.LayerNorm(reduction_axes=(1, 2, 3, 4), feature_axes=-1,
                                 name='norm1', **ln_kwargs)(x)
            elif self.block_norm == 'temporal_layer':
                x = nn.LayerNorm(reduction_axes=(1, 4), feature_axes=-1,
                                 name='norm1', **ln_kwargs)(x)
            else:  # 'layer' — per-frame, C only
                x = nn.LayerNorm(name='norm1', **ln_kwargs)(
                    x.reshape(B * T, H, W, C)).reshape(B, T, H, W, C)

        # --- Positional encoding on CSSM INPUT (DWConv, matches SHViT placement) ---
        # Applied before the CSSM so Q, K, V all receive position-aware features.
        # The CSSM's FFT mixing is translation-equivariant; this breaks that
        # equivariance with learned spatial priors, same role as SHViT's pos_conv
        # inside SingleHeadAttention.
        if self.use_pos_conv:
            if fast_path:
                pos = nn.Conv(
                    self.dim, kernel_size=(3, 3), padding='SAME',
                    feature_group_count=self.dim,
                    dtype=self.dtype, param_dtype=self.param_dtype,
                    name='pos_conv',
                )(x)
                x = x + pos
            else:
                # Apply per-frame (all T frames are identical for static images,
                # but this also works for real video input).
                B_pos = x.shape[0]
                T_pos = x.shape[1]
                x_flat = x.reshape(B_pos * T_pos, *x.shape[2:])
                pos = nn.Conv(
                    self.dim, kernel_size=(3, 3), padding='SAME',
                    feature_group_count=self.dim,
                    dtype=self.dtype, param_dtype=self.param_dtype,
                    name='pos_conv',
                )(x_flat)
                x = x + pos.reshape(x.shape)

        # --- CSSM ---
        if self.cssm_type == 'gdn':
            CSSM = GatedDeltaNetCSSM
            cssm_kwargs = dict(
                channels=self.dim,
                delta_key_dim=self.delta_key_dim,
                kernel_size=self.kernel_size,
                spectral_rho=self.spectral_rho,
                rope_mode=self.rope_mode,
                gate_type=self.gate_type,
                short_conv_size=self.short_conv_size,
                output_norm=self.output_norm,
                use_input_gates=self.use_input_gates,
                static_image_fast_path=self.static_image_fast_path,
                num_timesteps=self.num_timesteps,
                ssl_proj_dim=self.ssl_proj_dim,
                gate_proj_bias_init=self.gate_proj_bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        else:  # 'gated'
            CSSM = GatedCSSM
            cssm_kwargs = dict(
                channels=self.dim,
                dense_mixing=self.dense_mixing,
                gate_activation=self.gate_activation,
                kernel_size=self.kernel_size,
                spectral_rho=self.spectral_rho,
                rope_mode=self.rope_mode,
                short_conv_size=self.short_conv_size,
                short_conv_spatial_size=self.short_conv_spatial_size,
                block_size=self.block_size,
                mixing_rank=self.mixing_rank,
                gate_type=self.gate_type,
            )

        x = CSSM(**cssm_kwargs, name='cssm')(x)  # (B, T, H, W, C) or (B, H, W, C) in fast path

        # Decide output dimensionality: if we were handed real video (5D) and
        # the user didn't request T collapse via timestep_gate/fast_path, keep
        # T through the rest of the block. Every downstream CSSM block then
        # sees genuine temporal structure instead of a T-replicated single
        # frame (the old behavior of x_out = x[:, -1]).
        keep_time = (not fast_path) and (not self.timestep_gate) and (not is_image)

        if fast_path:
            x_out = x  # (B, H, W, C)
        elif self.timestep_gate:
            T_dim = x.shape[1]
            q = residual[:, 0].mean(axis=(1, 2))          # (B, C)
            k = x.mean(axis=(2, 3))                        # (B, T, C)
            q_t = jnp.broadcast_to(q[:, None, :], k.shape)
            gate_in = jnp.concatenate([q_t, k], axis=-1)
            hidden_dim = max(self.dim // 4, 8)
            h = nn.Dense(hidden_dim, dtype=self.dtype, param_dtype=self.param_dtype,
                         name='gate_fc1')(gate_in)
            h = nn.gelu(h)
            logits_t = nn.Dense(1, dtype=self.dtype, param_dtype=self.param_dtype,
                                name='gate_fc2')(h).squeeze(-1)
            prior = jnp.zeros(T_dim, dtype=self.dtype).at[-1].set(jnp.asarray(4.0, self.dtype))
            w = jax.nn.softmax(logits_t + prior[None, :], axis=-1)
            x_out = (x * w[:, :, None, None, None]).sum(axis=1)  # (B, H, W, C)
        elif keep_time:
            x_out = x  # (B, T, H, W, C) — preserve temporal dim through the block
        else:
            x_out = x[:, -1]  # (B, H, W, C)

        if self.output_act == 'gelu':
            x_out = jax.nn.gelu(x_out)
        elif self.output_act == 'silu':
            x_out = jax.nn.silu(x_out)

        # --- Residual: match x_out's dimensionality ---
        if fast_path:
            res_match = residual                # 4D
        elif keep_time:
            res_match = residual                # 5D, same shape as x_out
        else:
            res_match = residual[:, -1]         # 4D collapsed
        x_out = DropPath(self.drop_path)(x_out, deterministic)
        x_out = res_match + x_out

        # --- MLP path ---
        residual = x_out
        if keep_time:
            # Fold T into batch so LN + MLP run per-frame, then unfold. The
            # MLP weights are the same as the 4D path — (B*T, H, W, C) just
            # treats every frame as an independent sample.
            B_, T_, H_, W_, C_ = x_out.shape
            x_out_4d = x_out.reshape(B_ * T_, H_, W_, C_)
            x_out_4d = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype,
                                    name='norm2')(x_out_4d)
            x_out_4d = Mlp(
                hidden_dim=int(self.dim * self.mlp_ratio),
                out_dim=self.dim,
                use_dwconv=self.use_dwconv,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name='mlp'
            )(x_out_4d, deterministic)
            x_out_4d = DropPath(self.drop_path)(x_out_4d, deterministic)
            x_out = residual + x_out_4d.reshape(B_, T_, H_, W_, C_)
            return x_out  # (B, T, H, W, C)

        x_out = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype,
                             name='norm2')(x_out)
        x_out = Mlp(
            hidden_dim=int(self.dim * self.mlp_ratio),
            out_dim=self.dim,
            use_dwconv=self.use_dwconv,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='mlp'
        )(x_out, deterministic)
        x_out = DropPath(self.drop_path)(x_out, deterministic)
        x_out = residual + x_out

        return x_out  # (B, H, W, C)


class CSSMSHViT(nn.Module):
    """
    CSSM-SHViT: SHViT with CSSM in attention stages.

    Hierarchical 4-stage architecture:
    - ConvBlocks in stages 0-1 (efficient local processing)
    - CSSM blocks in stages 2-3 (temporal recurrence)

    With optional VideoRoPE-style spatiotemporal position encoding.
    Reference: https://arxiv.org/abs/2502.05173
    """
    num_classes: int = 1000
    embed_dims: Sequence[int] = (128, 256, 384, 512)
    depths: Sequence[int] = (1, 2, 4, 1)
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_cssm_stages: Sequence[bool] = (False, False, True, True)
    cssm_type: str = 'gated'
    dense_mixing: bool = False
    block_size: int = 32  # Block size for LMME channel mixing (only with gated + dense_mixing)
    mixing_rank: int = 0  # If > 0, use low-rank mixing (recommended: 4-16)
    gate_activation: str = 'softplus'  # 'softplus' for gated, 'sigmoid' for opponent
    num_timesteps: int = 8
    kernel_sizes: Sequence[int] = (15, 15, 5, 3)  # Kernel size per stage (stages 2,3 use CSSM)
    spectral_rho: float = 0.999  # Maximum spectral magnitude for stability
    rope_mode: str = 'none'  # 'spatiotemporal', 'temporal', or 'none'
    use_dwconv: bool = True  # DWConv in MLP (matches SHViT, adds params)
    output_act: str = 'none'  # Output activation after CSSM: 'gelu', 'silu', or 'none'
    short_conv_size: int = 4       # Temporal causal conv kernel size (0=disabled)
    short_conv_spatial_size: int = 3  # Spatial depthwise conv kernel size (0=disabled)
    # GDN-specific
    delta_key_dim: int = 2
    output_norm: str = 'rms'
    gate_type: str = 'factored'
    # Block norm
    block_norm: str = 'global_layer'
    use_pos_conv: bool = True        # Matches tog9av97 75% run. Helps under the default
                                     # 5-epoch warmup; can hurt under shorter warmup.
    gate_proj_bias_init: float = 0.0 # Bisect 2026-04-16: 1.0 regresses epoch-1 val by ~2× on ImageNet
    head_pool: str = 'max'           # 'max' (tog9av97 75% run) or 'mean'. Silent change to mean
                                     # post-tog9av97 cost ~3-5% ep1 val.
    # Static-image fast path (ImageNet speedup)
    static_image_fast_path: bool = False
    use_input_gates: bool = True  # B_k/B_v spatial input gates in GDN CSSM
    # Temporal SSL: per-block sown projection + stop-grad linear probe head
    ssl_proj_dim: int = 0       # 0=disabled. Forwarded to every CSSMSHViTBlock.
    linear_probe: bool = False  # When True, returns (logits, probe_logits) instead of logits.
    # Timestep gate: softmax pooling over CSSM timesteps, replacing x[:, -1].
    timestep_gate: bool = False
    # Mixed precision
    dtype: jnp.dtype = jnp.float32       # compute dtype (bf16 for mixed precision)
    param_dtype: jnp.dtype = jnp.float32 # weight storage dtype (keep fp32)

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        deterministic = not training

        # Handle video input - extract T and fold into batch so every early
        # (Conv) stage processes ALL T frames. Later, before the first CSSM
        # block, we reshape back to (B, T, H, W, C) so the CSSM block receives
        # genuine temporal structure. Previously we did ``x = x[:, -1]`` here,
        # which threw away T-1 frames — fine for static ImageNet (T was
        # synthetic) but catastrophic for real video training.
        is_video = (x.ndim == 5)
        if is_video:
            B_orig = x.shape[0]
            num_timesteps = x.shape[1]
            x = x.reshape(B_orig * num_timesteps, *x.shape[2:])  # (B*T, H, W, C)
        else:
            B_orig = x.shape[0]
            num_timesteps = self.num_timesteps

        B = x.shape[0]
        first_cssm_seen = False

        # Patch embedding (4x downsample) — also casts input from fp32 to self.dtype.
        x = PatchEmbed(self.embed_dims[0],
                       dtype=self.dtype, param_dtype=self.param_dtype,
                       name='patch_embed')(x, deterministic)

        # Stochastic depth
        total_depth = sum(self.depths)
        dp_rates = np.linspace(0, self.drop_path_rate, total_depth)
        dp_idx = 0

        # 4 stages
        for stage_idx in range(4):
            # Downsample between stages. Downsample is a 4D op (Conv stride=2);
            # if we're currently in 5D video mode, fold T into batch for it.
            if stage_idx > 0:
                if x.ndim == 5:
                    B_, T_, H_, W_, C_ = x.shape
                    x_4d = x.reshape(B_ * T_, H_, W_, C_)
                    x_4d = Downsample(self.embed_dims[stage_idx],
                                      dtype=self.dtype, param_dtype=self.param_dtype,
                                      name=f'downsample{stage_idx}')(x_4d, deterministic)
                    H2, W2, C2 = x_4d.shape[1], x_4d.shape[2], x_4d.shape[3]
                    x = x_4d.reshape(B_, T_, H2, W2, C2)
                else:
                    x = Downsample(self.embed_dims[stage_idx],
                                   dtype=self.dtype, param_dtype=self.param_dtype,
                                   name=f'downsample{stage_idx}')(x, deterministic)

            # Blocks
            for block_idx in range(self.depths[stage_idx]):
                if self.use_cssm_stages[stage_idx]:
                    # Before the FIRST CSSM block, unfold B*T -> (B, T, H, W, C)
                    # so the CSSM and every downstream CSSM block can see
                    # genuine temporal structure (the block now preserves T
                    # through its MLP path — see CSSMSHViTBlock).
                    if is_video and not first_cssm_seen:
                        H_cur, W_cur, C_cur = x.shape[1], x.shape[2], x.shape[3]
                        x = x.reshape(B_orig, num_timesteps, H_cur, W_cur, C_cur)
                        first_cssm_seen = True
                    x = CSSMSHViTBlock(
                        dim=self.embed_dims[stage_idx],
                        mlp_ratio=self.mlp_ratio,
                        drop_path=float(dp_rates[dp_idx]),
                        cssm_type=self.cssm_type,
                        dense_mixing=self.dense_mixing,
                        block_size=self.block_size,
                        mixing_rank=self.mixing_rank,
                        gate_activation=self.gate_activation,
                        num_timesteps=num_timesteps,  # Use extracted T for variable timesteps
                        kernel_size=self.kernel_sizes[stage_idx],
                        spectral_rho=self.spectral_rho,
                        rope_mode=self.rope_mode,
                        use_dwconv=self.use_dwconv,
                        output_act=self.output_act,
                        short_conv_size=self.short_conv_size,
                        short_conv_spatial_size=self.short_conv_spatial_size,
                        delta_key_dim=self.delta_key_dim,
                        output_norm=self.output_norm,
                        gate_type=self.gate_type,
                        block_norm=self.block_norm,
                        use_pos_conv=self.use_pos_conv,
                        gate_proj_bias_init=self.gate_proj_bias_init,
                        static_image_fast_path=self.static_image_fast_path,
                        use_input_gates=self.use_input_gates,
                        ssl_proj_dim=self.ssl_proj_dim,
                        timestep_gate=self.timestep_gate,
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                        name=f'stage{stage_idx}_block{block_idx}'
                    )(x, deterministic)
                else:
                    x = ConvBlock(
                        dim=self.embed_dims[stage_idx],
                        mlp_ratio=self.mlp_ratio,
                        drop_path=float(dp_rates[dp_idx]),
                        dtype=self.dtype,
                        param_dtype=self.param_dtype,
                        name=f'stage{stage_idx}_block{block_idx}'
                    )(x, deterministic)
                dp_idx += 1

        # Final norm
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype,
                         name='norm')(x)

        # Global pooling. For static 4D: pool over (H, W). For video 5D: pool
        # over (T, H, W) — the classifier sees a single spatiotemporal summary.
        # Default 'max' matches tog9av97 75% run; 'mean' matches SHViT baseline.
        if x.ndim == 5:
            pool_axes = (1, 2, 3)
        else:
            pool_axes = (1, 2)
        if self.head_pool == 'max':
            pooled = jnp.max(x, axis=pool_axes).astype(jnp.float32)
        else:
            pooled = jnp.mean(x, axis=pool_axes).astype(jnp.float32)
        # (B, C)

        # Sow pooled features so linear-probe eval scripts can read them.
        self.sow('intermediates', 'pooled_features', pooled)

        # Classification head (fp32)
        head_in = pooled
        if not deterministic and self.drop_rate > 0:
            head_in = nn.Dropout(self.drop_rate)(head_in, deterministic=False)
        logits = nn.Dense(self.num_classes, param_dtype=self.param_dtype,
                          name='head')(head_in)

        if self.linear_probe:
            # Stop-grad ensures the probe loss never updates the backbone.
            probe_in = jax.lax.stop_gradient(pooled)
            probe_logits = nn.Dense(self.num_classes, param_dtype=self.param_dtype,
                                    name='linear_probe_head')(probe_in)
            return logits, probe_logits

        return logits


# Factory functions

def cssm_shvit_s1(num_classes: int = 1000, num_timesteps: int = 8, cssm_type: str = 'gated', **kwargs):
    """CSSM-SHViT-S1 (~6M params + CSSM overhead)."""
    return CSSMSHViT(
        num_classes=num_classes,
        embed_dims=(64, 128, 256, 384),
        depths=(1, 2, 2, 1),
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )


def cssm_shvit_s2(num_classes: int = 1000, num_timesteps: int = 8, cssm_type: str = 'gated', **kwargs):
    """CSSM-SHViT-S2 (~11M params + CSSM overhead)."""
    return CSSMSHViT(
        num_classes=num_classes,
        embed_dims=(96, 192, 320, 448),
        depths=(1, 2, 3, 1),
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )


def cssm_shvit_s3(num_classes: int = 1000, num_timesteps: int = 8, cssm_type: str = 'gated', **kwargs):
    """CSSM-SHViT-S3 (~16M params + CSSM overhead)."""
    return CSSMSHViT(
        num_classes=num_classes,
        embed_dims=(112, 224, 352, 480),
        depths=(1, 2, 4, 1),
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )


def cssm_shvit_s4(num_classes: int = 1000, num_timesteps: int = 8, cssm_type: str = 'gated', **kwargs):
    """CSSM-SHViT-S4 (~22M params + CSSM overhead)."""
    return CSSMSHViT(
        num_classes=num_classes,
        embed_dims=(128, 256, 384, 512),
        depths=(1, 2, 4, 1),
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )
