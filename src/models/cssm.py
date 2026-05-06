"""
CSSM (Cepstral State Space Model) layers.

Implements both Standard and Gated Opponent CSSM layers with
options for Dense (multi-head) vs Depthwise spatial mixing.

Uses GOOM (Generalized Order of Magnitude) primitives for
numerically stable log-space computation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from .math import (cssm_scalar_scan_op, cssm_matrix_scan_op, cssm_3x3_matrix_scan_op,
                    cssm_general_matrix_scan_op,
                    make_block_scan_op,
                    complex64_to_linear_split, linear_split_to_complex64,
                    linear_split_scalar_scan_op, linear_split_2x2_scan_op,
                    linear_split_3x3_scan_op, linear_split_general_scan_op,
                    direct_scalar_scan_op, direct_2x2_scan_op,
                    direct_3x3_scan_op, direct_general_scan_op,
                    ssd_scan,
                    quadratic_scan, chunked_quadratic_scan)
from .goom import to_goom, from_goom


def _stable_spectral_magnitude(K: jnp.ndarray, rho: float = 0.999) -> jnp.ndarray:
    """
    Squash complex spectral magnitude to guarantee |K| < 1 for dynamic stability.

    Uses the formula K * rho / (1 + |K|) which:
    - Preserves phase exactly
    - Maps any magnitude to < rho (always < 1)
    - Is smooth and differentiable everywhere

    Args:
        K: Complex spectral coefficients
        rho: Maximum output magnitude (should be < 1 for stability)

    Returns:
        Squashed spectral coefficients with |output| < rho
    """
    K_mag = jnp.abs(K)
    return K * rho / (1.0 + K_mag)


def _rms_norm(x, weight, eps=1e-6):
    """RMSNorm: x / sqrt(mean(x²) + eps) * weight. No mean subtraction."""
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps) * weight


def log_add_exp(log_a: jnp.ndarray, log_b: jnp.ndarray) -> jnp.ndarray:
    """
    Numerically stable log(exp(log_a) + exp(log_b)) for log-space addition.

    Handles both real and complex (GOOM) log-space representations.
    For complex inputs, operates on the real part for stability while preserving phase.

    Args:
        log_a: First log-space value (real or complex)
        log_b: Second log-space value (real or complex)

    Returns:
        log(exp(log_a) + exp(log_b)) in the same dtype as inputs
    """
    is_complex = jnp.iscomplexobj(log_a) or jnp.iscomplexobj(log_b)
    if is_complex:
        # For GOOM: real part is log-magnitude, imag is phase
        # We do logsumexp on magnitudes, then average phases weighted by magnitudes
        log_mag_a = log_a.real if jnp.iscomplexobj(log_a) else log_a
        log_mag_b = log_b.real if jnp.iscomplexobj(log_b) else log_b
        phase_a = log_a.imag if jnp.iscomplexobj(log_a) else jnp.zeros_like(log_a)
        phase_b = log_b.imag if jnp.iscomplexobj(log_b) else jnp.zeros_like(log_b)

        # logsumexp for magnitudes
        max_log = jnp.maximum(log_mag_a, log_mag_b)
        log_sum_mag = max_log + jnp.log(jnp.exp(log_mag_a - max_log) + jnp.exp(log_mag_b - max_log))

        # Weighted average of phases (weights are exp(log_mag))
        weight_a = jnp.exp(log_mag_a - log_sum_mag)
        weight_b = jnp.exp(log_mag_b - log_sum_mag)
        phase_sum = weight_a * phase_a + weight_b * phase_b

        return log_sum_mag + 1j * phase_sum
    else:
        # Standard logsumexp for real values
        max_log = jnp.maximum(log_a, log_b)
        return max_log + jnp.log(jnp.exp(log_a - max_log) + jnp.exp(log_b - max_log))


def apply_rope(x: jnp.ndarray, mode: str = 'spatiotemporal', base: float = 10000.0) -> jnp.ndarray:
    """
    Apply relative position encoding via rotations (RoPE).

    Following VideoRoPE (Wei et al., ICML 2025):
    - Low-frequency allocation for temporal (mitigates periodic oscillations)
    - High-frequency allocation for spatial (fine-grained details)
    - Diagonal layout for spatial symmetry

    Reference: https://arxiv.org/abs/2502.05173

    Args:
        x: Input tensor (B, T, H, W, C) - in spatial domain before FFT
        mode: Position encoding mode:
            - 'spatiotemporal': Combined H, W, T encoding (VideoRoPE style, additive)
            - 'mrope': Multimodal RoPE (Qwen3.5 style, dedicated dims per axis, interleaved)
            - 'spatial_only': Only H, W encoding, no temporal (better length generalization)
            - 'temporal': Only T encoding
            - 'none': No position encoding
        base: Base for frequency computation (default 10000.0)

    Returns:
        Position-encoded tensor (B, T, H, W, C)
    """
    if mode == 'none':
        return x

    B, T, H, W, C = x.shape

    # Frequency bands per channel pair (standard RoPE formula)
    # Lower index = lower frequency (slower rotation)
    dim_indices = jnp.arange(0, C, 2)
    inv_freq = 1.0 / (base ** (dim_indices / C))  # (C/2,) - decreasing frequencies

    n_freq = len(inv_freq)

    if mode == 'temporal':
        # Temporal-only: rotate by timestep t
        t_pos = jnp.arange(T)  # (T,)
        theta = jnp.outer(t_pos, inv_freq)  # (T, C/2)
        theta = theta[:, None, None, :]  # (T, 1, 1, C/2) for broadcast

    elif mode == 'spatial_only':
        # Spatial-only: rotate by (h, w) position, no temporal encoding
        # This allows better length generalization since recurrence encodes time
        # Split frequencies between h and w (interleaved for diagonal layout)
        inv_freq_h = inv_freq[0::2]
        inv_freq_w = inv_freq[1::2]

        h_pos = jnp.arange(H)
        w_pos = jnp.arange(W)

        theta_h = jnp.outer(h_pos, inv_freq_h)  # (H, n_freq/2)
        theta_w = jnp.outer(w_pos, inv_freq_w)  # (W, n_freq/2)

        # Pad to C/2
        def pad_right(arr, target_len):
            curr_len = arr.shape[-1]
            if curr_len < target_len:
                pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_len - curr_len)]
                return jnp.pad(arr, pad_width)
            return arr

        theta_h_padded = pad_right(theta_h, n_freq)  # (H, C/2)
        theta_w_padded = pad_right(theta_w, n_freq)  # (W, C/2)

        # Combine: (H, W, C/2) - same for all timesteps (no temporal component)
        theta = (theta_h_padded[:, None, :] +
                 theta_w_padded[None, :, :])
        # Broadcast to (1, H, W, C/2) for proper broadcasting with x
        theta = theta[None, :, :, :]

    elif mode == 'mrope':
        # MRoPE (Multimodal RoPE) following Qwen3.5
        # Each channel pair rotates by exactly ONE axis (t, h, or w).
        # No additive combination. Interleaved layout: [t,h,w,t,h,w,...]

        n_t = n_freq // 3
        n_h = n_freq // 3
        n_w = n_freq - n_t - n_h  # remainder to width

        t_pos = jnp.arange(T)
        h_pos = jnp.arange(H)
        w_pos = jnp.arange(W)

        # Each axis uses its own slice of the frequency spectrum
        theta_t = jnp.outer(t_pos, inv_freq[:n_t])           # (T, n_t)
        theta_h = jnp.outer(h_pos, inv_freq[n_t:n_t+n_h])   # (H, n_h)
        theta_w = jnp.outer(w_pos, inv_freq[n_t+n_h:])       # (W, n_w)

        # Broadcast to (T, H, W, n_section)
        theta_t_full = jnp.broadcast_to(theta_t[:, None, None, :], (T, H, W, n_t))
        theta_h_full = jnp.broadcast_to(theta_h[None, :, None, :], (T, H, W, n_h))
        theta_w_full = jnp.broadcast_to(theta_w[None, None, :, :], (T, H, W, n_w))

        # Interleave: [t0,h0,w0, t1,h1,w1, ...]
        min_n = n_t  # n_t == n_h, n_w >= n_t
        interleaved = jnp.stack([
            theta_t_full,
            theta_h_full,
            theta_w_full[..., :min_n],
        ], axis=-1).reshape(T, H, W, min_n * 3)

        # Append remaining width dims if n_w > n_t
        if n_w > min_n:
            theta = jnp.concatenate([interleaved, theta_w_full[..., min_n:]], axis=-1)
        else:
            theta = interleaved

    else:  # spatiotemporal - following VideoRoPE
        # VideoRoPE insight: allocate LOW frequencies to temporal, HIGH to spatial
        # This prevents periodic oscillations in temporal dimension
        # Split: first 1/3 (low freq) -> temporal, remaining 2/3 (high freq) -> spatial

        n_temporal = n_freq // 3
        n_spatial = n_freq - n_temporal

        # Low frequencies for temporal (slow, smooth changes over time)
        inv_freq_t = inv_freq[:n_temporal]
        # High frequencies for spatial (fast, fine-grained position encoding)
        # Split remaining between h and w (diagonal layout for symmetry)
        inv_freq_h = inv_freq[n_temporal::2]  # Interleaved for diagonal layout
        inv_freq_w = inv_freq[n_temporal + 1::2]

        h_pos = jnp.arange(H)
        w_pos = jnp.arange(W)
        t_pos = jnp.arange(T)

        # Compute angles for each dimension
        theta_t = jnp.outer(t_pos, inv_freq_t)  # (T, n_temporal)
        theta_h = jnp.outer(h_pos, inv_freq_h)  # (H, n_spatial/2)
        theta_w = jnp.outer(w_pos, inv_freq_w)  # (W, n_spatial/2)

        # Broadcast and combine
        # Pad all to C/2 for final theta shape
        def pad_right(arr, target_len):
            curr_len = arr.shape[-1]
            if curr_len < target_len:
                pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_len - curr_len)]
                return jnp.pad(arr, pad_width)
            return arr

        theta_t_padded = pad_right(theta_t, n_freq)  # (T, C/2)
        theta_h_padded = pad_right(theta_h, n_freq)  # (H, C/2)
        theta_w_padded = pad_right(theta_w, n_freq)  # (W, C/2)

        # Combine: (T, H, W, C/2)
        # Each position (t, h, w) gets rotation from all three components
        theta = (theta_t_padded[:, None, None, :] +
                 theta_h_padded[None, :, None, :] +
                 theta_w_padded[None, None, :, :])

    # Apply rotation to channel pairs
    cos_theta = jnp.cos(theta)  # (T, H, W, C/2) or (T, 1, 1, C/2) or (1, H, W, C/2)
    sin_theta = jnp.sin(theta)

    # Split channels into even/odd pairs
    x_even = x[..., 0::2]  # (B, T, H, W, C/2)
    x_odd = x[..., 1::2]

    # Rotate: [cos -sin; sin cos] @ [even; odd]
    x_even_rot = x_even * cos_theta - x_odd * sin_theta
    x_odd_rot = x_even * sin_theta + x_odd * cos_theta

    # Interleave back to original channel order
    x_rot = jnp.stack([x_even_rot, x_odd_rot], axis=-1)  # (B, T, H, W, C/2, 2)
    x_rot = x_rot.reshape(B, T, H, W, C)

    return x_rot


def apply_learned_temporal_encoding(x: jnp.ndarray, temporal_embed: jnp.ndarray) -> jnp.ndarray:
    """
    Apply learned temporal position encoding (additive, not rotational).

    This is used with 'separate' position encoding mode where spatial uses RoPE
    and temporal uses learned embeddings that can interpolate for length generalization.

    Args:
        x: Input tensor (B, T, H, W, C)
        temporal_embed: Learned temporal embeddings (max_T, C)

    Returns:
        Position-encoded tensor (B, T, H, W, C)
    """
    B, T, H, W, C = x.shape
    max_T = temporal_embed.shape[0]

    if T <= max_T:
        # Use embeddings directly (or slice if T < max_T)
        t_embed = temporal_embed[:T]  # (T, C)
    else:
        # Interpolate for longer sequences
        # Use linear interpolation in embedding space
        indices = jnp.linspace(0, max_T - 1, T)
        lower_idx = jnp.floor(indices).astype(jnp.int32)
        upper_idx = jnp.minimum(lower_idx + 1, max_T - 1)
        alpha = indices - lower_idx

        lower_embed = temporal_embed[lower_idx]  # (T, C)
        upper_embed = temporal_embed[upper_idx]  # (T, C)
        t_embed = lower_embed * (1 - alpha[:, None]) + upper_embed * alpha[:, None]

    # Add temporal embedding: (T, C) -> (1, T, 1, 1, C)
    t_embed = t_embed[None, :, None, None, :]
    return x + t_embed


def get_sinusoidal_temporal_encoding(seq_len: int, dim: int, base: float = 10000.0) -> jnp.ndarray:
    """
    Generate sinusoidal temporal position encoding (like original Transformer).

    Unlike learned embeddings, sinusoidal encodings naturally extrapolate to
    longer sequences without interpolation artifacts.

    Args:
        seq_len: Number of timesteps
        dim: Embedding dimension
        base: Base for frequency computation (default 10000.0)

    Returns:
        Position encoding (seq_len, dim)
    """
    position = jnp.arange(seq_len)[:, None]  # (T, 1)
    div_term = jnp.exp(jnp.arange(0, dim, 2) * (-jnp.log(base) / dim))  # (dim/2,)

    # Compute sin for even indices, cos for odd indices
    pe_sin = jnp.sin(position * div_term)  # (T, dim/2)
    pe_cos = jnp.cos(position * div_term)  # (T, dim/2)

    # Interleave sin and cos: [sin0, cos0, sin1, cos1, ...]
    pe = jnp.stack([pe_sin, pe_cos], axis=-1).reshape(seq_len, -1)  # (T, dim)

    # Handle odd dim (trim if necessary)
    return pe[:, :dim]


def apply_sinusoidal_temporal_encoding(x: jnp.ndarray, base: float = 10000.0) -> jnp.ndarray:
    """
    Apply sinusoidal temporal position encoding (additive).

    This is used with 'sinusoidal' position encoding mode where spatial uses RoPE
    and temporal uses sinusoidal embeddings that naturally extrapolate to longer sequences.

    Args:
        x: Input tensor (B, T, H, W, C)
        base: Base for frequency computation

    Returns:
        Position-encoded tensor (B, T, H, W, C)
    """
    B, T, H, W, C = x.shape
    t_embed = get_sinusoidal_temporal_encoding(T, C, base)  # (T, C)

    # Add temporal embedding: (T, C) -> (1, T, 1, 1, C)
    t_embed = t_embed[None, :, None, None, :]
    return x + t_embed


def apply_temporal_rope_to_context(ctx: jnp.ndarray, base: float = 10000.0) -> jnp.ndarray:
    """
    Apply RoPE-style temporal position encoding to gate context.

    This allows gates to vary with timestep even when input is static (repeated image).
    Uses sinusoidal encoding like RoPE but adds rather than rotates (simpler for context).

    Args:
        ctx: Gate context tensor (B, T, C) - spatially pooled features
        base: Base for frequency computation

    Returns:
        Context with temporal position encoding (B, T, C)
    """
    B, T, C = ctx.shape

    # Create temporal position encoding (sinusoidal like RoPE)
    t_pos = jnp.arange(T)  # (T,)
    dim_indices = jnp.arange(0, C, 2)
    inv_freq = 1.0 / (base ** (dim_indices / C))  # (C/2,) - decreasing frequencies

    # Compute sinusoidal encoding
    theta = jnp.outer(t_pos, inv_freq)  # (T, C/2)
    sin_enc = jnp.sin(theta)  # (T, C/2)
    cos_enc = jnp.cos(theta)  # (T, C/2)

    # Interleave sin and cos to get (T, C)
    t_embed = jnp.stack([sin_enc, cos_enc], axis=-1).reshape(T, C)

    # Add to context (broadcast over batch)
    return ctx + t_embed[None, :, :]
class GatedCSSM(nn.Module):
    """
    Spectral Mamba — Mamba (S6) with spectral spatial convolution.

    Faithful adaptation of Mamba to 2D spatial + temporal data:
    - h_t = A_bar * h_{t-1} + B_bar * u_t    (state update, per freq bin)
    - y_t = C * h_t                           (output projection)

    Differences from standard Mamba:
    - State dim N=1 (spatial frequency bins provide per-channel expressivity)
    - Separable spatiotemporal short conv (spatial depthwise + temporal causal)
    - Spatial kernel operates in spectral domain via FFT

    Architecture (matches Mamba):
    1. x → Linear(2C) → (x_main, z)
    2. x_main → factored short conv (spatial dw + temporal causal) → SiLU
    3. x_main → spectral SSM (Δ, B, C from post-conv features, unconstrained B/C)
    4. output = SSM_out * SiLU(z) → Linear(C)

    Attributes:
        channels: Number of input/output channels
        kernel_size: Spatial kernel size for spectral convolution
        spectral_rho: Maximum spectral magnitude for stability (should be < 1)
        gate_activation: Activation for Δ decay gate ('softplus', 'sigmoid', 'exp')
        rope_mode: Position encoding mode ('spatiotemporal', 'temporal', 'none')
        rope_base: Base for RoPE frequency computation
        short_conv_size: Temporal causal conv kernel size (0=disabled, default 4)
        short_conv_spatial_size: Spatial depthwise conv kernel size (0=disabled, default 3)
        dense_mixing: If True, use channel mixing (LMME or low-rank)
        block_size: Size of channel blocks for LMME channel mixing
        mixing_rank: If > 0, use low-rank channel mixing instead of full LMME
    """
    channels: int
    dense_mixing: bool = False
    block_size: int = 32  # Block size for LMME channel mixing
    mixing_rank: int = 0  # If > 0, use low-rank mixing (recommended: 4-16)
    kernel_size: int = 15
    spectral_rho: float = 0.999
    gate_activation: str = 'softplus'
    rope_mode: str = 'none'  # 'spatiotemporal', 'temporal', or 'none'
    gate_rank: int = 0  # Unused, for API compatibility with GatedOpponentCSSM
    rope_base: float = 10000.0
    concat_xy: bool = True  # Unused, for API compatibility
    position_independent_gates: bool = False  # If True, compute gates from raw input (before pos encoding)
    short_conv_size: int = 4       # Temporal causal conv kernel size (0=disabled)
    short_conv_spatial_size: int = 3  # Spatial depthwise conv kernel size (0=disabled)
    gate_type: str = 'dense'  # 'dense' (Dense(C→H*W_freq)) or 'factored' (Dense(C→H) × Dense(C→W_freq))
    scan_mode: str = 'associative' # only for CUDA kernel activation for now

    def _compute_gate(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute input-dependent gate with specified activation."""
        if self.gate_activation == 'softplus':
            return nn.softplus(x)
        elif self.gate_activation == 'sigmoid':
            return nn.sigmoid(x) * 2.0  # Scale to [0, 2]
        elif self.gate_activation == 'exp':
            return jnp.exp(jnp.clip(x, -4, 2))  # Bounded exp
        else:
            return nn.softplus(x)

    @nn.compact
    def __call__(self, x: jnp.ndarray, injected_qkv_spatial=None) -> jnp.ndarray:
        """
        Forward pass with full Mamba-style gating in log-spectral domain.

        Args:
            x: Input tensor of shape (B, T, H, W, C)
            injected_qkv_spatial: Unused, for API compatibility.

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1  # rfft2 output width

        # Save raw input for position-independent gates
        x_raw = x if self.position_independent_gates else None

        # === Apply RoPE before FFT (VideoRoPE style) ===
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # === Channel Mixing Modes ===
        if self.dense_mixing:
            if self.mixing_rank > 0:
                # Low-rank channel mixing (efficient)
                return self._forward_lowrank_mixing(x, B, T, H, W, C, W_freq, x_raw=x_raw)
            else:
                # Full LMME (expensive, original implementation)
                return self._forward_dense_mixing(x, B, T, H, W, C, W_freq, x_raw=x_raw)
        else:
            return self._forward_depthwise(x, B, T, H, W, C, W_freq, x_raw=x_raw)

    def _forward_depthwise(self, x, B, T, H, W, C, W_freq, x_raw=None):
        """Depthwise forward pass — Mamba architecture with spectral spatial kernel."""

        # --- 0. Input expansion: x → (x_main, z) [Mamba-style] ---
        x_flat = x.reshape(B * T, H, W, C)
        xz = nn.Dense(2 * C, name='in_proj')(x_flat)  # (B*T, H, W, 2C)
        xz = xz.reshape(B, T, H, W, 2 * C)
        x_main = xz[..., :C]   # (B, T, H, W, C) — goes through conv + SSM
        z = xz[..., C:]         # (B, T, H, W, C) — SiLU gating branch

        # --- 1. Factored short conv (spatial dw + temporal causal) → SiLU ---
        if self.short_conv_spatial_size > 0:
            x_sp = x_main.reshape(B * T, H, W, C)
            x_sp = nn.Conv(C, kernel_size=(self.short_conv_spatial_size, self.short_conv_spatial_size),
                           feature_group_count=C, padding='SAME',
                           name='short_conv_spatial')(x_sp)
            x_main = x_sp.reshape(B, T, H, W, C)

        if self.short_conv_size > 0:
            x_1d = x_main.transpose(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
            x_1d = jnp.pad(x_1d, ((0, 0), (self.short_conv_size - 1, 0), (0, 0)))
            x_1d = nn.Conv(C, kernel_size=(self.short_conv_size,),
                           feature_group_count=C, padding='VALID',
                           name='short_conv_temporal')(x_1d)
            x_main = x_1d.reshape(B, H, W, T, C).transpose(0, 3, 1, 2, 4)

        x_main = jax.nn.silu(x_main)

        # --- 2. Kernel Generation (depthwise) ---
        k_spatial = self.param(
            'kernel',
            nn.initializers.normal(0.02),
            (C, self.kernel_size, self.kernel_size)
        )

        # --- 3. Spectral Transform of Kernel ---
        pad_h = max(0, (H - self.kernel_size) // 2)
        pad_w = max(0, (W - self.kernel_size) // 2)
        pad_h_after = H - self.kernel_size - pad_h
        pad_w_after = W - self.kernel_size - pad_w

        k_padded = jnp.pad(
            k_spatial,
            ((0, 0), (pad_h, max(0, pad_h_after)), (pad_w, max(0, pad_w_after))),
            mode='constant'
        )

        if self.kernel_size > H or self.kernel_size > W:
            start_h = (self.kernel_size - H) // 2
            start_w = (self.kernel_size - W) // 2
            k_padded = k_spatial[:, start_h:start_h+H, start_w:start_w+W]

        K_hat_raw = jnp.fft.rfft2(k_padded, axes=(1, 2))  # (C, H, W_freq)
        K_hat = _stable_spectral_magnitude(K_hat_raw, rho=self.spectral_rho)

        # --- 4. FFT Input (post-conv features) ---
        x_perm = x_main.transpose(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C, H, W_freq)

        # --- 5. Input-Dependent Gates (from post-conv features, like Mamba) ---
        ctx = x_main.mean(axis=(2, 3))  # (B, T, C)

        def _make_gate(ctx, name, H, W_freq):
            """Create per-frequency gate, dense or factored."""
            if self.gate_type == 'factored':
                g_h = nn.Dense(H, name=f'{name}_h')(ctx)      # (B, T, H)
                g_w = nn.Dense(W_freq, name=f'{name}_w')(ctx)  # (B, T, W_freq)
                return (g_h[..., :, None] * g_w[..., None, :]).reshape(B, T, 1, H, W_freq)
            else:  # 'dense'
                return nn.Dense(H * W_freq, name=name)(ctx).reshape(B, T, 1, H, W_freq)

        # Δ — per-frequency decay gate (softplus, same as Mamba)
        delta_raw = _make_gate(ctx, 'delta_proj', H, W_freq).reshape(B, T, H * W_freq)
        delta_freq = self._compute_gate(delta_raw)
        delta_freq = delta_freq.reshape(B, T, 1, H, W_freq)

        # B — unconstrained linear projection (like Mamba)
        B_proj = _make_gate(ctx, 'B_proj', H, W_freq)

        # C — unconstrained linear projection (like Mamba)
        C_proj = _make_gate(ctx, 'C_proj', H, W_freq)

        # --- 6-8. Scan (dispatch by scan_mode) ---
        if self.scan_mode == 'cuda':
            # Linear-space CUDA parallel scan (bypass GOOM)
            # Equivalent transforms:
            #   GOOM: K_log.real - delta  =>  linear: K_hat * exp(-delta)
            #   GOOM: U_log.real + log(delta)  =>  linear: U_mod * delta
            from .cuda_scan import cuda_complex_scan

            K_hat_broadcast = jnp.broadcast_to(
                K_hat[None, None, ...], U_hat.shape)
            A_t = (K_hat_broadcast * jnp.exp(-delta_freq)).astype(jnp.complex64)

            U_modulated = U_hat * B_proj
            U_t = (U_modulated * delta_freq).astype(jnp.complex64)

            X_hat = cuda_complex_scan(A_t, U_t)
        else:
            # --- 6. Convert to GOOM (log-space) ---
            K_log = to_goom(K_hat)
            K_log_broadcast = jnp.broadcast_to(K_log[None, None, ...], U_hat.shape)

            U_modulated = U_hat * B_proj
            U_log = to_goom(U_modulated)

            # --- 7. Apply Per-Frequency Δ Gating in Log-Space ---
            K_log_gated = (K_log_broadcast.real - delta_freq) + 1j * K_log_broadcast.imag
            U_log_gated = (U_log.real + jnp.log(delta_freq + 1e-8)) + 1j * U_log.imag

            # --- 8. Associative Scan (depthwise) ---
            _, X_log = jax.lax.associative_scan(
                cssm_scalar_scan_op, (K_log_gated, U_log_gated), axis=1
            )

            X_hat = from_goom(X_log)

        # --- 9. Apply C projection and Inverse Transform ---
        X_hat_modulated = X_hat * C_proj
        ssm_out = jnp.fft.irfft2(X_hat_modulated, s=(H, W), axes=(3, 4))
        ssm_out = ssm_out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)

        # --- 10. SiLU output gating (Mamba-style: y * SiLU(z)) ---
        output = ssm_out * jax.nn.silu(z)

        # --- 11. Output projection ---
        output = nn.Dense(C, name='out_proj')(output.reshape(B * T, H, W, C))
        return output.reshape(B, T, H, W, C)

    def _forward_lowrank_mixing(self, x, B, T, H, W, C, W_freq, x_raw=None):
        """
        Low-rank channel mixing: depthwise scan + low-rank projection.

        Much more efficient than full LMME:
        - Full LMME: O(C × block_size² × kernel_size²) params
        - Low-rank:  O(C × kernel_size² + 2 × C × rank) params

        Approach:
        1. Depthwise spatial-temporal scan (same as _forward_depthwise)
        2. Low-rank channel mixing after scan: out = h + V_up @ (V_down @ h)
        """
        rank = self.mixing_rank

        # --- 1. Depthwise Kernel (spatial only) ---
        k_spatial = self.param(
            'kernel',
            nn.initializers.normal(0.02),
            (C, self.kernel_size, self.kernel_size)
        )

        # --- 2. Low-rank channel mixing matrices ---
        # V_down projects C -> rank, V_up projects rank -> C
        # Use separate real/imag parts to properly handle complex frequency domain
        V_down_real = self.param(
            'V_down_real',
            nn.initializers.normal(0.02 / jnp.sqrt(rank)),
            (C, rank)
        )
        V_down_imag = self.param(
            'V_down_imag',
            nn.initializers.normal(0.02 / jnp.sqrt(rank)),
            (C, rank)
        )
        V_up_real = self.param(
            'V_up_real',
            nn.initializers.normal(0.02 / jnp.sqrt(rank)),
            (rank, C)
        )
        V_up_imag = self.param(
            'V_up_imag',
            nn.initializers.normal(0.02 / jnp.sqrt(rank)),
            (rank, C)
        )
        # Combine into complex matrices
        V_down = V_down_real + 1j * V_down_imag
        V_up = V_up_real + 1j * V_up_imag

        # Learnable mixing strength (start small for stability)
        mix_scale = self.param(
            'mix_scale',
            nn.initializers.constant(0.1),
            (1,)
        )

        # --- 3. Spectral Transform of Kernel ---
        pad_h = max(0, (H - self.kernel_size) // 2)
        pad_w = max(0, (W - self.kernel_size) // 2)
        pad_h_after = H - self.kernel_size - pad_h
        pad_w_after = W - self.kernel_size - pad_w

        k_padded = jnp.pad(
            k_spatial,
            ((0, 0), (pad_h, max(0, pad_h_after)), (pad_w, max(0, pad_w_after))),
            mode='constant'
        )

        if self.kernel_size > H or self.kernel_size > W:
            start_h = (self.kernel_size - H) // 2
            start_w = (self.kernel_size - W) // 2
            k_padded = k_spatial[:, start_h:start_h+H, start_w:start_w+W]

        K_hat_raw = jnp.fft.rfft2(k_padded, axes=(1, 2))  # (C, H, W_freq)
        K_hat = _stable_spectral_magnitude(K_hat_raw, rho=self.spectral_rho)

        # --- 4. FFT Input ---
        x_perm = x.transpose(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C, H, W_freq)

        # --- 5. Input-Dependent Gates ---
        # Use raw input (before position encoding) for gates if position_independent_gates is True
        gate_input = x_raw if x_raw is not None else x
        ctx = gate_input.mean(axis=(2, 3))  # (B, T, C)

        delta_raw = nn.Dense(H * W_freq, name='delta_gate')(ctx)
        delta_raw = delta_raw.reshape(B, T, H * W_freq)
        delta_freq = self._compute_gate(delta_raw)
        delta_freq = delta_freq.reshape(B, T, 1, H, W_freq)

        B_gate_raw = nn.Dense(H * W_freq, name='B_gate')(ctx)
        B_gate = nn.sigmoid(B_gate_raw).reshape(B, T, 1, H, W_freq)

        C_gate_raw = nn.Dense(H * W_freq, name='C_gate')(ctx)
        C_gate = nn.sigmoid(C_gate_raw).reshape(B, T, 1, H, W_freq)

        if self.scan_mode == 'cuda':
            from .cuda_scan import cuda_complex_scan

            K_hat_broadcast = jnp.broadcast_to(
                K_hat[None, None, ...], U_hat.shape)
            A_t = (K_hat_broadcast * jnp.exp(-delta_freq)).astype(jnp.complex64)

            U_modulated = U_hat * B_gate
            U_t = (U_modulated * delta_freq).astype(jnp.complex64)

            X_hat = cuda_complex_scan(A_t, U_t)
        else:
            # --- 6. Convert to GOOM (log-space) ---
            K_log = to_goom(K_hat)
            K_log_broadcast = jnp.broadcast_to(K_log[None, None, ...], U_hat.shape)

            U_modulated = U_hat * B_gate
            U_log = to_goom(U_modulated)

            # --- 7. Apply Per-Frequency Δ Gating ---
            K_log_gated = (K_log_broadcast.real - delta_freq) + 1j * K_log_broadcast.imag
            U_log_gated = (U_log.real + jnp.log(delta_freq + 1e-8)) + 1j * U_log.imag

            # --- 8. Associative Scan (depthwise) ---
            _, X_log = jax.lax.associative_scan(
                cssm_scalar_scan_op, (K_log_gated, U_log_gated), axis=1
            )
            
            X_hat = from_goom(X_log)

        # --- 9. Apply C gate ---
        X_hat_gated = X_hat * C_gate  # (B, T, C, H, W_freq)

        # --- 10. Low-rank channel mixing in frequency domain ---
        # X_hat_gated: (B, T, C, H, W_freq)
        # Reshape for matmul: (B, T, H, W_freq, C)
        X_for_mix = X_hat_gated.transpose(0, 1, 3, 4, 2)

        # Project down: (B, T, H, W_freq, C) @ (C, rank) -> (B, T, H, W_freq, rank)
        X_down = X_for_mix @ V_down

        # Project up: (B, T, H, W_freq, rank) @ (rank, C) -> (B, T, H, W_freq, C)
        X_mixed = X_down @ V_up

        # Residual connection with learned scale
        X_out = X_for_mix + mix_scale[0] * X_mixed  # (B, T, H, W_freq, C)

        # Back to (B, T, C, H, W_freq)
        X_out = X_out.transpose(0, 1, 4, 2, 3)

        # --- 11. Inverse Transform ---
        x_out = jnp.fft.irfft2(X_out, s=(H, W), axes=(3, 4))

        return x_out.transpose(0, 1, 3, 4, 2)

    def _forward_dense_mixing(self, x, B, T, H, W, C, W_freq, x_raw=None):
        """
        Dense mixing forward pass - LMME channel mixing.

        Uses block-diagonal LMME for O(C × block_size²) complexity.
        This enables spatial channel mixing required for tasks like Pathfinder.
        """
        block_size = min(self.block_size, C)
        if C % block_size != 0:
            # Pad channels to be divisible by block_size
            pad_c = block_size - (C % block_size)
            x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 0), (0, pad_c)))
            C_padded = C + pad_c
        else:
            C_padded = C
            pad_c = 0

        num_blocks = C_padded // block_size

        # --- 1. Block-Diagonal Kernel Generation ---
        # Each block has a (block_size × block_size) mixing matrix per spatial position
        # Kernel shape: (num_blocks, block_size, block_size, kernel_size, kernel_size)
        k_blocks = self.param(
            'kernel_blocks',
            nn.initializers.normal(0.02),
            (num_blocks, block_size, block_size, self.kernel_size, self.kernel_size)
        )

        # --- 2. Spectral Transform of Block Kernels ---
        pad_h = max(0, (H - self.kernel_size) // 2)
        pad_w = max(0, (W - self.kernel_size) // 2)
        pad_h_after = H - self.kernel_size - pad_h
        pad_w_after = W - self.kernel_size - pad_w

        # Pad to image size
        if self.kernel_size <= H and self.kernel_size <= W:
            k_padded = jnp.pad(
                k_blocks,
                ((0, 0), (0, 0), (0, 0),
                 (pad_h, max(0, pad_h_after)), (pad_w, max(0, pad_w_after))),
                mode='constant'
            )
        else:
            start_h = (self.kernel_size - H) // 2
            start_w = (self.kernel_size - W) // 2
            k_padded = k_blocks[:, :, :, start_h:start_h+H, start_w:start_w+W]

        # FFT each block's mixing matrix
        # Shape: (num_blocks, block_size, block_size, H, W_freq)
        K_hat_raw = jnp.fft.rfft2(k_padded, axes=(3, 4))
        K_hat = _stable_spectral_magnitude(K_hat_raw, rho=self.spectral_rho)

        # --- 3. FFT Input ---
        x_perm = x.transpose(0, 1, 4, 2, 3)  # (B, T, C_padded, H, W)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C_padded, H, W_freq)

        # Reshape to blocks: (B, T, C_padded, H, W_freq) -> (B, T, num_blocks, block_size, H, W_freq)
        U_hat_blocks = U_hat.reshape(B, T, num_blocks, block_size, H, W_freq)

        # --- 4. Input-Dependent Gates ---
        # Pool spatial dims for gating context: (B, T, C_padded) - use full channel context
        # Use raw input (before position encoding) for gates if position_independent_gates is True
        gate_input = x_raw if x_raw is not None else x
        ctx = gate_input.mean(axis=(2, 3))  # (B, T, C_padded)

        # Per-frequency, per-block decay gate
        n_gate_feats = num_blocks * H * W_freq
        delta_raw = nn.Dense(n_gate_feats, name='delta_gate')(ctx)  # (B, T, n_gate_feats)
        delta_raw = delta_raw.reshape(B, T, num_blocks, 1, H, W_freq)
        delta_freq = self._compute_gate(delta_raw)  # (B, T, num_blocks, 1, H, W_freq)

        # Input/output projection gates (per block)
        B_gate_raw = nn.Dense(n_gate_feats, name='B_gate')(ctx)  # (B, T, n_gate_feats)
        B_gate = nn.sigmoid(B_gate_raw).reshape(B, T, num_blocks, 1, H, W_freq)

        C_gate_raw = nn.Dense(n_gate_feats, name='C_gate')(ctx)  # (B, T, n_gate_feats)
        C_gate = nn.sigmoid(C_gate_raw).reshape(B, T, num_blocks, 1, H, W_freq)

        # --- 5. Prepare for Block Scan ---
        # Apply B gate to input
        U_modulated = U_hat_blocks * B_gate  # (B, T, num_blocks, block_size, H, W_freq)

        # Reshape kernel for scan: (num_blocks, block_size, block_size, H, W_freq)
        #   -> (B, T, H, W_freq, num_blocks, block_size, block_size)
        K_hat_perm = K_hat.transpose(0, 3, 4, 1, 2)  # (num_blocks, H, W_freq, block_size, block_size)
        K_hat_broadcast = jnp.broadcast_to(
            K_hat_perm[None, None, ...],
            (B, T, num_blocks, H, W_freq, block_size, block_size)
        )

        # Reshape input for scan: (B, T, num_blocks, block_size, H, W_freq)
        #   -> (B, T, H, W_freq, num_blocks, block_size)
        U_perm = U_modulated.transpose(0, 1, 4, 5, 2, 3)  # (B, T, H, W_freq, num_blocks, block_size)

        # Convert to GOOM
        K_log = to_goom(K_hat_broadcast)  # (B, T, num_blocks, H, W_freq, block_size, block_size)
        U_log = to_goom(U_perm)  # (B, T, H, W_freq, num_blocks, block_size)

        # Apply delta gating in log-space
        # delta_freq: (B, T, num_blocks, 1, H, W_freq) -> need to match K_log shape
        delta_for_K = delta_freq.transpose(0, 1, 4, 5, 2, 3)  # (B, T, H, W_freq, num_blocks, 1)
        delta_for_K = delta_for_K[..., None]  # (B, T, H, W_freq, num_blocks, 1, 1)

        # K_log needs to be: (B, T, H, W_freq, num_blocks, block_size, block_size)
        K_log = K_log.transpose(0, 1, 3, 4, 2, 5, 6)  # (B, T, H, W_freq, num_blocks, block_size, block_size)
        K_log_gated = (K_log.real - delta_for_K) + 1j * K_log.imag

        # U_log gating
        delta_for_U = delta_freq.transpose(0, 1, 4, 5, 2, 3)  # (B, T, H, W_freq, num_blocks, 1)
        U_log_gated = (U_log.real + jnp.log(delta_for_U + 1e-8)) + 1j * U_log.imag

        # --- 6. Block Scan ---
        # Reshape for scan: merge spatial freq dims with batch
        # (B, T, H, W_freq, num_blocks, ...) -> (B * H * W_freq, T, num_blocks, ...)
        K_flat = K_log_gated.reshape(B * H * W_freq, T, num_blocks, block_size, block_size)
        U_flat = U_log_gated.reshape(B * H * W_freq, T, num_blocks, block_size)

        # Scan over time dimension
        block_scan_op = make_block_scan_op(block_size)
        _, X_log_flat = jax.lax.associative_scan(
            block_scan_op, (K_flat, U_flat), axis=1
        )

        # --- 7. Reshape back and apply C gate ---
        X_log = X_log_flat.reshape(B, H, W_freq, T, num_blocks, block_size)
        X_log = X_log.transpose(0, 3, 4, 5, 1, 2)  # (B, T, num_blocks, block_size, H, W_freq)

        X_hat = from_goom(X_log)
        X_hat_modulated = X_hat * C_gate  # (B, T, num_blocks, block_size, H, W_freq)

        # Reshape to (B, T, C_padded, H, W_freq)
        X_hat_flat = X_hat_modulated.reshape(B, T, C_padded, H, W_freq)

        # --- 8. Inverse Transform ---
        x_out = jnp.fft.irfft2(X_hat_flat, s=(H, W), axes=(3, 4))  # (B, T, C_padded, H, W)

        # Remove padding and return
        x_out = x_out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C_padded)
        if pad_c > 0:
            x_out = x_out[..., :C]

        return x_out
# =============================================================================
# These are ablation controls that use regular arithmetic instead of log-space.
# They use sequential scan (jax.lax.scan) instead of associative scan.
# This tests whether log-space computation is necessary for performance.


class LinearCSSM(nn.Module):
    """
    Vanilla CSSM without log-space (ablation control).

    Uses regular spectral convolution with multiplication and addition:
        h_t = K * h_{t-1} + u_t

    This is a sequential scan (O(T)) instead of parallel (O(log T)).

    Attributes:
        channels: Number of input/output channels
        kernel_size: Spatial kernel size
        spectral_rho: Maximum spectral magnitude for stability
        rope_mode: Position encoding mode
        rope_base: RoPE base frequency
    """
    channels: int
    kernel_size: int = 15
    spectral_rho: float = 0.999
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    # API compatibility
    gate_activation: str = 'sigmoid'
    gate_rank: int = 0
    concat_xy: bool = True
    dense_mixing: bool = False
    block_size: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with linear spectral recurrence (no log-space).

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

        # === Apply RoPE before FFT ===
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # --- 1. Spatial Kernel ---
        k_shape = (C, self.kernel_size, self.kernel_size)
        k = self.param('kernel', nn.initializers.normal(0.02), k_shape)

        # Pad and FFT kernel
        pad_h = max(0, (H - self.kernel_size) // 2)
        pad_w = max(0, (W - self.kernel_size) // 2)
        pad_h_after = H - self.kernel_size - pad_h
        pad_w_after = W - self.kernel_size - pad_w

        def pad_kernel(k):
            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k[:, start_h:start_h+H, start_w:start_w+W]
            return jnp.pad(k, ((0, 0), (pad_h, max(0, pad_h_after)),
                               (pad_w, max(0, pad_w_after))), mode='constant')

        K_raw = jnp.fft.rfft2(pad_kernel(k), axes=(1, 2))
        K = _stable_spectral_magnitude(K_raw, rho=self.spectral_rho)  # (C, H, W_freq)

        # --- 2. FFT Input ---
        x_perm = x.transpose(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C, H, W_freq)

        # --- 3. Learnable decay ---
        decay = jnp.clip(self.param('decay', nn.initializers.constant(0.9), (C,)), 0.1, 0.99)
        decay_complex = decay[None, :, None, None].astype(jnp.complex64)  # (1, C, 1, 1)

        # --- 4. Sequential Scan (Linear Recurrence) ---
        # h_t = decay * K * h_{t-1} + u_t
        K_decay = decay_complex * K[None, ...]  # (1, C, H, W_freq)

        def scan_fn(h, u):
            # h: (B, C, H, W_freq), u: (B, C, H, W_freq)
            h_new = K_decay * h + u
            return h_new, h_new

        # Initial state: zeros
        h_init = jnp.zeros((B, C, H, W_freq), dtype=jnp.complex64)

        # Transpose U for scan: (B, T, C, H, W_freq) -> (T, B, C, H, W_freq)
        U_scan = U_hat.transpose(1, 0, 2, 3, 4)

        # Sequential scan over time
        _, H_out = jax.lax.scan(scan_fn, h_init, U_scan)

        # Transpose back: (T, B, C, H, W_freq) -> (B, T, C, H, W_freq)
        H_out = H_out.transpose(1, 0, 2, 3, 4)

        # --- 5. Inverse FFT ---
        x_out = jnp.fft.irfft2(H_out, s=(H, W), axes=(3, 4))
        return x_out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)


class LinearOpponentCSSM(nn.Module):
    """
    Opponent CSSM without log-space (ablation control).

    Uses regular spectral convolution with multiplication and subtraction:
        X_t = alpha * X_{t-1} - K_I * mu * Y_{t-1} + U
        Y_t = K_E * gamma * X_{t-1} + delta * Y_{t-1}

    This is a sequential scan (O(T)) instead of parallel (O(log T)).

    Attributes:
        channels: Number of input/output channels
        kernel_size: Spatial kernel size for E/I kernels
        spectral_rho: Maximum spectral magnitude for stability
        gate_activation: Activation for gates ('sigmoid', 'softplus')
        rope_mode: Position encoding mode
        rope_base: RoPE base frequency
    """
    channels: int
    kernel_size: int = 11
    spectral_rho: float = 0.999
    gate_activation: str = 'sigmoid'
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    # API compatibility
    gate_rank: int = 0
    concat_xy: bool = True
    dense_mixing: bool = False
    block_size: int = 1

    def _apply_gate_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.gate_activation == 'sigmoid':
            return nn.sigmoid(x)
        elif self.gate_activation == 'softplus':
            return nn.softplus(x)
        else:
            return nn.sigmoid(x)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with linear opponent circuit (no log-space).

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

        # === Apply RoPE before FFT ===
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # --- 1. Spatial Kernels (Excitation and Inhibition) ---
        k_shape = (C, self.kernel_size, self.kernel_size)
        k_exc = self.param('k_exc', nn.initializers.normal(0.02), k_shape)
        k_inh = self.param('k_inh', nn.initializers.normal(0.02), k_shape)

        pad_h = max(0, (H - self.kernel_size) // 2)
        pad_w = max(0, (W - self.kernel_size) // 2)
        pad_h_after = H - self.kernel_size - pad_h
        pad_w_after = W - self.kernel_size - pad_w

        def pad_kernel(k):
            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k[:, start_h:start_h+H, start_w:start_w+W]
            return jnp.pad(k, ((0, 0), (pad_h, max(0, pad_h_after)),
                               (pad_w, max(0, pad_w_after))), mode='constant')

        K_E_raw = jnp.fft.rfft2(pad_kernel(k_exc), axes=(1, 2))
        K_I_raw = jnp.fft.rfft2(pad_kernel(k_inh), axes=(1, 2))
        K_E = _stable_spectral_magnitude(K_E_raw, rho=self.spectral_rho)  # (C, H, W_freq)
        K_I = _stable_spectral_magnitude(K_I_raw, rho=self.spectral_rho)

        # --- 2. FFT Input ---
        x_perm = x.transpose(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C, H, W_freq)

        # --- 3. Input-Dependent Gates ---
        ctx = x.mean(axis=(2, 3))  # (B, T, C)
        n_gate_feats = H * W_freq

        # Decay gates
        alpha_raw = nn.Dense(n_gate_feats, name='alpha_gate')(ctx)
        alpha_freq = self._apply_gate_activation(alpha_raw).reshape(B, T, 1, H, W_freq)

        delta_raw = nn.Dense(n_gate_feats, name='delta_gate')(ctx)
        delta_freq = self._apply_gate_activation(delta_raw).reshape(B, T, 1, H, W_freq)

        # Coupling gates
        mu_raw = nn.Dense(n_gate_feats, name='mu_gate')(ctx)  # Y -> X inhibition
        mu_freq = self._apply_gate_activation(mu_raw).reshape(B, T, 1, H, W_freq)

        gamma_raw = nn.Dense(n_gate_feats, name='gamma_gate')(ctx)  # X -> Y excitation
        gamma_freq = self._apply_gate_activation(gamma_raw).reshape(B, T, 1, H, W_freq)

        # I/O gates
        B_gate_raw = nn.Dense(n_gate_feats, name='B_gate')(ctx)
        B_gate = nn.sigmoid(B_gate_raw).reshape(B, T, 1, H, W_freq)

        C_gate_raw = nn.Dense(n_gate_feats, name='C_gate')(ctx)
        C_gate = nn.sigmoid(C_gate_raw).reshape(B, T, 1, H, W_freq)

        # --- 4. Learnable Decay ---
        decay = jnp.clip(self.param('decay', nn.initializers.constant(0.9), (C,)), 0.1, 0.99)
        decay_complex = decay[None, None, :, None, None].astype(jnp.complex64)

        # --- 5. Build Transition Components ---
        K_E_b = K_E[None, None, ...]  # (1, 1, C, H, W_freq)
        K_I_b = K_I[None, None, ...]

        # Modulate input
        U_modulated = U_hat * B_gate

        # --- 6. Sequential Scan (Linear Opponent Recurrence) ---
        # X_t = alpha * decay * X_{t-1} - K_I * mu * Y_{t-1} + U
        # Y_t = K_E * gamma * X_{t-1} + delta * decay * Y_{t-1}

        def scan_fn(carry, inputs):
            X_prev, Y_prev = carry  # Each: (B, C, H, W_freq)
            U_t, alpha_t, delta_t, mu_t, gamma_t = inputs

            # Expand gates: (B, 1, H, W_freq) -> broadcast with (B, C, H, W_freq)
            # X update: excitation decays, inhibited by Y
            X_new = (alpha_t * decay_complex[0, 0] * X_prev
                     - K_I_b[0, 0] * mu_t * Y_prev
                     + U_t)

            # Y update: accumulates excitation from X
            Y_new = (K_E_b[0, 0] * gamma_t * X_prev
                     + delta_t * decay_complex[0, 0] * Y_prev)

            return (X_new, Y_new), X_new

        # Initial state: zeros
        X_init = jnp.zeros((B, C, H, W_freq), dtype=jnp.complex64)
        Y_init = jnp.zeros((B, C, H, W_freq), dtype=jnp.complex64)

        # Prepare inputs for scan: transpose time to first axis
        U_scan = U_modulated.transpose(1, 0, 2, 3, 4)  # (T, B, C, H, W_freq)
        alpha_scan = alpha_freq.transpose(1, 0, 2, 3, 4)  # (T, B, 1, H, W_freq)
        delta_scan = delta_freq.transpose(1, 0, 2, 3, 4)
        mu_scan = mu_freq.transpose(1, 0, 2, 3, 4)
        gamma_scan = gamma_freq.transpose(1, 0, 2, 3, 4)

        # Sequential scan over time
        _, X_out = jax.lax.scan(
            scan_fn,
            (X_init, Y_init),
            (U_scan, alpha_scan, delta_scan, mu_scan, gamma_scan)
        )

        # Transpose back: (T, B, C, H, W_freq) -> (B, T, C, H, W_freq)
        X_out = X_out.transpose(1, 0, 2, 3, 4)

        # Apply output gate
        X_out_gated = X_out * C_gate

        # --- 7. Inverse FFT ---
        x_out = jnp.fft.irfft2(X_out_gated, s=(H, W), axes=(3, 4))
        return x_out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)
class HGRUBilinearCSSM(nn.Module):
    """
    hGRU-style CSSM with 3x3 Interaction Channel (hgru_bi mode).

    Extends the 2x2 opponent dynamics with a third channel Z that learns
    to track X-Y interaction, providing a linear approximation to bilinear
    X*Y dynamics while maintaining associativity for parallel scans.

    ============================================================================
    THE 3-STATE SYSTEM
    ============================================================================

    State: [X, Y, Z] where:
        X = Excitatory state (receives input, inhibited by Y and Z)
        Y = Inhibitory state (excited by X and Z)
        Z = Interaction channel (learns to track X-Y correlation)

    Dynamics:
        X_t = decay_x·X - μ_I·K_I·Y - α_I·K_I·Z + U_X
        Y_t = μ_E·K_E·X + decay_y·Y + α_E·K_E·Z + U_Y
        Z_t = γ·X + δ·Y + ε·Z + U_Z

    ============================================================================
    WHY Z APPROXIMATES BILINEAR
    ============================================================================

    True hGRU: X_t = ... - α·K_I·X_{t-1}·Y_{t-1}  (can't do in associative scan)

    Our approach: Z learns a linear combination of X and Y history that
    captures their interaction patterns. When Z feeds back into X and Y:
        - α_I·K_I·Z acts like the inhibitory X*Y term
        - α_E·K_E·Z acts like the excitatory X*Y term

    The key insight: Z accumulates X+Y information over time, weighted by
    learnable coefficients γ, δ, ε. This ISN'T exactly X*Y, but it's a
    learnable proxy that:
        1. Depends on both X and Y history
        2. Has appropriate spatial structure (via K_I, K_E kernels)
        3. Maintains associativity for O(log T) parallel scan

    ============================================================================

    Attributes:
        channels: Number of input/output channels
        kernel_size: Spatial kernel size for E/I kernels
        spectral_rho: Maximum spectral magnitude for stability
        gate_activation: Activation for gates ('sigmoid', 'softplus')
        rope_mode: Position encoding mode
        rope_base: RoPE base frequency
        concat_xyz: If True, concatenate [X, Y, Z] and project to output
    """
    channels: int
    kernel_size: int = 11
    spectral_rho: float = 0.999
    gate_activation: str = 'sigmoid'
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    concat_xyz: bool = True
    readout_state: str = 'xyz'  # Which state(s) to use: 'xyz', 'x', 'y', 'z', 'xy', 'xz', 'yz'
    pre_output_act: str = 'none'  # Activation before output_proj: 'gelu', 'silu', or 'none'
    # API compatibility
    gate_rank: int = 0
    dense_mixing: bool = False
    block_size: int = 1

    def _apply_gate_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.gate_activation == 'sigmoid':
            return nn.sigmoid(x)
        elif self.gate_activation == 'softplus':
            return nn.softplus(x)
        else:
            return nn.sigmoid(x)

    def _apply_pre_output_act(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply activation before output_proj if configured."""
        if self.pre_output_act == 'gelu':
            return jax.nn.gelu(x)
        elif self.pre_output_act == 'silu':
            return jax.nn.silu(x)
        else:
            return x

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with 3x3 interaction channel dynamics using parallel scan.

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        from .math import cssm_3x3_matrix_scan_op
        from .goom import to_goom, from_goom

        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

        # === Apply RoPE before FFT ===
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # =====================================================================
        # 1. SPATIAL KERNELS (Xavier init)
        # =====================================================================
        k_shape = (C, self.kernel_size, self.kernel_size)
        k_exc = self.param('k_exc', nn.initializers.xavier_normal(), k_shape)
        k_inh = self.param('k_inh', nn.initializers.xavier_normal(), k_shape)

        pad_h = max(0, (H - self.kernel_size) // 2)
        pad_w = max(0, (W - self.kernel_size) // 2)
        pad_h_after = H - self.kernel_size - pad_h
        pad_w_after = W - self.kernel_size - pad_w

        def pad_kernel(k):
            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k[:, start_h:start_h+H, start_w:start_w+W]
            return jnp.pad(k, ((0, 0), (pad_h, max(0, pad_h_after)),
                               (pad_w, max(0, pad_w_after))), mode='constant')

        K_E_raw = jnp.fft.rfft2(pad_kernel(k_exc), axes=(1, 2))
        K_I_raw = jnp.fft.rfft2(pad_kernel(k_inh), axes=(1, 2))
        K_E = _stable_spectral_magnitude(K_E_raw, rho=self.spectral_rho)
        K_I = _stable_spectral_magnitude(K_I_raw, rho=self.spectral_rho)

        # =====================================================================
        # 2. PROJECT INPUT TO X, Y, Z PATHWAYS (fused Dense)
        # =====================================================================
        # Single Dense(3C) then split - efficient projection to all 3 channels
        x_flat = x.reshape(B * T, H, W, C)
        xyz_proj = nn.Dense(3 * C, name='input_proj')(x_flat)
        xyz_proj = xyz_proj.reshape(B, T, H, W, 3 * C)

        # Split into X, Y, Z input pathways
        x_input = xyz_proj[..., :C]
        y_input = xyz_proj[..., C:2*C]
        z_input = xyz_proj[..., 2*C:]

        # FFT all three pathways
        x_perm = x_input.transpose(0, 1, 4, 2, 3)
        y_perm = y_input.transpose(0, 1, 4, 2, 3)
        z_perm = z_input.transpose(0, 1, 4, 2, 3)
        U_X_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))
        U_Y_hat = jnp.fft.rfft2(y_perm, axes=(3, 4))
        U_Z_hat = jnp.fft.rfft2(z_perm, axes=(3, 4))

        # =====================================================================
        # 3. INPUT-DEPENDENT GATES (all from context)
        # =====================================================================
        ctx = x.mean(axis=(2, 3))  # (B, T, C)
        ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)
        n_gate_feats = H * W_freq

        # --- Decay gates (X, Y, Z self-connections) ---
        decay_x_raw = nn.Dense(n_gate_feats, name='decay_x_gate')(ctx)
        decay_x_freq = 0.1 + 0.89 * nn.sigmoid(decay_x_raw).reshape(B, T, 1, H, W_freq)

        decay_y_raw = nn.Dense(n_gate_feats, name='decay_y_gate')(ctx)
        decay_y_freq = 0.1 + 0.89 * nn.sigmoid(decay_y_raw).reshape(B, T, 1, H, W_freq)

        decay_z_raw = nn.Dense(n_gate_feats, name='decay_z_gate')(ctx)
        decay_z_freq = 0.1 + 0.89 * nn.sigmoid(decay_z_raw).reshape(B, T, 1, H, W_freq)

        # --- Linear coupling gates (same as 2x2) ---
        mu_inhib_raw = nn.Dense(n_gate_feats, name='mu_inhib_gate')(ctx)
        mu_inhib = self._apply_gate_activation(mu_inhib_raw).reshape(B, T, 1, H, W_freq)

        mu_excit_raw = nn.Dense(n_gate_feats, name='mu_excit_gate')(ctx)
        mu_excit = self._apply_gate_activation(mu_excit_raw).reshape(B, T, 1, H, W_freq)

        # --- Z coupling gates (Z's effect on X and Y) ---
        alpha_inhib_raw = nn.Dense(n_gate_feats, name='alpha_inhib_gate')(ctx)
        alpha_inhib = self._apply_gate_activation(alpha_inhib_raw).reshape(B, T, 1, H, W_freq)

        alpha_excit_raw = nn.Dense(n_gate_feats, name='alpha_excit_gate')(ctx)
        alpha_excit = self._apply_gate_activation(alpha_excit_raw).reshape(B, T, 1, H, W_freq)

        # --- Z receives from X and Y ---
        gamma_raw = nn.Dense(n_gate_feats, name='gamma_gate')(ctx)  # X -> Z
        gamma = self._apply_gate_activation(gamma_raw).reshape(B, T, 1, H, W_freq)

        delta_raw = nn.Dense(n_gate_feats, name='delta_gate')(ctx)  # Y -> Z
        delta = self._apply_gate_activation(delta_raw).reshape(B, T, 1, H, W_freq)

        # --- I/O gates ---
        B_gate_raw = nn.Dense(n_gate_feats, name='B_gate')(ctx)
        B_gate = nn.sigmoid(B_gate_raw).reshape(B, T, 1, H, W_freq)

        D_gate_raw = nn.Dense(n_gate_feats, name='D_gate')(ctx)
        D_gate = nn.sigmoid(D_gate_raw).reshape(B, T, 1, H, W_freq)

        E_gate_raw = nn.Dense(n_gate_feats, name='E_gate')(ctx)
        E_gate = nn.sigmoid(E_gate_raw).reshape(B, T, 1, H, W_freq)

        C_gate_raw = nn.Dense(n_gate_feats, name='C_gate')(ctx)
        C_gate = nn.sigmoid(C_gate_raw).reshape(B, T, 1, H, W_freq)

        # =====================================================================
        # 4. BUILD 3x3 TRANSITION MATRIX
        # =====================================================================
        # [X_t]   [decay_x    -μ_I·K_I    -α_I·K_I] [X_{t-1}]   [U_X]
        # [Y_t] = [μ_E·K_E     decay_y    +α_E·K_E] [Y_{t-1}] + [U_Y]
        # [Z_t]   [γ           δ           ε      ] [Z_{t-1}]   [U_Z]

        K_E_b = K_E[None, None, ...].astype(jnp.complex64)
        K_I_b = K_I[None, None, ...].astype(jnp.complex64)

        decay_x_c = decay_x_freq.astype(jnp.complex64)
        decay_y_c = decay_y_freq.astype(jnp.complex64)
        decay_z_c = decay_z_freq.astype(jnp.complex64)

        ones = jnp.ones_like(K_E_b)

        # Row 0: X update (inhibited by Y via K_I, inhibited by Z via K_I)
        A_00 = decay_x_c * ones
        A_01 = -1.0 * mu_inhib * K_I_b
        A_02 = -1.0 * alpha_inhib * K_I_b

        # Row 1: Y update (excited by X via K_E, excited by Z via K_E)
        A_10 = mu_excit * K_E_b
        A_11 = decay_y_c * ones
        A_12 = alpha_excit * K_E_b

        # Row 2: Z update (receives from X and Y, self-decay)
        A_20 = gamma * ones
        A_21 = delta * ones
        A_22 = decay_z_c * ones

        # Stack into (B, T, C, H, W_freq, 3, 3)
        row0 = jnp.stack([A_00, A_01, A_02], axis=-1)
        row1 = jnp.stack([A_10, A_11, A_12], axis=-1)
        row2 = jnp.stack([A_20, A_21, A_22], axis=-1)
        K_mat = jnp.stack([row0, row1, row2], axis=-2)

        # =====================================================================
        # 5. APPLY I/O GATES TO INPUT
        # =====================================================================
        U_X_gated = U_X_hat * B_gate
        U_Y_gated = U_Y_hat * D_gate
        U_Z_gated = U_Z_hat * E_gate
        U_vec = jnp.stack([U_X_gated, U_Y_gated, U_Z_gated], axis=-1)

        # =====================================================================
        # 6. CONVERT TO GOOM AND APPLY PARALLEL ASSOCIATIVE SCAN
        # =====================================================================
        K_log = to_goom(K_mat)
        U_log = to_goom(U_vec)

        # 3x3 linear scan with interaction channel
        _, State_log = jax.lax.associative_scan(
            cssm_3x3_matrix_scan_op, (K_log, U_log), axis=1
        )

        # =====================================================================
        # 7. CONVERT BACK AND APPLY OUTPUT GATE
        # =====================================================================
        XYZ_hat = from_goom(State_log)  # (B, T, C, H, W_freq, 3)
        XYZ_hat_gated = XYZ_hat * C_gate[..., None]

        # Select which states to use based on readout_state
        state_indices = {'x': [0], 'y': [1], 'z': [2],
                         'xy': [0, 1], 'xz': [0, 2], 'yz': [1, 2],
                         'xyz': [0, 1, 2]}
        indices = state_indices.get(self.readout_state, [0, 1, 2])

        if len(indices) == 3 and self.concat_xyz:
            # Original behavior: concatenate all and project
            XYZ_hat_gated = XYZ_hat_gated.transpose(0, 1, 2, 5, 3, 4)
            XYZ_hat_gated = XYZ_hat_gated.reshape(B, T, C * 3, H, -1)

            xyz_out = jnp.fft.irfft2(XYZ_hat_gated, s=(H, W), axes=(3, 4))
            xyz_out = xyz_out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, 3C)
            xyz_out = self._apply_pre_output_act(xyz_out)
            return nn.Dense(C, name='output_proj')(xyz_out)
        elif len(indices) == 1:
            # Single state output (x, y, or z) - no projection needed
            idx = indices[0]
            state_hat = XYZ_hat_gated[..., idx]
            state_out = jnp.fft.irfft2(state_hat, s=(H, W), axes=(3, 4))
            state_out = state_out.transpose(0, 1, 3, 4, 2)
            return self._apply_pre_output_act(state_out)
        else:
            # Multiple states: concatenate selected and project
            selected = [XYZ_hat_gated[..., i] for i in indices]
            # Stack and reshape for IFFT
            stacked = jnp.stack(selected, axis=-1)  # (B, T, C, H, W_freq, len(indices))
            stacked = stacked.transpose(0, 1, 2, 5, 3, 4)
            stacked = stacked.reshape(B, T, C * len(indices), H, -1)

            out = jnp.fft.irfft2(stacked, s=(H, W), axes=(3, 4))
            out = out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, len*C)
            out = self._apply_pre_output_act(out)
            return nn.Dense(C, name='output_proj')(out)


class TransformerCSSM(nn.Module):
    """
    Minimal transformer-like CSSM with Q, K, A states.

    This is a simplified version of HGRUBilinearCSSM that makes the
    attention parallel explicit:
    - Q (Query): like transformer queries, gets modulated by attention
    - K (Key): like transformer keys, interacts with Q
    - A (Attention): accumulates Q-K correlation, feeds back into Q

    The key insight: as the receptive field grows over timesteps (via
    kernel convolutions compounding), A accumulates Q-K interaction
    across this growing field - similar to iteratively growing attention.

    Dynamics:
        Q_t = decay_Q·Q + w·K·K + α·K·A + U_Q   (Q sees K and attention)
        K_t = w·K·Q + decay_K·K + U_K           (K sees Q, symmetric)
        A_t = γ·Q + γ·K + decay_A·A + U_A       (A accumulates Q+K)

    Key simplifications from HGRUBilinearCSSM:
    - One spatial kernel (not two K_E/K_I)
    - Symmetric Q↔K coupling
    - A is pure memory (no spatial kernel)
    - ~10 gates instead of 13+

    Attributes:
        channels: Number of input/output channels
        kernel_size: Spatial kernel size
        spectral_rho: Maximum spectral magnitude for stability
    """
    channels: int
    kernel_size: int = 11
    spectral_rho: float = 0.999
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    readout_state: str = 'qka'  # 'qka', 'q', 'k', 'a', 'qk', 'qa', 'ka'
    pre_output_act: str = 'none'
    position_independent_gates: bool = False  # Compute gates from raw input for length generalization
    # Ablation flags to test what makes TransformerCSSM work
    asymmetric_qk: bool = False      # If True, use separate weights for Q→K and K→Q
    no_feedback: bool = False        # If True, remove A→Q feedback (alpha = 0)
    no_spectral_clip: bool = False   # If True, skip spectral magnitude clipping
    use_layernorm: bool = False      # If True, add log-space LayerNorm before output
    no_k_state: bool = False         # If True, drop K entirely → 2×2 Q+A scan
    # Unused but kept for API compatibility
    gate_activation: str = 'sigmoid'
    concat_xy: bool = True
    gate_rank: int = 0
    dense_mixing: bool = False
    block_size: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, injected_qkv_spatial=None) -> jnp.ndarray:
        """
        Forward pass with minimal Q/K/A dynamics.

        Args:
            x: Input tensor of shape (B, T, H, W, C)
            injected_qkv_spatial: Unused; accepted for API compatibility with SimpleCSSM dispatch.

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        del injected_qkv_spatial
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

        # Save raw input for position-independent gates
        x_raw = x if self.position_independent_gates else None

        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # === SINGLE spatial kernel (simplified from two K_E/K_I) ===
        k_shape = (C, self.kernel_size, self.kernel_size)
        k_spatial = self.param('kernel', nn.initializers.xavier_normal(), k_shape)

        pad_h = max(0, (H - self.kernel_size) // 2)
        pad_w = max(0, (W - self.kernel_size) // 2)
        pad_h_after = H - self.kernel_size - pad_h
        pad_w_after = W - self.kernel_size - pad_w

        if self.kernel_size > H or self.kernel_size > W:
            start_h = (self.kernel_size - H) // 2
            start_w = (self.kernel_size - W) // 2
            k_padded = k_spatial[:, start_h:start_h+H, start_w:start_w+W]
        else:
            k_padded = jnp.pad(k_spatial, ((0, 0), (pad_h, max(0, pad_h_after)),
                                           (pad_w, max(0, pad_w_after))), mode='constant')

        K_hat_raw = jnp.fft.rfft2(k_padded, axes=(1, 2))
        # Ablation: no_spectral_clip - skip magnitude bounding
        if self.no_spectral_clip:
            K_hat = K_hat_raw
        else:
            K_hat = _stable_spectral_magnitude(K_hat_raw, rho=self.spectral_rho)

        # === Shared gate context ===
        gate_input = x_raw if x_raw is not None else x
        ctx = gate_input.mean(axis=(2, 3))
        if x_raw is None:
            ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)
        n_gate = H * W_freq

        K_b = K_hat[None, None, ...].astype(jnp.complex64)
        ones = jnp.ones_like(K_b)

        # =====================================================================
        # no_k_state: Lean 2×2 Q+A scan (drop K entirely)
        # =====================================================================
        if self.no_k_state:
            # --- Project input to Q, A pathways only ---
            x_flat = x.reshape(B * T, H, W, C)
            qa_proj = nn.Dense(2 * C, name='input_proj')(x_flat)
            qa_proj = qa_proj.reshape(B, T, H, W, 2 * C)

            q_input = qa_proj[..., :C]
            a_input = qa_proj[..., C:]

            U_Q_hat = jnp.fft.rfft2(q_input.transpose(0, 1, 4, 2, 3), axes=(3, 4))
            U_A_hat = jnp.fft.rfft2(a_input.transpose(0, 1, 4, 2, 3), axes=(3, 4))

            # --- Gates (7 total: decay_Q, decay_A, alpha, gamma, B_Q, B_A, C_gate) ---
            decay_Q = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='decay_Q')(ctx)).reshape(B, T, 1, H, W_freq)
            decay_A = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='decay_A')(ctx)).reshape(B, T, 1, H, W_freq)
            alpha = nn.sigmoid(nn.Dense(n_gate, name='alpha')(ctx)).reshape(B, T, 1, H, W_freq)
            gamma = nn.sigmoid(nn.Dense(n_gate, name='gamma')(ctx)).reshape(B, T, 1, H, W_freq)
            B_Q = nn.sigmoid(nn.Dense(n_gate, name='B_Q')(ctx)).reshape(B, T, 1, H, W_freq)
            B_A = nn.sigmoid(nn.Dense(n_gate, name='B_A')(ctx)).reshape(B, T, 1, H, W_freq)
            C_gate = nn.sigmoid(nn.Dense(n_gate, name='C_gate')(ctx)).reshape(B, T, 1, H, W_freq)

            # --- Build 2×2 transition matrix ---
            # [decay_Q    α·K ]   Q: self-decay + A→Q feedback via kernel
            # [  γ      decay_A]   A: accumulates Q + self-decay (no kernel)
            decay_Q_c = decay_Q.astype(jnp.complex64)
            decay_A_c = decay_A.astype(jnp.complex64)

            A_00 = decay_Q_c * ones      # Q self-decay
            A_01 = alpha * K_b           # A → Q via kernel (the feedback!)
            A_10 = gamma * ones          # Q → A (scalar, no kernel)
            A_11 = decay_A_c * ones      # A self-decay

            row0 = jnp.stack([A_00, A_01], axis=-1)
            row1 = jnp.stack([A_10, A_11], axis=-1)
            K_mat = jnp.stack([row0, row1], axis=-2)  # (B, T, C, H, W_freq, 2, 2)

            U_Q_gated = U_Q_hat * B_Q
            U_A_gated = U_A_hat * B_A
            U_vec = jnp.stack([U_Q_gated, U_A_gated], axis=-1)  # (B, T, C, H, W_freq, 2)

            # --- GOOM scan (2×2) ---
            K_log = to_goom(K_mat)
            U_log = to_goom(U_vec)

            _, State_log = jax.lax.associative_scan(
                cssm_matrix_scan_op, (K_log, U_log), axis=1
            )

            QA_hat = from_goom(State_log)  # (B, T, C, H, W_freq, 2)
            QA_hat_gated = QA_hat * C_gate[..., None]

            # --- Output: Q+A readout ---
            QA_hat_gated = QA_hat_gated.transpose(0, 1, 2, 5, 3, 4)
            QA_hat_gated = QA_hat_gated.reshape(B, T, C * 2, H, -1)

            out = jnp.fft.irfft2(QA_hat_gated, s=(H, W), axes=(3, 4))
            out = out.transpose(0, 1, 3, 4, 2)
            if self.pre_output_act == 'gelu':
                out = jax.nn.gelu(out)
            elif self.pre_output_act == 'silu':
                out = jax.nn.silu(out)
            return nn.Dense(C, name='output_proj')(out)

        # =====================================================================
        # Standard 3×3 Q+K+A scan
        # =====================================================================

        # === Project input to Q, K, A pathways ===
        x_flat = x.reshape(B * T, H, W, C)
        qka_proj = nn.Dense(3 * C, name='input_proj')(x_flat)
        qka_proj = qka_proj.reshape(B, T, H, W, 3 * C)

        q_input = qka_proj[..., :C]
        k_input = qka_proj[..., C:2*C]
        a_input = qka_proj[..., 2*C:]

        # FFT inputs
        U_Q_hat = jnp.fft.rfft2(q_input.transpose(0, 1, 4, 2, 3), axes=(3, 4))
        U_K_hat = jnp.fft.rfft2(k_input.transpose(0, 1, 4, 2, 3), axes=(3, 4))
        U_A_hat = jnp.fft.rfft2(a_input.transpose(0, 1, 4, 2, 3), axes=(3, 4))

        # === Input-dependent gates ===
        # 3 decay gates (bounded 0.1-0.99)
        decay_Q = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='decay_Q')(ctx)).reshape(B, T, 1, H, W_freq)
        decay_K = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='decay_K')(ctx)).reshape(B, T, 1, H, W_freq)
        decay_A = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='decay_A')(ctx)).reshape(B, T, 1, H, W_freq)

        # Q↔K coupling
        # Ablation: asymmetric_qk - use separate weights for each direction
        if self.asymmetric_qk:
            w_k_to_q = nn.sigmoid(nn.Dense(n_gate, name='w_k_to_q')(ctx)).reshape(B, T, 1, H, W_freq)
            w_q_to_k = nn.sigmoid(nn.Dense(n_gate, name='w_q_to_k')(ctx)).reshape(B, T, 1, H, W_freq)
        else:
            w_qk = nn.sigmoid(nn.Dense(n_gate, name='w_qk')(ctx)).reshape(B, T, 1, H, W_freq)
            w_k_to_q = w_qk  # Same weight for both directions (symmetric)
            w_q_to_k = w_qk

        # A→Q feedback (attention application)
        # Ablation: no_feedback - disable A→Q path
        if self.no_feedback:
            alpha = jnp.zeros((B, T, 1, H, W_freq))
        else:
            alpha = nn.sigmoid(nn.Dense(n_gate, name='alpha')(ctx)).reshape(B, T, 1, H, W_freq)

        # A accumulation rate (same for Q and K contribution)
        gamma = nn.sigmoid(nn.Dense(n_gate, name='gamma')(ctx)).reshape(B, T, 1, H, W_freq)

        # I/O gates
        B_Q = nn.sigmoid(nn.Dense(n_gate, name='B_Q')(ctx)).reshape(B, T, 1, H, W_freq)
        B_K = nn.sigmoid(nn.Dense(n_gate, name='B_K')(ctx)).reshape(B, T, 1, H, W_freq)
        B_A = nn.sigmoid(nn.Dense(n_gate, name='B_A')(ctx)).reshape(B, T, 1, H, W_freq)
        C_gate = nn.sigmoid(nn.Dense(n_gate, name='C_gate')(ctx)).reshape(B, T, 1, H, W_freq)

        # === Build 3x3 transition matrix ===
        # [decay_Q    w·K       α·K   ]   Q: sees K and A through kernel
        # [w·K        decay_K    0    ]   K: sees Q through kernel (symmetric)
        # [γ          γ        decay_A]   A: pure memory, no kernel

        zeros = jnp.zeros_like(decay_Q.astype(jnp.complex64) * ones)

        decay_Q_c = decay_Q.astype(jnp.complex64)
        decay_K_c = decay_K.astype(jnp.complex64)
        decay_A_c = decay_A.astype(jnp.complex64)

        # Row 0: Q update
        A_00 = decay_Q_c * ones           # Q self-decay (scalar, no kernel)
        A_01 = w_k_to_q * K_b             # K → Q via kernel
        A_02 = alpha * K_b                # A → Q via kernel (attention application!)

        # Row 1: K update
        A_10 = w_q_to_k * K_b             # Q → K via kernel (same as K→Q if symmetric)
        A_11 = decay_K_c * ones           # K self-decay (scalar)
        A_12 = zeros                      # No A → K

        # Row 2: A update (pure memory - NO kernel)
        A_20 = gamma * ones               # Q → A (scalar)
        A_21 = gamma * ones               # K → A (scalar, same weight)
        A_22 = decay_A_c * ones           # A self-decay (scalar)

        row0 = jnp.stack([A_00, A_01, A_02], axis=-1)
        row1 = jnp.stack([A_10, A_11, A_12], axis=-1)
        row2 = jnp.stack([A_20, A_21, A_22], axis=-1)
        K_mat = jnp.stack([row0, row1, row2], axis=-2)

        # Apply input gates
        U_Q_gated = U_Q_hat * B_Q
        U_K_gated = U_K_hat * B_K
        U_A_gated = U_A_hat * B_A
        U_vec = jnp.stack([U_Q_gated, U_K_gated, U_A_gated], axis=-1)

        # === GOOM scan ===
        from .math import cssm_3x3_matrix_scan_op

        K_log = to_goom(K_mat)
        U_log = to_goom(U_vec)

        _, State_log = jax.lax.associative_scan(
            cssm_3x3_matrix_scan_op, (K_log, U_log), axis=1
        )

        # Ablation: use_layernorm - apply LayerNorm in log-space before exp
        if self.use_layernorm:
            # LayerNorm parameters for each state
            ln_gamma = self.param('ln_gamma', nn.initializers.ones, (C, 1, 1, 3))
            ln_beta = self.param('ln_beta', nn.initializers.zeros, (C, 1, 1, 3))

            # State_log shape: (B, T, C, H, W_freq, 3)
            log_real = State_log.real
            log_imag = State_log.imag

            # LayerNorm over spatial frequencies (axes 3, 4 = H, W_freq)
            mean = log_real.mean(axis=(3, 4), keepdims=True)
            std = log_real.std(axis=(3, 4), keepdims=True) + 1e-6
            log_normalized = (log_real - mean) / std

            # Apply learnable scale and shift
            # ln_gamma/ln_beta: (C, 1, 1, 3) broadcast to (B, T, C, H, W_freq, 3)
            log_scaled = ln_gamma[None, None, :, :, :, :] * log_normalized + ln_beta[None, None, :, :, :, :]

            # Reconstruct complex and exp
            QKA_hat = jnp.exp(log_scaled + 1j * log_imag)
        else:
            QKA_hat = from_goom(State_log)

        QKA_hat_gated = QKA_hat * C_gate[..., None]

        # === Output ===
        state_indices = {'q': [0], 'k': [1], 'a': [2],
                         'qk': [0, 1], 'qa': [0, 2], 'ka': [1, 2],
                         'qka': [0, 1, 2]}
        indices = state_indices.get(self.readout_state, [0, 1, 2])

        if len(indices) == 3:
            QKA_hat_gated = QKA_hat_gated.transpose(0, 1, 2, 5, 3, 4)
            QKA_hat_gated = QKA_hat_gated.reshape(B, T, C * 3, H, -1)

            out = jnp.fft.irfft2(QKA_hat_gated, s=(H, W), axes=(3, 4))
            out = out.transpose(0, 1, 3, 4, 2)
            if self.pre_output_act == 'gelu':
                out = jax.nn.gelu(out)
            elif self.pre_output_act == 'silu':
                out = jax.nn.silu(out)
            return nn.Dense(C, name='output_proj')(out)
        elif len(indices) == 1:
            idx = indices[0]
            state_hat = QKA_hat_gated[..., idx]
            out = jnp.fft.irfft2(state_hat, s=(H, W), axes=(3, 4))
            out = out.transpose(0, 1, 3, 4, 2)
            if self.pre_output_act == 'gelu':
                out = jax.nn.gelu(out)
            elif self.pre_output_act == 'silu':
                out = jax.nn.silu(out)
            return out
        else:
            selected = [QKA_hat_gated[..., i] for i in indices]
            stacked = jnp.stack(selected, axis=-1)
            stacked = stacked.transpose(0, 1, 2, 5, 3, 4)
            stacked = stacked.reshape(B, T, C * len(indices), H, -1)

            out = jnp.fft.irfft2(stacked, s=(H, W), axes=(3, 4))
            out = out.transpose(0, 1, 3, 4, 2)
            if self.pre_output_act == 'gelu':
                out = jax.nn.gelu(out)
            elif self.pre_output_act == 'silu':
                out = jax.nn.silu(out)
            return nn.Dense(C, name='output_proj')(out)


# =============================================================================
# Multiplicative TransformerCSSM - Bilinear QKV with hGRU-style Spreading
# =============================================================================

class MultiplicativeTransformerCSSM(nn.Module):
    """
    Bilinear QKV Attention with hGRU-style Spreading (Simplified).

    A recurrent attention mechanism that captures the hGRU bilinear circuit
    in a QKV framework. **Simplified design**: kernels baked into diagonal of A,
    no separate bias vector.

    ============================================================================
    KEY FEATURES
    ============================================================================

    1. **3 Recurrent States**: Q, K, V (all evolve over time)
    2. **Each state spreads with its own kernel**: Q×W_Q, K×W_K, V×W_V
    3. **Cross-state contributions are direct**: Q→V and K→V have no extra convolution
    4. **Mamba-style input-dependent gates**: All 7 non-zero entries predicted from context
    5. **Uses existing `cssm_3x3_matrix_scan_op`**: Simple 2-tuple scan (A, b)

    ============================================================================
    KEY SIMPLIFICATION
    ============================================================================

    **Old design**: Separate bias vector c for kernels → complex 3-tuple scan
    **New design**: Kernels baked into diagonal of A → simpler 2-tuple scan (A, b)

    ============================================================================
    DYNAMICS
    ============================================================================

    Original Space (What We Compute):
        Q_new = (d_Q × W_Q × Q) + (d_K→Q × K) + U_Q
        K_new = (d_Q→K × Q) + (d_K × W_K × K) + U_K
        V_new = (d_Q→V × Q) + (d_K→V × K) + (d_V × W_V × V) + U_V

    Key insight:
    - **Diagonal terms** (self-recurrence): decay × kernel × state → state spreads spatially
    - **Off-diagonal terms** (cross-coupling): weight × state → direct contribution, no extra conv
    - **Input terms**: added via logsumexp

    ============================================================================
    TRANSITION MATRIX A (Kernels on Diagonal)
    ============================================================================

    In log-space:
    - Diagonal: `log(d × W) = log(d) + log(W)` — decay AND kernel combined
    - Off-diagonal: `log(d)` only — just the coupling weight
    - `-∞` entries: no V→Q or V→K feedback

                    q                    k                    v
    A = q  [ log(d_Q × W_Q)      log(d_K→Q)           -∞           ]
        k  [ log(d_Q→K)          log(d_K × W_K)       -∞           ]
        v  [ log(d_Q→V)          log(d_K→V)       log(d_V × W_V)   ]
             ↑                    ↑                    ↑
          kernel on diagonal    no kernel           kernel on diagonal

    ============================================================================
    UPDATE RULE
    ============================================================================

    Simple 2-tuple scan (no separate bias c):
        x_t = logsumexp(A @ x_{t-1}, b_t)

    Where:
    - `A @ x` in log-space: `(A @ x)_i = logsumexp_j(A[i,j] + x[j])`
    - `b_t` = input vector `[log_U_Q, log_U_K, log_U_V]`
    - Final logsumexp adds input (addition in original space)

    ============================================================================
    hGRU MAPPING
    ============================================================================

    | Component           | Behavior                                    |
    |---------------------|---------------------------------------------|
    | Q spreads via W_Q   | Q's receptive field grows over time         |
    | K spreads via W_K   | K's receptive field grows over time         |
    | Q + K → V (direct)  | Where Q and K meet, signal flows to V       |
    | V spreads via W_V   | V accumulates and spreads the attention sig |

    Attributes:
        channels: Number of input/output channels
        kernel_size: Spatial kernel size for lateral connections
        use_goom: If True, use complex log to handle negative values (GELU-friendly)
        rope_mode: Position encoding mode
        position_independent_gates: Compute gates from raw input for length generalization
    """
    channels: int
    kernel_size: int = 11
    use_goom: bool = True  # Handle negative values via complex phase
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    readout_state: str = 'v'  # 'v' for V output (attention-accumulated values)
    pre_output_act: str = 'none'
    position_independent_gates: bool = False  # Compute gates from raw input for length generalization
    # Unused but kept for API compatibility
    spectral_rho: float = 0.999
    block_size: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with bilinear QKV dynamics using 2-tuple scan.

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        from .math import cssm_3x3_matrix_scan_op

        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

        # Save raw input for position-independent gates
        x_raw = x if self.position_independent_gates else None

        # Optional position encoding
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # =====================================================================
        # 1. THREE SPATIAL KERNELS (W_Q, W_K, W_V)
        # =====================================================================
        k_shape = (C, self.kernel_size, self.kernel_size)
        kernel_q = self.param('kernel_q', nn.initializers.xavier_normal(), k_shape)
        kernel_k = self.param('kernel_k', nn.initializers.xavier_normal(), k_shape)
        kernel_v = self.param('kernel_v', nn.initializers.xavier_normal(), k_shape)

        def pad_kernel(k_spatial):
            """Pad kernel to input size for FFT."""
            pad_h = max(0, (H - self.kernel_size) // 2)
            pad_w = max(0, (W - self.kernel_size) // 2)
            pad_h_after = H - self.kernel_size - pad_h
            pad_w_after = W - self.kernel_size - pad_w

            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k_spatial[:, start_h:start_h+H, start_w:start_w+W]
            else:
                return jnp.pad(k_spatial, ((0, 0), (pad_h, max(0, pad_h_after)),
                                           (pad_w, max(0, pad_w_after))), mode='constant')

        def kernel_to_log_spectral(kernel):
            """FFT kernel and convert to log-spectral representation."""
            k_padded = pad_kernel(kernel)
            k_hat = jnp.fft.rfft2(k_padded, axes=(1, 2))  # (C, H, W_freq)
            if self.use_goom:
                return jnp.log(jnp.abs(k_hat) + 1e-8) + 1j * jnp.angle(k_hat)
            else:
                return jnp.log(jnp.abs(k_hat) + 1e-8)

        log_W_Q = kernel_to_log_spectral(kernel_q)  # (C, H, W_freq)
        log_W_K = kernel_to_log_spectral(kernel_k)
        log_W_V = kernel_to_log_spectral(kernel_v)

        # =====================================================================
        # 2. PROJECT INPUT TO Q, K, V PATHWAYS
        # =====================================================================
        x_flat = x.reshape(B * T, H, W, C)
        qkv_proj = nn.Dense(3 * C, name='qkv_proj')(x_flat)
        qkv_proj = qkv_proj.reshape(B, T, H, W, 3 * C)

        q_in = qkv_proj[..., :C]
        k_in = qkv_proj[..., C:2*C]
        v_in = qkv_proj[..., 2*C:]

        def to_log_spectral(y):
            """Convert spatial input to log-spectral representation."""
            y_transposed = y.transpose(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            y_hat = jnp.fft.rfft2(y_transposed, axes=(3, 4))  # (B, T, C, H, W_freq)
            if self.use_goom:
                return jnp.log(jnp.abs(y_hat) + 1e-8) + 1j * jnp.angle(y_hat)
            else:
                return jnp.log(jnp.abs(y_hat) + 1e-8)

        log_U_Q = to_log_spectral(q_in)
        log_U_K = to_log_spectral(k_in)
        log_U_V = to_log_spectral(v_in)

        # =====================================================================
        # 3. INPUT-DEPENDENT GATES (Mamba-style, ALL 7 non-zero entries)
        # =====================================================================
        gate_input = x_raw if x_raw is not None else x
        ctx = gate_input.mean(axis=(2, 3))  # (B, T, C) - spatial average for context
        if x_raw is None:
            ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)
        n_gate = H * W_freq

        # Self-decay exponents (diagonal, bounded 0.1-0.99 for stability)
        d_Q = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='d_Q')(ctx))
        d_K = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='d_K')(ctx))
        d_V = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='d_V')(ctx))

        # Q↔K cross-coupling (bilinear gating strength, 0-1)
        d_K_to_Q = nn.sigmoid(nn.Dense(n_gate, name='d_K_to_Q')(ctx))  # K gates Q
        d_Q_to_K = nn.sigmoid(nn.Dense(n_gate, name='d_Q_to_K')(ctx))  # Q gates K

        # Attention flow to V (how much Q and K contribute to V, 0-1)
        d_Q_to_V = nn.sigmoid(nn.Dense(n_gate, name='d_Q_to_V')(ctx))  # Q → V
        d_K_to_V = nn.sigmoid(nn.Dense(n_gate, name='d_K_to_V')(ctx))  # K → V

        # Reshape all for broadcasting: (B, T, 1, H, W_freq)
        d_Q = d_Q.reshape(B, T, 1, H, W_freq)
        d_K = d_K.reshape(B, T, 1, H, W_freq)
        d_V = d_V.reshape(B, T, 1, H, W_freq)
        d_K_to_Q = d_K_to_Q.reshape(B, T, 1, H, W_freq)
        d_Q_to_K = d_Q_to_K.reshape(B, T, 1, H, W_freq)
        d_Q_to_V = d_Q_to_V.reshape(B, T, 1, H, W_freq)
        d_K_to_V = d_K_to_V.reshape(B, T, 1, H, W_freq)

        # =====================================================================
        # 4. BUILD TRANSITION MATRIX A (KERNELS ON DIAGONAL)
        # =====================================================================
        # Diagonal entries: decay × kernel (combined)
        # log(d × W) = log(d) + log(W)
        dtype = jnp.complex64 if self.use_goom else jnp.float32

        # Expand kernels for broadcasting: (C, H, W_freq) -> (1, 1, C, H, W_freq)
        log_W_Q_exp = log_W_Q[None, None, :, :, :]
        log_W_K_exp = log_W_K[None, None, :, :, :]
        log_W_V_exp = log_W_V[None, None, :, :, :]

        # Diagonal: log(decay) + log(kernel) — decay AND kernel combined
        log_diag_Q = jnp.log(d_Q + 1e-8) + log_W_Q_exp  # d_Q × W_Q
        log_diag_K = jnp.log(d_K + 1e-8) + log_W_K_exp  # d_K × W_K
        log_diag_V = jnp.log(d_V + 1e-8) + log_W_V_exp  # d_V × W_V

        # Off-diagonal entries: coupling weight only (NO kernel)
        log_d_K_to_Q = jnp.log(d_K_to_Q + 1e-8)
        log_d_Q_to_K = jnp.log(d_Q_to_K + 1e-8)
        log_d_Q_to_V = jnp.log(d_Q_to_V + 1e-8)
        log_d_K_to_V = jnp.log(d_K_to_V + 1e-8)

        # Convert to complex for GOOM if needed
        if self.use_goom:
            log_diag_Q = log_diag_Q.astype(jnp.complex64)
            log_diag_K = log_diag_K.astype(jnp.complex64)
            log_diag_V = log_diag_V.astype(jnp.complex64)
            log_d_K_to_Q = log_d_K_to_Q.astype(jnp.complex64)
            log_d_Q_to_K = log_d_Q_to_K.astype(jnp.complex64)
            log_d_Q_to_V = log_d_Q_to_V.astype(jnp.complex64)
            log_d_K_to_V = log_d_K_to_V.astype(jnp.complex64)

        # -inf in log-space = 0 coefficient in original space (no V→Q or V→K feedback)
        neg_inf = jnp.full((B, T, C, H, W_freq), -1e8, dtype=dtype)

        #                q                    k                    v
        # q  [ log(d_Q × W_Q)      log(d_K→Q)           -∞           ]
        # k  [ log(d_Q→K)          log(d_K × W_K)       -∞           ]
        # v  [ log(d_Q→V)          log(d_K→V)       log(d_V × W_V)   ]
        #      ↑                    ↑                    ↑
        #   kernel on diagonal    no kernel           kernel on diagonal

        # Broadcast all to (B, T, C, H, W_freq)
        ones_shape = jnp.ones((B, T, C, H, W_freq), dtype=dtype)
        log_diag_Q_bc = log_diag_Q * ones_shape
        log_diag_K_bc = log_diag_K * ones_shape
        log_diag_V_bc = log_diag_V * ones_shape
        log_d_K_to_Q_bc = log_d_K_to_Q * ones_shape
        log_d_Q_to_K_bc = log_d_Q_to_K * ones_shape
        log_d_Q_to_V_bc = log_d_Q_to_V * ones_shape
        log_d_K_to_V_bc = log_d_K_to_V * ones_shape

        row_q = jnp.stack([log_diag_Q_bc, log_d_K_to_Q_bc, neg_inf], axis=-1)
        row_k = jnp.stack([log_d_Q_to_K_bc, log_diag_K_bc, neg_inf], axis=-1)
        row_v = jnp.stack([log_d_Q_to_V_bc, log_d_K_to_V_bc, log_diag_V_bc], axis=-1)

        A_mat = jnp.stack([row_q, row_k, row_v], axis=-2)  # (B, T, C, H, W_freq, 3, 3)

        # =====================================================================
        # 5. BUILD INPUT VECTOR b
        # =====================================================================
        b_vec = jnp.stack([log_U_Q, log_U_K, log_U_V], axis=-1)  # (B, T, C, H, W_freq, 3)

        # =====================================================================
        # 6. ASSOCIATIVE SCAN (2-tuple: A, b)
        # =====================================================================
        # Simple 2-tuple scan: x_t = logsumexp(A @ x_{t-1}, b_t)
        # No separate bias c — kernels are baked into diagonal of A
        _, state_log = jax.lax.associative_scan(
            cssm_3x3_matrix_scan_op, (A_mat, b_vec), axis=1
        )

        # state_log shape: (B, T, C, H, W_freq, 3) where last dim is [Q, K, V]
        log_Q = state_log[..., 0]  # (B, T, C, H, W_freq)
        log_K = state_log[..., 1]
        log_V = state_log[..., 2]

        # =====================================================================
        # 7. CONVERT TO SPATIAL AND OUTPUT
        # =====================================================================
        # LayerNorm in log-space for stability
        ln_gamma_v = self.param('ln_gamma_v', nn.initializers.ones, (C, 1, 1))
        ln_beta_v = self.param('ln_beta_v', nn.initializers.zeros, (C, 1, 1))

        def from_log_spectral(log_state, ln_gamma, ln_beta):
            """Convert log-spectral back to spatial domain with LayerNorm."""
            if self.use_goom:
                log_real = log_state.real
            else:
                log_real = log_state

            # LayerNorm over spatial frequencies (axes 3, 4 = H, W_freq)
            mean = log_real.mean(axis=(-2, -1), keepdims=True)
            std = log_real.std(axis=(-2, -1), keepdims=True) + 1e-6

            if self.use_goom:
                log_normalized = (log_state.real - mean) / std + 1j * log_state.imag
            else:
                log_normalized = (log_state - mean) / std

            # Learnable scale and shift
            log_scaled = ln_gamma[None, None, :, :, :] * log_normalized + ln_beta[None, None, :, :, :]

            # Now exp is safe (values centered around 0)
            state_hat = jnp.exp(log_scaled)

            # iFFT back to spatial
            return jnp.fft.irfft2(state_hat, s=(H, W), axes=(3, 4))

        # V is the output (accumulated attention-weighted values)
        V_spatial = from_log_spectral(log_V, ln_gamma_v, ln_beta_v)  # (B, T, C, H, W)

        # Optional pre-output activation
        if self.pre_output_act == 'gelu':
            V_spatial = jax.nn.gelu(V_spatial)
        elif self.pre_output_act == 'silu':
            V_spatial = jax.nn.silu(V_spatial)

        # Transpose and project
        output = V_spatial.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        return nn.Dense(C, name='output_proj')(output)


class GrowingTransformerCSSM(nn.Module):
    """
    Growing Attention Transformer with two-stage Q→[K,V] architecture.

    ============================================================================
    KEY FEATURES
    ============================================================================

    1. **Two-stage approach**: Q computed first (scalar scan), then [K, V] together
    2. **Triangular scan**: K→V flow, no V→K feedback (preserves associativity)
    3. **Growing attention**: V accumulates Q·K weighted signal over time
    4. **GOOM**: Uses to_goom/from_goom with custom VJP for gradient stability
    5. **Always spectral clip**: Kernels bounded by rho for dynamic stability
    6. **No LayerNorm**: Direct from_goom output (confirmed better)

    ============================================================================
    DYNAMICS
    ============================================================================

    **Stage 1: Q Scan (Independent)**

        Q_t = d_Q · W_Q · Q_{t-1} + B_Q · U_Q

    **Stage 2: [K, V] Triangular Scan**

        [K_t]   [ d_K·Q·W_K       0        ] [K_{t-1}]   [ B_K·Q·U_K ]
        [V_t] = [ Q·w_KV      d_V·W_V      ] [V_{t-1}] + [ B_V·U_V   ]

    ============================================================================
    GATES (9 total)
    ============================================================================

    | Gate | Range | Role |
    |------|-------|------|
    | d_Q | 0.1-0.99 | Q self-decay |
    | d_K | 0.1-0.99 | K self-decay |
    | d_V | 0.1-0.99 | V self-decay |
    | w_KV | 0-1 | K→V attention weight |
    | B_Q | 0-1 | Q input gate |
    | B_K | 0-1 | K input gate |
    | B_V | 0-1 | V input gate |
    | C_gate | 0-1 | Output gate |

    Attributes:
        channels: Number of input/output channels
        kernel_size: Spatial kernel size for lateral connections
        rope_mode: Position encoding mode
        position_independent_gates: Compute gates from raw input for length generalization
    """
    channels: int
    kernel_size: int = 11
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    position_independent_gates: bool = False
    spectral_rho: float = 0.999
    block_size: int = 1
    # Ablation flags
    shared_kernel: bool = False      # Use 1 kernel for Q/K/V
    additive_kv: bool = False        # Additive Q→V + K→V instead of Q·K→V
    # Legacy flags (ignored, kept for API compat)
    use_goom: bool = True
    spectral_clip: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with two-stage Q→[K,V] dynamics.

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

        # Save raw input for position-independent gates
        x_raw = x if self.position_independent_gates else None

        # Optional position encoding
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # =====================================================================
        # 1. SPATIAL KERNELS — always spectral clipped
        # =====================================================================
        k_shape = (C, self.kernel_size, self.kernel_size)

        def pad_kernel(k_spatial):
            pad_h = max(0, (H - self.kernel_size) // 2)
            pad_w = max(0, (W - self.kernel_size) // 2)
            pad_h_after = H - self.kernel_size - pad_h
            pad_w_after = W - self.kernel_size - pad_w
            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k_spatial[:, start_h:start_h+H, start_w:start_w+W]
            else:
                return jnp.pad(k_spatial, ((0, 0), (pad_h, max(0, pad_h_after)),
                                           (pad_w, max(0, pad_w_after))), mode='constant')

        def kernel_to_spectral(kernel):
            """FFT kernel with spectral clipping (always on)."""
            k_padded = pad_kernel(kernel)
            k_hat = jnp.fft.rfft2(k_padded, axes=(1, 2))
            return _stable_spectral_magnitude(k_hat, rho=self.spectral_rho)

        if self.shared_kernel:
            kernel_shared = self.param('kernel', nn.initializers.xavier_normal(), k_shape)
            K_hat_shared = kernel_to_spectral(kernel_shared)
            K_hat_Q = K_hat_shared
            K_hat_K = K_hat_shared
            K_hat_V = K_hat_shared
        else:
            kernel_q = self.param('kernel_q', nn.initializers.xavier_normal(), k_shape)
            kernel_k = self.param('kernel_k', nn.initializers.xavier_normal(), k_shape)
            kernel_v = self.param('kernel_v', nn.initializers.xavier_normal(), k_shape)
            K_hat_Q = kernel_to_spectral(kernel_q)  # (C, H, W_freq)
            K_hat_K = kernel_to_spectral(kernel_k)
            K_hat_V = kernel_to_spectral(kernel_v)

        # Broadcast kernels: (C, H, W_freq) -> (1, 1, C, H, W_freq)
        K_b_Q = K_hat_Q[None, None, ...].astype(jnp.complex64)
        K_b_K = K_hat_K[None, None, ...].astype(jnp.complex64)
        K_b_V = K_hat_V[None, None, ...].astype(jnp.complex64)

        # =====================================================================
        # 2. PROJECT INPUT TO Q, K, V PATHWAYS
        # =====================================================================
        x_flat = x.reshape(B * T, H, W, C)
        qkv_proj = nn.Dense(3 * C, name='qkv_proj')(x_flat)
        qkv_proj = qkv_proj.reshape(B, T, H, W, 3 * C)

        q_in = qkv_proj[..., :C]
        k_in = qkv_proj[..., C:2*C]
        v_in = qkv_proj[..., 2*C:]

        # Spectral domain (linear space, NOT log)
        U_Q_hat = jnp.fft.rfft2(q_in.transpose(0, 1, 4, 2, 3), axes=(3, 4))
        U_K_hat = jnp.fft.rfft2(k_in.transpose(0, 1, 4, 2, 3), axes=(3, 4))
        U_V_hat = jnp.fft.rfft2(v_in.transpose(0, 1, 4, 2, 3), axes=(3, 4))

        # =====================================================================
        # 3. INPUT-DEPENDENT GATES
        # =====================================================================
        gate_input = x_raw if x_raw is not None else x
        ctx = gate_input.mean(axis=(2, 3))
        if x_raw is None:
            ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)
        n_gate = H * W_freq

        d_Q = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='d_Q')(ctx))
        d_K = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='d_K')(ctx))
        d_V = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='d_V')(ctx))

        if self.additive_kv:
            d_Q_to_V = nn.sigmoid(nn.Dense(n_gate, name='d_Q_to_V')(ctx))
            d_K_to_V = nn.sigmoid(nn.Dense(n_gate, name='d_K_to_V')(ctx))
            d_Q_to_V = d_Q_to_V.reshape(B, T, 1, H, W_freq)
            d_K_to_V = d_K_to_V.reshape(B, T, 1, H, W_freq)
            w_KV = None
        else:
            w_KV = nn.sigmoid(nn.Dense(n_gate, name='w_KV')(ctx))
            w_KV = w_KV.reshape(B, T, 1, H, W_freq)
            d_Q_to_V = None
            d_K_to_V = None

        # I/O gates
        B_Q = nn.sigmoid(nn.Dense(n_gate, name='B_Q')(ctx))
        B_K = nn.sigmoid(nn.Dense(n_gate, name='B_K')(ctx))
        B_V = nn.sigmoid(nn.Dense(n_gate, name='B_V')(ctx))
        C_gate = nn.sigmoid(nn.Dense(n_gate, name='C_gate')(ctx))

        # Reshape for broadcasting: (B, T, 1, H, W_freq)
        d_Q = d_Q.reshape(B, T, 1, H, W_freq)
        d_K = d_K.reshape(B, T, 1, H, W_freq)
        d_V = d_V.reshape(B, T, 1, H, W_freq)
        B_Q = B_Q.reshape(B, T, 1, H, W_freq)
        B_K = B_K.reshape(B, T, 1, H, W_freq)
        B_V = B_V.reshape(B, T, 1, H, W_freq)
        C_gate = C_gate.reshape(B, T, 1, H, W_freq)

        # =====================================================================
        # 4. STAGE 1: Q SCAN (scalar, linear space → GOOM)
        # =====================================================================
        # Q_t = d_Q · W_Q · Q_{t-1} + B_Q · U_Q
        d_Q_c = d_Q.astype(jnp.complex64)
        A_Q = d_Q_c * K_b_Q              # (B, T, C, H, W_freq) linear space
        U_Q_gated = U_Q_hat * B_Q        # Input with gate

        A_Q_log = to_goom(A_Q)
        U_Q_log = to_goom(U_Q_gated)

        _, Q_log = jax.lax.associative_scan(
            cssm_scalar_scan_op, (A_Q_log, U_Q_log), axis=1
        )
        Q_hat = from_goom(Q_log)  # (B, T, C, H, W_freq) back to linear

        # =====================================================================
        # 5. STAGE 2: [K, V] TRANSITION MATRIX (linear space, then to_goom)
        # =====================================================================
        #        K              V
        # K [ d_K·Q·W_K        0       ]   K evolves, gated by Q
        # V [ Q·w_KV       d_V·W_V     ]   V accumulates K, weighted by Q

        d_K_c = d_K.astype(jnp.complex64)
        d_V_c = d_V.astype(jnp.complex64)
        zeros = jnp.zeros_like(d_K_c * K_b_K)

        diag_K = d_K_c * Q_hat * K_b_K          # K self-decay × Q gating × K kernel
        diag_V = d_V_c * K_b_V                  # V self-decay × V kernel

        if self.additive_kv:
            K_to_V = d_K_to_V.astype(jnp.complex64) * jnp.ones_like(K_b_K)
        else:
            K_to_V = Q_hat * w_KV               # Q-weighted K→V

        row_K = jnp.stack([diag_K, zeros], axis=-1)
        row_V = jnp.stack([K_to_V, diag_V], axis=-1)
        K_mat = jnp.stack([row_K, row_V], axis=-2)

        # Input vector with B gates
        U_K_gated = U_K_hat * Q_hat * B_K       # K input gated by Q and B_K
        if self.additive_kv:
            U_V_gated = U_V_hat * B_V + d_Q_to_V * Q_hat
        else:
            U_V_gated = U_V_hat * B_V
        U_vec = jnp.stack([U_K_gated, U_V_gated], axis=-1)

        # GOOM scan
        K_log = to_goom(K_mat)
        U_log = to_goom(U_vec)

        _, State_log = jax.lax.associative_scan(
            cssm_matrix_scan_op, (K_log, U_log), axis=1
        )

        # Convert back via GOOM (no LayerNorm)
        KV_hat = from_goom(State_log)  # (B, T, C, H, W_freq, 2)
        KV_hat_gated = KV_hat * C_gate[..., None]

        # =====================================================================
        # 6. CONVERT TO SPATIAL AND OUTPUT
        # =====================================================================
        V_hat = KV_hat_gated[..., 1]  # (B, T, C, H, W_freq)
        V_spatial = jnp.fft.irfft2(V_hat, s=(H, W), axes=(3, 4))

        output = V_spatial.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        return nn.Dense(C, name='output_proj')(output)


class MambaGrowingTransformerCSSM(nn.Module):
    """
    Mamba-style Growing Attention Transformer with 3D conv Q and [K,V] triangular scan.

    ============================================================================
    KEY FEATURES
    ============================================================================

    1. **Mamba-style Q**: Q computed via causal 3D convolution (not scan)
    2. **Single [K,V] triangular scan**: Only one scan needed (most efficient)
    3. **Growing attention**: V accumulates Q·K weighted signal over time
    4. **Hybrid**: Local Q (conv) + global K,V (scan)
    5. **GOOM**: Uses to_goom/from_goom with custom VJP for gradient stability
    6. **Always spectral clip**: Kernels bounded by rho for dynamic stability
    7. **No LayerNorm**: Direct from_goom output (confirmed better)

    ============================================================================
    DYNAMICS
    ============================================================================

    **Q Computation (Causal 3D Conv)**

        Q = GELU(Conv3D(input, kernel_size=(q_temporal, q_spatial, q_spatial)))

    The 3D conv is causal in time (only looks at current + past frames).

    **[K, V] Triangular Scan**

        [K_t]   [ d_K·Q·W_K       0        ] [K_{t-1}]   [ B_K·Q·U_K ]
        [V_t] = [ Q·w_KV      d_V·W_V      ] [V_{t-1}] + [ B_V·U_V   ]

    ============================================================================
    GATES (6 total)
    ============================================================================

    | Gate | Range | Role |
    |------|-------|------|
    | d_K | 0.1-0.99 | K self-decay |
    | d_V | 0.1-0.99 | V self-decay |
    | w_KV | 0-1 | K→V attention weight |
    | B_K | 0-1 | K input gate |
    | B_V | 0-1 | V input gate |
    | C_gate | 0-1 | Output gate |

    Attributes:
        channels: Number of input/output channels
        kernel_size: Spatial kernel size for K,V lateral connections
        q_temporal: Temporal extent of Q conv (default 3)
        q_spatial: Spatial extent of Q conv (default 5)
        rope_mode: Position encoding mode
        position_independent_gates: Compute gates from raw input for length generalization
    """
    channels: int
    kernel_size: int = 11
    q_temporal: int = 3   # Temporal extent of Q conv
    q_spatial: int = 5    # Spatial extent of Q conv
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    position_independent_gates: bool = False
    spectral_rho: float = 0.999
    block_size: int = 1
    # Ablation flags (kept for backward compat but use_goom/spectral_clip always on)
    shared_kernel: bool = False      # Use 1 kernel for K/V
    additive_kv: bool = False        # Additive Q→V + K→V instead of Q·K→V
    # Legacy flags (ignored, kept for API compat)
    use_goom: bool = True
    spectral_clip: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with Mamba-style Q conv and [K,V] triangular scan.

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

        # Save raw input for position-independent gates
        x_raw = x if self.position_independent_gates else None

        # Optional position encoding
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # =====================================================================
        # 1. Q COMPUTATION VIA CAUSAL 3D CONV (Mamba-style)
        # =====================================================================
        causal_pad_t = self.q_temporal - 1
        spatial_pad = self.q_spatial // 2

        x_padded = jnp.pad(
            x,
            ((0, 0),                          # batch
             (causal_pad_t, 0),               # time: causal (only past)
             (spatial_pad, spatial_pad),      # height: symmetric
             (spatial_pad, spatial_pad),      # width: symmetric
             (0, 0)),                         # channels
            mode='constant'
        )

        q_kernel = self.param(
            'q_conv_kernel',
            nn.initializers.xavier_normal(),
            (self.q_temporal, self.q_spatial, self.q_spatial, C, C)
        )

        Q_spatial = jax.lax.conv_general_dilated(
            x_padded, q_kernel,
            window_strides=(1, 1, 1),
            padding='VALID',
            dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'),
        )
        Q_spatial = jax.nn.gelu(Q_spatial)  # (B, T, H, W, C)

        # Q in spectral domain (linear space)
        Q_hat = jnp.fft.rfft2(Q_spatial.transpose(0, 1, 4, 2, 3), axes=(3, 4))

        # =====================================================================
        # 2. SPATIAL KERNELS (W_K, W_V) — always spectral clipped
        # =====================================================================
        k_shape = (C, self.kernel_size, self.kernel_size)

        def pad_kernel(k_spatial):
            pad_h = max(0, (H - self.kernel_size) // 2)
            pad_w = max(0, (W - self.kernel_size) // 2)
            pad_h_after = H - self.kernel_size - pad_h
            pad_w_after = W - self.kernel_size - pad_w
            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k_spatial[:, start_h:start_h+H, start_w:start_w+W]
            else:
                return jnp.pad(k_spatial, ((0, 0), (pad_h, max(0, pad_h_after)),
                                           (pad_w, max(0, pad_w_after))), mode='constant')

        def kernel_to_spectral(kernel):
            """FFT kernel with spectral clipping (always on)."""
            k_padded = pad_kernel(kernel)
            k_hat = jnp.fft.rfft2(k_padded, axes=(1, 2))
            return _stable_spectral_magnitude(k_hat, rho=self.spectral_rho)

        if self.shared_kernel:
            kernel_shared = self.param('kernel', nn.initializers.xavier_normal(), k_shape)
            K_hat_K = kernel_to_spectral(kernel_shared)
            K_hat_V = K_hat_K
        else:
            kernel_k = self.param('kernel_k', nn.initializers.xavier_normal(), k_shape)
            kernel_v = self.param('kernel_v', nn.initializers.xavier_normal(), k_shape)
            K_hat_K = kernel_to_spectral(kernel_k)  # (C, H, W_freq)
            K_hat_V = kernel_to_spectral(kernel_v)

        # Broadcast kernels: (C, H, W_freq) -> (1, 1, C, H, W_freq)
        K_b_K = K_hat_K[None, None, ...].astype(jnp.complex64)
        K_b_V = K_hat_V[None, None, ...].astype(jnp.complex64)

        # =====================================================================
        # 3. PROJECT INPUT TO K, V PATHWAYS (Q already computed)
        # =====================================================================
        x_flat = x.reshape(B * T, H, W, C)
        kv_proj = nn.Dense(2 * C, name='kv_proj')(x_flat)
        kv_proj = kv_proj.reshape(B, T, H, W, 2 * C)

        k_in = kv_proj[..., :C]
        v_in = kv_proj[..., C:]

        # Spectral domain (linear space, NOT log)
        U_K_hat = jnp.fft.rfft2(k_in.transpose(0, 1, 4, 2, 3), axes=(3, 4))
        U_V_hat = jnp.fft.rfft2(v_in.transpose(0, 1, 4, 2, 3), axes=(3, 4))

        # =====================================================================
        # 4. INPUT-DEPENDENT GATES
        # =====================================================================
        gate_input = x_raw if x_raw is not None else x
        ctx = gate_input.mean(axis=(2, 3))
        if x_raw is None:
            ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)
        n_gate = H * W_freq

        d_K = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='d_K')(ctx))
        d_V = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='d_V')(ctx))

        if self.additive_kv:
            d_Q_to_V = nn.sigmoid(nn.Dense(n_gate, name='d_Q_to_V')(ctx))
            d_K_to_V = nn.sigmoid(nn.Dense(n_gate, name='d_K_to_V')(ctx))
            d_Q_to_V = d_Q_to_V.reshape(B, T, 1, H, W_freq)
            d_K_to_V = d_K_to_V.reshape(B, T, 1, H, W_freq)
            w_KV = None
        else:
            w_KV = nn.sigmoid(nn.Dense(n_gate, name='w_KV')(ctx))
            w_KV = w_KV.reshape(B, T, 1, H, W_freq)
            d_Q_to_V = None
            d_K_to_V = None

        # I/O gates
        B_K = nn.sigmoid(nn.Dense(n_gate, name='B_K')(ctx))
        B_V = nn.sigmoid(nn.Dense(n_gate, name='B_V')(ctx))
        C_gate = nn.sigmoid(nn.Dense(n_gate, name='C_gate')(ctx))

        # Reshape for broadcasting: (B, T, 1, H, W_freq)
        d_K = d_K.reshape(B, T, 1, H, W_freq)
        d_V = d_V.reshape(B, T, 1, H, W_freq)
        B_K = B_K.reshape(B, T, 1, H, W_freq)
        B_V = B_V.reshape(B, T, 1, H, W_freq)
        C_gate = C_gate.reshape(B, T, 1, H, W_freq)

        # =====================================================================
        # 5. BUILD [K, V] TRANSITION MATRIX (linear space, then to_goom)
        # =====================================================================
        #        K              V
        # K [ d_K·Q·W_K        0       ]   K evolves, gated by Q
        # V [ Q·w_KV       d_V·W_V     ]   V accumulates K, weighted by Q

        d_K_c = d_K.astype(jnp.complex64)
        d_V_c = d_V.astype(jnp.complex64)
        zeros = jnp.zeros_like(d_K_c * K_b_K)

        # Diagonal entries (linear space)
        diag_K = d_K_c * Q_hat * K_b_K          # K self-decay × Q gating × K kernel
        diag_V = d_V_c * K_b_V                  # V self-decay × V kernel

        # Off-diagonal: K→V flow
        if self.additive_kv:
            K_to_V = d_K_to_V.astype(jnp.complex64) * jnp.ones_like(K_b_K)
        else:
            K_to_V = Q_hat * w_KV               # Q-weighted K→V (attention-like)

        row_K = jnp.stack([diag_K, zeros], axis=-1)
        row_V = jnp.stack([K_to_V, diag_V], axis=-1)
        K_mat = jnp.stack([row_K, row_V], axis=-2)

        # Input vector with B gates
        U_K_gated = U_K_hat * Q_hat * B_K       # K input gated by Q and B_K
        if self.additive_kv:
            U_V_gated = U_V_hat * B_V + d_Q_to_V * Q_hat  # V input + Q→V term
        else:
            U_V_gated = U_V_hat * B_V
        U_vec = jnp.stack([U_K_gated, U_V_gated], axis=-1)

        # GOOM scan
        K_log = to_goom(K_mat)
        U_log = to_goom(U_vec)

        _, State_log = jax.lax.associative_scan(
            cssm_matrix_scan_op, (K_log, U_log), axis=1
        )

        # Convert back via GOOM (no LayerNorm)
        KV_hat = from_goom(State_log)  # (B, T, C, H, W_freq, 2)
        KV_hat_gated = KV_hat * C_gate[..., None]

        # =====================================================================
        # 6. CONVERT TO SPATIAL AND OUTPUT
        # =====================================================================
        V_hat = KV_hat_gated[..., 1]  # (B, T, C, H, W_freq)
        V_spatial = jnp.fft.irfft2(V_hat, s=(H, W), axes=(3, 4))

        output = V_spatial.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        return nn.Dense(C, name='output_proj')(output)


class SpectralTransformerCSSM(nn.Module):
    """
    Spectral Transformer with correct spatial Q gating via spectral convolution.

    ============================================================================
    KEY DIFFERENCE FROM g_transformer
    ============================================================================

    **g_transformer (approximate):**
        Computes Q_hat × W_K_hat × K_hat (element-wise in spectral)
        = Q * W_K * K (cascade of convolutions in spatial)

    **spectral_transformer (correct):**
        Computes Q ⊙ (W_K * K) (element-wise Q gating AFTER convolution)
        Requires: spectral_conv(Q_hat, W_K_hat × K_hat)
                = FFT(IFFT(Q_hat) × IFFT(W_K_hat × K_hat))

    ============================================================================
    MATHEMATICAL BACKGROUND
    ============================================================================

    The convolution theorem states:
    - FFT(a * b) = FFT(a) ⊙ FFT(b)  [convolution → element-wise]
    - FFT(a ⊙ b) = FFT(a) ⊛ FFT(b)  [element-wise → convolution]

    So when we want Q ⊙ (W_K * K) in spatial domain:
    1. Compute W_K * K via spectral: W_K_hat ⊙ K_hat
    2. IFFT to spatial: (W_K * K)_spatial
    3. Multiply by Q_spatial element-wise
    4. FFT back to spectral (if needed for next step)

    ============================================================================
    TRADEOFF
    ============================================================================

    | Approach | Parallelism | What it computes |
    |----------|-------------|------------------|
    | g_transformer | O(log T) associative | Q * W_K * K (convolutions) |
    | spectral_transformer | O(T) sequential | Q ⊙ (W_K * K) (correct gating) |

    The spectral_transformer uses a sequential scan (jax.lax.scan) instead of
    associative_scan because the spatial gating operation doesn't compose
    associatively in log-space.

    ============================================================================
    DYNAMICS
    ============================================================================

    **Stage 1: Q Scan (same as g_transformer)**
        Q_t = d_Q × W_Q * Q_{t-1} + U_Q
        (Uses associative scan in spectral space)

    **Stage 2: [K, V] Sequential Scan with Spatial Q Gating**
        K_spatial_t = Q_t ⊙ (d_K × W_K * K_{t-1}) + Q_t ⊙ U_K
        V_spatial_t = Q_t ⊙ w_KV ⊙ K_t + d_V × W_V * V_{t-1} + U_V

        Each step:
        1. Apply decay and kernel in spectral: d × W_hat × state_hat
        2. IFFT to spatial
        3. Element-wise multiply by Q_spatial
        4. Add input contribution
        5. FFT back to spectral for next iteration

    Attributes:
        channels: Number of input/output channels
        kernel_size: Spatial kernel size for lateral connections
        use_goom: If True, use complex log for Q scan (handles negative values)
        rope_mode: Position encoding mode
        position_independent_gates: Compute gates from raw input
        spectral_rho: Spectral magnitude clipping threshold
    """
    channels: int
    kernel_size: int = 11
    use_goom: bool = True
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    position_independent_gates: bool = False
    spectral_rho: float = 0.999
    block_size: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with correct spatial Q gating.

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

        # Save raw input for position-independent gates
        x_raw = x if self.position_independent_gates else None

        # Optional position encoding
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # =====================================================================
        # 1. SPATIAL KERNELS (W_Q, W_K, W_V) with spectral clipping
        # =====================================================================
        k_shape = (C, self.kernel_size, self.kernel_size)
        kernel_q = self.param('kernel_q', nn.initializers.xavier_normal(), k_shape)
        kernel_k = self.param('kernel_k', nn.initializers.xavier_normal(), k_shape)
        kernel_v = self.param('kernel_v', nn.initializers.xavier_normal(), k_shape)

        def pad_kernel(k_spatial):
            """Pad kernel to input size for FFT."""
            pad_h = max(0, (H - self.kernel_size) // 2)
            pad_w = max(0, (W - self.kernel_size) // 2)
            pad_h_after = H - self.kernel_size - pad_h
            pad_w_after = W - self.kernel_size - pad_w

            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k_spatial[:, start_h:start_h+H, start_w:start_w+W]
            else:
                return jnp.pad(k_spatial, ((0, 0), (pad_h, max(0, pad_h_after)),
                                           (pad_w, max(0, pad_w_after))), mode='constant')

        # Get spectral kernels with stability clipping
        W_Q_hat = _stable_spectral_magnitude(
            jnp.fft.rfft2(pad_kernel(kernel_q), axes=(1, 2)), rho=self.spectral_rho
        )  # (C, H, W_freq)
        W_K_hat = _stable_spectral_magnitude(
            jnp.fft.rfft2(pad_kernel(kernel_k), axes=(1, 2)), rho=self.spectral_rho
        )
        W_V_hat = _stable_spectral_magnitude(
            jnp.fft.rfft2(pad_kernel(kernel_v), axes=(1, 2)), rho=self.spectral_rho
        )

        # =====================================================================
        # 2. PROJECT INPUT TO Q, K, V PATHWAYS
        # =====================================================================
        x_flat = x.reshape(B * T, H, W, C)
        qkv_proj = nn.Dense(3 * C, name='qkv_proj')(x_flat)
        qkv_proj = qkv_proj.reshape(B, T, H, W, 3 * C)

        q_in = qkv_proj[..., :C]
        k_in = qkv_proj[..., C:2*C]
        v_in = qkv_proj[..., 2*C:]

        # Convert to spectral for Q scan
        def to_spectral(y):
            """Convert spatial to spectral."""
            y_perm = y.transpose(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            return jnp.fft.rfft2(y_perm, axes=(3, 4))  # (B, T, C, H, W_freq)

        def to_log_spectral(y):
            """Convert spatial to log-spectral (for GOOM)."""
            y_hat = to_spectral(y)
            if self.use_goom:
                return jnp.log(jnp.abs(y_hat) + 1e-8) + 1j * jnp.angle(y_hat)
            else:
                return jnp.log(jnp.abs(y_hat) + 1e-8)

        log_U_Q = to_log_spectral(q_in)  # (B, T, C, H, W_freq)
        U_K = to_spectral(k_in)  # Keep in linear spectral for sequential scan
        U_V = to_spectral(v_in)

        # =====================================================================
        # 3. INPUT-DEPENDENT GATES
        # =====================================================================
        gate_input = x_raw if x_raw is not None else x
        ctx = gate_input.mean(axis=(2, 3))  # (B, T, C)
        if x_raw is None:
            ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)
        n_gate = H * W_freq

        # Self-decay exponents (bounded 0.1-0.99 for stability)
        d_Q = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='d_Q')(ctx))
        d_K = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='d_K')(ctx))
        d_V = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='d_V')(ctx))

        # K→V attention weight (0-1)
        w_KV = nn.sigmoid(nn.Dense(n_gate, name='w_KV')(ctx))

        # Reshape for broadcasting: (B, T, 1, H, W_freq)
        d_Q = d_Q.reshape(B, T, 1, H, W_freq)
        d_K = d_K.reshape(B, T, 1, H, W_freq)
        d_V = d_V.reshape(B, T, 1, H, W_freq)
        w_KV = w_KV.reshape(B, T, 1, H, W_freq)

        # =====================================================================
        # 4. STAGE 1: Q SCAN (Associative scan in log-spectral space)
        # =====================================================================
        # Q_t = d_Q × W_Q * Q_{t-1} + U_Q
        # This is the same as g_transformer - Q evolves independently

        dtype = jnp.complex64 if self.use_goom else jnp.float32

        # Convert kernel to log-spectral
        if self.use_goom:
            log_W_Q = jnp.log(jnp.abs(W_Q_hat) + 1e-8) + 1j * jnp.angle(W_Q_hat)
        else:
            log_W_Q = jnp.log(jnp.abs(W_Q_hat) + 1e-8)

        log_W_Q_exp = log_W_Q[None, None, :, :, :]  # (1, 1, C, H, W_freq)

        # Transition: log(decay × kernel)
        log_A_Q = jnp.log(d_Q + 1e-8) + log_W_Q_exp
        if self.use_goom:
            log_A_Q = log_A_Q.astype(jnp.complex64)

        # Run Q scan (associative)
        _, log_Q = jax.lax.associative_scan(
            cssm_scalar_scan_op, (log_A_Q, log_U_Q), axis=1
        )
        # log_Q shape: (B, T, C, H, W_freq)

        # Convert Q to spatial for gating
        if self.use_goom:
            Q_hat = jnp.exp(log_Q.real) * jnp.exp(1j * log_Q.imag)
        else:
            Q_hat = jnp.exp(log_Q)
        Q_spatial = jnp.fft.irfft2(Q_hat, s=(H, W), axes=(3, 4))  # (B, T, C, H, W)

        # =====================================================================
        # 5. STAGE 2: [K, V] SEQUENTIAL SCAN with SPATIAL Q GATING
        # =====================================================================
        # This is the key difference from g_transformer!
        # We use jax.lax.scan (sequential) instead of associative_scan
        # because spatial gating doesn't compose associatively.

        # Expand kernels: (C, H, W_freq) -> (1, C, H, W_freq)
        W_K_hat_exp = W_K_hat[None, :, :, :]
        W_V_hat_exp = W_V_hat[None, :, :, :]

        def sequential_step(carry, inputs):
            """
            One step of the [K, V] sequential recurrence.

            Dynamics:
                K_spatial_t = Q_t ⊙ (d_K × W_K * K_{t-1}) + Q_t ⊙ U_K_spatial
                V_spatial_t = Q_t ⊙ w_KV ⊙ K_t + d_V × W_V * V_{t-1} + U_V_spatial

            We keep K, V in spectral space between steps for efficiency,
            only converting to spatial for the Q gating operation.
            """
            K_hat_prev, V_hat_prev = carry  # (B, C, H, W_freq)
            Q_t, d_K_t, d_V_t, w_KV_t, U_K_t, U_V_t = inputs
            # Q_t: (B, C, H, W) spatial
            # d_K_t, d_V_t, w_KV_t: (B, 1, H, W_freq)
            # U_K_t, U_V_t: (B, C, H, W_freq) spectral

            # --- K update ---
            # 1. Apply decay and kernel in spectral: d_K × W_K × K_prev
            K_decay_hat = d_K_t * W_K_hat_exp * K_hat_prev  # (B, C, H, W_freq)

            # 2. IFFT to spatial for Q gating
            K_decay_spatial = jnp.fft.irfft2(K_decay_hat, s=(H, W), axes=(2, 3))  # (B, C, H, W)
            U_K_spatial = jnp.fft.irfft2(U_K_t, s=(H, W), axes=(2, 3))

            # 3. Element-wise Q gating in spatial (THE CORRECT OPERATION!)
            K_new_spatial = Q_t * K_decay_spatial + Q_t * U_K_spatial  # (B, C, H, W)

            # 4. FFT back to spectral for next iteration
            K_hat_new = jnp.fft.rfft2(K_new_spatial, axes=(2, 3))  # (B, C, H, W_freq)

            # --- V update ---
            # V gets Q-gated K contribution plus its own decay
            # 1. V decay in spectral
            V_decay_hat = d_V_t * W_V_hat_exp * V_hat_prev

            # 2. IFFT for Q gating of K→V contribution
            V_decay_spatial = jnp.fft.irfft2(V_decay_hat, s=(H, W), axes=(2, 3))
            U_V_spatial = jnp.fft.irfft2(U_V_t, s=(H, W), axes=(2, 3))

            # 3. K→V flow with Q gating in spatial
            # w_KV is in spectral, so we need to handle it carefully
            # For simplicity, treat w_KV as a scalar-like gate (average over spatial)
            w_KV_spatial = jnp.fft.irfft2(
                jnp.broadcast_to(w_KV_t, (B, C, H, W_freq)), s=(H, W), axes=(2, 3)
            )
            K_to_V_spatial = Q_t * w_KV_spatial * K_new_spatial

            # 4. Combine: V = K→V + decay + input
            V_new_spatial = K_to_V_spatial + V_decay_spatial + U_V_spatial

            # 5. FFT back to spectral
            V_hat_new = jnp.fft.rfft2(V_new_spatial, axes=(2, 3))

            return (K_hat_new, V_hat_new), (K_new_spatial, V_new_spatial)

        # Prepare inputs for scan: transpose T to first axis
        # Q_spatial: (B, T, C, H, W) -> (T, B, C, H, W)
        Q_scan = Q_spatial.transpose(1, 0, 2, 3, 4)
        d_K_scan = d_K.transpose(1, 0, 2, 3, 4)  # (T, B, 1, H, W_freq)
        d_V_scan = d_V.transpose(1, 0, 2, 3, 4)
        w_KV_scan = w_KV.transpose(1, 0, 2, 3, 4)
        U_K_scan = U_K.transpose(1, 0, 2, 3, 4)  # (T, B, C, H, W_freq)
        U_V_scan = U_V.transpose(1, 0, 2, 3, 4)

        # Initial states (zeros in spectral space)
        K_hat_init = jnp.zeros((B, C, H, W_freq), dtype=jnp.complex64)
        V_hat_init = jnp.zeros((B, C, H, W_freq), dtype=jnp.complex64)

        # Run sequential scan
        _, (K_spatial_all, V_spatial_all) = jax.lax.scan(
            sequential_step,
            (K_hat_init, V_hat_init),
            (Q_scan, d_K_scan, d_V_scan, w_KV_scan, U_K_scan, U_V_scan),
        )
        # K_spatial_all, V_spatial_all: (T, B, C, H, W)

        # Transpose back: (T, B, C, H, W) -> (B, T, C, H, W)
        V_spatial = V_spatial_all.transpose(1, 0, 2, 3, 4)

        # =====================================================================
        # 6. OUTPUT
        # =====================================================================
        # LayerNorm on V
        ln_gamma = self.param('ln_gamma', nn.initializers.ones, (C, 1, 1))
        ln_beta = self.param('ln_beta', nn.initializers.zeros, (C, 1, 1))

        # Normalize over spatial dims
        mean = V_spatial.mean(axis=(-2, -1), keepdims=True)
        std = V_spatial.std(axis=(-2, -1), keepdims=True) + 1e-6
        V_norm = (V_spatial - mean) / std
        V_scaled = ln_gamma[None, None, :, :, :] * V_norm + ln_beta[None, None, :, :, :]

        # Transpose and project: (B, T, C, H, W) -> (B, T, H, W, C)
        output = V_scaled.transpose(0, 1, 3, 4, 2)
        return nn.Dense(C, name='output_proj')(output)


class AdditiveCSSM(nn.Module):
    """
    Additive K/Q/V CSSM with triangular 3×3 scan.

    ============================================================================
    KEY FEATURES
    ============================================================================

    1. **Single shared kernel**: One spatial kernel W for Q and K spreading
    2. **Unidirectional Q→K coupling**: K sees Q via w_kq, but Q doesn't see K
    3. **Additive Q+K→V**: V accumulates Q and K independently (not Q·K)
    4. **Scalar V decay**: V has no spatial kernel, just decay

    ============================================================================
    DYNAMICS
    ============================================================================

    Q_t = (d_Q ⊙ W) * Q_{t-1} + U_Q           (Q evolves independently)
    K_t = (d_K ⊙ W) * K_{t-1} + (w_kq ⊙ W) * Q_{t-1} + U_K   (K sees Q)
    V_t = d_V · V_{t-1} + γ · Q_{t-1} + γ · K_{t-1} + U_V    (V accumulates Q+K)

    In log-space transition matrix (triangular for associativity):

               Q                  K                  V
    Q  [ log(d_Q⊙Ŵ)              -∞                -∞   ]
    K  [ log(w_kq⊙Ŵ)       log(d_K⊙Ŵ)              -∞   ]
    V  [   log γ             log γ            log d_V   ]

    ============================================================================
    KEY DIFFERENCES FROM TransformerCSSM
    ============================================================================

    | Feature | TransformerCSSM | AdditiveCSSM |
    |---------|-----------------|--------------|
    | States | Q, K, A | Q, K, V |
    | Q↔K coupling | Symmetric (w_qk both ways) | Unidirectional (w_kq only) |
    | Feedback | A→Q (recurrent loop) | None (strictly triangular) |
    | V/A kernel | Shared kernel K | Scalar only (d_V) |
    | Scan type | 3×3 coupled | 3×3 triangular |

    The simpler triangular structure may be easier to optimize while still
    capturing Q→K→V information flow.

    ============================================================================
    GATES (9 total)
    ============================================================================

    | Gate | Range | Role |
    |------|-------|------|
    | d_Q | 0.1-0.99 | Q self-decay |
    | d_K | 0.1-0.99 | K self-decay |
    | d_V | 0.1-0.99 | V self-decay (scalar, no spatial kernel) |
    | w_kq | 0-1 | Q→K coupling weight |
    | gamma | 0-1 | Q,K→V accumulation weight |
    | B_Q | 0-1 | Q input gate |
    | B_K | 0-1 | K input gate |
    | B_V | 0-1 | V input gate |
    | C_gate | 0-1 | Output gate |

    Uses GOOM (Generalized Order of Magnitude) log-space with custom VJP
    for gradient-stable computation. No LayerNorm in output path.

    Attributes:
        channels: Number of input/output channels
        kernel_size: Spatial kernel size for Q, K lateral connections
        rope_mode: Position encoding mode
        rope_base: Base for RoPE frequency computation
        spectral_rho: Maximum spectral magnitude for stability (used if spectral_clip=True)
        position_independent_gates: Compute gates from raw input for length generalization
        block_size: Unused, for API compatibility
    """
    channels: int
    kernel_size: int = 11
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    spectral_rho: float = 0.999
    position_independent_gates: bool = False
    block_size: int = 1
    no_k_state: bool = False         # Drop K → 2×2 Q+V scan (add_kqv_2)
    single_state: bool = False       # 1-state scalar scan (add_kqv_1)
    use_goom: bool = True            # Use GOOM custom VJP; False = standard jnp.log/exp
    gate_type: str = 'dense'         # 'dense' (per-frequency), 'channel' (per-channel), 'scalar' (single value)
    learned_init: bool = False       # Learn initial state for the recurrence (vs starting from zero)
    use_complex32: bool = False      # Linear-split scan: bf16 real + bf16 imag (skips GOOM entirely)
    use_complex16: bool = False      # Linear-split scan: fp8 real + fp8 imag (half the memory of complex32)
    use_ssd: bool = False            # Use SSD chunked scan instead of associative scan (legacy, use scan_mode)
    scan_mode: str = 'associative'   # 'associative', 'ssd', or 'quadratic'
    ssd_chunk_size: int = 8          # Chunk size for SSD / chunked quadratic
    use_delta: bool = False          # Enable delta-rule enhancements (beta gate, retrieval output)
    l2_norm_qk: bool = False         # L2-normalize q_in, k_in before gating+FFT

    @nn.compact
    def __call__(self, x: jnp.ndarray, injected_qkv_spatial=None) -> jnp.ndarray:
        """
        Forward pass with triangular Q→K→V dynamics (or Q→V if no_k_state).

        Args:
            x: Input tensor of shape (B, T, H, W, C)
            injected_qkv_spatial: If provided, replaces the spatial Q/K/V inputs
                before gating+FFT. Shape (B, T, H, W, C, n_states) real-valued.
                Gradients flow through gate+FFT+scan+readout with no spectral artifacts.

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

        # Resolve scan mode (legacy use_ssd flag takes priority if set)
        _scan_mode = self.scan_mode
        if self.use_ssd and _scan_mode == 'associative':
            _scan_mode = 'ssd'

        # Log-space conversion helpers (GOOM or standard)
        if self.use_goom:
            _to_log = to_goom
            _from_log = from_goom
        else:
            def _to_log(x):
                x_c = x.astype(jnp.complex64) if not jnp.iscomplexobj(x) else x
                return jnp.log(jnp.abs(x_c) + 1e-8) + 1j * jnp.angle(x_c)
            def _from_log(x):
                return jnp.exp(x)

        # Save raw input for position-independent gates
        x_raw = x if self.position_independent_gates else None

        # Optional position encoding
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # =====================================================================
        # 1. SINGLE SHARED SPATIAL KERNEL
        # =====================================================================
        k_shape = (C, self.kernel_size, self.kernel_size)

        def pad_kernel(k_spatial):
            """Pad kernel to input size for FFT."""
            pad_h = max(0, (H - self.kernel_size) // 2)
            pad_w = max(0, (W - self.kernel_size) // 2)
            pad_h_after = H - self.kernel_size - pad_h
            pad_w_after = W - self.kernel_size - pad_w

            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k_spatial[:, start_h:start_h+H, start_w:start_w+W]
            else:
                return jnp.pad(k_spatial, ((0, 0), (pad_h, max(0, pad_h_after)),
                                           (pad_w, max(0, pad_w_after))), mode='constant')

        # Single shared kernel for Q (and K if 3-state)
        kernel = self.param('kernel', nn.initializers.xavier_normal(), k_shape)
        k_padded = pad_kernel(kernel)
        k_hat = jnp.fft.rfft2(k_padded, axes=(1, 2))  # (C, H, W_freq)
        k_hat = _stable_spectral_magnitude(k_hat, rho=self.spectral_rho)

        # Broadcast kernel: (C, H, W_freq) -> (1, 1, C, H, W_freq)
        K_b = k_hat[None, None, ...].astype(jnp.complex64)
        ones = jnp.ones_like(K_b)

        # =====================================================================
        # 2. PROJECT INPUT (spatial — FFT deferred to after input gating)
        # =====================================================================
        x_flat = x.reshape(B * T, H, W, C)

        if self.single_state:
            x_proj = nn.Dense(C, name='x_proj')(x_flat).reshape(B, T, H, W, C)
        elif self.no_k_state:
            qv_proj = nn.Dense(2 * C, name='qv_proj')(x_flat).reshape(B, T, H, W, 2 * C)
            q_in = qv_proj[..., :C]
            v_in = qv_proj[..., C:]
        else:
            qkv_proj = nn.Dense(3 * C, name='qkv_proj')(x_flat).reshape(B, T, H, W, 3 * C)
            q_in = qkv_proj[..., :C]
            k_in = qkv_proj[..., C:2*C]
            v_in = qkv_proj[..., 2*C:]

        # =====================================================================
        # 3. INPUT-DEPENDENT GATES
        # =====================================================================
        gate_input = x_raw if x_raw is not None else x
        ctx = gate_input.mean(axis=(2, 3))  # (B, T, C) - spatial average for context
        if x_raw is None:
            ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)
        n_gate = H * W_freq
        # Whether I/O gates (B, C) are spatial (applied before FFT / after IFFT)
        _spatial_gates = self.gate_type in ('conv', 'channel', 'dense_decay')

        # Gate helper: creates a gate with the specified gate_type
        def _gate(name, bounded=False, spatial=False):
            """Create a gate.
            bounded=True → decay gate in [0.1, 0.99].
            spatial=True → I/O gate. For conv/channel/dense_decay: real spatial (B,T,H,W,C).
                           For dense/dense_io/scalar: spectral (same as decay).
            """
            if self.gate_type == 'conv':
                # 5×5×C×C conv for all gates
                gate_flat = gate_input.reshape(B * T, H, W, C)
                raw = nn.sigmoid(nn.Conv(
                    C, kernel_size=(5, 5), padding='SAME', name=name
                )(gate_flat))
                if spatial:
                    return raw.reshape(B, T, H, W, C)
                else:
                    raw = raw.mean(axis=(1, 2)).reshape(B, T, C, 1, 1)
                    return 0.1 + 0.89 * raw if bounded else raw
            elif self.gate_type == 'channel':
                if spatial:
                    gate_flat = gate_input.reshape(B * T, H, W, C)
                    raw = nn.sigmoid(nn.Conv(
                        C, kernel_size=(1, 1), padding='SAME', name=name
                    )(gate_flat))
                    return raw.reshape(B, T, H, W, C)
                else:
                    raw = nn.sigmoid(nn.Dense(C, name=name)(ctx))
                    raw = raw.reshape(B, T, C, 1, 1)
                    return 0.1 + 0.89 * raw if bounded else raw
            elif self.gate_type == 'dense_decay':
                # Decay: dense per-frequency | I/O: channel spatial
                if spatial:
                    gate_flat = gate_input.reshape(B * T, H, W, C)
                    raw = nn.sigmoid(nn.Conv(
                        C, kernel_size=(1, 1), padding='SAME', name=name
                    )(gate_flat))
                    return raw.reshape(B, T, H, W, C)
                else:
                    raw = nn.sigmoid(nn.Dense(n_gate, name=name)(ctx))
                    raw = raw.reshape(B, T, 1, H, W_freq)
                    return 0.1 + 0.89 * raw if bounded else raw
            elif self.gate_type == 'dense_io':
                # Decay: channel per-channel | I/O: dense per-frequency
                if spatial:
                    raw = nn.sigmoid(nn.Dense(n_gate, name=name)(ctx))
                    raw = raw.reshape(B, T, 1, H, W_freq)
                    return raw
                else:
                    raw = nn.sigmoid(nn.Dense(C, name=name)(ctx))
                    raw = raw.reshape(B, T, C, 1, 1)
                    return 0.1 + 0.89 * raw if bounded else raw
            elif self.gate_type == 'factored':
                # Separable H × W_freq outer product — 20× fewer params than dense
                row = nn.Dense(H, name=f'{name}_h')(ctx)       # (B, T, H)
                col = nn.Dense(W_freq, name=f'{name}_w')(ctx)  # (B, T, W_freq)
                raw = nn.sigmoid(row[..., :, None] + col[..., None, :])  # (B, T, H, W_freq)
                raw = raw.reshape(B, T, 1, H, W_freq)
                return 0.1 + 0.89 * raw if bounded else raw
            elif self.gate_type == 'spectral_conv':
                # 5×5 conv over input's spectral magnitude — per-channel
                gi_flat = gate_input.reshape(B * T, H, W, C)
                gi_spec = jnp.fft.rfft2(gi_flat.transpose(0, 3, 1, 2), axes=(2, 3))
                gi_mag = jnp.abs(gi_spec).transpose(0, 2, 3, 1)  # (B*T, H, W_freq, C)
                raw = nn.sigmoid(nn.Conv(
                    C, kernel_size=(5, 5), padding='SAME', name=name
                )(gi_mag))
                raw = raw.transpose(0, 3, 1, 2).reshape(B, T, C, H, W_freq)
                return 0.1 + 0.89 * raw if bounded else raw
            elif self.gate_type == 'low_rank':
                # Bottleneck Dense(C→8→n_gate) — true low-rank factorization
                hidden = nn.Dense(8, name=f'{name}_down')(ctx)
                raw = nn.sigmoid(nn.Dense(n_gate, name=f'{name}_up')(hidden))
                raw = raw.reshape(B, T, 1, H, W_freq)
                return 0.1 + 0.89 * raw if bounded else raw
            elif self.gate_type == 'scalar':
                out_dim, shape = 1, (B, T, 1, 1, 1)
            else:  # 'dense' (default)
                out_dim, shape = n_gate, (B, T, 1, H, W_freq)
            raw = nn.sigmoid(nn.Dense(out_dim, name=name)(ctx))
            raw = raw.reshape(*shape)
            return 0.1 + 0.89 * raw if bounded else raw

        def _to_spectral(spatial_in):
            """(B,T,H,W,C) → (B,T,C,H,W_freq) via rfft2."""
            return jnp.fft.rfft2(spatial_in.transpose(0, 1, 4, 2, 3), axes=(3, 4))

        def _gate_and_fft(spatial_in, b_gate):
            """Apply input gate and FFT.
            Conv: spatial gate × spatial input → FFT.
            Others: FFT → spectral gate × spectral input.
            """
            if _spatial_gates:
                return _to_spectral(spatial_in * b_gate)
            else:
                return _to_spectral(spatial_in) * b_gate

        def _prepend_init(A_log, U_log, n_states):
            """Prepend virtual timestep with identity transition and learned init.

            For the associative scan: prepending (I, init) means the scan
            output at position 1 becomes A[0] @ init + U[0], giving us a
            learned initial state. We strip position 0 from the output.
            """
            LOG_ZERO = jnp.complex64(-1e4 + 0j)
            if n_states > 1:
                eye = jnp.where(
                    jnp.eye(n_states, dtype=bool),
                    jnp.complex64(0 + 0j),
                    LOG_ZERO,
                )
                A_init = jnp.broadcast_to(eye, (B, 1, C, H, W_freq, n_states, n_states))
            else:
                # Scalar: identity = 1, log(1) = 0+0j
                A_init = jnp.zeros((B, 1, C, H, W_freq), dtype=jnp.complex64)

            init_shape = (1, 1, C, 1, 1, n_states) if n_states > 1 else (1, 1, C, 1, 1)
            init_real = self.param('init_real', nn.initializers.constant(-10.0), init_shape)
            init_imag = self.param('init_imag', nn.initializers.zeros, init_shape)
            U_init = jnp.broadcast_to(
                (init_real + 1j * init_imag).astype(jnp.complex64),
                U_log.shape[:1] + (1,) + U_log.shape[2:],
            )

            return (
                jnp.concatenate([A_init, A_log], axis=1),
                jnp.concatenate([U_init, U_log], axis=1),
            )

        if self.single_state:
            # =================================================================
            # 1-STATE PATH: single scalar scan
            # =================================================================
            # X_t = d · W · X_{t-1} + B · U
            d_X = _gate('d_X', bounded=True)
            B_X = _gate('B_X', spatial=True)
            C_gate = _gate('C_gate', spatial=True)

            # Transition: d_X · kernel (linear space)
            A_X = d_X.astype(jnp.complex64) * K_b

            # Input with gate + FFT
            U_X_hat = _gate_and_fft(x_proj, B_X)

            if _scan_mode == 'quadratic':
                # Quadratic (attention) form: fully parallel T×T matmul
                A_c = A_X.astype(jnp.complex64)
                U_c = U_X_hat.astype(jnp.complex64)
                L = self.ssd_chunk_size
                if T <= L:
                    V_hat = quadratic_scan(A_c, U_c)
                else:
                    V_hat = chunked_quadratic_scan(A_c, U_c, chunk_size=L)
            elif _scan_mode == 'ssd' or self.use_ssd:
                # SSD chunked scan: works in linear complex space
                A_c = A_X.astype(jnp.complex64)
                U_c = U_X_hat.astype(jnp.complex64)
                V_hat = ssd_scan(A_c, U_c, chunk_size=self.ssd_chunk_size)
            elif self.use_complex32 or self.use_complex16:
                # Linear-split: skip GOOM, split (re_bf16, im_bf16) directly
                # Priority over pallas: parallel associative_scan compiles faster under pmap
                A_c = A_X.astype(jnp.complex64)
                U_c = U_X_hat.astype(jnp.complex64)

                if self.learned_init:
                    # Need log-space init for prepend, then convert back
                    A_log = _to_log(A_c)
                    U_log = _to_log(U_c)
                    A_log, U_log = _prepend_init(A_log, U_log, n_states=1)
                    A_c = _from_log(A_log)
                    U_c = _from_log(U_log)

                A_re, A_im = complex64_to_linear_split(A_c, jnp.float8_e4m3fn if self.use_complex16 else jnp.bfloat16)
                U_re, U_im = complex64_to_linear_split(U_c, jnp.float8_e4m3fn if self.use_complex16 else jnp.bfloat16)
                _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                    linear_split_scalar_scan_op,
                    (A_re, A_im, U_re, U_im), axis=1)
                V_hat = linear_split_to_complex64(U_re_out, U_im_out)

                if self.learned_init:
                    V_hat = V_hat[:, 1:]
            elif _scan_mode == 'pallas':
                # Fused Pallas GPU kernel: sequential scan in registers
                from .pallas_scan import pallas_scalar_scan
                A_c = A_X.astype(jnp.complex64)
                U_c = U_X_hat.astype(jnp.complex64)
                if self.learned_init:
                    init_shape = (1, 1, C, 1, 1)
                    init_real = self.param('init_real', nn.initializers.constant(-10.0), init_shape)
                    init_imag = self.param('init_imag', nn.initializers.zeros, init_shape)
                    h0 = jnp.broadcast_to(
                        (init_real + 1j * init_imag).astype(jnp.complex64),
                        (B, C, H, W_freq))
                    V_hat = pallas_scalar_scan(A_c, U_c, h0=h0)
                else:
                    V_hat = pallas_scalar_scan(A_c, U_c)
            else:
                # Standard GOOM path
                A_log = _to_log(A_X)
                U_log = _to_log(U_X_hat)

                if self.learned_init:
                    A_log, U_log = _prepend_init(A_log, U_log, n_states=1)

                _, X_log = jax.lax.associative_scan(
                    cssm_scalar_scan_op, (A_log, U_log), axis=1
                )

                if self.learned_init:
                    X_log = X_log[:, 1:]
                V_hat = _from_log(X_log)  # (B, T, C, H, W_freq)

        else:
            # Shared gates for 2-state and 3-state paths
            d_Q = _gate('d_Q', bounded=True)
            d_V = _gate('d_V', bounded=True)
            gamma = _gate('gamma')
            B_Q = _gate('B_Q', spatial=True)
            B_V = _gate('B_V', spatial=True)
            C_gate = _gate('C_gate', spatial=True)

        if self.single_state:
            pass  # Already computed V_hat above

        elif self.no_k_state:
            # =================================================================
            # 2-STATE PATH: Q + V (no K)
            # =================================================================
            #        Q              V
            # Q  [ d_Q·K_b          0       ]
            # V  [ γ·ones        d_V·ones   ]

            d_Q_c = d_Q.astype(jnp.complex64)
            d_V_c = d_V.astype(jnp.complex64)
            zeros = jnp.zeros_like(d_Q_c * K_b)

            A_00 = d_Q_c * K_b         # Q self-decay with kernel
            A_01 = zeros               # No V→Q
            A_10 = gamma * ones        # Q→V (scalar)
            A_11 = d_V_c * ones        # V self-decay (scalar)

            row_Q = jnp.stack([A_00, A_01], axis=-1)
            row_V = jnp.stack([A_10, A_11], axis=-1)
            K_mat = jnp.stack([row_Q, row_V], axis=-2)

            U_Q_hat = _gate_and_fft(q_in, B_Q)
            U_V_hat = _gate_and_fft(v_in, B_V)
            U_vec = jnp.stack([U_Q_hat, U_V_hat], axis=-1)

            if _scan_mode in ('quadratic', 'ssd') or self.use_ssd:
                # Cascaded: decompose 2×2 triangular into 2 scalar SSMs
                a_Q = (d_Q_c * K_b).astype(jnp.complex64)
                a_V = (d_V_c * ones).astype(jnp.complex64)
                gamma_c = (gamma * ones).astype(jnp.complex64)
                L = self.ssd_chunk_size

                if _scan_mode == 'quadratic':
                    _scan_fn = lambda A, X: (quadratic_scan(A, X) if T <= L
                                             else chunked_quadratic_scan(A, X, chunk_size=L))
                else:
                    _scan_fn = lambda A, X: ssd_scan(A, X, chunk_size=L)

                Q_all = _scan_fn(a_Q, U_Q_hat.astype(jnp.complex64))
                Q_prev = jnp.concatenate(
                    [jnp.zeros_like(Q_all[:, :1]), Q_all[:, :-1]], axis=1)

                U_V_eff = gamma_c * Q_prev + U_V_hat.astype(jnp.complex64)
                V_hat = _scan_fn(a_V, U_V_eff)
                self.sow('intermediates', 'Q_hat', Q_all)
            elif self.use_complex32 or self.use_complex16:
                # Linear-split: skip GOOM, split (re_bf16, im_bf16) directly
                K_c = K_mat.astype(jnp.complex64)
                U_c = U_vec.astype(jnp.complex64)

                if self.learned_init:
                    K_log = _to_log(K_c)
                    U_log = _to_log(U_c)
                    K_log, U_log = _prepend_init(K_log, U_log, n_states=2)
                    K_c = _from_log(K_log)
                    U_c = _from_log(U_log)

                K_re, K_im = complex64_to_linear_split(K_c, jnp.float8_e4m3fn if self.use_complex16 else jnp.bfloat16)
                U_re, U_im = complex64_to_linear_split(U_c, jnp.float8_e4m3fn if self.use_complex16 else jnp.bfloat16)
                _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                    linear_split_2x2_scan_op,
                    (K_re, K_im, U_re, U_im), axis=1)
                QV_hat = linear_split_to_complex64(U_re_out, U_im_out)

                if self.learned_init:
                    QV_hat = QV_hat[:, 1:]
                V_hat = QV_hat[..., 1]
                self.sow('intermediates', 'Q_hat', QV_hat[..., 0])
            else:
                # Scan (2×2)
                K_log = _to_log(K_mat)
                U_log = _to_log(U_vec)

                if self.learned_init:
                    K_log, U_log = _prepend_init(K_log, U_log, n_states=2)

                _, State_log = jax.lax.associative_scan(
                    cssm_matrix_scan_op, (K_log, U_log), axis=1
                )

                if self.learned_init:
                    State_log = State_log[:, 1:]
                QV_hat = _from_log(State_log)  # (B, T, C, H, W_freq, 2)
                V_hat = QV_hat[..., 1]
                self.sow('intermediates', 'Q_hat', QV_hat[..., 0])

        else:
            # =================================================================
            # 3-STATE PATH: Q + K + V
            # =================================================================
            d_K = _gate('d_K', bounded=True)
            w_kq = _gate('w_kq')
            B_K = _gate('B_K', spatial=True)
            # Delta-rule selective forgetting gate
            beta = _gate('beta') if self.use_delta else None

            #            Q              K              V
            # Q  [ d_Q·K_b           0              0   ]
            # K  [ w_kq·K_b       d_K·K_b           0   ]
            # V  [ γ·ones          γ·ones   d_V·(1-β)·ones ]

            d_Q_c = d_Q.astype(jnp.complex64)
            d_K_c = d_K.astype(jnp.complex64)
            d_V_c = d_V.astype(jnp.complex64)
            # Effective V decay: d_V * (1 - beta) allows full memory wipe when beta→1
            d_V_eff = d_V_c * (1 - beta) if beta is not None else d_V_c
            zeros = jnp.zeros_like(d_Q_c * ones)

            A_00 = d_Q_c * K_b
            A_01 = zeros
            A_02 = zeros
            A_10 = w_kq * K_b
            A_11 = d_K_c * K_b
            A_12 = zeros
            A_20 = gamma * ones
            A_21 = gamma * ones
            A_22 = d_V_eff * ones

            row_Q = jnp.stack([A_00, A_01, A_02], axis=-1)
            row_K = jnp.stack([A_10, A_11, A_12], axis=-1)
            row_V = jnp.stack([A_20, A_21, A_22], axis=-1)
            K_mat = jnp.stack([row_Q, row_K, row_V], axis=-2)

            # L2-normalize q_in, k_in per-pixel across channels (delta-rule stabilization)
            if self.l2_norm_qk:
                q_in = q_in / (jnp.linalg.norm(q_in, axis=-1, keepdims=True) + 1e-6)
                k_in = k_in / (jnp.linalg.norm(k_in, axis=-1, keepdims=True) + 1e-6)

            # Capture spatial Q/K/V inputs for gradient analysis
            self.sow('intermediates', 'qkv_spatial',
                      jnp.stack([q_in, k_in, v_in], axis=-1))
            if injected_qkv_spatial is not None:
                q_in = injected_qkv_spatial[..., 0]
                k_in = injected_qkv_spatial[..., 1]
                v_in = injected_qkv_spatial[..., 2]

            U_Q_hat = _gate_and_fft(q_in, B_Q)
            U_K_hat = _gate_and_fft(k_in, B_K)
            U_V_hat = _gate_and_fft(v_in, B_V)
            U_vec = jnp.stack([U_Q_hat, U_K_hat, U_V_hat], axis=-1)

            # Scan (3×3)
            if _scan_mode in ('quadratic', 'ssd') or self.use_ssd:
                # Cascaded: decompose 3×3 triangular into 3 scalar SSMs
                # Q_t = a_Q * Q_{t-1} + U_Q
                # K_t = a_K * K_{t-1} + w_kq·K_b · Q_{t-1} + U_K
                # V_t = a_V * V_{t-1} + γ · Q_{t-1} + γ · K_{t-1} + U_V
                a_Q = (d_Q_c * K_b).astype(jnp.complex64)
                a_K = (d_K_c * K_b).astype(jnp.complex64)
                a_V = (d_V_eff * ones).astype(jnp.complex64)
                w_kq_c = (w_kq * K_b).astype(jnp.complex64)
                gamma_c = (gamma * ones).astype(jnp.complex64)
                L = self.ssd_chunk_size

                if _scan_mode == 'quadratic':
                    _scan_fn = lambda A, X: (quadratic_scan(A, X) if T <= L
                                             else chunked_quadratic_scan(A, X, chunk_size=L))
                else:
                    _scan_fn = lambda A, X: ssd_scan(A, X, chunk_size=L)

                # 1. Q scan (independent)
                Q_all = _scan_fn(a_Q, U_Q_hat.astype(jnp.complex64))
                Q_prev = jnp.concatenate(
                    [jnp.zeros_like(Q_all[:, :1]), Q_all[:, :-1]], axis=1)

                # 2. K scan (depends on Q)
                U_K_eff = w_kq_c * Q_prev + U_K_hat.astype(jnp.complex64)
                K_all = _scan_fn(a_K, U_K_eff)
                K_prev = jnp.concatenate(
                    [jnp.zeros_like(K_all[:, :1]), K_all[:, :-1]], axis=1)

                # 3. V scan (depends on Q and K)
                U_V_eff = gamma_c * Q_prev + gamma_c * K_prev + U_V_hat.astype(jnp.complex64)
                V_hat = _scan_fn(a_V, U_V_eff)
                self.sow('intermediates', 'Q_hat', Q_all)
                self.sow('intermediates', 'K_hat', K_all)
            elif self.use_complex32 or self.use_complex16:
                # Linear-split: skip GOOM, split (re_bf16, im_bf16) directly
                K_c = K_mat.astype(jnp.complex64)
                U_c = U_vec.astype(jnp.complex64)

                if self.learned_init:
                    K_log = _to_log(K_c)
                    U_log = _to_log(U_c)
                    K_log, U_log = _prepend_init(K_log, U_log, n_states=3)
                    K_c = _from_log(K_log)
                    U_c = _from_log(U_log)

                K_re, K_im = complex64_to_linear_split(K_c, jnp.float8_e4m3fn if self.use_complex16 else jnp.bfloat16)
                U_re, U_im = complex64_to_linear_split(U_c, jnp.float8_e4m3fn if self.use_complex16 else jnp.bfloat16)
                _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                    linear_split_3x3_scan_op,
                    (K_re, K_im, U_re, U_im), axis=1)
                QKV_hat = linear_split_to_complex64(U_re_out, U_im_out)

                if self.learned_init:
                    QKV_hat = QKV_hat[:, 1:]
                V_hat = QKV_hat[..., 2]
                self.sow('intermediates', 'Q_hat', QKV_hat[..., 0])
                self.sow('intermediates', 'K_hat', QKV_hat[..., 1])
            else:
                K_log = _to_log(K_mat)
                U_log = _to_log(U_vec)

                if self.learned_init:
                    K_log, U_log = _prepend_init(K_log, U_log, n_states=3)

                _, State_log = jax.lax.associative_scan(
                    cssm_3x3_matrix_scan_op, (K_log, U_log), axis=1
                )

                if self.learned_init:
                    State_log = State_log[:, 1:]
                QKV_hat = _from_log(State_log)  # (B, T, C, H, W_freq, 3)
                V_hat = QKV_hat[..., 2]
                self.sow('intermediates', 'Q_hat', QKV_hat[..., 0])
                self.sow('intermediates', 'K_hat', QKV_hat[..., 1])

        # Capture scan states for visualization (no-op when 'intermediates' not mutable)
        self.sow('intermediates', 'V_hat', V_hat)

        # Delta-rule retrieval: phase-only deconvolution V_hat * conj(Q_hat) / |Q_hat|
        if self.use_delta and not self.single_state:
            # Q_hat is available from all scan paths via sow (but we need the tensor directly)
            # For cascaded (SSD/quadratic): Q_all was computed separately
            # For matrix scans: QKV_hat[..., 0] or QV_hat[..., 0]
            if not self.no_k_state:
                if (_scan_mode in ('quadratic', 'ssd') or self.use_ssd):
                    Q_hat_retrieved = Q_all  # from cascaded 3-state path
                else:
                    Q_hat_retrieved = QKV_hat[..., 0]
            else:
                if (_scan_mode in ('quadratic', 'ssd') or self.use_ssd):
                    Q_hat_retrieved = Q_all  # from cascaded 2-state path
                else:
                    Q_hat_retrieved = QV_hat[..., 0]
            Q_hat_mag = jnp.abs(Q_hat_retrieved) + 1e-6
            V_hat = V_hat * jnp.conj(Q_hat_retrieved) / Q_hat_mag

        # =====================================================================
        # OUTPUT: apply C gate, spectral → spatial, project
        # =====================================================================
        if _spatial_gates:
            # Conv/channel: C_gate is spatial (B,T,H,W,C) — apply after IFFT
            V_spatial = jnp.fft.irfft2(V_hat, s=(H, W), axes=(3, 4))  # (B, T, C, H, W)
            output = V_spatial.transpose(0, 1, 3, 4, 2) * C_gate
        else:
            # Others: C_gate is spectral-compatible — apply before IFFT
            V_hat_gated = V_hat * C_gate
            V_spatial = jnp.fft.irfft2(V_hat_gated, s=(H, W), axes=(3, 4))
            output = V_spatial.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)

        return nn.Dense(C, name='output_proj')(output)


class DeltaNetCSSM(nn.Module):
    """
    Spectral DeltaNet CSSM — single-state delta rule in spectral domain.

    Implements the Gated DeltaNet (arXiv:2412.06464) update rule adapted to
    per-channel, per-frequency-bin scalar recurrence:

        A_t = alpha_t · W_hat · (1 - beta_t · |k̂_norm_t|²)
        U_t = beta_t · v̂_t · conj(k̂_norm_t)
        h_t = A_t · h_{t-1} + U_t        (scalar SSM, existing scan)
        o_hat_t = h_t · q̂_norm_t          (query-based retrieval)

    Gates (5 total):
        alpha   [0.1, 0.99]  Global state decay (bounded)
        beta    [0, 1]       Delta gate: coupled erase + write strength
        B_k     [0, 1]       Key input gate
        B_v     [0, 1]       Value input gate
        C_gate  [0, 1]       Output gate

    Attributes:
        channels: Number of input/output channels
        kernel_size: Spatial kernel size
        spectral_rho: Maximum spectral magnitude for stability
        gate_type: Gate parameterization ('channel', 'dense', 'scalar', etc.)
        use_goom: Use GOOM log-space for scan
        use_complex32: Use linear-split bf16 scan
        scan_mode: 'associative', 'ssd', or 'quadratic'
        ssd_chunk_size: Chunk size for SSD
        learned_init: Learn initial recurrence state
    """
    channels: int
    kernel_size: int = 11
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    spectral_rho: float = 0.999
    position_independent_gates: bool = False
    block_size: int = 1
    gate_type: str = 'channel'
    use_goom: bool = True
    use_complex32: bool = False
    use_complex16: bool = False
    scan_mode: str = 'associative'
    ssd_chunk_size: int = 8
    learned_init: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, injected_qkv_spatial=None) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, H, W, C)
            injected_qkv_spatial: Unused, for API compatibility.

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

        # Resolve scan mode
        _scan_mode = self.scan_mode

        # Log-space conversion helpers
        if self.use_goom:
            _to_log = to_goom
            _from_log = from_goom
        else:
            def _to_log(x):
                x_c = x.astype(jnp.complex64) if not jnp.iscomplexobj(x) else x
                return jnp.log(jnp.abs(x_c) + 1e-8) + 1j * jnp.angle(x_c)
            def _from_log(x):
                return jnp.exp(x)

        # Save raw input for position-independent gates
        x_raw = x if self.position_independent_gates else None

        # Optional position encoding
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # =====================================================================
        # 1. SPATIAL KERNEL
        # =====================================================================
        k_shape = (C, self.kernel_size, self.kernel_size)

        def pad_kernel(k_spatial):
            """Pad kernel to input size for FFT."""
            pad_h = max(0, (H - self.kernel_size) // 2)
            pad_w = max(0, (W - self.kernel_size) // 2)
            pad_h_after = H - self.kernel_size - pad_h
            pad_w_after = W - self.kernel_size - pad_w
            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k_spatial[:, start_h:start_h+H, start_w:start_w+W]
            else:
                return jnp.pad(k_spatial, ((0, 0), (pad_h, max(0, pad_h_after)),
                                           (pad_w, max(0, pad_w_after))), mode='constant')

        kernel = self.param('kernel', nn.initializers.xavier_normal(), k_shape)
        k_padded = pad_kernel(kernel)
        W_hat = jnp.fft.rfft2(k_padded, axes=(1, 2))  # (C, H, W_freq)
        W_hat = _stable_spectral_magnitude(W_hat, rho=self.spectral_rho)
        W_hat = W_hat[None, None, ...].astype(jnp.complex64)  # (1, 1, C, H, W_freq)

        # =====================================================================
        # 2. PROJECT INPUT -> q, k, v
        # =====================================================================
        x_flat = x.reshape(B * T, H, W, C)
        qkv_proj = nn.Dense(3 * C, name='qkv_proj')(x_flat).reshape(B, T, H, W, 3 * C)
        q_in = qkv_proj[..., :C]
        k_in = qkv_proj[..., C:2*C]
        v_in = qkv_proj[..., 2*C:]

        # =====================================================================
        # 3. GATES
        # =====================================================================
        gate_input = x_raw if x_raw is not None else x
        ctx = gate_input.mean(axis=(2, 3))  # (B, T, C)
        if x_raw is None:
            ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)
        n_gate = H * W_freq
        _spatial_gates = self.gate_type in ('conv', 'channel', 'dense_decay')

        def _gate(name, bounded=False, spatial=False):
            """Create a gate (same logic as AdditiveCSSM)."""
            if self.gate_type == 'channel':
                if spatial:
                    gate_flat = gate_input.reshape(B * T, H, W, C)
                    raw = nn.sigmoid(nn.Conv(
                        C, kernel_size=(1, 1), padding='SAME', name=name
                    )(gate_flat))
                    return raw.reshape(B, T, H, W, C)
                else:
                    raw = nn.sigmoid(nn.Dense(C, name=name)(ctx))
                    raw = raw.reshape(B, T, C, 1, 1)
                    return 0.1 + 0.89 * raw if bounded else raw
            elif self.gate_type == 'factored':
                row = nn.Dense(H, name=f'{name}_h')(ctx)
                col = nn.Dense(W_freq, name=f'{name}_w')(ctx)
                raw = nn.sigmoid(row[..., :, None] + col[..., None, :])
                raw = raw.reshape(B, T, 1, H, W_freq)
                return 0.1 + 0.89 * raw if bounded else raw
            elif self.gate_type == 'scalar':
                raw = nn.sigmoid(nn.Dense(1, name=name)(ctx))
                raw = raw.reshape(B, T, 1, 1, 1)
                return 0.1 + 0.89 * raw if bounded else raw
            else:  # 'dense'
                raw = nn.sigmoid(nn.Dense(n_gate, name=name)(ctx))
                raw = raw.reshape(B, T, 1, H, W_freq)
                return 0.1 + 0.89 * raw if bounded else raw

        def _to_spectral(spatial_in):
            """(B,T,H,W,C) -> (B,T,C,H,W_freq) via rfft2."""
            return jnp.fft.rfft2(spatial_in.transpose(0, 1, 4, 2, 3), axes=(3, 4))

        def _gate_and_fft(spatial_in, b_gate):
            """Apply input gate and FFT."""
            if _spatial_gates:
                return _to_spectral(spatial_in * b_gate)
            else:
                return _to_spectral(spatial_in) * b_gate

        # Gates
        alpha = _gate('alpha', bounded=True)    # [0.1, 0.99] global decay
        beta = _gate('beta')                     # [0, 1] delta gate
        B_k = _gate('B_k', spatial=True)         # key input gate
        B_v = _gate('B_v', spatial=True)         # value input gate
        C_gate = _gate('C_gate', spatial=True)   # output gate

        # =====================================================================
        # 4. FFT INPUTS + L2 NORMALIZATION
        # =====================================================================
        k_hat = _gate_and_fft(k_in, B_k)    # (B, T, C, H, W_freq)
        v_hat = _gate_and_fft(v_in, B_v)    # (B, T, C, H, W_freq)
        q_hat = _to_spectral(q_in)          # no input gate on query

        # L2-normalize k and q in spectral domain (per channel)
        # By Parseval's theorem, equivalent to spatial L2 norm
        def _spectral_l2_norm(z):
            """L2-normalize complex tensor across freq bins (axes 3,4) per channel."""
            norm = jnp.sqrt(jnp.sum(jnp.abs(z) ** 2, axis=(3, 4), keepdims=True) + 1e-6)
            return z / norm

        k_hat_norm = _spectral_l2_norm(k_hat)
        q_hat_norm = _spectral_l2_norm(q_hat)

        # =====================================================================
        # 5. BUILD TRANSITION A_t AND INPUT U_t
        # =====================================================================
        # |k_hat_norm|^2 per freq bin -- erasure pattern
        k_power = jnp.abs(k_hat_norm) ** 2   # (B, T, C, H, W_freq), real

        # A_t = alpha * W_hat * (1 - beta * |k_hat_norm|^2)
        A_t = alpha.astype(jnp.complex64) * W_hat * (1.0 - beta * k_power)
        # Clamp transition magnitude for stability (belt-and-suspenders with GOOM)
        A_t = _stable_spectral_magnitude(A_t, rho=self.spectral_rho)

        # U_t = beta * v_hat * conj(k_hat_norm) -- coupled erase-write
        U_t = beta * v_hat * jnp.conj(k_hat_norm)

        # =====================================================================
        # 6. SCALAR SCAN (reuses existing infrastructure)
        # =====================================================================
        if _scan_mode in ('quadratic', 'ssd'):
            A_c = A_t.astype(jnp.complex64)
            U_c = U_t.astype(jnp.complex64)
            L = self.ssd_chunk_size
            if _scan_mode == 'quadratic':
                if T <= L:
                    h_all = quadratic_scan(A_c, U_c)
                else:
                    h_all = chunked_quadratic_scan(A_c, U_c, chunk_size=L)
            else:
                h_all = ssd_scan(A_c, U_c, chunk_size=L)
        elif self.use_complex32 or self.use_complex16:
            # Linear-split bf16 scan
            # Priority over pallas: parallel associative_scan compiles faster under pmap
            A_c = A_t.astype(jnp.complex64)
            U_c = U_t.astype(jnp.complex64)

            if self.learned_init:
                A_log = _to_log(A_c)
                U_log = _to_log(U_c)
                A_init = jnp.zeros((B, 1, C, H, W_freq), dtype=jnp.complex64)
                init_shape = (1, 1, C, 1, 1)
                init_real = self.param('init_real', nn.initializers.constant(-10.0), init_shape)
                init_imag = self.param('init_imag', nn.initializers.zeros, init_shape)
                U_init = jnp.broadcast_to(
                    (init_real + 1j * init_imag).astype(jnp.complex64),
                    (B, 1, C, H, W_freq))
                A_log = jnp.concatenate([A_init, A_log], axis=1)
                U_log = jnp.concatenate([U_init, U_log], axis=1)
                A_c = _from_log(A_log)
                U_c = _from_log(U_log)

            A_re, A_im = complex64_to_linear_split(A_c, jnp.float8_e4m3fn if self.use_complex16 else jnp.bfloat16)
            U_re, U_im = complex64_to_linear_split(U_c, jnp.float8_e4m3fn if self.use_complex16 else jnp.bfloat16)
            _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                linear_split_scalar_scan_op,
                (A_re, A_im, U_re, U_im), axis=1)
            h_all = linear_split_to_complex64(U_re_out, U_im_out)

            if self.learned_init:
                h_all = h_all[:, 1:]
        elif _scan_mode == 'pallas':
            # Fused Pallas GPU kernel: sequential scan in registers
            from .pallas_scan import pallas_scalar_scan
            A_c = A_t.astype(jnp.complex64)
            U_c = U_t.astype(jnp.complex64)
            if self.learned_init:
                init_shape = (1, 1, C, 1, 1)
                init_real = self.param('init_real', nn.initializers.constant(-10.0), init_shape)
                init_imag = self.param('init_imag', nn.initializers.zeros, init_shape)
                h0 = jnp.broadcast_to(
                    (init_real + 1j * init_imag).astype(jnp.complex64),
                    (B, C, H, W_freq))
                h_all = pallas_scalar_scan(A_c, U_c, h0=h0)
            else:
                h_all = pallas_scalar_scan(A_c, U_c)
        else:
            # GOOM log-space scan
            A_log = _to_log(A_t)
            U_log = _to_log(U_t)

            if self.learned_init:
                A_init = jnp.zeros((B, 1, C, H, W_freq), dtype=jnp.complex64)
                init_shape = (1, 1, C, 1, 1)
                init_real = self.param('init_real', nn.initializers.constant(-10.0), init_shape)
                init_imag = self.param('init_imag', nn.initializers.zeros, init_shape)
                U_init = jnp.broadcast_to(
                    (init_real + 1j * init_imag).astype(jnp.complex64),
                    (B, 1, C, H, W_freq))
                A_log = jnp.concatenate([A_init, A_log], axis=1)
                U_log = jnp.concatenate([U_init, U_log], axis=1)

            _, h_log = jax.lax.associative_scan(
                cssm_scalar_scan_op, (A_log, U_log), axis=1)

            if self.learned_init:
                h_log = h_log[:, 1:]
            h_all = _from_log(h_log)  # (B, T, C, H, W_freq)

        # =====================================================================
        # 7. RETRIEVAL: h * q_norm
        # =====================================================================
        o_hat = h_all * q_hat_norm  # (B, T, C, H, W_freq)

        # =====================================================================
        # 8. OUTPUT: gate + iFFT + project
        # =====================================================================
        if _spatial_gates:
            o_spatial = jnp.fft.irfft2(o_hat, s=(H, W), axes=(3, 4))  # (B, T, C, H, W)
            output = o_spatial.transpose(0, 1, 3, 4, 2) * C_gate
        else:
            o_hat_gated = o_hat * C_gate
            o_spatial = jnp.fft.irfft2(o_hat_gated, s=(H, W), axes=(3, 4))
            output = o_spatial.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)

        return nn.Dense(C, name='output_proj')(output)


class MatrixDeltaNetCSSM(nn.Module):
    """
    Matrix-state DeltaNet CSSM — full delta rule with d_k×d_v state matrix.

    Extends DeltaNetCSSM from scalar state to matrix state per frequency bin,
    giving content-addressable memory via the DGM recurrence:

        M_t = alpha_t · (I - beta_t · k̂·k̂†)        transition (d_k×d_k)
        Δ_t = beta_t · v̂·k̂†                         rank-1 write (d_v×d_k)
        S_t = M_t · S_{t-1} + Δ_t                   matrix state update
        o_hat_t = S_t @ q̂_norm_t                     query readout (d_v)

    The state S is d_v × d_k per head per frequency bin. Each column of S
    evolves independently under the same M_t, so we fold d_v into the channel
    dimension and reuse existing 2×2 / 3×3 matrix scan infrastructure.

    Attributes:
        channels: Number of input/output channels (C = n_h × d_k)
        delta_key_dim: Key dimension d_k (2 or 3). d_v = d_k, n_h = C // d_k.
        kernel_size: Spatial kernel size
        spectral_rho: Maximum spectral magnitude for stability
        gate_type: Gate parameterization ('channel', 'dense', 'scalar', 'factored')
        use_goom: Use GOOM log-space for scan
        use_complex32: Use linear-split bf16 scan
        scan_mode: 'associative' (only associative supported for matrix scan)
        ssd_chunk_size: Chunk size for SSD (unused, kept for API compat)
        learned_init: Learn initial recurrence state
    """
    channels: int
    delta_key_dim: int = 2
    kernel_size: int = 11
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    spectral_rho: float = 0.999
    position_independent_gates: bool = False
    block_size: int = 1
    gate_type: str = 'channel'
    use_goom: bool = True
    use_complex32: bool = False
    use_complex16: bool = False
    scan_mode: str = 'associative'
    ssd_chunk_size: int = 8
    learned_init: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, injected_qkv_spatial=None) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, H, W, C)
            injected_qkv_spatial: Unused, for API compatibility.

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1
        d_k = self.delta_key_dim
        d_v = d_k  # Square state matrix
        n_h = C // d_k
        assert C % d_k == 0, f"channels={C} must be divisible by delta_key_dim={d_k}"

        # Log-space conversion helpers
        if self.use_goom:
            _to_log = to_goom
            _from_log = from_goom
        else:
            def _to_log(x):
                x_c = x.astype(jnp.complex64) if not jnp.iscomplexobj(x) else x
                return jnp.log(jnp.abs(x_c) + 1e-8) + 1j * jnp.angle(x_c)
            def _from_log(x):
                return jnp.exp(x)

        # Save raw input for position-independent gates
        x_raw = x if self.position_independent_gates else None

        # Optional position encoding
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # =====================================================================
        # 1. SPATIAL KERNEL (per head, shared across d_k dims within head)
        # =====================================================================
        k_shape = (n_h, self.kernel_size, self.kernel_size)

        def pad_kernel(k_spatial):
            """Pad kernel to input size for FFT."""
            pad_h = max(0, (H - self.kernel_size) // 2)
            pad_w = max(0, (W - self.kernel_size) // 2)
            pad_h_after = H - self.kernel_size - pad_h
            pad_w_after = W - self.kernel_size - pad_w
            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k_spatial[:, start_h:start_h+H, start_w:start_w+W]
            else:
                return jnp.pad(k_spatial, ((0, 0), (pad_h, max(0, pad_h_after)),
                                           (pad_w, max(0, pad_w_after))), mode='constant')

        kernel = self.param('kernel', nn.initializers.xavier_normal(), k_shape)
        k_padded = pad_kernel(kernel)
        W_hat = jnp.fft.rfft2(k_padded, axes=(1, 2))  # (n_h, H, W_freq)
        W_hat = _stable_spectral_magnitude(W_hat, rho=self.spectral_rho)
        W_hat = W_hat.astype(jnp.complex64)  # (n_h, H, W_freq)

        # =====================================================================
        # 2. PROJECT INPUT -> q, k, v per head
        # =====================================================================
        x_flat = x.reshape(B * T, H, W, C)
        qkv_proj = nn.Dense(3 * C, name='qkv_proj')(x_flat).reshape(B, T, H, W, 3 * C)
        q_in = qkv_proj[..., :C]        # (B, T, H, W, C)
        k_in = qkv_proj[..., C:2*C]
        v_in = qkv_proj[..., 2*C:]

        # Reshape into heads: (B, T, H, W, n_h, d_k)
        q_heads = q_in.reshape(B, T, H, W, n_h, d_k)
        k_heads = k_in.reshape(B, T, H, W, n_h, d_k)
        v_heads = v_in.reshape(B, T, H, W, n_h, d_v)

        # =====================================================================
        # 3. GATES (per head, shared across d_k dims)
        # =====================================================================
        gate_input = x_raw if x_raw is not None else x
        ctx = gate_input.mean(axis=(2, 3))  # (B, T, C)
        if x_raw is None:
            ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)
        n_gate = H * W_freq
        _spatial_gates = self.gate_type in ('conv', 'channel', 'dense_decay')

        def _gate(name, bounded=False, spatial=False):
            """Create a gate (same logic as DeltaNetCSSM)."""
            if self.gate_type == 'channel':
                if spatial:
                    gate_flat = gate_input.reshape(B * T, H, W, C)
                    raw = nn.sigmoid(nn.Conv(
                        C, kernel_size=(1, 1), padding='SAME', name=name
                    )(gate_flat))
                    return raw.reshape(B, T, H, W, C)
                else:
                    raw = nn.sigmoid(nn.Dense(n_h, name=name)(ctx))
                    raw = raw.reshape(B, T, n_h, 1, 1)
                    return 0.1 + 0.89 * raw if bounded else raw
            elif self.gate_type == 'factored':
                row = nn.Dense(H, name=f'{name}_h')(ctx)
                col = nn.Dense(W_freq, name=f'{name}_w')(ctx)
                raw = nn.sigmoid(row[..., :, None] + col[..., None, :])
                raw = raw.reshape(B, T, 1, H, W_freq)
                return 0.1 + 0.89 * raw if bounded else raw
            elif self.gate_type == 'scalar':
                raw = nn.sigmoid(nn.Dense(1, name=name)(ctx))
                raw = raw.reshape(B, T, 1, 1, 1)
                return 0.1 + 0.89 * raw if bounded else raw
            else:  # 'dense'
                raw = nn.sigmoid(nn.Dense(n_gate, name=name)(ctx))
                raw = raw.reshape(B, T, 1, H, W_freq)
                return 0.1 + 0.89 * raw if bounded else raw

        # Gates: alpha (decay), beta (delta), C_gate (output)
        alpha = _gate('alpha', bounded=True)    # [0.1, 0.99]
        beta = _gate('beta')                     # [0, 1]
        C_gate = _gate('C_gate', spatial=True)   # output gate

        # =====================================================================
        # 4. FFT per head + L2 normalization
        # =====================================================================
        # Transpose to (B, T, n_h, d_k, H, W) then rfft2
        q_spec = jnp.fft.rfft2(
            q_heads.transpose(0, 1, 4, 5, 2, 3), axes=(4, 5)
        )  # (B, T, n_h, d_k, H, W_freq)
        k_spec = jnp.fft.rfft2(
            k_heads.transpose(0, 1, 4, 5, 2, 3), axes=(4, 5)
        )  # (B, T, n_h, d_k, H, W_freq)
        v_spec = jnp.fft.rfft2(
            v_heads.transpose(0, 1, 4, 5, 2, 3), axes=(4, 5)
        )  # (B, T, n_h, d_v, H, W_freq)

        # L2-normalize k and q per head across freq bins
        def _spectral_l2_norm(z):
            """L2-normalize across freq dims (axes 4,5) per head per d_k component."""
            norm = jnp.sqrt(jnp.sum(jnp.abs(z) ** 2, axis=(4, 5), keepdims=True) + 1e-6)
            return z / norm

        def _key_vector_l2_norm(z):
            """L2-normalize across d_k dim (axis 3) per head per freq bin.

            Ensures ||k||²=1 per freq bin so that (I - β·k·k†) is a
            contraction with eigenvalues {1, ..., 1-β}, all ≤ 1.
            Without this, ||k||²=d_k and the erase eigenvalue is 1-d_k·β,
            which goes negative and causes instability for d_k≥2.
            """
            norm = jnp.sqrt(jnp.sum(jnp.abs(z) ** 2, axis=3, keepdims=True) + 1e-6)
            return z / norm

        k_hat_norm = _key_vector_l2_norm(_spectral_l2_norm(k_spec))  # (B, T, n_h, d_k, H, W_freq)
        q_hat_norm = _spectral_l2_norm(q_spec)   # (B, T, n_h, d_k, H, W_freq)
        v_hat = v_spec                             # (B, T, n_h, d_v, H, W_freq)

        # =====================================================================
        # 5. BUILD M_t (d_k×d_k transition) and Δ_t (d_v×d_k write)
        # =====================================================================
        # k outer product: k[..., i] * conj(k[..., j]) -> (..., d_k, d_k, H, W_freq)
        # k_hat_norm: (B, T, n_h, d_k, H, W_freq)
        k_outer = (k_hat_norm[..., :, None, :, :]           # (B, T, n_h, d_k, 1, H, W_freq)
                   * jnp.conj(k_hat_norm[..., None, :, :, :]))  # (B, T, n_h, 1, d_k, H, W_freq)
        # k_outer: (B, T, n_h, d_k, d_k, H, W_freq)

        # Identity matrix for d_k: (d_k, d_k) -> (1, 1, 1, d_k, d_k, 1, 1)
        I_dk = jnp.eye(d_k, dtype=jnp.complex64).reshape(1, 1, 1, d_k, d_k, 1, 1)

        # Broadcast alpha/beta to (B, T, n_h_or_1, 1, 1, H_or_1, W_freq_or_1)
        if self.gate_type == 'channel':
            # alpha: (B, T, n_h, 1, 1) -> (B, T, n_h, 1, 1, 1, 1)
            alpha_bc = alpha[..., None, None]
            beta_bc = beta[..., None, None]
        else:
            # alpha: (B, T, 1, H, W_freq) -> (B, T, 1, 1, 1, H, W_freq)
            alpha_bc = alpha[:, :, :, None, None, :, :]
            beta_bc = beta[:, :, :, None, None, :, :]

        # W_hat: (n_h, H, W_freq) -> (1, 1, n_h, 1, 1, H, W_freq)
        W_hat_bc = W_hat[None, None, :, None, None, :, :]

        # M_t = alpha * W_hat * (I - beta * k·k†)
        # Shape: (B, T, n_h, d_k, d_k, H, W_freq)
        # Eigenvalues are bounded: |alpha|<0.99, |W_hat|<rho, |eig(I-beta*k*k†)|≤1
        # (with ||k||²=1 from _key_vector_l2_norm), so max|eig(M_t)| < 0.99*rho.
        # No element-wise clamping — it would distort the matrix structure and
        # suppress gradients without improving stability.
        M_t = alpha_bc.astype(jnp.complex64) * W_hat_bc * (I_dk - beta_bc * k_outer)

        # Δ_t = beta * v·k† outer product
        # v_hat: (B, T, n_h, d_v, H, W_freq), k_hat_norm: (B, T, n_h, d_k, H, W_freq)
        vk_outer = (v_hat[..., :, None, :, :]                   # (B, T, n_h, d_v, 1, H, W_freq)
                    * jnp.conj(k_hat_norm[..., None, :, :, :]))  # (B, T, n_h, 1, d_k, H, W_freq)
        # vk_outer: (B, T, n_h, d_v, d_k, H, W_freq)
        Delta_t = beta_bc * vk_outer

        # =====================================================================
        # 6. MATRIX SCAN
        # =====================================================================
        # Each column j of S evolves: S_t[:, j] = M_t @ S_{t-1}[:, j] + Δ_t[:, j]
        # Fold d_v into channel dim so we scan d_v copies with same M_t.
        #
        # Rearrange for scan ops which expect: K (..., d_k, d_k), U (..., d_k)
        # Move spatial dims (H, W_freq) before matrix dims:

        # M_t: (B, T, n_h, d_k, d_k, H, W_freq) -> (B, T, n_h, H, W_freq, d_k, d_k)
        M_scan = M_t.transpose(0, 1, 2, 5, 6, 3, 4)

        # Delta_t: (B, T, n_h, d_v, d_k, H, W_freq) -> (B, T, n_h, d_v, H, W_freq, d_k)
        Delta_scan = Delta_t.transpose(0, 1, 2, 3, 5, 6, 4)

        # Broadcast M across d_v: (B, T, n_h, H, W_freq, d_k, d_k) -> (B, T, n_h*d_v, H, W_freq, d_k, d_k)
        M_scan_rep = jnp.repeat(M_scan, d_v, axis=2)

        # Flatten d_v into channel: (B, T, n_h, d_v, H, W_freq, d_k) -> (B, T, n_h*d_v, H, W_freq, d_k)
        Delta_scan_flat = Delta_scan.reshape(B, T, n_h * d_v, H, W_freq, d_k)

        # Dispatch scan
        # Priority: complex32/16 associative scan > pallas > log-space
        # complex32/16 uses parallel associative_scan which compiles much faster
        # under pmap than sequential jax.lax.scan (O(log T) vs O(T) XLA ops)
        _scan_mode = self.scan_mode
        if self.use_complex32 or self.use_complex16:
            K_c = M_scan_rep.astype(jnp.complex64)
            U_c = Delta_scan_flat.astype(jnp.complex64)
            K_re, K_im = complex64_to_linear_split(K_c, jnp.float8_e4m3fn if self.use_complex16 else jnp.bfloat16)
            U_re, U_im = complex64_to_linear_split(U_c, jnp.float8_e4m3fn if self.use_complex16 else jnp.bfloat16)
            if d_k == 2:
                _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                    linear_split_2x2_scan_op,
                    (K_re, K_im, U_re, U_im), axis=1)
            elif d_k == 3:
                _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                    linear_split_3x3_scan_op,
                    (K_re, K_im, U_re, U_im), axis=1)
            else:  # general d_k
                _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                    linear_split_general_scan_op,
                    (K_re, K_im, U_re, U_im), axis=1)
            S_flat = linear_split_to_complex64(U_re_out, U_im_out)
        elif _scan_mode == 'pallas' and d_k == 2:
            # Fused Pallas GPU kernel (d_k=2 only)
            from .pallas_scan import pallas_matrix_2x2_scan
            M_c = M_scan_rep.astype(jnp.complex64)
            U_c = Delta_scan_flat.astype(jnp.complex64)
            S_flat = pallas_matrix_2x2_scan(M_c, U_c)
        else:
            # GOOM log-space scan (default, numerically stable)
            K_log = _to_log(M_scan_rep)
            U_log = _to_log(Delta_scan_flat)
            if d_k == 2:
                _, S_log = jax.lax.associative_scan(
                    cssm_matrix_scan_op, (K_log, U_log), axis=1)
            elif d_k == 3:
                _, S_log = jax.lax.associative_scan(
                    cssm_3x3_matrix_scan_op, (K_log, U_log), axis=1)
            else:  # general d_k
                _, S_log = jax.lax.associative_scan(
                    cssm_general_matrix_scan_op, (K_log, U_log), axis=1)
            S_flat = _from_log(S_log)
        # S_flat: (B, T, n_h*d_v, H, W_freq, d_k)

        # Unfold: (B, T, n_h*d_v, H, W_freq, d_k) -> (B, T, n_h, d_v, H, W_freq, d_k)
        S_t = S_flat.reshape(B, T, n_h, d_v, H, W_freq, d_k)

        # =====================================================================
        # 7. READOUT: o = S @ q  per head per freq
        # =====================================================================
        # S_t: (B, T, n_h, d_v, H, W_freq, d_k)
        # q_hat_norm: (B, T, n_h, d_k, H, W_freq) -> (B, T, n_h, 1, H, W_freq, d_k)
        q_read = q_hat_norm.transpose(0, 1, 2, 4, 5, 3)[:, :, :, None, :, :, :]
        # o = sum over d_k
        o_hat = jnp.sum(S_t * q_read, axis=-1)  # (B, T, n_h, d_v, H, W_freq)

        # =====================================================================
        # 8. OUTPUT: reshape heads -> channels, gate, iFFT, project
        # =====================================================================
        # (B, T, n_h, d_v, H, W_freq) -> (B, T, C, H, W_freq) where C = n_h * d_v
        o_hat = o_hat.reshape(B, T, C, H, W_freq)

        if _spatial_gates:
            o_spatial = jnp.fft.irfft2(o_hat, s=(H, W), axes=(3, 4))  # (B, T, C, H, W)
            output = o_spatial.transpose(0, 1, 3, 4, 2) * C_gate
        else:
            o_hat_gated = o_hat * C_gate
            o_spatial = jnp.fft.irfft2(o_hat_gated, s=(H, W), axes=(3, 4))
            output = o_spatial.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)

        return nn.Dense(C, name='mat_deltanet_output_proj')(output)


class GatedDeltaNetCSSM(nn.Module):
    """
    Gated DeltaNet CSSM — closer to arXiv:2412.06464 with input gates,
    short temporal conv, output norm, and SiLU gating.

    Extends MatrixDeltaNetCSSM with:
        - Input gates B_k, B_v (spatial, 1x1 conv)
        - Causal depthwise conv1d along T for q, k, v (local temporal context)
        - SiLU activation after short conv (matching paper: Conv → SiLU → L2Norm)
        - RMSNorm (or LayerNorm) before output gating
        - SiLU output gating (unbounded, richer modulation than sigmoid)

    Recurrence is identical to MatrixDeltaNetCSSM:
        M_t = alpha_t · W_hat · (I - beta_t · k·k†)        transition (d_k×d_k)
        Δ_t = beta_t · v·k†                                 rank-1 write (d_v×d_k)
        S_t = M_t · S_{t-1} + Δ_t                           matrix state update
        o_hat_t = S_t @ q_norm_t                             query readout (d_v)

    Attributes:
        channels: Number of input/output channels (C = n_h × d_k)
        delta_key_dim: Key dimension d_k (2 or 3). d_v = d_k, n_h = C // d_k.
        kernel_size: Spatial kernel size
        spectral_rho: Maximum spectral magnitude for stability
        gate_type: Gate parameterization ('channel', 'dense', 'scalar', 'factored')
        use_goom: Use GOOM log-space for scan
        use_complex32: Use linear-split bf16 scan
        scan_mode: 'associative' (only associative supported for matrix scan)
        ssd_chunk_size: Chunk size for SSD (unused, kept for API compat)
        learned_init: Learn initial recurrence state
        short_conv_size: Temporal conv kernel size (0=disabled)
        output_norm: Output norm before gating ('rms', 'layer', 'none')
        use_input_gates: Enable B_k, B_v spatial input gates
        output_gate_act: Output gate activation ('silu' or 'sigmoid')
    """
    channels: int
    delta_key_dim: int = 2
    kernel_size: int = 11
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    spectral_rho: float = 0.999
    position_independent_gates: bool = False
    block_size: int = 1
    gate_type: str = 'channel'
    use_goom: bool = True
    use_log: bool = True                 # Use log-space scan; False = direct complex multiply
    use_complex32: bool = False          # bfloat16 split precision (only with use_log=False)
    use_complex16: bool = False
    scan_mode: str = 'associative'
    ssd_chunk_size: int = 8
    learned_init: bool = False
    # --- New (vs MatrixDeltaNetCSSM) ---
    short_conv_size: int = 4
    output_norm: str = 'rms'
    use_input_gates: bool = True
    output_gate_act: str = 'silu'
    use_spectral_l2_norm: bool = True    # L2-normalize q,k across freq bins (ablatable)
    qkv_conv_size: int = 1               # QKV projection spatial extent (1=Dense, 3/5/7=spatial conv → one-shot cross-freq)
    qkv_conv_separable: bool = True      # Use depthwise-separable conv for QKV (much fewer params)
    cross_freq_conv_size: int = 0        # 1D conv across W_freq on output (0=disabled, 3/5=cross-freq mixing)
    no_spectral_clamp: bool = False      # Skip spectral magnitude squash on W_hat
    # --- InT-style spatial attention ---
    int_attention_mode: str = 'none'     # 'none', 'spectral', 'elementwise', 'qk' — spatial attention on GDN output
    attn_kernel_size: int = 7            # Spatial kernel for spectral attention state (gdn_int)
    int_attn_dim: int = 16              # Hidden dim for Q·K attention projections (gdn_int_qk)
    # --- Static-image fast path (ImageNet optimization) ---
    static_image_fast_path: bool = False  # When True and input is 4D (B,H,W,C), skip T-replication
    num_timesteps: int = 0                # Required when static_image_fast_path=True (scan length)
    # --- Temporal SSL projection head (sown intermediates for contrastive loss) ---
    ssl_proj_dim: int = 0                 # 0=disabled. >0 enables MLP projection + L2-norm + sow
    # --- gate_proj bias init. 0.0 is the tog9av97 75% run value — bisect on
    # 2026-04-16 confirmed 1.0 regresses epoch-1 val by ~2× on ImageNet.
    gate_proj_bias_init: float = 0.0
    # Mixed precision: bf16 compute for Dense/Conv, fp32 weight storage, fp32 FFT.
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, injected_qkv_spatial=None) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, H, W, C) — or (B, H, W, C) when
               static_image_fast_path=True.
            injected_qkv_spatial: Unused, for API compatibility.

        Returns:
            Output tensor of shape (B, T, H, W, C), or (B, H, W, C) in fast path.
        """
        if self.static_image_fast_path and x.ndim == 4:
            # --- Preconditions for fast path correctness ---
            assert self.ssl_proj_dim == 0, \
                "static_image_fast_path is incompatible with ssl_proj_dim>0 " \
                "(SSL needs per-timestep features which the fast path collapses)"
            assert self.short_conv_size == 0, \
                "static_image_fast_path requires short_conv_size=0"
            assert self.use_input_gates is False, \
                "static_image_fast_path requires use_input_gates=False"
            assert self.rope_mode == 'none', \
                "static_image_fast_path requires rope_mode='none'"
            assert self.position_independent_gates is False, \
                "static_image_fast_path requires position_independent_gates=False"
            assert self.int_attention_mode == 'none', \
                "static_image_fast_path requires int_attention_mode='none'"
            assert self.learned_init is False, \
                "static_image_fast_path requires learned_init=False"
            assert self.num_timesteps > 0, \
                "static_image_fast_path requires num_timesteps > 0 to be passed explicitly"

            B_, H_, W_, C_ = x.shape
            T_ = self.num_timesteps
            W_freq_ = W_ // 2 + 1
            d_k_ = self.delta_key_dim
            d_v_ = d_k_
            n_h_ = C_ // d_k_
            assert C_ % d_k_ == 0, f"channels={C_} must be divisible by delta_key_dim={d_k_}"

            # =============================================================
            # T-INVARIANT WORK (no T axis)
            # =============================================================

            # 1. Spatial kernel FFT → W_hat (n_h, H, W_freq)
            k_shape_ = (n_h_, self.kernel_size, self.kernel_size)

            def _pad_kernel_fp(k_spatial):
                pad_h = max(0, (H_ - self.kernel_size) // 2)
                pad_w = max(0, (W_ - self.kernel_size) // 2)
                pad_h_after = H_ - self.kernel_size - pad_h
                pad_w_after = W_ - self.kernel_size - pad_w
                if self.kernel_size > H_ or self.kernel_size > W_:
                    start_h = (self.kernel_size - H_) // 2
                    start_w = (self.kernel_size - W_) // 2
                    return k_spatial[:, start_h:start_h+H_, start_w:start_w+W_]
                else:
                    return jnp.pad(k_spatial,
                                   ((0, 0),
                                    (pad_h, max(0, pad_h_after)),
                                    (pad_w, max(0, pad_w_after))),
                                   mode='constant')

            kernel_fp = self.param('kernel', nn.initializers.xavier_normal(), k_shape_)
            k_padded_fp = _pad_kernel_fp(kernel_fp)
            W_hat_fp = jnp.fft.rfft2(k_padded_fp, axes=(1, 2))  # (n_h, H, W_freq)
            if not self.no_spectral_clamp:
                W_hat_fp = _stable_spectral_magnitude(W_hat_fp, rho=self.spectral_rho)
            W_hat_fp = W_hat_fp.astype(jnp.complex64)

            # 2. QKV projection (shared with slow path: same name='qkv_proj')
            if self.qkv_conv_size > 1:
                if self.qkv_conv_separable:
                    qkv_proj = nn.Conv(
                        C_, kernel_size=(self.qkv_conv_size, self.qkv_conv_size),
                        feature_group_count=C_, padding='SAME',
                        dtype=self.dtype, param_dtype=self.param_dtype,
                        name='qkv_dw'
                    )(x)
                    qkv_proj = nn.Dense(
                        3 * C_,
                        dtype=self.dtype, param_dtype=self.param_dtype,
                        name='qkv_proj'
                    )(qkv_proj)
                else:
                    qkv_proj = nn.Conv(
                        3 * C_, kernel_size=(self.qkv_conv_size, self.qkv_conv_size),
                        padding='SAME',
                        dtype=self.dtype, param_dtype=self.param_dtype,
                        name='qkv_proj'
                    )(x)
            else:
                qkv_proj = nn.Dense(
                    3 * C_,
                    dtype=self.dtype, param_dtype=self.param_dtype,
                    name='qkv_proj'
                )(x)
            # qkv_proj: (B, H, W, 3C)
            q_in = qkv_proj[..., :C_]
            k_in = qkv_proj[..., C_:2*C_]
            v_in = qkv_proj[..., 2*C_:]

            # 3. short_conv SKIPPED (precondition)
            # 4. B_k, B_v SKIPPED (precondition use_input_gates=False)

            # 5. Output gate projection (T-invariant: uses x directly)
            # Bias +1: silu(z+1) ≈ 0.92 at init vs silu(z) ≈ 0.40, keeping the
            # gate near pass-through so gradients flow through the CSSM from step 0
            # (same principle as LSTM forget-gate bias init).
            gate_val_fp = nn.Dense(
                C_,
                bias_init=nn.initializers.constant(self.gate_proj_bias_init),
                dtype=self.dtype, param_dtype=self.param_dtype,
                name='gate_proj'
            )(x)  # (B, H, W, C)
            if self.output_gate_act == 'silu':
                gate_val_fp = jax.nn.silu(gate_val_fp)
            else:
                gate_val_fp = nn.sigmoid(gate_val_fp)

            # 6. Reshape to heads. FFT requires real fp32/fp64 input; cast up
            #    explicitly so bf16 inputs still produce complex64 specs.
            q_heads_fp = q_in.reshape(B_, H_, W_, n_h_, d_k_).astype(jnp.float32)
            k_heads_fp = k_in.reshape(B_, H_, W_, n_h_, d_k_).astype(jnp.float32)
            v_heads_fp = v_in.reshape(B_, H_, W_, n_h_, d_v_).astype(jnp.float32)

            # Transpose to (B, n_h, d_k, H, W) then rfft2
            q_spec_fp = jnp.fft.rfft2(
                q_heads_fp.transpose(0, 3, 4, 1, 2), axes=(3, 4)
            )  # (B, n_h, d_k, H, W_freq) complex64
            k_spec_fp = jnp.fft.rfft2(
                k_heads_fp.transpose(0, 3, 4, 1, 2), axes=(3, 4)
            )
            v_spec_fp = jnp.fft.rfft2(
                v_heads_fp.transpose(0, 3, 4, 1, 2), axes=(3, 4)
            )

            # 7. L2 norms (no T axis; adjust axes vs slow path)
            def _spectral_l2_norm_fp(z):
                # z shape (B, n_h, d_k, H, W_freq); normalize across (H, W_freq) = (3, 4)
                if not self.use_spectral_l2_norm:
                    return z
                norm = jnp.sqrt(jnp.sum(jnp.abs(z) ** 2, axis=(3, 4), keepdims=True) + 1e-6)
                return z / norm

            def _key_vector_l2_norm_fp(z):
                # axis 2 is d_k in 5D fast-path layout
                norm = jnp.sqrt(jnp.sum(jnp.abs(z) ** 2, axis=2, keepdims=True) + 1e-6)
                return z / norm

            k_hat_norm_fp = _key_vector_l2_norm_fp(_spectral_l2_norm_fp(k_spec_fp))
            q_hat_norm_fp = _spectral_l2_norm_fp(q_spec_fp)
            v_hat_fp = v_spec_fp
            # All shape (B, n_h, d_k/d_v, H, W_freq)

            # 8. k_outer, vk_outer (T-invariant)
            # k_hat_norm_fp[..., :, None, :, :] has axes (B, n_h, d_k, 1, H, W_freq)
            # k_hat_norm_fp[..., None, :, :, :] has axes (B, n_h, 1, d_k, H, W_freq)
            k_outer_fp = (k_hat_norm_fp[..., :, None, :, :]
                          * jnp.conj(k_hat_norm_fp[..., None, :, :, :]))
            # shape (B, n_h, d_k, d_k, H, W_freq)
            vk_outer_fp = (v_hat_fp[..., :, None, :, :]
                           * jnp.conj(k_hat_norm_fp[..., None, :, :, :]))
            # shape (B, n_h, d_v, d_k, H, W_freq)

            # =============================================================
            # T-VARYING WORK (cheap: just gate ctx + Dense)
            # =============================================================

            # 9. ctx → broadcast to T → temporal encoding
            ctx_static = x.mean(axis=(1, 2))              # (B, C)
            ctx_T = jnp.broadcast_to(
                ctx_static[:, None, :], (B_, T_, C_)
            )                                              # (B, T, C)
            if self.rope_mode == 'learned_t':
                t_embed = self.param('temporal_embed',
                                     nn.initializers.zeros, (T_, C_))
                ctx_T = ctx_T + t_embed[None, :, :]
            else:
                ctx_T = apply_temporal_rope_to_context(ctx_T, base=self.rope_base)

            # 10. alpha_t, beta_t via gate Dense layers (inline, gate_type-aware)
            _gate_dense_kw = dict(dtype=self.dtype, param_dtype=self.param_dtype)

            def _gate_fast(name, bounded=False):
                if self.gate_type == 'channel':
                    raw = nn.sigmoid(nn.Dense(n_h_, name=name, **_gate_dense_kw)(ctx_T))  # (B, T, n_h)
                    raw = raw.reshape(B_, T_, n_h_, 1, 1)               # (B, T, n_h, 1, 1)
                    return 0.1 + 0.89 * raw if bounded else raw
                elif self.gate_type == 'factored':
                    row = nn.Dense(H_, name=f'{name}_h', **_gate_dense_kw)(ctx_T)         # (B, T, H)
                    col = nn.Dense(W_freq_, name=f'{name}_w', **_gate_dense_kw)(ctx_T)    # (B, T, W_freq)
                    raw = nn.sigmoid(row[..., :, None] + col[..., None, :])  # (B, T, H, W_freq)
                    raw = raw.reshape(B_, T_, 1, H_, W_freq_)
                    return 0.1 + 0.89 * raw if bounded else raw
                elif self.gate_type == 'scalar':
                    raw = nn.sigmoid(nn.Dense(1, name=name, **_gate_dense_kw)(ctx_T))     # (B, T, 1)
                    raw = raw.reshape(B_, T_, 1, 1, 1)
                    return 0.1 + 0.89 * raw if bounded else raw
                else:  # 'dense'
                    n_gate_ = H_ * W_freq_
                    raw = nn.sigmoid(nn.Dense(n_gate_, name=name, **_gate_dense_kw)(ctx_T))
                    raw = raw.reshape(B_, T_, 1, H_, W_freq_)
                    return 0.1 + 0.89 * raw if bounded else raw

            alpha_fp = _gate_fast('alpha', bounded=True)   # (B, T, *, *, *)  bf16 or fp32
            beta_fp = _gate_fast('beta')

            # Cast gates to fp32 then complex64 for the spectral math below.
            alpha_fp = alpha_fp.astype(jnp.float32)
            beta_fp = beta_fp.astype(jnp.float32)

            # Diagnostic sow: for analyses that measure how gates evolve across
            # timesteps (e.g. are they flat or t-varying?).
            self.sow('intermediates', 'alpha_t', alpha_fp)
            self.sow('intermediates', 'beta_t', beta_fp)

            # 11. Build M_seq, Delta_seq
            if self.gate_type == 'channel':
                alpha_bc = alpha_fp[..., None, None]       # (B, T, n_h, 1, 1, 1, 1)
                beta_bc = beta_fp[..., None, None]
            else:
                # alpha_fp shape (B, T, 1, H, W_freq) → insert (d_k, d_k) axes
                alpha_bc = alpha_fp[:, :, :, None, None, :, :]  # (B, T, 1, 1, 1, H, W_freq)
                beta_bc = beta_fp[:, :, :, None, None, :, :]

            # W_hat: (n_h, H, W_freq) → (1, 1, n_h, 1, 1, H, W_freq)
            W_hat_bc = W_hat_fp[None, None, :, None, None, :, :]
            # k_outer_fp: (B, n_h, d_k, d_k, H, W_freq) → insert T=1 axis at position 1
            k_outer_bc = k_outer_fp[:, None]    # (B, 1, n_h, d_k, d_k, H, W_freq)
            vk_outer_bc = vk_outer_fp[:, None]  # (B, 1, n_h, d_v, d_k, H, W_freq)

            I_dk = jnp.eye(d_k_, dtype=jnp.complex64).reshape(1, 1, 1, d_k_, d_k_, 1, 1)

            M_seq = alpha_bc.astype(jnp.complex64) * W_hat_bc * (I_dk - beta_bc * k_outer_bc)
            # M_seq shape: (B, T, n_h, d_k, d_k, H, W_freq)
            Delta_seq = beta_bc * vk_outer_bc
            # Delta_seq shape: (B, T, n_h, d_v, d_k, H, W_freq)

            # 12. Match slow path's layout for scan
            # Transpose M: (0,1,2,5,6,3,4) → (B, T, n_h, H, W_freq, d_k, d_k)
            M_scan_fp = M_seq.transpose(0, 1, 2, 5, 6, 3, 4)
            # Transpose Delta: (0,1,2,3,5,6,4) → (B, T, n_h, d_v, H, W_freq, d_k)
            Delta_scan_fp = Delta_seq.transpose(0, 1, 2, 3, 5, 6, 4)
            # Repeat M along d_v head axis
            M_scan_rep_fp = jnp.repeat(M_scan_fp, d_v_, axis=2)
            # (B, T, n_h*d_v, H, W_freq, d_k, d_k)
            Delta_scan_flat_fp = Delta_scan_fp.reshape(
                B_, T_, n_h_ * d_v_, H_, W_freq_, d_k_
            )

            # 13. Sequential scan in "linear split" complex32 (bf16 real + bf16 imag).
            # Halves the scan carry memory bandwidth vs complex64; matmul still
            # promotes to fp32 internally for precision.
            M_xs = jnp.moveaxis(M_scan_rep_fp, 1, 0)       # complex64 (T, B, ...)
            U_xs = jnp.moveaxis(Delta_scan_flat_fp, 1, 0)  # complex64

            M_re_xs, M_im_xs = complex64_to_linear_split(M_xs, jnp.bfloat16)
            U_re_xs, U_im_xs = complex64_to_linear_split(U_xs, jnp.bfloat16)

            def _scan_step_ls(carry, inp):
                S_re, S_im = carry
                M_re, M_im, U_re, U_im = inp
                # Complex matmul M @ S:
                #   (M_re + j M_im) @ (S_re + j S_im)
                #   = (M_re@S_re - M_im@S_im) + j (M_re@S_im + M_im@S_re)
                # Promote to fp32 for the einsum, cast back to bf16.
                Mr = M_re.astype(jnp.float32)
                Mi = M_im.astype(jnp.float32)
                Sr = S_re.astype(jnp.float32)
                Si = S_im.astype(jnp.float32)
                out_re_f32 = (
                    jnp.einsum('...ij,...j->...i', Mr, Sr)
                    - jnp.einsum('...ij,...j->...i', Mi, Si)
                )
                out_im_f32 = (
                    jnp.einsum('...ij,...j->...i', Mr, Si)
                    + jnp.einsum('...ij,...j->...i', Mi, Sr)
                )
                new_S_re = (out_re_f32 + U_re.astype(jnp.float32)).astype(jnp.bfloat16)
                new_S_im = (out_im_f32 + U_im.astype(jnp.float32)).astype(jnp.bfloat16)
                return (new_S_re, new_S_im), None

            S0_re = jnp.zeros_like(U_re_xs[0])
            S0_im = jnp.zeros_like(U_im_xs[0])
            # unroll=T compiles into a straight-line graph so XLA can fuse all
            # the 2x2 einsums with the surrounding rfft/irfft ops.
            (S_final_re, S_final_im), _ = jax.lax.scan(
                _scan_step_ls, (S0_re, S0_im),
                (M_re_xs, M_im_xs, U_re_xs, U_im_xs),
                unroll=T_,
            )
            # Cast back to complex64 for the readout einsum + irfft2.
            S_final = linear_split_to_complex64(S_final_re, S_final_im)
            # S_final: (B, n_h*d_v, H, W_freq, d_k)

            # 14. Readout: o = S_final @ q_hat_norm (T-invariant q)
            S_final_r = S_final.reshape(B_, n_h_, d_v_, H_, W_freq_, d_k_)
            # q_hat_norm_fp: (B, n_h, d_k, H, W_freq)
            # → transpose (0,1,3,4,2) → (B, n_h, H, W_freq, d_k)
            # → insert d_v broadcast axis at position 2 → (B, n_h, 1, H, W_freq, d_k)
            q_read_fp = q_hat_norm_fp.transpose(0, 1, 3, 4, 2)[:, :, None, :, :, :]
            o_hat_fp = jnp.sum(S_final_r * q_read_fp, axis=-1)  # (B, n_h, d_v, H, W_freq)
            o_hat_fp = o_hat_fp.reshape(B_, C_, H_, W_freq_)

            # 15. Optional cross-freq conv (same as slow path, but no T dim)
            if self.cross_freq_conv_size > 0:
                o_flat_fp = o_hat_fp.reshape(B_ * C_ * H_, W_freq_)
                o_re = o_flat_fp.real[..., None]
                o_im = o_flat_fp.imag[..., None]
                pad = self.cross_freq_conv_size // 2
                o_re = jnp.pad(o_re, ((0, 0), (pad, pad), (0, 0)))
                o_im = jnp.pad(o_im, ((0, 0), (pad, pad), (0, 0)))
                o_re = nn.Conv(1, kernel_size=(self.cross_freq_conv_size,),
                               padding='VALID',
                               dtype=self.dtype, param_dtype=self.param_dtype,
                               name='cross_freq_re')(o_re)
                o_im = nn.Conv(1, kernel_size=(self.cross_freq_conv_size,),
                               padding='VALID',
                               dtype=self.dtype, param_dtype=self.param_dtype,
                               name='cross_freq_im')(o_im)
                o_hat_fp = (o_re[..., 0] + 1j * o_im[..., 0]).reshape(B_, C_, H_, W_freq_)

            # 16. iFFT → (B, C, H, W) → transpose → (B, H, W, C). irfft2 returns
            # fp32; cast back to compute dtype for the downstream Dense ops.
            o_spatial_fp = jnp.fft.irfft2(o_hat_fp, s=(H_, W_), axes=(2, 3))
            output_fp = o_spatial_fp.transpose(0, 2, 3, 1).astype(self.dtype)

            # 17. Output norm (same param names)
            if self.output_norm == 'rms':
                rms_scale_fp = self.param('rms_scale', nn.initializers.ones, (C_,))
                output_fp = _rms_norm(output_fp, rms_scale_fp.astype(self.dtype))
            elif self.output_norm == 'layer':
                output_fp = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype,
                                         name='output_layernorm')(output_fp)

            # 18. Output gate multiply + output projection
            output_fp = output_fp * gate_val_fp
            return nn.Dense(
                C_,
                dtype=self.dtype, param_dtype=self.param_dtype,
                name='output_proj'
            )(output_fp)

        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1
        d_k = self.delta_key_dim
        d_v = d_k  # Square state matrix
        n_h = C // d_k
        assert C % d_k == 0, f"channels={C} must be divisible by delta_key_dim={d_k}"

        # Log-space conversion helpers
        if self.use_goom:
            _to_log = to_goom
            _from_log = from_goom
        else:
            def _to_log(x):
                x_c = x.astype(jnp.complex64) if not jnp.iscomplexobj(x) else x
                return jnp.log(jnp.abs(x_c) + 1e-8) + 1j * jnp.angle(x_c)
            def _from_log(x):
                return jnp.exp(x)

        # Save raw input for position-independent gates
        x_raw = x if self.position_independent_gates else None

        # Optional position encoding
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # =====================================================================
        # 1. SPATIAL KERNEL (per head, shared across d_k dims within head)
        # =====================================================================
        k_shape = (n_h, self.kernel_size, self.kernel_size)

        def pad_kernel(k_spatial):
            """Pad kernel to input size for FFT."""
            pad_h = max(0, (H - self.kernel_size) // 2)
            pad_w = max(0, (W - self.kernel_size) // 2)
            pad_h_after = H - self.kernel_size - pad_h
            pad_w_after = W - self.kernel_size - pad_w
            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k_spatial[:, start_h:start_h+H, start_w:start_w+W]
            else:
                return jnp.pad(k_spatial, ((0, 0), (pad_h, max(0, pad_h_after)),
                                           (pad_w, max(0, pad_w_after))), mode='constant')

        kernel = self.param('kernel', nn.initializers.xavier_normal(), k_shape)
        k_padded = pad_kernel(kernel)
        W_hat = jnp.fft.rfft2(k_padded, axes=(1, 2))  # (n_h, H, W_freq)
        if not self.no_spectral_clamp:
            W_hat = _stable_spectral_magnitude(W_hat, rho=self.spectral_rho)
        W_hat = W_hat.astype(jnp.complex64)  # (n_h, H, W_freq)

        # =====================================================================
        # 2. QKV PROJECTION (Dense or spatial conv for cross-freq coupling)
        # =====================================================================
        x_flat = x.reshape(B * T, H, W, C)
        if self.qkv_conv_size > 1:
            if self.qkv_conv_separable:
                # Depthwise-separable: spatial mixing (depthwise) then channel mixing (pointwise)
                # Depthwise: K×K conv, groups=C → spatial neighbors mix per-channel
                qkv_proj = nn.Conv(C, kernel_size=(self.qkv_conv_size, self.qkv_conv_size),
                                   feature_group_count=C, padding='SAME',
                                   name='qkv_dw')(x_flat)
                # Pointwise: 1×1 conv C → 3C → channel mixing
                qkv_proj = nn.Dense(3 * C, name='qkv_proj')(qkv_proj)
            else:
                # Full conv: K×K with full channel mixing (expensive)
                qkv_proj = nn.Conv(3 * C, kernel_size=(self.qkv_conv_size, self.qkv_conv_size),
                                   padding='SAME', name='qkv_proj')(x_flat)
        else:
            qkv_proj = nn.Dense(3 * C, name='qkv_proj')(x_flat)
        qkv_proj = qkv_proj.reshape(B, T, H, W, 3 * C)
        q_in = qkv_proj[..., :C]        # (B, T, H, W, C)
        k_in = qkv_proj[..., C:2*C]
        v_in = qkv_proj[..., 2*C:]

        # =====================================================================
        # 3. SHORT TEMPORAL CONV (NEW: causal depthwise conv1d + SiLU)
        # =====================================================================
        if self.short_conv_size > 0:
            def _short_conv(x_5d, name):
                """(B, T, H, W, C) -> (B, T, H, W, C) with causal temporal conv + SiLU."""
                B_, T_, H_, W_, C_ = x_5d.shape
                # (B, T, H, W, C) -> (B*H*W, T, C)
                x_1d = x_5d.transpose(0, 2, 3, 1, 4).reshape(B_ * H_ * W_, T_, C_)
                # Causal left-pad
                x_1d = jnp.pad(x_1d, ((0, 0), (self.short_conv_size - 1, 0), (0, 0)))
                # Depthwise conv
                x_1d = nn.Conv(C_, kernel_size=(self.short_conv_size,),
                               feature_group_count=C_, padding='VALID', name=name)(x_1d)
                x_1d = jax.nn.silu(x_1d)
                # (B*H*W, T, C) -> (B, T, H, W, C)
                return x_1d.reshape(B_, H_, W_, T_, C_).transpose(0, 3, 1, 2, 4)

            q_in = _short_conv(q_in, 'short_conv_q')
            k_in = _short_conv(k_in, 'short_conv_k')
            v_in = _short_conv(v_in, 'short_conv_v')

        # =====================================================================
        # 4. INPUT GATES B_k, B_v (NEW: spatial input gates)
        # =====================================================================
        gate_input = x_raw if x_raw is not None else x
        ctx = gate_input.mean(axis=(2, 3))  # (B, T, C)
        if x_raw is None:
            if self.rope_mode == 'learned_t':
                T_d, C_d = ctx.shape[-2], ctx.shape[-1]
                t_embed = self.param('temporal_embed',
                                     nn.initializers.zeros, (T_d, C_d))
                ctx = ctx + t_embed[None, :, :]
            else:
                ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)
        n_gate = H * W_freq
        _spatial_gates = self.gate_type in ('conv', 'channel', 'dense_decay')

        def _gate(name, bounded=False, spatial=False):
            """Create a gate (same logic as MatrixDeltaNetCSSM)."""
            if self.gate_type == 'channel':
                if spatial:
                    gate_flat = gate_input.reshape(B * T, H, W, C)
                    raw = nn.sigmoid(nn.Conv(
                        C, kernel_size=(1, 1), padding='SAME', name=name
                    )(gate_flat))
                    return raw.reshape(B, T, H, W, C)
                else:
                    raw = nn.sigmoid(nn.Dense(n_h, name=name)(ctx))
                    raw = raw.reshape(B, T, n_h, 1, 1)
                    return 0.1 + 0.89 * raw if bounded else raw
            elif self.gate_type == 'factored':
                if spatial:
                    # Spatial-domain factored gate: H × W, broadcast over C
                    row = nn.Dense(H, name=f'{name}_h')(ctx)
                    col = nn.Dense(W, name=f'{name}_w')(ctx)
                    raw = nn.sigmoid(row[..., :, None] + col[..., None, :])
                    raw = raw.reshape(B, T, H, W, 1)
                    return raw
                else:
                    row = nn.Dense(H, name=f'{name}_h')(ctx)
                    col = nn.Dense(W_freq, name=f'{name}_w')(ctx)
                    raw = nn.sigmoid(row[..., :, None] + col[..., None, :])
                    raw = raw.reshape(B, T, 1, H, W_freq)
                    return 0.1 + 0.89 * raw if bounded else raw
            elif self.gate_type == 'scalar':
                if spatial:
                    raw = nn.sigmoid(nn.Dense(1, name=name)(ctx))
                    raw = raw.reshape(B, T, 1, 1, 1)
                    return raw
                else:
                    raw = nn.sigmoid(nn.Dense(1, name=name)(ctx))
                    raw = raw.reshape(B, T, 1, 1, 1)
                    return 0.1 + 0.89 * raw if bounded else raw
            else:  # 'dense'
                if spatial:
                    raw = nn.sigmoid(nn.Dense(H * W, name=name)(ctx))
                    raw = raw.reshape(B, T, H, W, 1)
                    return raw
                else:
                    raw = nn.sigmoid(nn.Dense(n_gate, name=name)(ctx))
                    raw = raw.reshape(B, T, 1, H, W_freq)
                    return 0.1 + 0.89 * raw if bounded else raw

        if self.use_input_gates:
            B_k = _gate('B_k', spatial=True)
            B_v = _gate('B_v', spatial=True)
            k_in = k_in * B_k
            v_in = v_in * B_v

        # =====================================================================
        # 5. OUTPUT GATE PROJECTION (NEW: computed early, SiLU or sigmoid)
        # =====================================================================
        gate_val = nn.Dense(C, bias_init=nn.initializers.constant(self.gate_proj_bias_init),
                             name='gate_proj')(x_flat)  # (B*T, H, W, C)
        gate_val = gate_val.reshape(B, T, H, W, C)
        if self.output_gate_act == 'silu':
            gate_val = jax.nn.silu(gate_val)
        else:
            gate_val = nn.sigmoid(gate_val)

        # =====================================================================
        # 6. GATES alpha, beta (recurrence gates, no C_gate)
        # =====================================================================
        alpha = _gate('alpha', bounded=True)    # [0.1, 0.99]
        beta = _gate('beta')                     # [0, 1]

        # Diagnostic sow for per-t gate variation analyses.
        self.sow('intermediates', 'alpha_t', alpha)
        self.sow('intermediates', 'beta_t', beta)

        # =====================================================================
        # 7. RESHAPE TO HEADS + FFT + L2 NORM
        # =====================================================================
        # Reshape into heads: (B, T, H, W, n_h, d_k)
        q_heads = q_in.reshape(B, T, H, W, n_h, d_k)
        k_heads = k_in.reshape(B, T, H, W, n_h, d_k)
        v_heads = v_in.reshape(B, T, H, W, n_h, d_v)

        # Transpose to (B, T, n_h, d_k, H, W) then rfft2
        q_spec = jnp.fft.rfft2(
            q_heads.transpose(0, 1, 4, 5, 2, 3), axes=(4, 5)
        )  # (B, T, n_h, d_k, H, W_freq)
        k_spec = jnp.fft.rfft2(
            k_heads.transpose(0, 1, 4, 5, 2, 3), axes=(4, 5)
        )  # (B, T, n_h, d_k, H, W_freq)
        v_spec = jnp.fft.rfft2(
            v_heads.transpose(0, 1, 4, 5, 2, 3), axes=(4, 5)
        )  # (B, T, n_h, d_v, H, W_freq)

        # L2-normalize k and q per head across freq bins
        def _spectral_l2_norm(z):
            """L2-normalize across freq dims (axes 4,5) per head per d_k component."""
            if not self.use_spectral_l2_norm:
                return z
            norm = jnp.sqrt(jnp.sum(jnp.abs(z) ** 2, axis=(4, 5), keepdims=True) + 1e-6)
            return z / norm

        def _key_vector_l2_norm(z):
            """L2-normalize across d_k dim (axis 3) per head per freq bin.
            Ensures ||k||²=1 so (I - β·k·k†) is a contraction. Always on."""
            norm = jnp.sqrt(jnp.sum(jnp.abs(z) ** 2, axis=3, keepdims=True) + 1e-6)
            return z / norm

        k_hat_norm = _key_vector_l2_norm(_spectral_l2_norm(k_spec))
        q_hat_norm = _spectral_l2_norm(q_spec)
        v_hat = v_spec

        # =====================================================================
        # 8. BUILD M_t (d_k×d_k transition) and Δ_t (d_v×d_k write)
        # =====================================================================
        k_outer = (k_hat_norm[..., :, None, :, :]
                   * jnp.conj(k_hat_norm[..., None, :, :, :]))
        I_dk = jnp.eye(d_k, dtype=jnp.complex64).reshape(1, 1, 1, d_k, d_k, 1, 1)

        # Broadcast alpha/beta
        if self.gate_type == 'channel':
            alpha_bc = alpha[..., None, None]
            beta_bc = beta[..., None, None]
        else:
            alpha_bc = alpha[:, :, :, None, None, :, :]
            beta_bc = beta[:, :, :, None, None, :, :]

        W_hat_bc = W_hat[None, None, :, None, None, :, :]

        M_t = alpha_bc.astype(jnp.complex64) * W_hat_bc * (I_dk - beta_bc * k_outer)

        vk_outer = (v_hat[..., :, None, :, :]
                    * jnp.conj(k_hat_norm[..., None, :, :, :]))
        Delta_t = beta_bc * vk_outer

        # =====================================================================
        # 9. MATRIX SCAN
        # =====================================================================
        M_scan = M_t.transpose(0, 1, 2, 5, 6, 3, 4)
        Delta_scan = Delta_t.transpose(0, 1, 2, 3, 5, 6, 4)
        M_scan_rep = jnp.repeat(M_scan, d_v, axis=2)
        Delta_scan_flat = Delta_scan.reshape(B, T, n_h * d_v, H, W_freq, d_k)

        _scan_mode = self.scan_mode
        if _scan_mode == 'pallas' and d_k == 2:
            from .pallas_scan import pallas_matrix_2x2_scan
            M_c = M_scan_rep.astype(jnp.complex64)
            U_c = Delta_scan_flat.astype(jnp.complex64)
            S_flat = pallas_matrix_2x2_scan(M_c, U_c)
        elif self.use_log:
            # Log-space scan (GOOM or standard log)
            K_log = _to_log(M_scan_rep)
            U_log = _to_log(Delta_scan_flat)
            if d_k == 2:
                _, S_log = jax.lax.associative_scan(
                    cssm_matrix_scan_op, (K_log, U_log), axis=1)
            elif d_k == 3:
                _, S_log = jax.lax.associative_scan(
                    cssm_3x3_matrix_scan_op, (K_log, U_log), axis=1)
            else:  # general d_k
                _, S_log = jax.lax.associative_scan(
                    cssm_general_matrix_scan_op, (K_log, U_log), axis=1)
            S_flat = _from_log(S_log)
        else:
            # Direct scan — no log transform
            K_c = M_scan_rep.astype(jnp.complex64)
            U_c = Delta_scan_flat.astype(jnp.complex64)
            if self.use_complex32 or self.use_complex16:
                # bfloat16/fp8 split precision
                split_dtype = jnp.float8_e4m3fn if self.use_complex16 else jnp.bfloat16
                K_re, K_im = complex64_to_linear_split(K_c, split_dtype)
                U_re, U_im = complex64_to_linear_split(U_c, split_dtype)
                if d_k == 2:
                    _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                        linear_split_2x2_scan_op,
                        (K_re, K_im, U_re, U_im), axis=1)
                elif d_k == 3:
                    _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                        linear_split_3x3_scan_op,
                        (K_re, K_im, U_re, U_im), axis=1)
                else:
                    _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                        linear_split_general_scan_op,
                        (K_re, K_im, U_re, U_im), axis=1)
                S_flat = linear_split_to_complex64(U_re_out, U_im_out)
            else:
                # Full complex64 precision
                if d_k == 2:
                    _, S_flat = jax.lax.associative_scan(
                        direct_2x2_scan_op, (K_c, U_c), axis=1)
                elif d_k == 3:
                    _, S_flat = jax.lax.associative_scan(
                        direct_3x3_scan_op, (K_c, U_c), axis=1)
                else:
                    _, S_flat = jax.lax.associative_scan(
                        direct_general_scan_op, (K_c, U_c), axis=1)

        S_t = S_flat.reshape(B, T, n_h, d_v, H, W_freq, d_k)

        # =====================================================================
        # 10. READOUT: o = S @ q per head per freq
        # =====================================================================
        q_read = q_hat_norm.transpose(0, 1, 2, 4, 5, 3)[:, :, :, None, :, :, :]
        o_hat = jnp.sum(S_t * q_read, axis=-1)  # (B, T, n_h, d_v, H, W_freq)
        o_hat = o_hat.reshape(B, T, C, H, W_freq)

        # =====================================================================
        # 10b. CROSS-FREQUENCY CONV (optional: mix nearby freq bins)
        # =====================================================================
        if self.cross_freq_conv_size > 0:
            # o_hat: (B, T, C, H, W_freq) — apply 1D depthwise conv across W_freq
            # Reshape to (B*T*C*H, W_freq, 1) for Conv, then back
            # We treat real and imag parts separately since nn.Conv is real-valued
            o_flat = o_hat.reshape(B * T * C * H, W_freq)
            o_re = o_flat.real[..., None]  # (N, W_freq, 1)
            o_im = o_flat.imag[..., None]  # (N, W_freq, 1)
            # Symmetric padding to preserve W_freq size
            pad = self.cross_freq_conv_size // 2
            o_re = jnp.pad(o_re, ((0, 0), (pad, pad), (0, 0)))
            o_im = jnp.pad(o_im, ((0, 0), (pad, pad), (0, 0)))
            o_re = nn.Conv(1, kernel_size=(self.cross_freq_conv_size,),
                           padding='VALID', name='cross_freq_re')(o_re)
            o_im = nn.Conv(1, kernel_size=(self.cross_freq_conv_size,),
                           padding='VALID', name='cross_freq_im')(o_im)
            o_hat = (o_re[..., 0] + 1j * o_im[..., 0]).reshape(B, T, C, H, W_freq)

        # =====================================================================
        # 11. OUTPUT: iFFT → norm → gate → project (MODIFIED vs MatrixDeltaNetCSSM)
        # =====================================================================
        o_spatial = jnp.fft.irfft2(o_hat, s=(H, W), axes=(3, 4))  # (B, T, C, H, W)
        output = o_spatial.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)

        # Output norm
        if self.output_norm == 'rms':
            rms_scale = self.param('rms_scale', nn.initializers.ones, (C,))
            output = _rms_norm(output, rms_scale)
        elif self.output_norm == 'layer':
            output = nn.LayerNorm(name='output_layernorm')(output)

        # =====================================================================
        # 11b. SPATIAL ATTENTION (InT-style, optional)
        # Modes:
        #   'spectral'    — Channel collapse → FFT → scalar scan w/ spectral conv → iFFT → sigmoid
        #   'elementwise' — Dense(C→1) collapse → temporal EMA → sigmoid gate
        #   'qk'          — Q·K dot product → temporal EMA → sigmoid gate
        # =====================================================================
        if self.int_attention_mode == 'spectral':
            # Original InT: channel collapse + spectral conv recurrence
            attn_input = nn.Dense(1, name='attn_collapse')(output).squeeze(-1)  # (B, T, H, W)

            # Attention spatial kernel (separate from main W_hat)
            attn_ks = self.attn_kernel_size
            attn_k = self.param('attn_kernel', nn.initializers.xavier_normal(),
                                (1, attn_ks, attn_ks))
            if attn_ks > H or attn_ks > W:
                sh = (attn_ks - H) // 2
                sw = (attn_ks - W) // 2
                attn_k_padded = attn_k[:, sh:sh+H, sw:sw+W]
            else:
                ph = (H - attn_ks) // 2
                pw = (W - attn_ks) // 2
                attn_k_padded = jnp.pad(attn_k,
                    ((0, 0), (ph, H - attn_ks - ph), (pw, W - attn_ks - pw)))
            W_attn = jnp.fft.rfft2(attn_k_padded, axes=(1, 2))          # (1, H, W_freq)
            W_attn = _stable_spectral_magnitude(W_attn, rho=self.spectral_rho)
            W_attn = W_attn.squeeze(0)                                    # (H, W_freq)

            gamma = nn.sigmoid(
                self.param('attn_gamma', nn.initializers.constant(2.0), (1,)))

            attn_spec = jnp.fft.rfft2(attn_input, axes=(2, 3))           # (B, T, H, W_freq)

            K_attn = jnp.broadcast_to(
                (gamma * W_attn).astype(jnp.complex64)[None, None],
                (B, T, H, W_freq))
            U_attn = ((1 - gamma) * attn_spec).astype(jnp.complex64)

            if self.use_complex32:
                K_re, K_im = complex64_to_linear_split(K_attn, jnp.bfloat16)
                U_re, U_im = complex64_to_linear_split(U_attn, jnp.bfloat16)
                _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                    linear_split_scalar_scan_op, (K_re, K_im, U_re, U_im), axis=1)
                attn_state = linear_split_to_complex64(U_re_out, U_im_out)
            else:
                K_log = _to_log(K_attn)
                U_log = _to_log(U_attn)
                _, S_log = jax.lax.associative_scan(
                    cssm_scalar_scan_op, (K_log, U_log), axis=1)
                attn_state = _from_log(S_log)

            attn_spatial = jnp.fft.irfft2(attn_state, s=(H, W), axes=(2, 3))
            attn_map = nn.sigmoid(attn_spatial)[..., None]                # (B, T, H, W, 1)
            output = output * attn_map

        elif self.int_attention_mode in ('elementwise', 'qk'):
            if self.int_attention_mode == 'qk':
                # Q·K attention: project to low-dim, dot product → scalar
                d_a = self.int_attn_dim
                q_attn = nn.Dense(d_a, name='attn_q')(output)  # (B, T, H, W, d_a)
                k_attn = nn.Dense(d_a, name='attn_k')(output)  # (B, T, H, W, d_a)
                attn_logits = jnp.sum(q_attn * k_attn, axis=-1) / jnp.sqrt(d_a)  # (B, T, H, W)
            else:  # elementwise
                # Channel collapse → scalar per spatial location
                attn_logits = nn.Dense(1, name='attn_collapse')(output).squeeze(-1)  # (B, T, H, W)

            # Temporal scalar recurrence: s_t = gamma * s_{t-1} + (1 - gamma) * input_t
            # Uses log-space scan for numerical stability (same as spectral variant)
            gamma = nn.sigmoid(
                self.param('attn_gamma', nn.initializers.constant(2.0), (1,)))

            K_attn = jnp.broadcast_to(gamma, (B, T, H, W)).astype(jnp.complex64)
            U_attn = ((1 - gamma) * attn_logits).astype(jnp.complex64)

            if self.use_complex32:
                K_re, K_im = complex64_to_linear_split(K_attn, jnp.bfloat16)
                U_re, U_im = complex64_to_linear_split(U_attn, jnp.bfloat16)
                _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                    linear_split_scalar_scan_op, (K_re, K_im, U_re, U_im), axis=1)
                attn_state = linear_split_to_complex64(U_re_out, U_im_out).real
            else:
                K_log = _to_log(K_attn)
                U_log = _to_log(U_attn)
                _, S_log = jax.lax.associative_scan(
                    cssm_scalar_scan_op, (K_log, U_log), axis=1)
                attn_state = _from_log(S_log).real

            attn_map = nn.sigmoid(attn_state)[..., None]  # (B, T, H, W, 1)
            output = output * attn_map

        # Output gate
        output = output * gate_val

        # --- Temporal SSL projection (sown for LeJEPA-recurrent loss) ---
        # Un-normalized: SIGReg needs embeddings that can match N(0, I) including
        # both moments and shape. L2-normalizing forces them onto the unit sphere
        # and would mask collapse at z=0.
        if self.ssl_proj_dim > 0:
            pooled_t = output.mean(axis=(2, 3)).astype(jnp.float32)            # (B, T, C)
            pooled_t = nn.Dense(self.ssl_proj_dim * 4, name='ssl_proj_in',
                                param_dtype=jnp.float32)(pooled_t)
            pooled_t = nn.gelu(pooled_t)
            pooled_t = nn.Dense(self.ssl_proj_dim, name='ssl_proj_out',
                                param_dtype=jnp.float32)(pooled_t)            # (B, T, P)
            self.sow('intermediates', 'cssm_temporal_proj', pooled_t)

        return nn.Dense(C, name='output_proj')(output)


class SpatialAttentionCSSM(nn.Module):
    """Standard multi-head self-attention as a CSSM-compatible block.

    Implements pre-norm transformer block(s) that plug into SimpleCSSM's
    residual framework. Returns delta for external x + cssm(x) residual.
    """
    channels: int
    num_heads: int = 4
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0
    layer_scale_init: float = 1e-6
    use_temporal_attn: bool = False  # If True, add temporal attention after spatial
    # Unused CSSM kwargs (accepted for interface compat)
    kernel_size: int = 11
    block_size: int = 1
    rope_mode: str = 'none'

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B, T, H, W, C = x.shape
        x_in = x  # Save for delta computation
        head_dim = C // self.num_heads
        scale = head_dim ** -0.5
        has_rng = self.has_rng('dropout')

        # --- Learnable 2D spatial position embedding (ViT-style) ---
        pos_embed = self.param('pos_embed', nn.initializers.normal(0.02), (1, 1, H, W, C))
        x = x + pos_embed

        # --- Spatial attention (per-frame) ---
        # Reshape: (B, T, H, W, C) -> (B*T, H*W, C)
        x_flat = x.reshape(B * T, H * W, C)

        # Pre-norm
        x_norm = nn.LayerNorm(name='norm1')(x_flat)

        # Multi-head self-attention
        qkv = nn.Dense(3 * C, name='attn_qkv')(x_norm)
        qkv = qkv.reshape(B * T, H * W, 3, self.num_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B*T, N, heads, head_dim)
        out = jax.nn.dot_product_attention(q, k, v, scale=scale)
        out = out.reshape(B * T, H * W, C)
        out = nn.Dense(C, name='attn_proj')(out)

        # Layer scale + drop path + residual
        gamma1 = self.param('gamma1', nn.initializers.constant(self.layer_scale_init), (C,))
        out = out * gamma1
        if self.drop_path_rate > 0.0 and has_rng:
            rng = self.make_rng('dropout')
            keep = 1.0 - self.drop_path_rate
            mask = jax.random.bernoulli(rng, keep, (out.shape[0], 1, 1))
            out = out * mask / keep
        x_flat = x_flat + out

        # --- MLP ---
        residual = x_flat
        x_flat = nn.LayerNorm(name='norm2')(x_flat)
        x_flat = nn.Dense(int(C * self.mlp_ratio), name='mlp_fc1')(x_flat)
        x_flat = nn.gelu(x_flat)
        x_flat = nn.Dense(C, name='mlp_fc2')(x_flat)
        gamma2 = self.param('gamma2', nn.initializers.constant(self.layer_scale_init), (C,))
        x_flat = x_flat * gamma2
        if self.drop_path_rate > 0.0 and has_rng:
            rng = self.make_rng('dropout')
            keep = 1.0 - self.drop_path_rate
            mask = jax.random.bernoulli(rng, keep, (x_flat.shape[0], 1, 1))
            x_flat = x_flat * mask / keep
        x_flat = residual + x_flat

        # Reshape back: (B*T, H*W, C) -> (B, T, H, W, C)
        x_out = x_flat.reshape(B, T, H, W, C)

        # --- Optional temporal attention ---
        if self.use_temporal_attn:
            # (B, T, H, W, C) -> (B*H*W, T, C)
            x_temp = x_out.transpose(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
            residual_t = x_temp
            x_temp = nn.LayerNorm(name='norm_t')(x_temp)
            # Temporal MHA
            qkv_t = nn.Dense(3 * C, name='t_attn_qkv')(x_temp)
            qkv_t = qkv_t.reshape(B * H * W, T, 3, self.num_heads, head_dim)
            q_t, k_t, v_t = qkv_t[:, :, 0], qkv_t[:, :, 1], qkv_t[:, :, 2]
            out_t = jax.nn.dot_product_attention(q_t, k_t, v_t, scale=scale)
            out_t = out_t.reshape(B * H * W, T, C)
            out_t = nn.Dense(C, name='t_attn_proj')(out_t)
            gamma_t = self.param('gamma_t', nn.initializers.constant(self.layer_scale_init), (C,))
            out_t = out_t * gamma_t
            if self.drop_path_rate > 0.0 and has_rng:
                rng = self.make_rng('dropout')
                keep = 1.0 - self.drop_path_rate
                mask = jax.random.bernoulli(rng, keep, (out_t.shape[0], 1, 1))
                out_t = out_t * mask / keep
            x_temp = residual_t + out_t
            x_out = x_temp.reshape(B, H, W, T, C).transpose(0, 3, 1, 2, 4)

        # Return delta for SimpleCSSM's external residual: x + cssm(x)
        return x_out - x_in


class Mamba2SeqCSSM(nn.Module):
    """Mamba-2 on flattened tokens — pure 1D sequence model (no spatial structure).

    Flattens (B, T, H, W, C) → (B, T*H*W, C), runs Mamba-2 style SSM,
    then reshapes back. Tests whether spatial inductive bias helps vs
    pure sequence processing.

    Architecture (matches Mamba-2):
        1. in_proj(2*inner_dim) → (x_main, z)
        2. Causal 1D depthwise conv → SiLU
        3. dt = softplus(Dense), B_proj = Dense(N), C_proj = Dense(N)
        4. Discretize: A_bar = exp(-dt), B_bar = dt * B_proj
        5. N independent scalar scans via ssd_scan
        6. Readout: y = sum(S * C_proj, axis=-1)
        7. output = norm(y) * silu(z) → out_proj(C)

    Returns delta for SimpleCSSM's external residual.
    """
    channels: int
    state_dim: int = 16
    short_conv_size: int = 4
    expand_factor: int = 2
    ssd_chunk_size: int = 8
    flatten_mode: str = 'temporal_spatial'  # 'temporal_spatial' or 'per_frame'
    # Unused CSSM kwargs (accepted for interface compat)
    kernel_size: int = 11
    block_size: int = 1
    rope_mode: str = 'none'

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B, T, H, W, C = x.shape
        x_in = x
        inner_dim = self.expand_factor * C
        N = self.state_dim

        # === Flatten to 1D sequence ===
        if self.flatten_mode == 'per_frame':
            # Flatten spatial only, keep T separate: (B, T, H*W, C)
            x = x.reshape(B, T, H * W, C)
            L = H * W
            # Merge batch and time: (B*T, L, C)
            x = x.reshape(B * T, L, C)
            effective_B = B * T
        else:
            # Full flatten: (B, T*H*W, C)
            x = x.reshape(B, T * H * W, C)
            L = T * H * W
            effective_B = B

        # Pad L to be divisible by ssd_chunk_size
        chunk = self.ssd_chunk_size
        pad_len = (chunk - L % chunk) % chunk
        if pad_len > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))
        L_padded = L + pad_len

        # === 1. Input expansion: x → (x_main, z) ===
        xz = nn.Dense(2 * inner_dim, name='in_proj')(x)  # (eB, L_padded, 2*inner)
        x_main = xz[..., :inner_dim]
        z = xz[..., inner_dim:]

        # === 2. Causal 1D depthwise conv → SiLU ===
        if self.short_conv_size > 0:
            x_main = jnp.pad(x_main, ((0, 0), (self.short_conv_size - 1, 0), (0, 0)))
            x_main = nn.Conv(inner_dim, kernel_size=(self.short_conv_size,),
                             feature_group_count=inner_dim, padding='VALID',
                             name='short_conv')(x_main)
        x_main = jax.nn.silu(x_main)

        # === 3. Input-dependent SSM parameters ===
        dt_raw = nn.Dense(inner_dim, name='dt_proj')(x_main)  # (eB, L_padded, inner)
        dt = jax.nn.softplus(dt_raw)

        B_proj = nn.Dense(N, name='B_proj')(x_main)  # (eB, L_padded, N)
        C_proj = nn.Dense(N, name='C_proj')(x_main)  # (eB, L_padded, N)

        # === 4. Discretize ===
        # Learned per-channel log decay (negative, like Mamba)
        A_log = self.param('A_log', nn.initializers.normal(0.02), (inner_dim,))
        A = -jnp.exp(A_log)  # (inner_dim,) — negative real

        # A_bar = exp(A * dt): (eB, L_padded, inner_dim)
        # Clamp A*dt to prevent exp() underflow → 0 → log(0)=NaN inside ssd_scan
        A_bar = jnp.exp(jnp.clip(A[None, None, :] * dt, a_min=-20.0))
        # B_bar = dt * B_proj: expand to (eB, L_padded, inner_dim, N)
        B_bar = dt[..., :, None] * B_proj[..., None, :]  # (eB, L_padded, inner_dim, N)

        # === 5. Run N independent scalar scans ===
        # x_main: (eB, L_padded, inner_dim)
        # For each state dim n, run: h_t[n] = A_bar * h_{t-1}[n] + B_bar[n] * x_main
        # Input for scan: x_main * B_bar → (eB, L_padded, inner_dim, N)
        U = x_main[..., :, None] * B_bar  # (eB, L_padded, inner_dim, N)

        # Reshape for ssd_scan: merge inner_dim and N → (eB, L_padded, inner_dim*N)
        A_scan = jnp.broadcast_to(A_bar[..., None], U.shape)  # (eB, L_padded, inner_dim, N)
        A_scan = A_scan.reshape(effective_B, L_padded, inner_dim * N)
        U_scan = U.reshape(effective_B, L_padded, inner_dim * N)

        # ssd_scan handles the chunked associative scan
        # ssd_scan returns complex64 (it uses log-space internally), take .real since we're in real space
        S = ssd_scan(A_scan, U_scan, chunk_size=chunk).real  # (eB, L_padded, inner_dim*N)
        S = S.reshape(effective_B, L_padded, inner_dim, N)

        # === 6. Readout: y = sum(S * C_proj, axis=-1) ===
        y = jnp.sum(S * C_proj[:, :, None, :], axis=-1)  # (eB, L_padded, inner_dim)

        # === 7. Output: norm(y) * silu(z) → out_proj ===
        rms_scale = self.param('rms_scale', nn.initializers.ones, (inner_dim,))
        y = _rms_norm(y, rms_scale)
        output = y * jax.nn.silu(z)
        output = nn.Dense(C, name='out_proj')(output)  # (eB, L_padded, C)

        # Remove padding
        output = output[:, :L]

        # === Reshape back to 5D ===
        if self.flatten_mode == 'per_frame':
            output = output.reshape(B, T, H, W, C)
        else:
            output = output.reshape(B, T, H, W, C)

        return output - x_in


class GDNSeqCSSM(nn.Module):
    """Gated DeltaNet on flattened tokens — pure 1D sequence model (no spatial structure).

    Same delta rule as GatedDeltaNetCSSM but NO FFT/spectral operations:
        M_t = alpha * (I - beta * k·k^T)        transition (dk×dk)
        Δ_t = beta * v·k^T                       rank-1 write
        S_t = M_t · S_{t-1} + Δ_t               state update
        o_t = S_t @ q                            readout

    Flattens (B,T,H,W,C) → (B, T*H*W, C) then runs the delta rule
    recurrence on the 1D token sequence.

    Returns delta for SimpleCSSM's external residual.
    """
    channels: int
    delta_key_dim: int = 2
    short_conv_size: int = 4
    flatten_mode: str = 'temporal_spatial'
    # Unused CSSM kwargs (accepted for interface compat)
    kernel_size: int = 11
    block_size: int = 1
    rope_mode: str = 'none'

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B, T, H, W, C = x.shape
        x_in = x
        d_k = self.delta_key_dim
        d_v = d_k
        n_h = C // d_k
        assert C % d_k == 0, f"channels={C} must be divisible by delta_key_dim={d_k}"

        # === Flatten to 1D sequence ===
        if self.flatten_mode == 'per_frame':
            x = x.reshape(B * T, H * W, C)
            L = H * W
            effective_B = B * T
        else:
            x = x.reshape(B, T * H * W, C)
            L = T * H * W
            effective_B = B

        # === 1. QKV projection ===
        qkv = nn.Dense(3 * C, name='qkv_proj')(x)  # (eB, L, 3C)
        q_in = qkv[..., :C]
        k_in = qkv[..., C:2*C]
        v_in = qkv[..., 2*C:]

        # === 2. Short temporal conv + SiLU ===
        if self.short_conv_size > 0:
            def _short_conv(x_3d, name):
                """(eB, L, C) → (eB, L, C) with causal conv + SiLU."""
                x_pad = jnp.pad(x_3d, ((0, 0), (self.short_conv_size - 1, 0), (0, 0)))
                x_out = nn.Conv(C, kernel_size=(self.short_conv_size,),
                                feature_group_count=C, padding='VALID', name=name)(x_pad)
                return jax.nn.silu(x_out)

            q_in = _short_conv(q_in, 'short_conv_q')
            k_in = _short_conv(k_in, 'short_conv_k')
            v_in = _short_conv(v_in, 'short_conv_v')

        # === 3. Output gate (SiLU, computed early) ===
        gate_val = jax.nn.silu(nn.Dense(C, name='gate_proj')(x))  # (eB, L, C)

        # === 4. Gates alpha, beta ===
        ctx = x.mean(axis=1)  # (eB, C)
        alpha_raw = nn.sigmoid(nn.Dense(n_h, name='alpha')(ctx))  # (eB, n_h)
        alpha = 0.1 + 0.89 * alpha_raw  # [0.1, 0.99]
        beta = nn.sigmoid(nn.Dense(n_h, name='beta')(ctx))  # (eB, n_h)

        # === 5. Reshape to heads ===
        q_heads = q_in.reshape(effective_B, L, n_h, d_k)
        k_heads = k_in.reshape(effective_B, L, n_h, d_k)
        v_heads = v_in.reshape(effective_B, L, n_h, d_v)

        # L2-normalize k across d_k dim
        k_norm = jnp.sqrt(jnp.sum(k_heads ** 2, axis=-1, keepdims=True) + 1e-6)
        k_heads = k_heads / k_norm

        # === 6. Build M_t and Delta_t ===
        # k_outer: (eB, L, n_h, d_k, d_k)
        k_outer = k_heads[..., :, None] * k_heads[..., None, :]
        I_dk = jnp.eye(d_k, dtype=jnp.float32)

        # Broadcast gates: alpha (eB, n_h) → (eB, 1, n_h, 1, 1)
        alpha_bc = alpha[:, None, :, None, None]
        beta_bc = beta[:, None, :, None, None]

        # M_t = alpha * (I - beta * k·kT): (eB, L, n_h, d_k, d_k)
        M_t = alpha_bc * (I_dk - beta_bc * k_outer)

        # Delta_t = beta * v·kT: (eB, L, n_h, d_v, d_k)
        vk_outer = v_heads[..., :, None] * k_heads[..., None, :]
        Delta_t = beta_bc * vk_outer

        # === 7. Matrix scan ===
        # Reshape for scan: merge n_h into batch, repeat M per d_v row
        # M: (eB, L, n_h, d_k, d_k) → (eB, L, n_h*d_v, d_k, d_k) via repeat
        M_scan = jnp.repeat(M_t, d_v, axis=2)  # (eB, L, n_h*d_v, d_k, d_k)
        # Delta: (eB, L, n_h, d_v, d_k) → (eB, L, n_h*d_v, d_k)
        Delta_scan = Delta_t.reshape(effective_B, L, n_h * d_v, d_k)

        # Cast to complex for scan ops (real-valued, imag=0)
        M_c = M_scan.astype(jnp.complex64)
        U_c = Delta_scan.astype(jnp.complex64)

        if d_k == 2:
            _, S_flat = jax.lax.associative_scan(
                direct_2x2_scan_op, (M_c, U_c), axis=1)
        elif d_k == 3:
            _, S_flat = jax.lax.associative_scan(
                direct_3x3_scan_op, (M_c, U_c), axis=1)
        else:
            _, S_flat = jax.lax.associative_scan(
                direct_general_scan_op, (M_c, U_c), axis=1)

        S_t = S_flat.real.reshape(effective_B, L, n_h, d_v, d_k)

        # === 8. Readout: o = S @ q ===
        q_read = q_heads[:, :, :, None, :]  # (eB, L, n_h, 1, d_k)
        o = jnp.sum(S_t * q_read, axis=-1)  # (eB, L, n_h, d_v)
        output = o.reshape(effective_B, L, C)

        # === 9. Output: RMSNorm → gate → project ===
        rms_scale = self.param('rms_scale', nn.initializers.ones, (C,))
        output = _rms_norm(output, rms_scale)
        output = output * gate_val
        output = nn.Dense(C, name='output_proj')(output)

        # === Reshape back to 5D ===
        if self.flatten_mode == 'per_frame':
            output = output.reshape(B, T, H, W, C)
        else:
            output = output.reshape(B, T, H, W, C)

        return output - x_in


class ConvSSMCSSM(nn.Module):
    """ConvSSM — Real-valued depthwise conv + temporal scan (NVlabs arXiv:2310.19694).

    Keeps 5D (B,T,H,W,C) structure — HAS spatial inductive bias.
    All operations in real spatial domain — NO Fourier transform.

    Architecture:
        1. Depthwise spatial conv (kernel_size × kernel_size) for spatial mixing
        2. Learned per-channel scalar decay A ∈ (0, 1) via sigmoid
        3. Input gate: B_bar = Dense(C) per-timestep
        4. State update: S_t = A * S_{t-1} + B_bar * conv(x_t)  (per spatial position)
        5. Output gate: C_bar = Dense(C), y = S * C_bar
        6. Output projection Dense(C)

    No FFT, no in_proj expansion, no z-branch, no SiLU gating.

    Returns delta for SimpleCSSM's external residual.
    """
    channels: int
    kernel_size: int = 3
    # Unused CSSM kwargs (accepted for interface compat)
    block_size: int = 1
    rope_mode: str = 'none'

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B, T, H, W, C = x.shape
        x_in = x

        # === 1. Depthwise spatial conv for spatial mixing ===
        x_flat = x.reshape(B * T, H, W, C)
        x_conv = nn.Conv(C, kernel_size=(self.kernel_size, self.kernel_size),
                         feature_group_count=C, padding='SAME',
                         name='spatial_conv')(x_flat)  # (B*T, H, W, C)
        x_conv = x_conv.reshape(B, T, H, W, C)

        # === 2. Learned per-channel decay A ∈ (0, 1) ===
        A_logit = self.param('A_logit', nn.initializers.constant(2.0), (C,))
        A = nn.sigmoid(A_logit)  # (C,) in (0, 1)
        # Broadcast: (C,) → (B, T, H, W, C)
        A_broadcast = jnp.broadcast_to(A[None, None, None, None, :], (B, T, H, W, C))

        # === 3. Input gate B_bar ===
        B_bar = nn.Dense(C, name='B_proj')(x_flat).reshape(B, T, H, W, C)

        # Modulated input
        U = B_bar * x_conv  # (B, T, H, W, C)

        # === 4. Scalar scan: S_t = A * S_{t-1} + U_t per spatial position ===
        # Reshape for scan: (B, T, H*W*C)
        A_scan = A_broadcast.reshape(B, T, H * W * C)
        U_scan = U.reshape(B, T, H * W * C)

        _, S_scan = jax.lax.associative_scan(
            direct_scalar_scan_op,
            (A_scan.astype(jnp.complex64), U_scan.astype(jnp.complex64)),
            axis=1
        )
        S = S_scan.real.reshape(B, T, H, W, C)

        # === 5. Output gate C_bar ===
        C_bar = nn.Dense(C, name='C_proj')(x_flat).reshape(B, T, H, W, C)
        output = S * C_bar

        # === 6. Output projection ===
        output = nn.Dense(C, name='out_proj')(output.reshape(B * T, H, W, C))
        output = output.reshape(B, T, H, W, C)

        return output - x_in


def _fft_conv_1d_per_channel(x, k, axis):
    """Per-channel 1D circular convolution via FFT along `axis`.

    x: (..., L_axis, ..., C)  real
    k: (C, L_axis)             real
    Returns y of same shape as x.
    """
    L = k.shape[-1]
    Xf = jnp.fft.rfft(x, n=L, axis=axis)                                 # (..., L//2+1, ..., C)
    Kf = jnp.fft.rfft(k, n=L, axis=-1)                                   # (C, L//2+1)
    shape_bc = [1] * x.ndim
    shape_bc[axis] = Kf.shape[-1]
    shape_bc[-1] = Kf.shape[0]
    Kf_bc = jnp.transpose(Kf, (1, 0)).reshape(shape_bc)                  # broadcast-ready
    Yf = Xf * Kf_bc
    return jnp.fft.irfft(Yf, n=L, axis=axis)


class _S4DKernel1D(nn.Module):
    """1D S4D diagonal-complex state-space kernel. Returns real kernel (H, L).

    S4D (Gu et al. 2022, "On the Parameterization and Initialization of Diagonal
    State Space Models") is the diagonal variant of S4 — drops the DPLR low-rank
    correction while keeping almost all the performance on LRA/continuous signals.
    We use it here as the per-axis kernel for S4ND.

    State stored as conjugate pairs (half the total state dim); output kernel
    recovered via 2 * Re(sum). All complex math in complex64, real cast at end.

    Fields:
      H: channels (one SSM per channel)
      N: total state dim (half is stored; output doubled via conjugate symmetry)
      L: output kernel length
    """
    H: int
    N: int
    L: int

    @nn.compact
    def __call__(self):
        half_N = max(self.N // 2, 1)

        # Λ = -exp(Λ_log) + i * Λ_im. Negative real part → contractive discrete kernel.
        Lambda_log = self.param(
            'Lambda_log',
            nn.initializers.constant(jnp.log(0.5)),
            (half_N,),
        )
        Lambda_imag = self.param(
            'Lambda_imag',
            lambda k, s: jax.random.uniform(k, s, minval=0.0, maxval=2.0 * jnp.pi),
            (half_N,),
        )
        # B and C stored as (2, half_N) / (H, 2, half_N) real (= re/im split).
        B_ri = self.param(
            'B_ri',
            lambda k, s: jax.random.normal(k, s) / jnp.sqrt(2.0 * half_N),
            (2, half_N),
        )
        C_ri = self.param(
            'C_ri',
            lambda k, s: jax.random.normal(k, s) / jnp.sqrt(half_N),
            (self.H, 2, half_N),
        )
        log_Delta = self.param(
            'log_Delta',
            lambda k, s: jax.random.uniform(
                k, s, minval=jnp.log(1e-3), maxval=jnp.log(1e-1)),
            (self.H,),
        )

        Lambda_log = Lambda_log.astype(jnp.float32)
        Lambda_imag = Lambda_imag.astype(jnp.float32)
        Lambda = (-jnp.exp(Lambda_log) + 1j * Lambda_imag).astype(jnp.complex64)  # (half_N,)
        B_cplx = (B_ri[0] + 1j * B_ri[1]).astype(jnp.complex64)                  # (half_N,)
        C_cplx = (C_ri[:, 0] + 1j * C_ri[:, 1]).astype(jnp.complex64)            # (H, half_N)
        Delta = jnp.exp(log_Delta.astype(jnp.float32))                           # (H,)

        # Bilinear (Tustin) discretization:
        #   Ā = (1 + Δ/2 · Λ) / (1 - Δ/2 · Λ);   B̄ = Δ B / (1 - Δ/2 · Λ)
        dL = (Delta[:, None] * Lambda[None, :]).astype(jnp.complex64)            # (H, half_N)
        A_bar = (1.0 + 0.5 * dL) / (1.0 - 0.5 * dL)                              # (H, half_N)
        B_bar = Delta[:, None].astype(jnp.complex64) * B_cplx[None, :] / (1.0 - 0.5 * dL)

        # K̄[l] = C̄ Ā^l B̄ summed over state. Use log-powered Ā for speed.
        # (H, half_N, L): A_bar^l via exp(l * log Ā).
        log_A = jnp.log(A_bar + 1e-12)                                           # (H, half_N) complex
        powers = jnp.arange(self.L, dtype=jnp.float32)
        A_pow = jnp.exp(log_A[..., None] * powers[None, None, :])                # (H, half_N, L)
        # kernel[h, l] = sum_n C[h,n] * A_bar^l[h,n] * B_bar[h,n]
        k_cplx = jnp.einsum('hn,hnl,hn->hl', C_cplx, A_pow, B_bar)               # (H, L)
        return (2.0 * jnp.real(k_cplx)).astype(jnp.float32)                      # real (H, L)


class S4NDCSSM(nn.Module):
    """S4ND — separable per-axis S4 along spatial dims (2D variant).

    Reference: Nguyen et al. "S4ND: Modeling Images and Videos as Multidimensional
    Signals Using State Spaces" (NeurIPS 2022). The paper's S4ND uses DPLR;
    this port uses the simpler S4D (diagonal-only) parameterization per axis,
    which is functionally equivalent when not using HiPPO-LegS init.

    Architecture:
      1. Per-channel 1D S4D kernel for H axis → (C, H).
      2. Per-channel 1D S4D kernel for W axis → (C, W).
      3. Apply separable 2D convolution via two 1D FFT convs (H then W).
      4. Optional bidirectional branch (reversed kernels) summed in.
      5. Per-channel D-term skip scale.
      6. Output projection Dense(C).

    No temporal state: each of the T frames is processed independently
    (frames are the batch axis here). For pathfinder (T=1) this is the natural
    single-frame spatial encoder.

    Returns delta for SimpleCSSM's external residual.
    """
    channels: int
    d_state: int = 64
    bidirectional: bool = True
    dropout: float = 0.0
    # Unused CSSM interface kwargs (accepted for compat with SimpleCSSM dispatch).
    kernel_size: int = 3
    block_size: int = 1
    rope_mode: str = 'none'

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B, T, H, W, C = x.shape
        x_in = x

        # Per-axis kernels: one per channel, per axis.
        kH = _S4DKernel1D(H=C, N=self.d_state, L=H, name='kH')()             # (C, H)
        kW = _S4DKernel1D(H=C, N=self.d_state, L=W, name='kW')()             # (C, W)

        # Reshape into (B*T, H, W, C) for spatial convolution.
        xf = x.reshape(B * T, H, W, C).astype(jnp.float32)

        # Forward branch: FFT conv along H, then W.
        yf = _fft_conv_1d_per_channel(xf, kH, axis=1)
        yf = _fft_conv_1d_per_channel(yf, kW, axis=2)

        if self.bidirectional:
            # Reverse branch: flip kernels along length axis.
            kH_r = kH[:, ::-1]
            kW_r = kW[:, ::-1]
            yf_bw = _fft_conv_1d_per_channel(xf, kH_r, axis=1)
            yf_bw = _fft_conv_1d_per_channel(yf_bw, kW_r, axis=2)
            yf = yf + yf_bw

        # Per-channel D-term skip scale (standard S4 D).
        D = self.param('D', nn.initializers.ones, (C,))
        yf = yf + D[None, None, None, :].astype(jnp.float32) * xf

        yf = yf.reshape(B, T, H, W, C)
        if self.dropout > 0:
            yf = nn.Dropout(rate=self.dropout, deterministic=False)(yf)

        # Cast back and project.
        y = yf.astype(x.dtype)
        y = nn.Dense(C, name='out_proj')(y.reshape(B * T, H, W, C))
        y = y.reshape(B, T, H, W, C)
        return y - x_in


class _S4NDLegSKernel1D(nn.Module):
    """1D S4 kernel with HiPPO-LegS DPLR initialization (canonical S4-LegS).

    Faithful re-implementation of the S4 LegS kernel from Gu et al. 2021/2022.
    Uses conjugate-pair compression (store half_N = N // 2 complex state
    components; full-N kernel recovered via `2 · Re(...)`) — same convention
    as `_S4DKernel1D`. DPLR rank-1 correction acts in the half-N space; for
    real-valued P, Q this gives the correct full-N kernel under the conjugate
    trick.

    HiPPO-LegS init (Gu 2022 Sec 4.3 / S4 paper Appendix C):
        Λ_n     = -1/2 + i·π·n   (n = 0..half_N-1)   eigenvalues
        P_n=Q_n = √(n + 1/2)                          rank-1 correction
        B_n     = √(2n + 1)                           LegS impulse measure
        Δ_log   ~ U[log 1e-3, log 1e-1]               per-channel log step

    Discretization: bilinear (Tustin)
        Ā = (I - Δ/2 A)⁻¹ (I + Δ/2 A)
        B̄ = (I - Δ/2 A)⁻¹ Δ B
    Kernel: K[l] = 2 · Re(Cᵀ Ā^l B̄) via brute-force `lax.scan`.
    """
    H: int      # channels
    N: int      # total state dim (half_N = N // 2 stored)
    L: int      # output kernel length

    @nn.compact
    def __call__(self):
        half_N = max(self.N // 2, 1)
        n_idx = jnp.arange(half_N, dtype=jnp.float32)

        # HiPPO-LegS init (in conjugate-paired half-N representation).
        Lambda_log_init = jnp.log(jnp.full((half_N,), 0.5))   # -log(re(Λ)) → re(Λ) = -1/2
        Lambda_im_init  = jnp.pi * n_idx                       # iπn
        PQ_init         = jnp.sqrt(n_idx + 0.5)                # P_n = Q_n = √(n+1/2)
        B_init          = jnp.sqrt(2.0 * n_idx + 1.0)          # B_n = √(2n+1) (LegS measure)

        Lambda_log = self.param('Lambda_log',
            lambda k, s: Lambda_log_init, (half_N,))
        Lambda_im = self.param('Lambda_im',
            lambda k, s: Lambda_im_init, (half_N,))
        P_real = self.param('P', lambda k, s: PQ_init, (half_N,))
        Q_real = self.param('Q', lambda k, s: PQ_init, (half_N,))
        B_real = self.param('B', lambda k, s: B_init, (half_N,))
        # C is per-channel complex (re/im split).
        C_ri = self.param('C_ri',
            lambda k, s: jax.random.normal(k, s) / jnp.sqrt(half_N),
            (self.H, 2, half_N))
        log_Delta = self.param('log_Delta',
            lambda k, s: jax.random.uniform(
                k, s, minval=jnp.log(1e-3), maxval=jnp.log(1e-1)),
            (self.H,))

        # Build complex tensors. Re(Λ) is forced negative for stability.
        Lambda_re = -jnp.exp(Lambda_log.astype(jnp.float32))
        Lambda    = (Lambda_re + 1j * Lambda_im.astype(jnp.float32)).astype(jnp.complex64)
        P_c       = P_real.astype(jnp.complex64)
        Q_c       = Q_real.astype(jnp.complex64)
        B_c       = B_real.astype(jnp.complex64)
        C_c       = (C_ri[:, 0] + 1j * C_ri[:, 1]).astype(jnp.complex64)  # (H, half_N)
        Delta     = jnp.exp(log_Delta.astype(jnp.float32))                # (H,)

        # DPLR matrix in half_N complex space: A = diag(Λ) - P Qᵀ.
        A_full = jnp.diag(Lambda) - jnp.outer(P_c, Q_c)                   # (half_N, half_N)
        I_N    = jnp.eye(half_N, dtype=jnp.complex64)

        # Per-channel kernel via bilinear discretization + brute-force scan.
        def kernel_per_channel(dt, C_h):
            dt_c   = dt.astype(jnp.complex64)
            M_lhs  = I_N - 0.5 * dt_c * A_full
            M_rhs  = I_N + 0.5 * dt_c * A_full
            A_bar  = jnp.linalg.solve(M_lhs, M_rhs)                       # (half_N, half_N)
            B_bar  = jnp.linalg.solve(M_lhs, dt_c * B_c)                  # (half_N,)

            # K[l] = C_hᵀ Ā^l B̄  for l = 0..L-1; h_0 = B̄.
            def step(h, _):
                k_l = jnp.dot(C_h, h)
                h_next = A_bar @ h
                return h_next, k_l
            _, K_l = jax.lax.scan(step, B_bar, None, length=self.L)
            return K_l                                                     # (L,) complex

        K_complex = jax.vmap(kernel_per_channel)(Delta, C_c)               # (H, L)
        # Conjugate-pair symmetry: full-N real kernel = 2 · Re(half-N complex).
        return (2.0 * K_complex.real).astype(jnp.float32)                  # (H, L)


class S4NDFullCSSM(nn.Module):
    """True S4ND with HiPPO-LegS DPLR initialization.

    Reference: Nguyen et al. "S4ND: Modeling Images and Videos as Multidimensional
    Signals Using State Spaces" (NeurIPS 2022). Each spatial axis gets its own
    1D S4 kernel parameterized as DPLR with HiPPO-LegS init — the full
    canonical S4 family from Gu et al., not the simplified S4D-diagonal that
    `S4NDCSSM` (cssm_type='s4nd') uses.

    Architecture: identical to `S4NDCSSM` except the per-axis kernel is
    `_S4NDLegSKernel1D` (DPLR + LegS) instead of `_S4DKernel1D` (diagonal +
    random init).
    """
    channels: int
    d_state: int = 64
    bidirectional: bool = True
    dropout: float = 0.0
    # Unused CSSM interface kwargs (accepted for SimpleCSSM dispatch compat).
    kernel_size: int = 3
    block_size: int = 1
    rope_mode: str = 'none'

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B, T, H, W, C = x.shape
        x_in = x

        kH = _S4NDLegSKernel1D(H=C, N=self.d_state, L=H, name='kH')()       # (C, H)
        kW = _S4NDLegSKernel1D(H=C, N=self.d_state, L=W, name='kW')()       # (C, W)

        xf = x.reshape(B * T, H, W, C).astype(jnp.float32)
        yf = _fft_conv_1d_per_channel(xf, kH, axis=1)
        yf = _fft_conv_1d_per_channel(yf, kW, axis=2)

        if self.bidirectional:
            kH_r = kH[:, ::-1]
            kW_r = kW[:, ::-1]
            yf_bw = _fft_conv_1d_per_channel(xf, kH_r, axis=1)
            yf_bw = _fft_conv_1d_per_channel(yf_bw, kW_r, axis=2)
            yf = yf + yf_bw

        D = self.param('D', nn.initializers.ones, (C,))
        yf = yf + D[None, None, None, :].astype(jnp.float32) * xf

        yf = yf.reshape(B, T, H, W, C)
        if self.dropout > 0:
            yf = nn.Dropout(rate=self.dropout, deterministic=False)(yf)

        y = yf.astype(x.dtype)
        y = nn.Dense(C, name='out_proj')(y.reshape(B * T, H, W, C))
        y = y.reshape(B, T, H, W, C)
        return y - x_in


class ConvS5CSSM(nn.Module):
    """Official NVlabs ConvSSM / ConvS5: diagonal complex SSM with conv B/C
    projections and associative parallel scan along T.

    Reference: Smith et al. "Convolutional State Space Models for Long-Range
    Spatiotemporal Modeling" (NeurIPS 2023, NVlabs/ConvSSM).

    Key differences from our simpler `ConvSSMCSSM`:
      - state_dim > 1 (N complex state components per spatial location per channel)
      - Complex diagonal Λ (decay + oscillation), not real sigmoid scalar
      - Conv-parameterized B and C (3x3 or 5x5 depthwise-group Conv)
      - Associative parallel scan (same contract — uses `direct_scalar_scan_op`)

    Architecture:
      1. Per-state-component complex poles Λ = -exp(a_log) + i * a_imag.
      2. Per-channel log-step Δ; discretize via ZOH: Ā[c,n] = exp(Δ[c] · Λ[n]).
      3. B_bar, C_bar = conv(x) with (C*N) output channels, depthwise-group.
      4. S_t = Ā ⊙ S_{t-1} + B_bar_t · x_t  (per spatial position, per channel)
         via jax.lax.associative_scan — same operator as ConvSSMCSSM.
      5. y_t = 2 · Re(S_t · C_bar_t).sum(N axis).
      6. Output projection Dense(C).

    Returns delta for SimpleCSSM's external residual.
    """
    channels: int
    state_dim: int = 16
    kernel_size: int = 3
    num_groups: int = 1
    # Unused CSSM interface kwargs.
    block_size: int = 1
    rope_mode: str = 'none'

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B_b, T, H, W, C = x.shape
        x_in = x
        ks = self.kernel_size
        N = self.state_dim

        # === 1. Diagonal complex poles Λ (shared across channels, N state comps). ===
        a_log = self.param('a_log', nn.initializers.constant(jnp.log(0.5)), (N,))
        a_imag = self.param(
            'a_imag',
            lambda k, s: jax.random.uniform(k, s, minval=0.0, maxval=2.0 * jnp.pi),
            (N,),
        )
        Lambda = (-jnp.exp(a_log.astype(jnp.float32))
                  + 1j * a_imag.astype(jnp.float32)).astype(jnp.complex64)    # (N,)

        # === 2. Per-channel log step size; ZOH discretization Ā = exp(Δ Λ). ===
        log_Delta = self.param(
            'log_Delta',
            lambda k, s: jax.random.uniform(
                k, s, minval=jnp.log(1e-3), maxval=jnp.log(1e-1)),
            (C,),
        )
        Delta = jnp.exp(log_Delta.astype(jnp.float32))                        # (C,)
        A_bar = jnp.exp(Delta[:, None].astype(jnp.complex64) * Lambda[None, :])  # (C, N)

        # === 3. Conv-parameterized B and C (depthwise-group: each C input channel
        #       produces N state components independently). ===
        xf4 = x.reshape(B_b * T, H, W, C)
        B_conv = nn.Conv(
            features=C * N, kernel_size=(ks, ks),
            feature_group_count=C, padding='SAME', name='B_conv',
        )(xf4)                                                                # (B*T, H, W, C*N)
        C_conv = nn.Conv(
            features=C * N, kernel_size=(ks, ks),
            feature_group_count=C, padding='SAME', name='C_conv',
        )(xf4)
        Bx = B_conv.reshape(B_b, T, H, W, C, N).astype(jnp.complex64)         # scan input
        Cy = C_conv.reshape(B_b, T, H, W, C, N).astype(jnp.complex64)         # output mix

        # === 4. Associative scan along T. ===
        # A_bar broadcast to (B, T, H, W, C, N). Same pole every (b, h, w, c, n) —
        # only T-axis accumulates state.
        A_bT = jnp.broadcast_to(
            A_bar[None, None, None, None, :, :], Bx.shape,
        ).astype(jnp.complex64)

        # Flatten (H, W) into the channel axis so the scan key is (B, T, HWCN).
        # Actually direct_scalar_scan_op operates element-wise on arrays;
        # axis=1 is the T axis, everything else is treated per-element.
        _, hidden = jax.lax.associative_scan(
            direct_scalar_scan_op,
            (A_bT, Bx),
            axis=1,
        )  # hidden: (B, T, H, W, C, N) complex

        # === 5. Readout: y = 2 * Re(hidden · C_bar).sum(-1). ===
        y_cplx = (hidden * Cy).sum(axis=-1)                                   # (B, T, H, W, C) cplx
        y = (2.0 * jnp.real(y_cplx)).astype(x.dtype)

        # === 6. Output projection. ===
        y = nn.Dense(C, name='out_proj')(y.reshape(B_b * T, H, W, C))
        y = y.reshape(B_b, T, H, W, C)
        return y - x_in


class NoFFTCSSM(nn.Module):
    """Mamba-style SSM with NO FFT — operates entirely in pixel domain.

    Same architecture as GatedCSSM but without any spectral transform:
        1. in_proj(2C) → (x_main, z)
        2. Optional short temporal conv → SiLU
        3. Per-channel scalar decay A ∈ (0,1) via sigmoid
        4. Input-dependent gates B, C, Delta from spatially-pooled context
           projected to (H, W) spatial positions (NOT frequency bins)
        5. Scalar scan: S_t = A^delta * S_{t-1} + delta * B * x_main
        6. Output: C * S, then RMSNorm → SiLU(z) gating → out_proj

    No FFT, no spatial convolution, no frequency-selective gating.
    Tests whether the FFT provides spatial structure beyond what
    per-pixel operations can achieve.

    Returns delta for SimpleCSSM's external residual.
    """
    channels: int
    kernel_size: int = 1  # unused, for API compat
    spectral_rho: float = 0.999  # unused
    gate_activation: str = 'softplus'
    gate_type: str = 'dense'  # 'dense' (full H*W) or 'factored' (separable H + W)
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    position_independent_gates: bool = False
    short_conv_size: int = 4
    short_conv_spatial_size: int = 0
    block_size: int = 1

    def _compute_gate(self, x):
        if self.gate_activation == 'softplus':
            return nn.softplus(x)
        elif self.gate_activation == 'sigmoid':
            return nn.sigmoid(x) * 2.0
        else:
            return nn.softplus(x)

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B, T, H, W, C = x.shape
        x_in = x

        # === 0. Input expansion: x → (x_main, z) ===
        x_flat = x.reshape(B * T, H, W, C)
        xz = nn.Dense(2 * C, name='in_proj')(x_flat)
        xz = xz.reshape(B, T, H, W, 2 * C)
        x_main = xz[..., :C]
        z = xz[..., C:]

        # === 1. Optional short convs → SiLU ===
        if self.short_conv_spatial_size > 0:
            x_sp = x_main.reshape(B * T, H, W, C)
            x_sp = nn.Conv(C, kernel_size=(self.short_conv_spatial_size, self.short_conv_spatial_size),
                           feature_group_count=C, padding='SAME',
                           name='short_conv_spatial')(x_sp)
            x_main = x_sp.reshape(B, T, H, W, C)

        if self.short_conv_size > 0:
            x_1d = x_main.transpose(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
            x_1d = jnp.pad(x_1d, ((0, 0), (self.short_conv_size - 1, 0), (0, 0)))
            x_1d = nn.Conv(C, kernel_size=(self.short_conv_size,),
                           feature_group_count=C, padding='VALID',
                           name='short_conv_temporal')(x_1d)
            x_main = x_1d.reshape(B, H, W, T, C).transpose(0, 3, 1, 2, 4)

        x_main = jax.nn.silu(x_main)

        # === 2. Per-channel scalar decay ===
        A_logit = self.param('A_logit', nn.initializers.constant(2.0), (C,))
        A = nn.sigmoid(A_logit)  # (C,) in (0, 1)

        # === 3. Input-dependent gates from spatially-pooled context ===
        ctx = x_main.mean(axis=(2, 3))  # (B, T, C)

        def _spatial_gate(ctx, name):
            """Project ctx to (B, T, H, W, 1) gate. Dense or factored."""
            if self.gate_type == 'factored':
                row = nn.Dense(H, name=f'{name}_h')(ctx)  # (B, T, H)
                col = nn.Dense(W, name=f'{name}_w')(ctx)  # (B, T, W)
                return (row[..., :, None] + col[..., None, :]).reshape(B, T, H, W, 1)
            else:  # dense
                return nn.Dense(H * W, name=name)(ctx).reshape(B, T, H, W, 1)

        delta_raw = _spatial_gate(ctx, 'delta_proj')
        delta = self._compute_gate(delta_raw)

        B_proj = _spatial_gate(ctx, 'B_proj')
        C_proj = _spatial_gate(ctx, 'C_proj')

        # === 4. Discretize ===
        A_bar = jnp.broadcast_to(A[None, None, None, None, :], (B, T, H, W, C))
        A_bar = A_bar ** delta  # per-pixel decay

        U = B_proj * x_main * delta  # (B, T, H, W, C)

        # === 5. Scalar scan per spatial position ===
        A_scan = A_bar.reshape(B, T, H * W * C)
        U_scan = U.reshape(B, T, H * W * C)

        _, S_scan = jax.lax.associative_scan(
            direct_scalar_scan_op,
            (A_scan.astype(jnp.complex64), U_scan.astype(jnp.complex64)),
            axis=1
        )
        S = S_scan.real.reshape(B, T, H, W, C)

        # === 6. Output: C gate → RMSNorm → SiLU(z) → project ===
        ssm_out = S * C_proj

        rms_scale = self.param('rms_scale', nn.initializers.ones, (C,))
        ms = jnp.mean(ssm_out ** 2, axis=-1, keepdims=True)
        ssm_out = ssm_out * jax.lax.rsqrt(ms + 1e-6) * rms_scale

        output = ssm_out * jax.nn.silu(z)
        output = nn.Dense(C, name='out_proj')(output.reshape(B * T, H, W, C))
        output = output.reshape(B, T, H, W, C)

        return output - x_in


class NoGateCSSM(nn.Module):
    """Minimal spectral CSSM — ONLY the spectral kernel recurrence.

    Isolates the pure spectral convolution dynamics:
        S_hat_t(f) = K_hat(f) * S_hat_{t-1}(f) + X_hat_t(f)
        y_t = iFFT(S_hat_t)

    No in_proj, no SiLU, no gating, no B/C/Delta.
    Just: FFT → spectral recurrence → iFFT → out_proj.

    With kernel_size=1:  uniform decay, no spatial mixing → should fail
    With kernel_size>1:  spectral conv in transition → tests if kernel alone suffices

    Returns delta for SimpleCSSM's external residual.
    """
    channels: int
    kernel_size: int = 11
    spectral_rho: float = 0.999
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    short_conv_size: int = 4      # accepted for interface compat, unused
    short_conv_spatial_size: int = 0
    block_size: int = 1
    position_independent_gates: bool = False
    gate_type: str = 'dense'

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

        # === 1. Spectral kernel (learned, fixed) ===
        k_spatial = self.param('kernel', nn.initializers.normal(0.02),
                               (C, self.kernel_size, self.kernel_size))

        ks = self.kernel_size
        if ks > H or ks > W:
            start_h = (ks - H) // 2
            start_w = (ks - W) // 2
            k_padded = k_spatial[:, start_h:start_h+H, start_w:start_w+W]
        else:
            pad_h = (H - ks) // 2
            pad_w = (W - ks) // 2
            pad_h_after = H - ks - pad_h
            pad_w_after = W - ks - pad_w
            k_padded = jnp.pad(k_spatial,
                               ((0, 0), (pad_h, max(0, pad_h_after)),
                                (pad_w, max(0, pad_w_after))),
                               mode='constant')

        K_hat = _stable_spectral_magnitude(
            jnp.fft.rfft2(k_padded, axes=(1, 2)), rho=self.spectral_rho
        )  # (C, H, W_freq)

        # === 2. FFT input (raw, no in_proj, no nonlinearity) ===
        x_perm = x.transpose(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C, H, W_freq)

        K_broadcast = jnp.broadcast_to(
            K_hat[None, None].astype(jnp.complex64), (B, T, C, H, W_freq)
        )

        # === 3. Scalar scan: S_hat_t = K * S_hat_{t-1} + U_hat_t ===
        _, S_hat = jax.lax.associative_scan(
            direct_scalar_scan_op,
            (K_broadcast, U_hat.astype(jnp.complex64)),
            axis=1
        )

        # === 4. iFFT → out_proj (linear only) ===
        ssm_out = jnp.fft.irfft2(S_hat, s=(H, W), axes=(3, 4))
        ssm_out = ssm_out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)

        output = nn.Dense(C, name='out_proj')(ssm_out.reshape(B * T, H, W, C))
        return output.reshape(B, T, H, W, C) - x


class DirectConvParallelCSSM(nn.Module):
    """Theory-validation timing variant: depthwise 2D conv + scalar SSM
    + *explicit* length-T temporal conv (no FFT, no associative-scan trick).

    Spatial mixing: real-valued depthwise 2D conv (no FFT).
    Temporal recurrence S_t = a·S_{t-1} + y_t is unrolled into its impulse
    response [a^0, a^1, ..., a^{T-1}] and applied as a causal Toeplitz matmul
    along the time axis. The kernel literally has length T per channel and the
    Toeplitz operator is materialized as a (T, T, C) tensor, so compute is
    O(T²·B·H·W·C) and memory is O(T²·C + B·T·HW·C). Crashes (OOM) at moderate
    T — that's the *point*: this is the cost FFT and associative-scan both
    avoid, and the panel exists to demonstrate why those tricks matter.

    Used by `benchmarks/bench_image_timing.py` Panel T (theory validation) to
    represent the "Parallel CSSM" curve — meant to grow superlinearly relative
    to fCSSM (FFT, O(T log T)) and crash at large T. Not for training.

    Returns delta for SimpleCSSM's external residual.
    """
    channels: int
    kernel_size: int = 5
    spectral_rho: float = 0.999
    # Interface-compat fields (accepted but unused).
    short_conv_size: int = 4
    short_conv_spatial_size: int = 0
    block_size: int = 1
    rope_mode: str = 'none'

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B, T, H, W, C = x.shape
        x_in = x

        # 1. Depthwise spatial conv (real, kernel_size × kernel_size) on input.
        x_flat = x.reshape(B * T, H, W, C)
        y_flat = nn.Conv(C, kernel_size=(self.kernel_size, self.kernel_size),
                         feature_group_count=C, padding='SAME',
                         name='spatial_conv')(x_flat)
        y = y_flat.reshape(B, T, H, W, C)

        # 2. Per-channel scalar damping a ∈ (-rho, rho) for stability.
        a_logit = self.param('a_logit', nn.initializers.zeros, (C,))
        a = self.spectral_rho * jnp.tanh(a_logit)            # (C,)

        # 3. Materialize the length-T impulse response and full Toeplitz
        #    operator. Each row of K is the causal kernel at one output
        #    position; the kernel grows linearly with T and the operator
        #    quadratically — exactly what FFT lets us avoid.
        t_idx = jnp.arange(T, dtype=jnp.float32)             # (T,)
        impulse = a.astype(jnp.float32)[None, :] ** t_idx[:, None]   # (T, C)

        diff = jnp.arange(T)[:, None] - jnp.arange(T)[None, :]       # (T, T)
        causal = diff >= 0
        K = jnp.where(
            causal[..., None],
            impulse[jnp.clip(diff, 0)],                       # (T, T, C)
            0.0,
        )

        # S[b, t, h, w, c] = sum_{j<=t} K[t, j, c] * y[b, j, h, w, c]
        S = jnp.einsum('tjc,bjhwc->bthwc', K, y).astype(y.dtype)

        # 4. Output projection.
        output = nn.Dense(C, name='out_proj')(S.reshape(B * T, H, W, C))
        return output.reshape(B, T, H, W, C) - x_in


class DirectConvSeqCSSM(nn.Module):
    """Theory-validation timing variant: depthwise 2D conv + scalar SSM
    + sequential temporal scan.

    Identical to DirectConvParallelCSSM except the temporal scan uses
    `jax.lax.scan` instead of `jax.lax.associative_scan`. This makes the
    temporal recurrence O(T) sequential — the asymptotic baseline that a
    parallel scan beats.

    Used by `benchmarks/bench_image_timing.py` Panel A (theory validation) to
    represent the "Direct CSSM (sequential)" curve. Not intended for training.

    Returns delta for SimpleCSSM's external residual.
    """
    channels: int
    kernel_size: int = 5
    spectral_rho: float = 0.999
    short_conv_size: int = 4
    short_conv_spatial_size: int = 0
    block_size: int = 1
    rope_mode: str = 'none'

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B, T, H, W, C = x.shape
        x_in = x

        x_flat = x.reshape(B * T, H, W, C)
        y_flat = nn.Conv(C, kernel_size=(self.kernel_size, self.kernel_size),
                         feature_group_count=C, padding='SAME',
                         name='spatial_conv')(x_flat)
        y = y_flat.reshape(B, T, H, W, C)

        a_logit = self.param('a_logit', nn.initializers.zeros, (C,))
        a = self.spectral_rho * jnp.tanh(a_logit)
        a_bcast = a[None, None, None, :]  # broadcast over (B, H, W, C)

        # Sequential scan along T: transpose to T-major, scan, transpose back.
        y_t_first = y.transpose(1, 0, 2, 3, 4)
        S_init = jnp.zeros_like(y_t_first[0])

        def step(S_prev, y_t):
            S_new = a_bcast * S_prev + y_t
            return S_new, S_new

        _, S_t_first = jax.lax.scan(step, S_init, y_t_first)
        S = S_t_first.transpose(1, 0, 2, 3, 4)

        output = nn.Dense(C, name='out_proj')(S.reshape(B * T, H, W, C))
        return output.reshape(B, T, H, W, C) - x_in


def _circ_dwconv2d_batched(x, k):
    """Per-channel circular depthwise 2D conv in real space (no FFT).

    x, k both shape (..., H, W, C). Each (B*T)-th sample carries its own
    (H, W, C) kernel. Output: same shape as x.

    Cost: O(N·H²·W²·C) per call where N = prod(batch). Used by the
    real-space sCSSM variants where the kernel is the temporal damping
    operator that gets composed across scan steps.
    """
    *batch, H, W, C = x.shape
    Bf = int(np.prod(batch)) if batch else 1
    x_flat = x.reshape(Bf, H, W, C)
    k_flat = k.reshape(Bf, H, W, C)
    # Wrap-pad x to (Bf, 2H-1, 2W-1, C) so VALID conv with (H, W) kernel
    # produces (H, W) output that is the circular conv.
    x_pad = jnp.pad(x_flat, ((0, 0), (H - 1, 0), (W - 1, 0), (0, 0)), mode='wrap')
    x_pad = x_pad[:, :2 * H - 1, :2 * W - 1, :]

    def _one(x_one, k_one):
        out = jax.lax.conv_general_dilated(
            x_one[None], k_one[..., None, :],
            window_strides=(1, 1), padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=C,
        )
        return out[0]

    out_flat = jax.vmap(_one)(x_pad, k_flat)
    return out_flat.reshape(*batch, H, W, C)


class NoGateRealSpaceParallelCSSM(nn.Module):
    """sCSSM-without-FFT: same K×K kernel, applied as temporal damping in
    real space, parallel scan via associative_scan with kernel-conv combiner.

    Recurrence: S_t = k ⊛ S_{t-1} + k ⊛ x_t, where k is depthwise (C, K, K)
    and ⊛ is real-space circular depthwise convolution. The associative_scan
    carry is (k_eff, b_eff); the combiner is
        ((k_a, b_a), (k_b, b_b)) ↦ (k_b ⊛ k_a, k_b ⊛ b_a + b_b)
    where ⊛ is depthwise conv in real space. Each combine call does 2
    real-space convs at O(H²·W²·C) each — vs O(H·W·log HW) per step under
    FFT — so the panel can demonstrate the per-step cost of avoiding FFT.

    Kernel is padded to (C, H, W) at init so the carry has fixed shape
    across scan rounds (the natural growing kernel saturates at image size
    via circular wrap; semantically equivalent to letting it grow freely
    once it exceeds H×W).
    """
    channels: int
    kernel_size: int = 5
    spectral_rho: float = 0.999
    short_conv_size: int = 4
    short_conv_spatial_size: int = 0
    block_size: int = 1
    rope_mode: str = 'none'

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B, T, H, W, C = x.shape
        x_in = x
        ks = self.kernel_size

        k_spatial = self.param('kernel', nn.initializers.normal(0.02),
                               (C, ks, ks))
        if ks > H or ks > W:
            sh, sw = (ks - H) // 2, (ks - W) // 2
            k_pad = k_spatial[:, sh:sh + H, sw:sw + W]
        else:
            ph, pw = (H - ks) // 2, (W - ks) // 2
            pha, pwa = max(0, H - ks - ph), max(0, W - ks - pw)
            k_pad = jnp.pad(k_spatial, ((0, 0), (ph, pha), (pw, pwa)),
                            mode='constant')
        # Stability: bound the per-channel L2 norm of the kernel so the
        # repeated conv composition stays in a reasonable range.
        norm = jnp.linalg.norm(k_pad.reshape(C, -1), axis=-1).reshape(C, 1, 1)
        k_pad = k_pad * (self.spectral_rho / (norm + 1e-8))
        k_hwc = k_pad.transpose(1, 2, 0)                            # (H, W, C)

        # Apply k as spatial filter to each x_t to get U_t = k ⊛ x_t.
        k_b = jnp.broadcast_to(k_hwc[None, None], (B, T, H, W, C))
        U = _circ_dwconv2d_batched(x, k_b)                          # (B, T, H, W, C)

        # Associative scan with kernel-conv combiner.
        K_carry = jnp.broadcast_to(k_hwc[None, None], (B, T, H, W, C))

        def combiner(left, right):
            k_l, b_l = left
            k_r, b_r = right
            new_k = _circ_dwconv2d_batched(k_l, k_r)
            new_b = _circ_dwconv2d_batched(b_l, k_r) + b_r
            return (new_k, new_b)

        _, S = jax.lax.associative_scan(combiner, (K_carry, U), axis=1)

        output = nn.Dense(C, name='out_proj')(S.reshape(B * T, H, W, C))
        return output.reshape(B, T, H, W, C) - x_in


class NoGateRealSpaceSeqCSSM(nn.Module):
    """sCSSM-without-FFT, sequential variant: same recurrence as
    NoGateRealSpaceParallelCSSM but executed via `jax.lax.scan` — each step
    applies the K×K kernel ⊛ to (S_{t-1} + x_t) in real space. T sequential
    convs, no kernel composition (carry is just S_{t-1}, fixed shape).
    """
    channels: int
    kernel_size: int = 5
    spectral_rho: float = 0.999
    short_conv_size: int = 4
    short_conv_spatial_size: int = 0
    block_size: int = 1
    rope_mode: str = 'none'

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B, T, H, W, C = x.shape
        x_in = x
        ks = self.kernel_size

        k_spatial = self.param('kernel', nn.initializers.normal(0.02),
                               (C, ks, ks))
        if ks > H or ks > W:
            sh, sw = (ks - H) // 2, (ks - W) // 2
            k_pad = k_spatial[:, sh:sh + H, sw:sw + W]
        else:
            ph, pw = (H - ks) // 2, (W - ks) // 2
            pha, pwa = max(0, H - ks - ph), max(0, W - ks - pw)
            k_pad = jnp.pad(k_spatial, ((0, 0), (ph, pha), (pw, pwa)),
                            mode='constant')
        norm = jnp.linalg.norm(k_pad.reshape(C, -1), axis=-1).reshape(C, 1, 1)
        k_pad = k_pad * (self.spectral_rho / (norm + 1e-8))
        k_hwc = k_pad.transpose(1, 2, 0)                            # (H, W, C)

        # Wrap-pad k_hwc into the per-step conv shape; we'll broadcast a
        # fresh batch dim each step.
        x_t_first = x.transpose(1, 0, 2, 3, 4)                      # (T, B, H, W, C)
        S_init = jnp.zeros_like(x_t_first[0])

        def step(S_prev, x_t):
            # S_t = k ⊛ (S_{t-1} + x_t)
            inp = S_prev + x_t                                      # (B, H, W, C)
            k_bcast = jnp.broadcast_to(k_hwc[None], inp.shape)      # (B, H, W, C)
            S_next = _circ_dwconv2d_batched(inp, k_bcast)
            return S_next, S_next

        _, S_t_first = jax.lax.scan(step, S_init, x_t_first)
        S = S_t_first.transpose(1, 0, 2, 3, 4)                      # (B, T, H, W, C)

        output = nn.Dense(C, name='out_proj')(S.reshape(B * T, H, W, C))
        return output.reshape(B, T, H, W, C) - x_in


class DirectConvAssocCSSM(nn.Module):
    """Theory-validation timing variant: depthwise 2D conv + scalar SSM
    + parallel temporal scan via associative_scan with scalar summary.

    Same architecture as DirectConvParallelCSSM and DirectConvSeqCSSM
    (real-space depthwise spatial conv + per-channel scalar damping a),
    but the temporal recurrence S_t = a · S_{t-1} + y_t is computed via
    `jax.lax.associative_scan`. The combiner closed-form keeps the carry
    summary (a_eff, b_eff) at SCALAR shape per channel — never materializes
    a length-T impulse response. O(T log T) FLOPs, no kernel growth.

    Pairs with DirectConvParallelCSSM (Toeplitz materialization → grows)
    and DirectConvSeqCSSM (sequential scan) for the apples-to-apples
    "three temporal-scan algorithms, identical real-space spatial mixing"
    comparison in benchmarks/bench_image_timing.py Panel T.

    Returns delta for SimpleCSSM's external residual.
    """
    channels: int
    kernel_size: int = 5
    spectral_rho: float = 0.999
    short_conv_size: int = 4
    short_conv_spatial_size: int = 0
    block_size: int = 1
    rope_mode: str = 'none'

    @nn.compact
    def __call__(self, x, injected_qkv_spatial=None):
        B, T, H, W, C = x.shape
        x_in = x

        x_flat = x.reshape(B * T, H, W, C)
        y_flat = nn.Conv(C, kernel_size=(self.kernel_size, self.kernel_size),
                         feature_group_count=C, padding='SAME',
                         name='spatial_conv')(x_flat)
        y = y_flat.reshape(B, T, H, W, C)

        a_logit = self.param('a_logit', nn.initializers.zeros, (C,))
        a = self.spectral_rho * jnp.tanh(a_logit)            # (C,)
        a_b = jnp.broadcast_to(a[None, None, None, None, :], (B, T, H, W, C))

        # Associative scalar scan: combiner closed-form keeps (a_eff, b_eff)
        # at the same shape as a single scan element — no kernel materialization.
        a_scan = a_b.reshape(B, T, H * W * C).astype(jnp.complex64)
        y_scan = y.reshape(B, T, H * W * C).astype(jnp.complex64)
        _, S_scan = jax.lax.associative_scan(
            direct_scalar_scan_op, (a_scan, y_scan), axis=1
        )
        S = S_scan.real.reshape(B, T, H, W, C)

        output = nn.Dense(C, name='out_proj')(S.reshape(B * T, H, W, C))
        return output.reshape(B, T, H, W, C) - x_in
