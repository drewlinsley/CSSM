"""
CSSM (Cepstral State Space Model) layers.

Implements both Standard and Gated Opponent CSSM layers with
options for Dense (multi-head) vs Depthwise spatial mixing.

Uses GOOM (Generalized Order of Magnitude) primitives for
numerically stable log-space computation.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn

from .math import (cssm_scalar_scan_op, cssm_matrix_scan_op, cssm_3x3_matrix_scan_op,
                    make_block_scan_op,
                    complex64_to_linear_split, linear_split_to_complex64,
                    linear_split_scalar_scan_op, linear_split_2x2_scan_op,
                    linear_split_3x3_scan_op,
                    ssd_scan)
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
            - 'spatiotemporal': Combined H, W, T encoding (VideoRoPE style)
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
    Mamba-style Gated CSSM with full input-dependent integration.

    Implements the complete Mamba formulation in log-spectral domain:
    - h_t = A_bar * h_{t-1} + B_bar * u_t    (state update)
    - y_t = C * h_t                           (output projection)

    Where:
    - A_bar = K * exp(-Δ): Input-dependent per-frequency decay
    - B_bar: Input-dependent per-frequency input projection
    - C: Input-dependent per-frequency output projection

    **Channel Mixing Modes:**
    - `dense_mixing=False`: Depthwise - each channel independent, O(C) complexity
    - `dense_mixing=True, mixing_rank=0`: Full LMME - O(C × block_size²) params (expensive!)
    - `dense_mixing=True, mixing_rank>0`: Low-rank LMME - O(C × rank) params (recommended)

    Attributes:
        channels: Number of input/output channels
        dense_mixing: If True, use channel mixing (LMME or low-rank)
        block_size: Size of channel blocks for LMME (default 32)
        mixing_rank: If > 0, use low-rank channel mixing instead of full LMME.
                     Recommended: 4-16. Set to 0 for full block×block matrices.
        kernel_size: Spatial kernel size
        spectral_rho: Maximum spectral magnitude for stability (should be < 1)
        gate_activation: Activation for decay gate ('softplus', 'sigmoid', 'exp')
        rope_mode: Position encoding mode ('spatiotemporal', 'temporal', 'none')
        rope_base: Base for RoPE frequency computation
        gate_rank: Unused (for API compatibility with GatedOpponentCSSM)
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
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with full Mamba-style gating in log-spectral domain.

        Args:
            x: Input tensor of shape (B, T, H, W, C)

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
        """Depthwise forward pass - each channel independent."""

        # --- 1. Kernel Generation (depthwise) ---
        k_spatial = self.param(
            'kernel',
            nn.initializers.normal(0.02),
            (C, self.kernel_size, self.kernel_size)
        )

        # --- 2. Spectral Transform of Kernel ---
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

        # --- 3. FFT Input ---
        x_perm = x.transpose(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C, H, W_freq)

        # --- 4. Input-Dependent Gates (Mamba-style) ---
        # Use raw input (before position encoding) for gates if position_independent_gates is True
        # This makes gates invariant to sequence position, improving length generalization
        gate_input = x_raw if x_raw is not None else x
        ctx = gate_input.mean(axis=(2, 3))  # (B, T, C)

        # Per-frequency decay gate
        delta_raw = nn.Dense(H * W_freq, name='delta_gate')(ctx)
        delta_raw = delta_raw.reshape(B, T, H * W_freq)
        delta_freq = self._compute_gate(delta_raw)
        delta_freq = delta_freq.reshape(B, T, 1, H, W_freq)

        # Input/output projection gates
        B_gate_raw = nn.Dense(H * W_freq, name='B_gate')(ctx)
        B_gate = nn.sigmoid(B_gate_raw).reshape(B, T, 1, H, W_freq)

        C_gate_raw = nn.Dense(H * W_freq, name='C_gate')(ctx)
        C_gate = nn.sigmoid(C_gate_raw).reshape(B, T, 1, H, W_freq)

        # --- 5. Convert to GOOM (log-space) ---
        K_log = to_goom(K_hat)
        K_log_broadcast = jnp.broadcast_to(K_log[None, None, ...], U_hat.shape)

        U_modulated = U_hat * B_gate
        U_log = to_goom(U_modulated)

        # --- 6. Apply Per-Frequency Δ Gating in Log-Space ---
        K_log_gated = (K_log_broadcast.real - delta_freq) + 1j * K_log_broadcast.imag
        U_log_gated = (U_log.real + jnp.log(delta_freq + 1e-8)) + 1j * U_log.imag

        # --- 7. Associative Scan (depthwise) ---
        _, X_log = jax.lax.associative_scan(
            cssm_scalar_scan_op, (K_log_gated, U_log_gated), axis=1
        )

        # --- 8. Apply C gate and Inverse Transform ---
        X_hat = from_goom(X_log)
        X_hat_modulated = X_hat * C_gate
        x_out = jnp.fft.irfft2(X_hat_modulated, s=(H, W), axes=(3, 4))

        return x_out.transpose(0, 1, 3, 4, 2)

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

        # --- 9. Apply C gate ---
        X_hat = from_goom(X_log)
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
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with minimal Q/K/A dynamics.

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
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
    use_ssd: bool = False            # Use SSD chunked scan instead of associative scan
    ssd_chunk_size: int = 8          # Chunk size for SSD

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with triangular Q→K→V dynamics (or Q→V if no_k_state).

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

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

            if self.use_ssd:
                # SSD chunked scan: works in linear complex space
                A_c = A_X.astype(jnp.complex64)
                U_c = U_X_hat.astype(jnp.complex64)
                V_hat = ssd_scan(A_c, U_c, chunk_size=self.ssd_chunk_size)
            elif self.use_complex32:
                # Linear-split: skip GOOM, split (re_bf16, im_bf16) directly
                A_c = A_X.astype(jnp.complex64)
                U_c = U_X_hat.astype(jnp.complex64)

                if self.learned_init:
                    # Need log-space init for prepend, then convert back
                    A_log = _to_log(A_c)
                    U_log = _to_log(U_c)
                    A_log, U_log = _prepend_init(A_log, U_log, n_states=1)
                    A_c = _from_log(A_log)
                    U_c = _from_log(U_log)

                A_re, A_im = complex64_to_linear_split(A_c)
                U_re, U_im = complex64_to_linear_split(U_c)
                _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                    linear_split_scalar_scan_op,
                    (A_re, A_im, U_re, U_im), axis=1)
                V_hat = linear_split_to_complex64(U_re_out, U_im_out)

                if self.learned_init:
                    V_hat = V_hat[:, 1:]
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

            if self.use_ssd:
                # SSD cascaded: decompose 2×2 triangular into 2 scalar SSMs
                a_Q = (d_Q_c * K_b).astype(jnp.complex64)
                a_V = (d_V_c * ones).astype(jnp.complex64)
                gamma_c = (gamma * ones).astype(jnp.complex64)

                Q_all = ssd_scan(a_Q, U_Q_hat.astype(jnp.complex64),
                                 chunk_size=self.ssd_chunk_size)
                Q_prev = jnp.concatenate(
                    [jnp.zeros_like(Q_all[:, :1]), Q_all[:, :-1]], axis=1)

                U_V_eff = gamma_c * Q_prev + U_V_hat.astype(jnp.complex64)
                V_hat = ssd_scan(a_V, U_V_eff, chunk_size=self.ssd_chunk_size)
            elif self.use_complex32:
                # Linear-split: skip GOOM, split (re_bf16, im_bf16) directly
                K_c = K_mat.astype(jnp.complex64)
                U_c = U_vec.astype(jnp.complex64)

                if self.learned_init:
                    K_log = _to_log(K_c)
                    U_log = _to_log(U_c)
                    K_log, U_log = _prepend_init(K_log, U_log, n_states=2)
                    K_c = _from_log(K_log)
                    U_c = _from_log(U_log)

                K_re, K_im = complex64_to_linear_split(K_c)
                U_re, U_im = complex64_to_linear_split(U_c)
                _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                    linear_split_2x2_scan_op,
                    (K_re, K_im, U_re, U_im), axis=1)
                QV_hat = linear_split_to_complex64(U_re_out, U_im_out)

                if self.learned_init:
                    QV_hat = QV_hat[:, 1:]
                V_hat = QV_hat[..., 1]
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

        else:
            # =================================================================
            # 3-STATE PATH: Q + K + V
            # =================================================================
            d_K = _gate('d_K', bounded=True)
            w_kq = _gate('w_kq')
            B_K = _gate('B_K', spatial=True)

            #            Q              K              V
            # Q  [ d_Q·K_b           0              0   ]
            # K  [ w_kq·K_b       d_K·K_b           0   ]
            # V  [ γ·ones          γ·ones        d_V·ones ]

            d_Q_c = d_Q.astype(jnp.complex64)
            d_K_c = d_K.astype(jnp.complex64)
            d_V_c = d_V.astype(jnp.complex64)
            zeros = jnp.zeros_like(d_Q_c * ones)

            A_00 = d_Q_c * K_b
            A_01 = zeros
            A_02 = zeros
            A_10 = w_kq * K_b
            A_11 = d_K_c * K_b
            A_12 = zeros
            A_20 = gamma * ones
            A_21 = gamma * ones
            A_22 = d_V_c * ones

            row_Q = jnp.stack([A_00, A_01, A_02], axis=-1)
            row_K = jnp.stack([A_10, A_11, A_12], axis=-1)
            row_V = jnp.stack([A_20, A_21, A_22], axis=-1)
            K_mat = jnp.stack([row_Q, row_K, row_V], axis=-2)

            U_Q_hat = _gate_and_fft(q_in, B_Q)
            U_K_hat = _gate_and_fft(k_in, B_K)
            U_V_hat = _gate_and_fft(v_in, B_V)
            U_vec = jnp.stack([U_Q_hat, U_K_hat, U_V_hat], axis=-1)

            # Scan (3×3)
            if self.use_ssd:
                # SSD cascaded: decompose 3×3 triangular into 3 scalar SSMs
                # Q_t = a_Q * Q_{t-1} + U_Q
                # K_t = a_K * K_{t-1} + w_kq·K_b · Q_{t-1} + U_K
                # V_t = a_V * V_{t-1} + γ · Q_{t-1} + γ · K_{t-1} + U_V
                a_Q = (d_Q_c * K_b).astype(jnp.complex64)
                a_K = (d_K_c * K_b).astype(jnp.complex64)
                a_V = (d_V_c * ones).astype(jnp.complex64)
                w_kq_c = (w_kq * K_b).astype(jnp.complex64)
                gamma_c = (gamma * ones).astype(jnp.complex64)
                L = self.ssd_chunk_size

                # 1. Q scan (independent)
                Q_all = ssd_scan(a_Q, U_Q_hat.astype(jnp.complex64), chunk_size=L)
                Q_prev = jnp.concatenate(
                    [jnp.zeros_like(Q_all[:, :1]), Q_all[:, :-1]], axis=1)

                # 2. K scan (depends on Q)
                U_K_eff = w_kq_c * Q_prev + U_K_hat.astype(jnp.complex64)
                K_all = ssd_scan(a_K, U_K_eff, chunk_size=L)
                K_prev = jnp.concatenate(
                    [jnp.zeros_like(K_all[:, :1]), K_all[:, :-1]], axis=1)

                # 3. V scan (depends on Q and K)
                U_V_eff = gamma_c * Q_prev + gamma_c * K_prev + U_V_hat.astype(jnp.complex64)
                V_hat = ssd_scan(a_V, U_V_eff, chunk_size=L)
            elif self.use_complex32:
                # Linear-split: skip GOOM, split (re_bf16, im_bf16) directly
                K_c = K_mat.astype(jnp.complex64)
                U_c = U_vec.astype(jnp.complex64)

                if self.learned_init:
                    K_log = _to_log(K_c)
                    U_log = _to_log(U_c)
                    K_log, U_log = _prepend_init(K_log, U_log, n_states=3)
                    K_c = _from_log(K_log)
                    U_c = _from_log(U_log)

                K_re, K_im = complex64_to_linear_split(K_c)
                U_re, U_im = complex64_to_linear_split(U_c)
                _, _, U_re_out, U_im_out = jax.lax.associative_scan(
                    linear_split_3x3_scan_op,
                    (K_re, K_im, U_re, U_im), axis=1)
                QKV_hat = linear_split_to_complex64(U_re_out, U_im_out)

                if self.learned_init:
                    QKV_hat = QKV_hat[:, 1:]
                V_hat = QKV_hat[..., 2]
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

        # Capture V_hat for visualization (no-op when 'intermediates' not mutable)
        self.sow('intermediates', 'V_hat', V_hat)

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
