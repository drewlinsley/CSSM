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

from .math import cssm_scalar_scan_op, cssm_matrix_scan_op, make_block_scan_op
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


class StandardCSSM(nn.Module):
    """
    Standard Log-Spectral State Space Model.

    Implements the recurrence: h_t = K * h_{t-1} + u_t
    in the spectral (frequency) domain using FFT, with computation
    performed in log-space using GOOM for numerical stability.

    Attributes:
        channels: Number of input/output channels
        dense_mixing: If True, use multi-head parameter sharing (more efficient).
                     If False, use fully independent depthwise kernels.
        kernel_size: Spatial kernel size (square)
        spectral_rho: Maximum spectral magnitude for stability (should be < 1)
        rope_mode: Position encoding mode ('spatiotemporal', 'temporal', 'none')
        rope_base: Base for RoPE frequency computation
        concat_xy: Unused (for API compatibility with GatedOpponentCSSM)
        gate_activation: Unused (for API compatibility with GatedOpponentCSSM)
        gate_rank: Unused (for API compatibility with GatedOpponentCSSM)
        block_size: Unused (for API compatibility with GatedOpponentCSSM)
    """
    channels: int
    dense_mixing: bool = False
    kernel_size: int = 15
    spectral_rho: float = 0.999
    rope_mode: str = 'none'  # 'spatiotemporal', 'temporal', or 'none'
    rope_base: float = 10000.0
    concat_xy: bool = True  # Unused, for API compatibility
    gate_activation: str = 'sigmoid'  # Unused, for API compatibility
    gate_rank: int = 0  # Unused, for API compatibility
    block_size: int = 1  # Unused, for API compatibility

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, H, W, C)
               B = batch, T = time, H/W = spatial, C = channels

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape

        # === Apply RoPE before FFT (VideoRoPE style) ===
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # --- 1. Kernel Generation ---
        if self.dense_mixing:
            # Multi-Head Dense Mixing to save parameters
            # Head dimension of 32 is a good balance
            head_dim = 32
            num_heads = max(1, C // head_dim)

            # Param shape: (num_heads, kernel_size, kernel_size)
            # Each head's kernel is shared across head_dim channels
            k_param = self.param(
                'kernel',
                nn.initializers.normal(0.02),
                (num_heads, self.kernel_size, self.kernel_size)
            )

            # Broadcast heads to match channels
            # (num_heads, K, K) -> (C, K, K) by repeating each kernel
            k_spatial = jnp.repeat(k_param, C // num_heads, axis=0)
        else:
            # Independent Depthwise: each channel has its own kernel
            k_spatial = self.param(
                'kernel',
                nn.initializers.normal(0.02),
                (C, self.kernel_size, self.kernel_size)
            )

        # --- 2. Spectral Transform ---
        # Pad kernel to image size (handle edge cases)
        pad_h = max(0, (H - self.kernel_size) // 2)
        pad_w = max(0, (W - self.kernel_size) // 2)
        # Ensure symmetric padding to match H, W exactly
        pad_h_after = H - self.kernel_size - pad_h
        pad_w_after = W - self.kernel_size - pad_w

        k_padded = jnp.pad(
            k_spatial,
            ((0, 0), (pad_h, max(0, pad_h_after)), (pad_w, max(0, pad_w_after))),
            mode='constant'
        )

        # If kernel is larger than image, crop instead
        if self.kernel_size > H or self.kernel_size > W:
            start_h = (self.kernel_size - H) // 2
            start_w = (self.kernel_size - W) // 2
            k_padded = k_spatial[:, start_h:start_h+H, start_w:start_w+W]

        # RFFT over spatial dims (axes 1, 2 of kernel -> H, W)
        K_hat_raw = jnp.fft.rfft2(k_padded, axes=(1, 2))  # (C, H, W_freq)
        # Squash spectral magnitude to ensure stability (|K| < 1)
        K_hat = _stable_spectral_magnitude(K_hat_raw, rho=self.spectral_rho)

        # Reshape input for FFT: (B, T, H, W, C) -> (B, T, C, H, W)
        x_perm = x.transpose(0, 1, 4, 2, 3)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C, H, W_freq)

        # --- 3. Cepstral Scan ---
        # Convert to GOOM representation (numerically stable log-space)
        K_log = to_goom(K_hat)  # (C, H, W_freq)
        # Broadcast to match input shape: (C, H, W_f) -> (B, T, C, H, W_f)
        K_log = jnp.broadcast_to(K_log[None, None, ...], U_hat.shape)
        U_log = to_goom(U_hat)

        # Scan over time dimension (axis 1)
        _, X_log = jax.lax.associative_scan(
            cssm_scalar_scan_op, (K_log, U_log), axis=1
        )

        # --- 4. Inverse Transform ---
        # Convert back from GOOM representation
        X_hat = from_goom(X_log)
        x_out = jnp.fft.irfft2(X_hat, s=(H, W), axes=(3, 4))

        # Back to (B, T, H, W, C)
        return x_out.transpose(0, 1, 3, 4, 2)


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


class GatedOpponentCSSM(nn.Module):
    """
    Gated Opponent CSSM with biologically-plausible X<->Y coupled oscillator.

    Implements a 2x2 state transition matrix with Mamba-style input-dependent gates:
    - Diagonal: decay terms (alpha for X, delta for Y) - per-frequency
    - Off-diagonal: coupling (mu for inhibition Y->X, gamma for excitation X->Y) - per-frequency
    - B gate: Input projection (controls how input enters state) - per-frequency
    - C gate: Output projection (controls how state exits) - per-frequency

    Uses GOOM primitives for numerically stable log-space computation.

    **Channel Mixing Modes (block_size parameter):**
    - block_size=1 (default): Depthwise - no channel mixing within E or I pools
    - block_size>1: LMME - channels mix within blocks in both E and I pools
      This enables cross-channel horizontal connections like hGRU.

    Attributes:
        channels: Number of input/output channels
        dense_mixing: If True, use multi-head parameter sharing (legacy, prefer block_size)
        block_size: Channel mixing block size (1=depthwise, >1=LMME within E/I pools)
        kernel_size: Spatial kernel size for excitation/inhibition kernels
        spectral_rho: Maximum spectral magnitude for stability (should be < 1)
        concat_xy: If True, concat [X,Y] and project to C channels
        gate_activation: Activation for gates ('sigmoid', 'softplus_clamped', 'tanh_scaled')
        rope_mode: Position encoding mode ('spatiotemporal', 'temporal', 'none')
        rope_base: Base for RoPE frequency computation
    """
    channels: int
    dense_mixing: bool = False
    block_size: int = 1  # 1=depthwise (no channel mixing), >1=LMME channel mixing
    kernel_size: int = 11
    spectral_rho: float = 0.999
    concat_xy: bool = True
    gate_activation: str = 'sigmoid'
    gate_rank: int = 0  # If > 0, use low-rank gates: Dense(C -> rank) -> Dense(rank -> n_feats)
    rope_mode: str = 'none'  # 'spatiotemporal', 'temporal', or 'none'
    rope_base: float = 10000.0

    def _compute_gate(self, ctx: jnp.ndarray, output_size: int, name: str) -> jnp.ndarray:
        """Compute gate with optional low-rank bottleneck."""
        if self.gate_rank > 0:
            # Low-rank: ctx -> rank -> output_size
            x = nn.Dense(self.gate_rank, name=f'{name}_down')(ctx)
            x = nn.gelu(x)
            return nn.Dense(output_size, name=f'{name}_up')(x)
        else:
            # Full rank: ctx -> output_size
            return nn.Dense(output_size, name=name)(ctx)

    def _apply_gate_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply configured activation to gate values."""
        if self.gate_activation == 'sigmoid':
            return nn.sigmoid(x)  # [0, 1] - safe default
        elif self.gate_activation == 'softplus_clamped':
            return jnp.minimum(nn.softplus(x), 2.0)  # [0, 2] with clamp
        elif self.gate_activation == 'tanh_scaled':
            return (jnp.tanh(x) + 1) * 0.5  # [0, 1] centered around 0.5
        else:
            return nn.sigmoid(x)  # fallback to safe default

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with full Mamba-style per-frequency gating.

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1  # rfft2 output width

        # Route based on block_size
        if self.block_size > 1:
            return self._forward_lmme(x, B, T, H, W, C, W_freq)

        # === Apply RoPE before FFT (VideoRoPE style) ===
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # --- 1. Spatial Kernels ---
        k_shape = (C, self.kernel_size, self.kernel_size)
        k_exc = self.param('k_exc', nn.initializers.normal(0.02), k_shape)
        k_inh = self.param('k_inh', nn.initializers.normal(0.02), k_shape)

        # Pad and FFT kernels
        pad_h = max(0, (H - self.kernel_size) // 2)
        pad_w = max(0, (W - self.kernel_size) // 2)
        pad_h_after = H - self.kernel_size - pad_h
        pad_w_after = W - self.kernel_size - pad_w

        def pad_kernel(k):
            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k[:, start_h:start_h+H, start_w:start_w+W]
            return jnp.pad(
                k,
                ((0, 0), (pad_h, max(0, pad_h_after)), (pad_w, max(0, pad_w_after))),
                mode='constant'
            )

        # Batch kernel FFTs: stack and do single FFT call (2 -> 1 FFT)
        k_stacked = jnp.stack([pad_kernel(k_exc), pad_kernel(k_inh)], axis=0)  # (2, C, H, W)
        K_stacked_raw = jnp.fft.rfft2(k_stacked, axes=(2, 3))  # (2, C, H, W_f)
        K_E_raw, K_I_raw = K_stacked_raw[0], K_stacked_raw[1]
        # Squash coupling kernel magnitudes to ensure 2x2 matrix spectral radius < 1
        K_E_spec = _stable_spectral_magnitude(K_E_raw, rho=self.spectral_rho)
        K_I_spec = _stable_spectral_magnitude(K_I_raw, rho=self.spectral_rho)

        # --- 2. FFT Input ---
        x_perm = x.transpose(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C, H, W_freq)

        # --- 3. Input-Dependent Per-Channel Per-Frequency Gates (1x1 conv style) ---
        # Pool spatial dims for gating context: (B, T, H, W, C) -> (B, T, C)
        ctx = x.mean(axis=(2, 3))

        # Add temporal position encoding to context
        # This allows gates to vary with timestep even when input is static (repeated image)
        ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)

        # Gate output size: per-channel AND per-frequency (like 1x1 conv)
        gate_size = C * H * W_freq

        # === Per-channel, per-frequency decay gates (diagonal elements) ===
        # alpha: X self-decay, delta: Y self-decay
        alpha_raw = self._compute_gate(ctx, gate_size, 'alpha_gate')  # (B, T, C*H*W_freq)
        alpha_freq = self._apply_gate_activation(alpha_raw).reshape(B, T, C, H, W_freq)

        delta_raw = self._compute_gate(ctx, gate_size, 'delta_gate')  # (B, T, C*H*W_freq)
        delta_freq = self._apply_gate_activation(delta_raw).reshape(B, T, C, H, W_freq)

        # === Per-channel, per-frequency coupling gates (off-diagonal elements) ===
        # mu: inhibition Y->X, gamma: excitation X->Y
        mu_raw = self._compute_gate(ctx, gate_size, 'mu_gate')  # (B, T, C*H*W_freq)
        mu_freq = self._apply_gate_activation(mu_raw).reshape(B, T, C, H, W_freq)

        gamma_raw = self._compute_gate(ctx, gate_size, 'gamma_gate')  # (B, T, C*H*W_freq)
        gamma_freq = self._apply_gate_activation(gamma_raw).reshape(B, T, C, H, W_freq)

        # === B gate: Per-channel, per-frequency input projection ===
        B_gate_raw = self._compute_gate(ctx, gate_size, 'B_gate')  # (B, T, C*H*W_freq)
        B_gate = nn.sigmoid(B_gate_raw).reshape(B, T, C, H, W_freq)

        # === C gate: Per-channel, per-frequency output projection ===
        C_gate_raw = self._compute_gate(ctx, gate_size, 'C_gate')  # (B, T, C*H*W_freq)
        C_gate = nn.sigmoid(C_gate_raw).reshape(B, T, C, H, W_freq)

        # --- 4. Learnable Decay Parameter ---
        # Base decay is learnable per-channel (initialized near 0.9)
        decay_init = nn.initializers.constant(0.9)
        decay_param = self.param('decay', decay_init, (C,))
        decay = jnp.clip(decay_param, 0.1, 0.99)  # Keep in stable range

        # --- 5. Build Transition Matrix ---
        # Target shape: (B, T, C, H, W_freq, 2, 2)

        # Broadcast kernels: (C, H, W_freq) -> (1, 1, C, H, W_freq)
        K_E = K_E_spec[None, None, ...]
        K_I = K_I_spec[None, None, ...]

        # Broadcast decay: (C,) -> (1, 1, C, 1, 1)
        b_decay = decay[None, None, :, None, None]
        decay_complex = b_decay.astype(jnp.complex64)

        # Build 2x2 transition matrix elements with per-frequency gates
        # A_xx = decay * alpha_freq (self-connection for X)
        # A_xy = -K_I * mu_freq (inhibition from Y to X, negative)
        # A_yx = K_E * gamma_freq (excitation from X to Y)
        # A_yy = decay * delta_freq (self-connection for Y)

        A_xx = decay_complex * alpha_freq * jnp.ones_like(K_E)
        A_xy = -1.0 * K_I * mu_freq  # Negative for inhibition
        A_yx = K_E * gamma_freq
        A_yy = decay_complex * delta_freq * jnp.ones_like(K_E)

        # Stack into (B, T, C, H, W_freq, 2, 2)
        row0 = jnp.stack([A_xx, A_xy], axis=-1)
        row1 = jnp.stack([A_yx, A_yy], axis=-1)
        K_mat = jnp.stack([row0, row1], axis=-2)

        # --- 6. Apply B gate to input (Mamba-style input projection) ---
        U_modulated = U_hat * B_gate  # (B, T, C, H, W_freq)

        # Input drives X channel (index 0), Y channel gets zero input
        U_vec = jnp.stack([U_modulated, jnp.zeros_like(U_modulated)], axis=-1)  # (..., 2)

        # --- 7. Log-Space Scan (using GOOM) ---
        K_log = to_goom(K_mat)
        U_log = to_goom(U_vec)

        _, State_log = jax.lax.associative_scan(
            cssm_matrix_scan_op, (K_log, U_log), axis=1
        )

        # --- 8. Apply C gate to output and Inverse Transform ---
        if self.concat_xy:
            # State_log has shape (..., 2) where last dim is [X, Y]
            XY_hat = from_goom(State_log)  # (B, T, C, H, W_freq, 2)

            # Apply C gate to both X and Y channels before IFFT
            # C_gate: (B, T, 1, H, W_freq) -> broadcast over C and apply to both X,Y
            XY_hat_modulated = XY_hat * C_gate[..., None]  # (B, T, C, H, W_freq, 2)

            # Move channel dim for batched IFFT: (B, T, C, H, W_freq, 2) -> (B, T, C, 2, H, W_freq)
            XY_hat_modulated = XY_hat_modulated.transpose(0, 1, 2, 5, 3, 4)
            # Reshape to batch the 2 channels: (B, T, C*2, H, W_freq)
            XY_hat_modulated = XY_hat_modulated.reshape(B, T, C * 2, H, -1)
            xy_out = jnp.fft.irfft2(XY_hat_modulated, s=(H, W), axes=(3, 4))  # (B, T, C*2, H, W)
            xy_out = xy_out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C*2)
            return nn.Dense(C, name='output_proj')(xy_out)
        else:
            # Only need Y channel (index 1)
            Y_log = State_log[..., 1]
            Y_hat = from_goom(Y_log)

            # Apply C gate to Y channel before IFFT
            Y_hat_modulated = Y_hat * C_gate  # (B, T, C, H, W_freq)

            y_out = jnp.fft.irfft2(Y_hat_modulated, s=(H, W), axes=(3, 4))
            y_out = y_out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)
            return y_out

    def _forward_lmme(self, x, B, T, H, W, C, W_freq):
        """
        LMME forward pass with channel mixing within E and I pools.

        This implements hGRU-style horizontal connections where:
        - E neurons can influence other E neurons in the same block
        - I neurons can influence other I neurons in the same block
        - E↔I coupling happens across channels within blocks

        The state per block is [X_block, Y_block] = 2*block_size dimensions.
        Transition is a (2*block_size) × (2*block_size) matrix per spatial frequency.
        """
        block_size = min(self.block_size, C)

        # Pad channels if needed
        if C % block_size != 0:
            pad_c = block_size - (C % block_size)
            x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 0), (0, pad_c)))
            C_padded = C + pad_c
        else:
            C_padded = C
            pad_c = 0

        num_blocks = C_padded // block_size

        # === Apply RoPE before FFT ===
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # --- 1. Block-Diagonal Kernels for E and I ---
        # Shape: (num_blocks, block_size, block_size, kernel_size, kernel_size)
        # This allows channels within a block to mix in both E and I pools
        k_E_blocks = self.param(
            'k_E_blocks',
            nn.initializers.normal(0.02),
            (num_blocks, block_size, block_size, self.kernel_size, self.kernel_size)
        )
        k_I_blocks = self.param(
            'k_I_blocks',
            nn.initializers.normal(0.02),
            (num_blocks, block_size, block_size, self.kernel_size, self.kernel_size)
        )

        # --- 2. Pad and FFT kernels ---
        pad_h = max(0, (H - self.kernel_size) // 2)
        pad_w = max(0, (W - self.kernel_size) // 2)
        pad_h_after = H - self.kernel_size - pad_h
        pad_w_after = W - self.kernel_size - pad_w

        def pad_block_kernel(k):
            # k: (num_blocks, block_size, block_size, kernel_size, kernel_size)
            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k[..., start_h:start_h+H, start_w:start_w+W]
            return jnp.pad(
                k,
                ((0, 0), (0, 0), (0, 0),
                 (pad_h, max(0, pad_h_after)), (pad_w, max(0, pad_w_after))),
                mode='constant'
            )

        K_E_padded = pad_block_kernel(k_E_blocks)
        K_I_padded = pad_block_kernel(k_I_blocks)

        # FFT: (num_blocks, block_size, block_size, H, W) -> (num_blocks, block_size, block_size, H, W_freq)
        K_E_hat_raw = jnp.fft.rfft2(K_E_padded, axes=(3, 4))
        K_I_hat_raw = jnp.fft.rfft2(K_I_padded, axes=(3, 4))

        # Stabilize spectral magnitude
        K_E_hat = _stable_spectral_magnitude(K_E_hat_raw, rho=self.spectral_rho)
        K_I_hat = _stable_spectral_magnitude(K_I_hat_raw, rho=self.spectral_rho)

        # --- 3. FFT Input ---
        x_perm = x.transpose(0, 1, 4, 2, 3)  # (B, T, C_padded, H, W)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C_padded, H, W_freq)

        # Reshape to blocks: (B, T, num_blocks, block_size, H, W_freq)
        U_hat_blocks = U_hat.reshape(B, T, num_blocks, block_size, H, W_freq)

        # --- 4. Input-Dependent Per-Channel Gates (1x1 conv style) ---
        ctx = x.mean(axis=(2, 3))  # (B, T, C_padded) - use full channel context for expressivity

        # Add temporal position encoding to context
        # This allows gates to vary with timestep even when input is static (repeated image)
        ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)

        # Per-channel gate size (full 1x1 conv equivalent)
        n_gate_feats = C_padded * H * W_freq  # Per-channel, per-frequency

        # Decay gates for E and I - shape (B, T, num_blocks, H, W_freq, block_size, 1)
        # Each channel within a block gets its own decay
        alpha_raw = self._compute_gate(ctx, n_gate_feats, 'alpha_gate')
        alpha_freq = self._apply_gate_activation(alpha_raw).reshape(B, T, num_blocks, block_size, H, W_freq)
        alpha_freq = alpha_freq.transpose(0, 1, 2, 4, 5, 3)[..., None]  # (B, T, num_blocks, H, W_freq, block_size, 1)

        delta_raw = self._compute_gate(ctx, n_gate_feats, 'delta_gate')
        delta_freq = self._apply_gate_activation(delta_raw).reshape(B, T, num_blocks, block_size, H, W_freq)
        delta_freq = delta_freq.transpose(0, 1, 2, 4, 5, 3)[..., None]  # (B, T, num_blocks, H, W_freq, block_size, 1)

        # Coupling gates - per-channel modulation of the coupling kernels
        mu_raw = self._compute_gate(ctx, n_gate_feats, 'mu_gate')
        mu_freq = self._apply_gate_activation(mu_raw).reshape(B, T, num_blocks, block_size, H, W_freq)
        mu_freq = mu_freq.transpose(0, 1, 2, 4, 5, 3)[..., None]  # (B, T, num_blocks, H, W_freq, block_size, 1)

        gamma_raw = self._compute_gate(ctx, n_gate_feats, 'gamma_gate')
        gamma_freq = self._apply_gate_activation(gamma_raw).reshape(B, T, num_blocks, block_size, H, W_freq)
        gamma_freq = gamma_freq.transpose(0, 1, 2, 4, 5, 3)[..., None]  # (B, T, num_blocks, H, W_freq, block_size, 1)

        # B and C gates - per-channel input/output gating
        B_gate_raw = self._compute_gate(ctx, n_gate_feats, 'B_gate')
        B_gate = nn.sigmoid(B_gate_raw).reshape(B, T, num_blocks, block_size, H, W_freq)

        C_gate_raw = self._compute_gate(ctx, n_gate_feats, 'C_gate')
        C_gate = nn.sigmoid(C_gate_raw).reshape(B, T, num_blocks, block_size, H, W_freq)

        # --- 5. Learnable Decay ---
        decay_init = nn.initializers.constant(0.9)
        decay_param = self.param('decay', decay_init, (num_blocks,))
        decay = jnp.clip(decay_param, 0.1, 0.99)
        # Shape: (1, 1, num_blocks, 1, 1, 1, 1) to broadcast with (B, T, num_blocks, H, W_freq, bs, bs)
        decay = decay[None, None, :, None, None, None, None].astype(jnp.complex64)

        # --- 6. Build 2*block_size × 2*block_size Transition Matrix ---
        # K_E_hat, K_I_hat: (num_blocks, block_size, block_size, H, W_freq)
        # Transpose to (1, 1, num_blocks, H, W_freq, block_size, block_size)
        K_E = K_E_hat.transpose(0, 3, 4, 1, 2)[None, None, ...]  # (1, 1, num_blocks, H, W_freq, bs, bs)
        K_I = K_I_hat.transpose(0, 3, 4, 1, 2)[None, None, ...]

        # Build identity matrices for diagonal blocks
        # Shape: (1, 1, 1, 1, 1, bs, bs) to broadcast with (B, T, num_blocks, H, W_freq, bs, bs)
        eye_block = jnp.eye(block_size, dtype=jnp.complex64)
        eye_block = eye_block[None, None, None, None, None, :, :]

        # A_XX = decay * alpha * I (E self-decay, diagonal)
        A_XX = decay * alpha_freq * eye_block

        # A_XY = -K_I * mu (I -> E inhibition, with channel mixing)
        A_XY = -1.0 * K_I * mu_freq

        # A_YX = K_E * gamma (E -> I excitation, with channel mixing)
        A_YX = K_E * gamma_freq

        # A_YY = decay * delta * I (I self-decay, diagonal)
        A_YY = decay * delta_freq * eye_block

        # Stack into (B, T, num_blocks, H, W_freq, 2*block_size, 2*block_size)
        # Top row: [A_XX, A_XY]
        # Bottom row: [A_YX, A_YY]
        top_row = jnp.concatenate([A_XX, A_XY], axis=-1)  # (..., bs, 2*bs)
        bottom_row = jnp.concatenate([A_YX, A_YY], axis=-1)
        K_mat = jnp.concatenate([top_row, bottom_row], axis=-2)  # (..., 2*bs, 2*bs)

        # Broadcast to full batch/time dims
        K_mat = jnp.broadcast_to(K_mat, (B, T, num_blocks, H, W_freq, 2*block_size, 2*block_size))

        # --- 7. Prepare Input Vector ---
        # Apply B gate
        U_modulated = U_hat_blocks * B_gate  # (B, T, num_blocks, block_size, H, W_freq)

        # Transpose for scan: (B, T, num_blocks, H, W_freq, block_size)
        U_perm = U_modulated.transpose(0, 1, 2, 4, 5, 3)

        # Input drives E (X), I (Y) gets zero: [U, 0]
        zeros = jnp.zeros_like(U_perm)
        U_vec = jnp.concatenate([U_perm, zeros], axis=-1)  # (B, T, num_blocks, H, W_freq, 2*block_size)

        # --- 8. Log-Space Scan ---
        K_log = to_goom(K_mat)
        U_log = to_goom(U_vec)

        # Flatten spatial frequencies with batch for scan
        K_flat = K_log.reshape(B * num_blocks * H * W_freq, T, 2*block_size, 2*block_size)
        U_flat = U_log.reshape(B * num_blocks * H * W_freq, T, 2*block_size)

        # Scan over time
        block_scan_op = make_block_scan_op(2 * block_size)
        _, State_log_flat = jax.lax.associative_scan(
            block_scan_op, (K_flat, U_flat), axis=1
        )

        # Reshape back: (B, num_blocks, H, W_freq, T, 2*block_size)
        State_log = State_log_flat.reshape(B, num_blocks, H, W_freq, T, 2*block_size)
        State_log = State_log.transpose(0, 4, 1, 2, 3, 5)  # (B, T, num_blocks, H, W_freq, 2*block_size)

        # --- 9. Extract E and I, Apply C gate, Inverse Transform ---
        State_hat = from_goom(State_log)

        # Split into E and I
        E_hat = State_hat[..., :block_size]  # (B, T, num_blocks, H, W_freq, block_size)
        I_hat = State_hat[..., block_size:]

        # Apply C gate (per-channel gating)
        # C_gate: (B, T, num_blocks, block_size, H, W_freq)
        C_gate_expanded = C_gate.transpose(0, 1, 2, 4, 5, 3)  # (B, T, num_blocks, H, W_freq, block_size)
        E_hat_gated = E_hat * C_gate_expanded
        I_hat_gated = I_hat * C_gate_expanded

        if self.concat_xy:
            # Concat E and I, then project
            EI_hat = jnp.concatenate([E_hat_gated, I_hat_gated], axis=-1)  # (..., 2*block_size)

            # Reshape for IFFT: (B, T, num_blocks, H, W_freq, 2*block_size)
            #   -> (B, T, num_blocks * 2 * block_size, H, W_freq)
            EI_hat_flat = EI_hat.transpose(0, 1, 2, 5, 3, 4)  # (B, T, num_blocks, 2*bs, H, W_freq)
            EI_hat_flat = EI_hat_flat.reshape(B, T, num_blocks * 2 * block_size, H, W_freq)

            ei_out = jnp.fft.irfft2(EI_hat_flat, s=(H, W), axes=(3, 4))  # (B, T, C*2, H, W)
            ei_out = ei_out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C*2)

            # Remove padding if needed
            if pad_c > 0:
                ei_out = ei_out[..., :C*2]

            return nn.Dense(C, name='output_proj')(ei_out)
        else:
            # Only return I (inhibition output)
            I_hat_flat = I_hat_gated.transpose(0, 1, 2, 5, 3, 4)  # (B, T, num_blocks, bs, H, W_freq)
            I_hat_flat = I_hat_flat.reshape(B, T, C_padded, H, W_freq)

            i_out = jnp.fft.irfft2(I_hat_flat, s=(H, W), axes=(3, 4))
            i_out = i_out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C_padded)

            if pad_c > 0:
                i_out = i_out[..., :C]

            return i_out


class BilinearOpponentCSSM(nn.Module):
    """
    3x3 Bilinear Opponent CSSM with hGRU-style X² inhibition.

    State: [X, Y, Z] where:
    - X: Excitatory state
    - Y: Inhibitory state (accumulates over time)
    - Z: Delayed X (Z_t = X_{t-1})

    Key feature: Inhibition includes X*Z term, giving X² dynamics:
    X_t = U - K_I*ν*X*Z - K_I*μ*Y + α*X
        ≈ U - K_I*ν*X² - K_I*μ*Y + α*X  (hGRU-style!)

    In log-space, X*Z becomes log(X) + log(Z) — just addition!

    Attributes:
        channels: Number of input/output channels
        kernel_size: Spatial kernel size for excitation/inhibition kernels
        spectral_rho: Maximum spectral magnitude for stability (should be < 1)
        gate_activation: Activation for gates ('sigmoid', 'softplus')
        rope_mode: Position encoding mode ('spatiotemporal', 'temporal', 'none')
        rope_base: Base for RoPE frequency computation
    """
    channels: int
    kernel_size: int = 11
    spectral_rho: float = 0.999
    gate_activation: str = 'sigmoid'
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    gate_rank: int = 0  # If > 0, use low-rank gates: Dense(C -> rank) -> Dense(rank -> n_feats)
    # API compatibility attributes
    concat_xy: bool = True
    dense_mixing: bool = False
    block_size: int = 32

    def _low_rank_gate(self, ctx: jnp.ndarray, n_out: int, name: str) -> jnp.ndarray:
        """Low-rank gate projection to reduce parameters."""
        if self.gate_rank > 0:
            # Bottleneck: C -> rank -> n_out
            x = nn.Dense(self.gate_rank, name=f'{name}_down')(ctx)
            x = nn.gelu(x)
            x = nn.Dense(n_out, name=f'{name}_up')(x)
        else:
            # Direct projection: C -> n_out
            x = nn.Dense(n_out, name=name)(ctx)
        return x

    def _apply_gate_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply configured activation to gate values."""
        if self.gate_activation == 'sigmoid':
            return nn.sigmoid(x)
        elif self.gate_activation == 'softplus':
            return nn.softplus(x)
        else:
            return nn.sigmoid(x)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with 3x3 hGRU-style dynamics.

        Uses linear 3x3 state dynamics with Z = X_{t-1} for delayed self-inhibition.
        The X update includes: -K_I*ν*Z (delayed inhibition) and -K_I*μ*Y (surround inhibition).

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

        # Route to LMME if block_size > 1
        if self.block_size > 1:
            return self._forward_lmme(x, B, T, H, W, C, W_freq)

        from .math import cssm_3x3_matrix_scan_op

        # === Apply RoPE before FFT ===
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # --- 1. Spatial Kernels ---
        k_shape = (C, self.kernel_size, self.kernel_size)
        k_exc = self.param('k_exc', nn.initializers.normal(0.02), k_shape)
        k_inh = self.param('k_inh', nn.initializers.normal(0.02), k_shape)

        # Pad and FFT kernels
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

        # --- 2. FFT Input ---
        x_perm = x.transpose(0, 1, 4, 2, 3)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C, H, W_freq)

        # --- 3. Input-Dependent Gates ---
        ctx = x.mean(axis=(2, 3))  # (B, T, C)
        n_gate_feats = H * W_freq

        # Decay gates - use low-rank if gate_rank > 0
        alpha_raw = self._low_rank_gate(ctx, n_gate_feats, 'alpha_gate')
        alpha_freq = self._apply_gate_activation(alpha_raw).reshape(B, T, 1, H, W_freq)

        delta_raw = self._low_rank_gate(ctx, n_gate_feats, 'delta_gate')
        delta_freq = self._apply_gate_activation(delta_raw).reshape(B, T, 1, H, W_freq)

        # Coupling gates
        mu_raw = self._low_rank_gate(ctx, n_gate_feats, 'mu_gate')  # Y -> X inhibition
        mu_freq = self._apply_gate_activation(mu_raw).reshape(B, T, 1, H, W_freq)

        nu_raw = self._low_rank_gate(ctx, n_gate_feats, 'nu_gate')  # Z -> X inhibition (delayed self)
        nu_freq = self._apply_gate_activation(nu_raw).reshape(B, T, 1, H, W_freq)

        gamma_raw = self._low_rank_gate(ctx, n_gate_feats, 'gamma_gate')  # X -> Y excitation
        gamma_freq = self._apply_gate_activation(gamma_raw).reshape(B, T, 1, H, W_freq)

        # I/O gates
        B_gate_raw = self._low_rank_gate(ctx, n_gate_feats, 'B_gate')
        B_gate = nn.sigmoid(B_gate_raw).reshape(B, T, 1, H, W_freq)

        C_gate_raw = self._low_rank_gate(ctx, n_gate_feats, 'C_gate')
        C_gate = nn.sigmoid(C_gate_raw).reshape(B, T, 1, H, W_freq)

        # --- 4. Learnable Decay ---
        decay = jnp.clip(self.param('decay', nn.initializers.constant(0.9), (C,)), 0.1, 0.99)
        b_decay = decay[None, None, :, None, None].astype(jnp.complex64)

        # --- 5. Build 3x3 Transition Matrix ---
        K_E_b = K_E[None, None, ...]  # (1, 1, C, H, W_freq)
        K_I_b = K_I[None, None, ...]

        # Matrix elements (in linear space, will convert to log)
        # All elements need shape (B, T, C, H, W_freq) for stacking
        target_shape = (B, T, C, H, W_freq)
        ones_full = jnp.broadcast_to(jnp.ones_like(K_E_b), target_shape)
        zeros_full = jnp.broadcast_to(jnp.zeros_like(K_E_b), target_shape)

        # Row 0 (X update): X_t = α*X - K_I*μ*Y - K_I*ν*Z + U
        # Linear delayed self-inhibition via Z (approximates hGRU's X² term)
        A_xx = b_decay * alpha_freq * ones_full
        A_xy = -1.0 * K_I_b * mu_freq  # Negative for inhibition from Y
        A_xz = -1.0 * K_I_b * nu_freq  # Negative for delayed self-inhibition from Z

        # Row 1 (Y update): Y_t = K_E*γ*X + δ*Y + 0*Z
        A_yx = K_E_b * gamma_freq
        A_yy = b_decay * delta_freq * ones_full
        A_yz = zeros_full

        # Row 2 (Z update): Z_t = 1*X + 0*Y + 0*Z  (Z = X_{t-1})
        A_zx = ones_full
        A_zy = zeros_full
        A_zz = zeros_full

        # Stack into 3x3: (B, T, C, H, W_freq, 3, 3)
        row0 = jnp.stack([A_xx, A_xy, A_xz], axis=-1)
        row1 = jnp.stack([A_yx, A_yy, A_yz], axis=-1)
        row2 = jnp.stack([A_zx, A_zy, A_zz], axis=-1)
        K_mat = jnp.stack([row0, row1, row2], axis=-2)

        # --- 6. Input vector: [U, 0, 0] ---
        U_modulated = U_hat * B_gate
        zeros = jnp.zeros_like(U_modulated)
        U_vec = jnp.stack([U_modulated, zeros, zeros], axis=-1)  # (B, T, C, H, W_freq, 3)

        # --- 7. Log-Space Scan ---
        K_log = to_goom(K_mat)
        U_log = to_goom(U_vec)

        # Use linear 3x3 scan (delayed self-inhibition via A_xz)
        _, State_log = jax.lax.associative_scan(
            cssm_3x3_matrix_scan_op, (K_log, U_log), axis=1
        )

        # --- 8. Output: Use X state ---
        X_log = State_log[..., 0]  # Take X component (B, T, C, H, W_freq)
        X_hat = from_goom(X_log)
        X_hat_modulated = X_hat * C_gate

        x_out = jnp.fft.irfft2(X_hat_modulated, s=(H, W), axes=(3, 4))
        return x_out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)

    def _forward_lmme(self, x, B, T, H, W, C, W_freq):
        """
        LMME forward pass with channel mixing for 3x3 bilinear CSSM.

        Uses 3*block_size × 3*block_size transition matrices where:
        - X, Y, Z each have block_size channels that can mix
        - Cross-channel spatial mixing via K_E (excitation) and K_I (inhibition)
        - Enables association field-style curve propagation

        State per block: [X_block, Y_block, Z_block] = 3*block_size dimensions
        """
        from .math import make_block_scan_op

        block_size = min(self.block_size, C)

        # Pad channels if needed
        if C % block_size != 0:
            pad_c = block_size - (C % block_size)
            x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 0), (0, pad_c)))
            C_padded = C + pad_c
        else:
            C_padded = C
            pad_c = 0

        num_blocks = C_padded // block_size

        # === Apply RoPE before FFT ===
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # --- 1. Block-Diagonal Kernels for E and I ---
        # Shape: (num_blocks, block_size, block_size, kernel_size, kernel_size)
        k_E_blocks = self.param(
            'k_E_blocks',
            nn.initializers.normal(0.02),
            (num_blocks, block_size, block_size, self.kernel_size, self.kernel_size)
        )
        k_I_blocks = self.param(
            'k_I_blocks',
            nn.initializers.normal(0.02),
            (num_blocks, block_size, block_size, self.kernel_size, self.kernel_size)
        )

        # --- 2. Pad and FFT kernels ---
        pad_h = max(0, (H - self.kernel_size) // 2)
        pad_w = max(0, (W - self.kernel_size) // 2)
        pad_h_after = H - self.kernel_size - pad_h
        pad_w_after = W - self.kernel_size - pad_w

        def pad_block_kernel(k):
            if self.kernel_size > H or self.kernel_size > W:
                start_h = (self.kernel_size - H) // 2
                start_w = (self.kernel_size - W) // 2
                return k[..., start_h:start_h+H, start_w:start_w+W]
            return jnp.pad(
                k,
                ((0, 0), (0, 0), (0, 0),
                 (pad_h, max(0, pad_h_after)), (pad_w, max(0, pad_w_after))),
                mode='constant'
            )

        K_E_padded = pad_block_kernel(k_E_blocks)
        K_I_padded = pad_block_kernel(k_I_blocks)

        # FFT: (num_blocks, bs, bs, H, W) -> (num_blocks, bs, bs, H, W_freq)
        K_E_hat_raw = jnp.fft.rfft2(K_E_padded, axes=(3, 4))
        K_I_hat_raw = jnp.fft.rfft2(K_I_padded, axes=(3, 4))

        K_E_hat = _stable_spectral_magnitude(K_E_hat_raw, rho=self.spectral_rho)
        K_I_hat = _stable_spectral_magnitude(K_I_hat_raw, rho=self.spectral_rho)

        # --- 3. FFT Input ---
        x_perm = x.transpose(0, 1, 4, 2, 3)  # (B, T, C_padded, H, W)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C_padded, H, W_freq)

        # Reshape to blocks: (B, T, num_blocks, block_size, H, W_freq)
        U_hat_blocks = U_hat.reshape(B, T, num_blocks, block_size, H, W_freq)

        # --- 4. Input-Dependent Gates ---
        ctx = x.mean(axis=(2, 3))  # (B, T, C_padded)

        n_gate_feats = num_blocks * H * W_freq

        # Decay gates - shape (B, T, num_blocks, H, W_freq, 1, 1)
        # Use low-rank gates if gate_rank > 0 to reduce parameters
        alpha_raw = self._low_rank_gate(ctx, n_gate_feats, 'alpha_gate')
        alpha_freq = self._apply_gate_activation(alpha_raw).reshape(B, T, num_blocks, H, W_freq, 1, 1)

        delta_raw = self._low_rank_gate(ctx, n_gate_feats, 'delta_gate')
        delta_freq = self._apply_gate_activation(delta_raw).reshape(B, T, num_blocks, H, W_freq, 1, 1)

        # Coupling gates
        mu_raw = self._low_rank_gate(ctx, n_gate_feats, 'mu_gate')  # Y -> X
        mu_freq = self._apply_gate_activation(mu_raw).reshape(B, T, num_blocks, H, W_freq, 1, 1)

        nu_raw = self._low_rank_gate(ctx, n_gate_feats, 'nu_gate')  # Z -> X (delayed self-inhibition)
        nu_freq = self._apply_gate_activation(nu_raw).reshape(B, T, num_blocks, H, W_freq, 1, 1)

        gamma_raw = self._low_rank_gate(ctx, n_gate_feats, 'gamma_gate')  # X -> Y
        gamma_freq = self._apply_gate_activation(gamma_raw).reshape(B, T, num_blocks, H, W_freq, 1, 1)

        # B and C gates
        B_gate_raw = self._low_rank_gate(ctx, n_gate_feats, 'B_gate')
        B_gate = nn.sigmoid(B_gate_raw).reshape(B, T, num_blocks, 1, H, W_freq)

        C_gate_raw = self._low_rank_gate(ctx, n_gate_feats, 'C_gate')
        C_gate = nn.sigmoid(C_gate_raw).reshape(B, T, num_blocks, 1, H, W_freq)

        # --- 5. Learnable Decay ---
        decay_param = self.param('decay', nn.initializers.constant(0.9), (num_blocks,))
        decay = jnp.clip(decay_param, 0.1, 0.99)
        decay = decay[None, None, :, None, None, None, None].astype(jnp.complex64)

        # --- 6. Build 3*block_size × 3*block_size Transition Matrix ---
        # K_E_hat, K_I_hat: (num_blocks, bs, bs, H, W_freq)
        # Transpose to (1, 1, num_blocks, H, W_freq, bs, bs)
        K_E = K_E_hat.transpose(0, 3, 4, 1, 2)[None, None, ...]
        K_I = K_I_hat.transpose(0, 3, 4, 1, 2)[None, None, ...]

        # Identity block for diagonal terms
        eye_block = jnp.eye(block_size, dtype=jnp.complex64)
        eye_block = eye_block[None, None, None, None, None, :, :]

        # Zero block
        zero_block = jnp.zeros((block_size, block_size), dtype=jnp.complex64)
        zero_block = zero_block[None, None, None, None, None, :, :]

        # Row 0 (X update): X = α*X - K_I*μ*Y - K_I*ν*Z + U
        A_XX = decay * alpha_freq * eye_block  # Self-decay (diagonal)
        A_XY = -1.0 * K_I * mu_freq            # Y -> X inhibition (full block mixing)
        A_XZ = -1.0 * K_I * nu_freq            # Z -> X delayed inhibition (full block mixing)

        # Row 1 (Y update): Y = K_E*γ*X + δ*Y
        A_YX = K_E * gamma_freq                # X -> Y excitation (full block mixing)
        A_YY = decay * delta_freq * eye_block  # Self-decay (diagonal)
        A_YZ = zero_block                      # No Z -> Y

        # Row 2 (Z update): Z = X (Z tracks X with 1-step delay)
        A_ZX = eye_block                       # X -> Z (identity, diagonal)
        A_ZY = zero_block                      # No Y -> Z
        A_ZZ = zero_block                      # No Z -> Z (Z is replaced each step)

        # Broadcast all blocks to full shape
        full_shape = (B, T, num_blocks, H, W_freq, block_size, block_size)
        A_XX = jnp.broadcast_to(A_XX, full_shape)
        A_XY = jnp.broadcast_to(A_XY, full_shape)
        A_XZ = jnp.broadcast_to(A_XZ, full_shape)
        A_YX = jnp.broadcast_to(A_YX, full_shape)
        A_YY = jnp.broadcast_to(A_YY, full_shape)
        A_YZ = jnp.broadcast_to(A_YZ, full_shape)
        A_ZX = jnp.broadcast_to(A_ZX, full_shape)
        A_ZY = jnp.broadcast_to(A_ZY, full_shape)
        A_ZZ = jnp.broadcast_to(A_ZZ, full_shape)

        # Stack into 3x3 block matrix: (B, T, num_blocks, H, W_freq, 3*bs, 3*bs)
        # Top row: [A_XX, A_XY, A_XZ]
        top_row = jnp.concatenate([A_XX, A_XY, A_XZ], axis=-1)     # (..., bs, 3*bs)
        mid_row = jnp.concatenate([A_YX, A_YY, A_YZ], axis=-1)     # (..., bs, 3*bs)
        bot_row = jnp.concatenate([A_ZX, A_ZY, A_ZZ], axis=-1)     # (..., bs, 3*bs)
        K_mat = jnp.concatenate([top_row, mid_row, bot_row], axis=-2)  # (..., 3*bs, 3*bs)

        # --- 7. Prepare Input Vector ---
        U_modulated = U_hat_blocks * B_gate  # (B, T, num_blocks, block_size, H, W_freq)

        # Transpose for scan: (B, T, num_blocks, H, W_freq, block_size)
        U_perm = U_modulated.transpose(0, 1, 2, 4, 5, 3)

        # Input drives X only: [U, 0, 0]
        zeros = jnp.zeros_like(U_perm)
        U_vec = jnp.concatenate([U_perm, zeros, zeros], axis=-1)  # (..., 3*block_size)

        # --- 8. Log-Space Scan ---
        K_log = to_goom(K_mat)
        U_log = to_goom(U_vec)

        # Flatten for scan
        K_flat = K_log.reshape(B * num_blocks * H * W_freq, T, 3*block_size, 3*block_size)
        U_flat = U_log.reshape(B * num_blocks * H * W_freq, T, 3*block_size)

        # Scan over time
        block_scan_op = make_block_scan_op(3 * block_size)
        _, State_log_flat = jax.lax.associative_scan(
            block_scan_op, (K_flat, U_flat), axis=1
        )

        # Reshape back
        State_log = State_log_flat.reshape(B, num_blocks, H, W_freq, T, 3*block_size)
        State_log = State_log.transpose(0, 4, 1, 2, 3, 5)  # (B, T, num_blocks, H, W_freq, 3*bs)

        # --- 9. Extract X, Apply C gate, Inverse Transform ---
        State_hat = from_goom(State_log)

        # Take X component (first block_size elements)
        X_hat = State_hat[..., :block_size]  # (B, T, num_blocks, H, W_freq, block_size)

        # Apply C gate and transpose
        X_hat_gated = X_hat.transpose(0, 1, 2, 5, 3, 4)  # (B, T, num_blocks, bs, H, W_freq)
        X_hat_gated = X_hat_gated * C_gate

        # Reshape to channels
        X_hat_channels = X_hat_gated.reshape(B, T, C_padded, H, W_freq)

        # Inverse FFT
        x_out = jnp.fft.irfft2(X_hat_channels, s=(H, W), axes=(3, 4))
        x_out = x_out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C_padded)

        # Remove padding if needed
        if pad_c > 0:
            x_out = x_out[..., :C]

        return x_out


# =============================================================================
# LINEAR ABLATION VARIANTS (No Log-Space)
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


class HGRUStyleCSSM(nn.Module):
    """
    hGRU-style CSSM with linear opponent dynamics via PARALLEL associative scan.

    Implements linearized hGRU equations using GOOM log-space:

        X_t = decay_x·X_{t-1} − μ_inhib·K_I·Y_{t-1} + U_X
        Y_t = decay_y·Y_{t-1} + μ_excit·K_E·X_{t-1} + U_Y

    Key design choices:
    - ALL parameters are input-dependent gates (decay, coupling, I/O)
    - Separate gates for inhibition vs excitation pathways
    - Both X and Y receive transformed input (not just X)
    - Xavier initialization for spatial kernels
    - No bilinear terms (clean linear dynamics for associative scan)

    In log-spectral (GOOM) domain:
    - Convolution → Multiplication → Addition
    - Stable numerical dynamics without clipping

    Attributes:
        channels: Number of input/output channels
        kernel_size: Spatial kernel size for E/I kernels
        spectral_rho: Maximum spectral magnitude for stability
        gate_activation: Activation for gates ('sigmoid', 'softplus')
        rope_mode: Position encoding mode
        rope_base: RoPE base frequency
        concat_xy: If True, concatenate [X, Y] and project to output
    """
    channels: int
    kernel_size: int = 11
    spectral_rho: float = 0.999
    gate_activation: str = 'sigmoid'
    rope_mode: str = 'none'
    rope_base: float = 10000.0
    concat_xy: bool = True
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

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with linear opponent dynamics using parallel scan.

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        from .goom import to_goom, from_goom

        B, T, H, W, C = x.shape
        W_freq = W // 2 + 1

        # === Apply RoPE before FFT ===
        if self.rope_mode != 'none':
            x = apply_rope(x, mode=self.rope_mode, base=self.rope_base)

        # --- 1. Spatial Kernels (Xavier init) ---
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
        K_E = _stable_spectral_magnitude(K_E_raw, rho=self.spectral_rho)  # (C, H, W_freq)
        K_I = _stable_spectral_magnitude(K_I_raw, rho=self.spectral_rho)

        # --- 2. Project input to X and Y pathways (fused) ---
        # Single Dense(2C) then split - more efficient than two Dense(C)
        x_flat = x.reshape(B * T, H, W, C)
        xy_proj = nn.Dense(2 * C, name='input_proj')(x_flat)  # (B*T, H, W, 2C)
        xy_proj = xy_proj.reshape(B, T, H, W, 2 * C)

        # Split into X and Y input pathways
        x_input = xy_proj[..., :C]   # (B, T, H, W, C)
        y_input = xy_proj[..., C:]   # (B, T, H, W, C)

        # FFT both pathways
        x_perm = x_input.transpose(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        y_perm = y_input.transpose(0, 1, 4, 2, 3)
        U_X_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C, H, W_freq)
        U_Y_hat = jnp.fft.rfft2(y_perm, axes=(3, 4))

        # --- 3. Input-Dependent Gates (ALL gates from context) ---
        ctx = x.mean(axis=(2, 3))  # (B, T, C)
        ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)
        n_gate_feats = H * W_freq

        # Decay gates (input-dependent, not static!)
        decay_x_raw = nn.Dense(n_gate_feats, name='decay_x_gate')(ctx)
        decay_x_freq = 0.1 + 0.89 * nn.sigmoid(decay_x_raw).reshape(B, T, 1, H, W_freq)  # [0.1, 0.99]

        decay_y_raw = nn.Dense(n_gate_feats, name='decay_y_gate')(ctx)
        decay_y_freq = 0.1 + 0.89 * nn.sigmoid(decay_y_raw).reshape(B, T, 1, H, W_freq)  # [0.1, 0.99]

        # Separate coupling gates for each direction
        mu_inhib_raw = nn.Dense(n_gate_feats, name='mu_inhib_gate')(ctx)  # Y -> X inhibition
        mu_inhib = self._apply_gate_activation(mu_inhib_raw).reshape(B, T, 1, H, W_freq)

        mu_excit_raw = nn.Dense(n_gate_feats, name='mu_excit_gate')(ctx)  # X -> Y excitation
        mu_excit = self._apply_gate_activation(mu_excit_raw).reshape(B, T, 1, H, W_freq)

        # I/O gates
        B_gate_raw = nn.Dense(n_gate_feats, name='B_gate')(ctx)  # X input gate
        B_gate = nn.sigmoid(B_gate_raw).reshape(B, T, 1, H, W_freq)

        D_gate_raw = nn.Dense(n_gate_feats, name='D_gate')(ctx)  # Y input gate
        D_gate = nn.sigmoid(D_gate_raw).reshape(B, T, 1, H, W_freq)

        C_gate_raw = nn.Dense(n_gate_feats, name='C_gate')(ctx)  # Output gate
        C_gate = nn.sigmoid(C_gate_raw).reshape(B, T, 1, H, W_freq)

        # --- 4. Build 2x2 Transition Matrix ---
        # Broadcast kernels: (C, H, W_freq) -> (1, 1, C, H, W_freq)
        K_E_b = K_E[None, None, ...].astype(jnp.complex64)
        K_I_b = K_I[None, None, ...].astype(jnp.complex64)

        # Convert decay gates to complex
        decay_x_c = decay_x_freq.astype(jnp.complex64)
        decay_y_c = decay_y_freq.astype(jnp.complex64)

        # Linear opponent dynamics:
        # X_t = decay_x·X − μ_inhib·K_I·Y + U_X   (inhibition from Y)
        # Y_t = decay_y·Y + μ_excit·K_E·X + U_Y   (excitation from X)
        #
        # 2x2 matrix: [decay_x        -μ_inhib·K_I]
        #             [μ_excit·K_E     decay_y    ]

        A_xx = decay_x_c  # (B, T, 1, H, W_freq) broadcasts to (B, T, C, H, W_freq)
        A_xy = -1.0 * mu_inhib * K_I_b  # Negative for inhibition
        A_yx = mu_excit * K_E_b  # Positive for excitation
        A_yy = decay_y_c

        # Stack into (B, T, C, H, W_freq, 2, 2)
        row0 = jnp.stack([A_xx * jnp.ones_like(K_E_b), A_xy], axis=-1)
        row1 = jnp.stack([A_yx, A_yy * jnp.ones_like(K_E_b)], axis=-1)
        K_mat = jnp.stack([row0, row1], axis=-2)

        # --- 5. Apply I/O gates to input ---
        U_X_gated = U_X_hat * B_gate  # X pathway input
        U_Y_gated = U_Y_hat * D_gate  # Y pathway input
        U_vec = jnp.stack([U_X_gated, U_Y_gated], axis=-1)  # (..., 2)

        # --- 6. Convert to GOOM and Apply Parallel Associative Scan ---
        K_log = to_goom(K_mat)
        U_log = to_goom(U_vec)

        # Linear 2x2 scan operator (no bilinear terms)
        # PARALLEL scan! O(log T) complexity
        _, State_log = jax.lax.associative_scan(
            cssm_matrix_scan_op, (K_log, U_log), axis=1
        )

        # --- 7. Convert back from GOOM and apply output gate ---
        if self.concat_xy:
            XY_hat = from_goom(State_log)  # (B, T, C, H, W_freq, 2)
            XY_hat_gated = XY_hat * C_gate[..., None]

            # Reshape for IFFT
            XY_hat_gated = XY_hat_gated.transpose(0, 1, 2, 5, 3, 4)  # (B, T, C, 2, H, W_freq)
            XY_hat_gated = XY_hat_gated.reshape(B, T, C * 2, H, -1)

            xy_out = jnp.fft.irfft2(XY_hat_gated, s=(H, W), axes=(3, 4))
            xy_out = xy_out.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, 2C)
            return nn.Dense(C, name='output_proj')(xy_out)
        else:
            # Use X channel (index 0)
            X_log = State_log[..., 0]
            X_hat = from_goom(X_log)
            X_hat_gated = X_hat * C_gate

            x_out = jnp.fft.irfft2(X_hat_gated, s=(H, W), axes=(3, 4))
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
        K_hat = _stable_spectral_magnitude(K_hat_raw, rho=self.spectral_rho)

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

        # === Input-dependent gates (MINIMAL set) ===
        # Use raw input (before position encoding) for gates if position_independent_gates is True
        # This makes gates invariant to sequence position, improving length generalization
        gate_input = x_raw if x_raw is not None else x
        ctx = gate_input.mean(axis=(2, 3))
        if x_raw is None:  # Only apply temporal RoPE to context if using position-encoded input
            ctx = apply_temporal_rope_to_context(ctx, base=self.rope_base)
        n_gate = H * W_freq

        # 3 decay gates (bounded 0.1-0.99)
        decay_Q = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='decay_Q')(ctx)).reshape(B, T, 1, H, W_freq)
        decay_K = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='decay_K')(ctx)).reshape(B, T, 1, H, W_freq)
        decay_A = 0.1 + 0.89 * nn.sigmoid(nn.Dense(n_gate, name='decay_A')(ctx)).reshape(B, T, 1, H, W_freq)

        # Q↔K coupling (single weight, symmetric)
        w_qk = nn.sigmoid(nn.Dense(n_gate, name='w_qk')(ctx)).reshape(B, T, 1, H, W_freq)

        # A→Q feedback (attention application)
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

        K_b = K_hat[None, None, ...].astype(jnp.complex64)
        ones = jnp.ones_like(K_b)
        zeros = jnp.zeros_like(decay_Q.astype(jnp.complex64) * ones)

        decay_Q_c = decay_Q.astype(jnp.complex64)
        decay_K_c = decay_K.astype(jnp.complex64)
        decay_A_c = decay_A.astype(jnp.complex64)

        # Row 0: Q update
        A_00 = decay_Q_c * ones           # Q self-decay (scalar, no kernel)
        A_01 = w_qk * K_b                 # K → Q via kernel
        A_02 = alpha * K_b                # A → Q via kernel (attention application!)

        # Row 1: K update (symmetric with Q)
        A_10 = w_qk * K_b                 # Q → K via kernel (same weight as K→Q)
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
