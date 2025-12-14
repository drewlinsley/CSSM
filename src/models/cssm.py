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

from .math import cssm_scalar_scan_op, cssm_matrix_scan_op
from .goom import to_goom, from_goom


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
    """
    channels: int
    dense_mixing: bool = False
    kernel_size: int = 15

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

        # --- 1. Kernel Generation ---
        if self.dense_mixing:
            # Multi-Head Dense Mixing to save parameters
            # Head dimension of 32 is a good balance
            head_dim = 32
            num_heads = max(1, C // head_dim)

            # Param shape: (num_heads, kernel_size, kernel_size, head_dim)
            k_param = self.param(
                'kernel',
                nn.initializers.normal(0.02),
                (num_heads, self.kernel_size, self.kernel_size, head_dim)
            )

            # Reshape and broadcast heads to match channels
            # (num_heads, K, K, head_dim) -> (C, K, K)
            k_spatial = jnp.repeat(k_param, C // num_heads, axis=0)
            k_spatial = k_spatial.reshape(C, self.kernel_size, self.kernel_size)
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
        K_hat = jnp.fft.rfft2(k_padded, axes=(1, 2))  # (C, H, W_freq)

        # Reshape input for FFT: (B, T, H, W, C) -> (B, T, C, H, W)
        x_perm = x.transpose(0, 1, 4, 2, 3)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C, H, W_freq)

        # --- 3. Cepstral Scan ---
        # Convert to GOOM representation (numerically stable log-space)
        K_log = to_goom(K_hat)[None, None, ...]  # Broadcast over B, T
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


class GatedOpponentCSSM(nn.Module):
    """
    Gated Opponent CSSM with biologically-plausible X<->Y coupled oscillator.

    Implements a 2x2 state transition matrix with:
    - Diagonal: decay terms (alpha for X, delta for Y)
    - Off-diagonal: coupling (mu for inhibition X->Y, gamma for excitation Y->X)

    Input gating is controlled by dense layers on pooled spatial context.
    Uses GOOM primitives for numerically stable log-space computation.

    Attributes:
        channels: Number of input/output channels
        dense_mixing: If True, use multi-head parameter sharing
        kernel_size: Spatial kernel size for excitation/inhibition kernels
    """
    channels: int
    dense_mixing: bool = False
    kernel_size: int = 15

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, H, W, C)

        Returns:
            Output tensor of shape (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape

        # --- 1. Controller (Input Gating) ---
        # Pool spatial dims for gating context: (B, T, H, W, C) -> (B, T, C)
        ctx = x.mean(axis=(2, 3))

        # Decay Gates (Diagonal elements of transition matrix)
        alpha = nn.sigmoid(nn.Dense(C, name='alpha_gate')(ctx))  # X decay
        delta = nn.sigmoid(nn.Dense(C, name='delta_gate')(ctx))  # Y decay

        # Coupling Gates (Off-diagonal elements)
        mu = nn.softplus(nn.Dense(C, name='mu_gate')(ctx))      # Inhibition X->Y
        gamma = nn.softplus(nn.Dense(C, name='gamma_gate')(ctx)) # Excitation Y->X

        # --- 2. Learnable Decay Parameter ---
        # Base decay is learnable per-channel (initialized near 0.9)
        decay_init = nn.initializers.constant(0.9)
        decay_param = self.param('decay', decay_init, (C,))
        decay = jnp.clip(decay_param, 0.1, 0.99)  # Keep in stable range

        # --- 3. Spatial Kernels ---
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

        K_E_spec = jnp.fft.rfft2(pad_kernel(k_exc), axes=(1, 2))  # (C, H, W_f)
        K_I_spec = jnp.fft.rfft2(pad_kernel(k_inh), axes=(1, 2))

        # --- 4. Build Transition Matrix ---
        # Target shape: (B, T, C, H, W_f, 2, 2)

        # Broadcast kernels: (C, H, W_f) -> (1, 1, C, H, W_f)
        K_E = K_E_spec[None, None, ...]
        K_I = K_I_spec[None, None, ...]

        # Broadcast gates: (B, T, C) -> (B, T, C, 1, 1)
        b_alpha = alpha[..., None, None]
        b_delta = delta[..., None, None]
        b_mu = mu[..., None, None]
        b_gamma = gamma[..., None, None]

        # Broadcast decay: (C,) -> (1, 1, C, 1, 1)
        b_decay = decay[None, None, :, None, None]

        # Convert to complex for log-space operations
        decay_complex = b_decay.astype(jnp.complex64)

        # Build 2x2 transition matrix elements
        # A_xx = decay * alpha (self-connection for X)
        # A_xy = -K_I * mu (inhibition from Y to X, negative)
        # A_yx = K_E * gamma (excitation from X to Y)
        # A_yy = decay * delta (self-connection for Y)

        A_xx = decay_complex * b_alpha
        A_xy = -1.0 * K_I * b_mu  # Negative for inhibition
        A_yx = K_E * b_gamma
        A_yy = decay_complex * b_delta

        # Stack into (B, T, C, H, W_f, 2, 2)
        row0 = jnp.stack([A_xx, A_xy], axis=-1)
        row1 = jnp.stack([A_yx, A_yy], axis=-1)
        K_mat = jnp.stack([row0, row1], axis=-2)

        # --- 5. Input Vector ---
        # FFT the input
        x_perm = x.transpose(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        U_hat = jnp.fft.rfft2(x_perm, axes=(3, 4))  # (B, T, C, H, W_f)

        # Input drives X channel (index 0), Y channel gets zero input
        # Use small positive value instead of zero to avoid log(0)
        small_val = jnp.full_like(U_hat, 1e-10)
        U_vec = jnp.stack([U_hat, small_val], axis=-1)  # (..., 2)

        # --- 6. Log-Space Scan (using GOOM) ---
        K_log = to_goom(K_mat)
        U_log = to_goom(U_vec)

        _, State_log = jax.lax.associative_scan(
            cssm_matrix_scan_op, (K_log, U_log), axis=1
        )

        # --- 7. Output ---
        # Extract Y channel (index 1) as output
        Y_log = State_log[..., 1]
        Y_hat = from_goom(Y_log)
        y_out = jnp.fft.irfft2(Y_hat, s=(H, W), axes=(3, 4))

        # Back to (B, T, H, W, C)
        return y_out.transpose(0, 1, 3, 4, 2)
