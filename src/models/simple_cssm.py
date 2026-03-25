"""
SimpleCSSM: Clean architecture for plugging different CSSM variants.

Architecture:
    Conv -> act -> norm -> maxpool
    Conv -> act -> norm -> maxpool
    + Position Embeddings (spatiotemporal RoPE by default)
    CSSM block(s)
    Frame selection (last or all)
    Norm -> act -> spatial pool -> norm -> head
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

from .cssm import GatedCSSM, HGRUBilinearCSSM, TransformerCSSM, MultiplicativeTransformerCSSM, GrowingTransformerCSSM, MambaGrowingTransformerCSSM, SpectralTransformerCSSM, AdditiveCSSM, DeltaNetCSSM, MatrixDeltaNetCSSM, GatedDeltaNetCSSM, SpatialAttentionCSSM, Mamba2SeqCSSM, GDNSeqCSSM, ConvSSMCSSM, apply_rope, apply_learned_temporal_encoding, apply_sinusoidal_temporal_encoding


# Registry of CSSM variants
CSSM_REGISTRY = {
    'gated': GatedCSSM,
    'hgru_bi': HGRUBilinearCSSM,          # Primary: 3x3 with E/I kernels
    'transformer': TransformerCSSM,       # Minimal Q/K/A (additive, uses GOOM)
    'mult_transformer': MultiplicativeTransformerCSSM,  # Multiplicative Q/K/A (log-space linear)
    'g_transformer': GrowingTransformerCSSM,  # Growing attention: Q→[K,V] triangular scan
    'mg_transformer': MambaGrowingTransformerCSSM,  # Mamba-style: 3D conv Q + [K,V] scan
    'spectral_transformer': SpectralTransformerCSSM,  # Correct spatial Q gating via spectral conv
    'kqv': TransformerCSSM,               # Alias for transformer (K/Q/V-inspired naming)
    'add_kqv': AdditiveCSSM,              # Additive Q→K→V with shared kernel, triangular 3x3
    'add_kqv_2': AdditiveCSSM,            # 2-state Q→V only (no K), 2x2 scan
    'add_kqv_1': AdditiveCSSM,            # 1-state scalar scan (spread + decay)
    'add_delta': AdditiveCSSM,            # Delta-enhanced AdditiveCSSM (beta gate + retrieval output)
    'deltanet': DeltaNetCSSM,             # Spectral DeltaNet: single-state delta rule
    'deltanet_d2': MatrixDeltaNetCSSM,    # Matrix DeltaNet, d_k=2
    'deltanet_d3': MatrixDeltaNetCSSM,    # Matrix DeltaNet, d_k=3
    'gdn': GatedDeltaNetCSSM,                # Gated DeltaNet, d_k=2 (default)
    'gdn_d2': GatedDeltaNetCSSM,              # Gated DeltaNet, d_k=2
    'gdn_d3': GatedDeltaNetCSSM,              # Gated DeltaNet, d_k=3
    'gdn_int': GatedDeltaNetCSSM,                 # Gated DeltaNet + spectral InT spatial attention
    'gdn_int_elem': GatedDeltaNetCSSM,           # Gated DeltaNet + elementwise spatial attention
    'gdn_int_qk': GatedDeltaNetCSSM,             # Gated DeltaNet + Q·K spatial attention
    'spatial_attn': SpatialAttentionCSSM,      # Spatial-only multi-head self-attention
    'spatiotemporal_attn': SpatialAttentionCSSM,  # Spatial + temporal attention
    'mamba2_seq': Mamba2SeqCSSM,              # Mamba-2 on flattened 1D tokens (no spatial)
    'gdn_seq': GDNSeqCSSM,                   # Gated DeltaNet on flattened 1D tokens (no spatial)
    'conv_ssm': ConvSSMCSSM,                  # ConvSSM: spatial conv + temporal scan (NVlabs)
}


class SimpleCSSM(nn.Module):
    """
    Simple CSSM architecture with clean stem and readout.

    Attributes:
        num_classes: Number of output classes (2 for Pathfinder)
        embed_dim: Embedding dimension after stem
        depth: Number of CSSM blocks
        cssm_type: Which CSSM variant to use
        kernel_size: CSSM spatial kernel size
        frame_readout: 'last' (single frame) or 'all' (spatiotemporal pool)
        norm_type: 'layer' or 'batch'
        pos_embed: Position embedding type:
            - 'spatiotemporal': Combined H, W, T RoPE (VideoRoPE style)
            - 'spatial_only': Only H, W RoPE, no temporal (better length generalization)
            - 'separate': Spatial RoPE + learned temporal
            - 'sinusoidal': Spatial RoPE + sinusoidal temporal (natural length extrapolation)
            - 'temporal': Only T RoPE
            - 'learnable': 2D learnable spatial embeddings
            - 'none': No position encoding
        act_type: Nonlinearity type (softplus, gelu, relu)
        pool_type: Final pooling type (mean, max)
        seq_len: Number of temporal recurrence steps
        max_seq_len: Maximum sequence length for learned temporal embeddings (used with 'separate')
    """
    num_classes: int = 2
    embed_dim: int = 32
    depth: int = 1
    cssm_type: str = 'hgru_bi'
    kernel_size: int = 15
    block_size: int = 1          # Channel mixing block size (1=depthwise, >1=block mixing)
    frame_readout: str = 'last'  # 'last' or 'all'
    norm_type: str = 'layer'     # 'layer', 'batch', or 'instance' (global default, overridden by stem_norm/body_norm)
    stem_norm: str = ''          # Stem norm type: 'layer', 'batch', 'instance', or '' (use norm_type)
    body_norm: str = ''          # Body/readout norm type: 'layer', 'batch', 'instance', or '' (use norm_type)
    pos_embed: str = 'spatiotemporal'  # 'spatiotemporal', 'spatial_only', 'separate', 'sinusoidal', 'temporal', 'learnable', 'none'
    act_type: str = 'softplus'   # 'softplus', 'gelu', 'relu'
    pool_type: str = 'mean'      # 'mean' or 'max'
    seq_len: int = 8             # Temporal sequence length
    max_seq_len: int = 32        # Max sequence length for learned temporal embeddings (used with 'separate')
    position_independent_gates: bool = False  # Compute gates from raw input (before pos encoding) for length generalization
    use_goom: bool = True        # Use GOOM primitives for log-space scan (vs standard log/exp)
    use_log: bool = True          # Use log-space scan; False = direct complex multiply
    # Ablation flags for g_transformer and mg_transformer
    shared_kernel: bool = False  # Use 1 shared kernel for Q/K/V instead of separate
    additive_kv: bool = False    # Additive Q→V + K→V instead of multiplicative Q·K→V
    spectral_clip: bool = False  # Apply spectral magnitude clipping to kernels
    # mg_transformer 3D conv config
    q_temporal: int = 3          # Temporal extent of Q conv in mg_transformer
    q_spatial: int = 5           # Spatial extent of Q conv in mg_transformer
    # Ablation flags for transformer (TransformerCSSM)
    asymmetric_qk: bool = False      # Use separate weights for Q→K and K→Q
    no_feedback: bool = False        # Remove A→Q feedback
    no_spectral_clip: bool = False   # Skip spectral magnitude clipping
    use_layernorm: bool = False      # Add log-space LayerNorm before output
    no_k_state: bool = False         # Drop K entirely → 2×2 Q+A scan
    transformer_readout: str = 'qka' # Which states to read out: 'qka', 'q', 'k', 'a', etc.
    gate_type: str = 'dense'         # Gate type for AdditiveCSSM: 'dense', 'channel', 'scalar'
    n_register_tokens: int = 0       # Number of learnable register tokens prepended to temporal sequence
    learned_init: bool = False       # Learn initial state for CSSM recurrence (vs starting from zero)
    stem_mode: str = 'default'       # 'default' (conv+norm+act+pool per layer) or 'pathtracker' (1x1 conv, no downsample)
    stem_layers: int = 2             # Number of stem layers (each = conv+norm+act+pool = 2x downsample)
    stem_norm_order: str = 'post'    # 'pre' (norm→act→conv) or 'post' (conv→act→norm)
    use_complex32: bool = False      # Phase-split scan: bf16 mag + bf16 phase (halves scan memory)
    use_complex16: bool = False      # FP8 scan: fp8 real + fp8 imag (quarter memory vs complex64)
    use_ssd: bool = False            # Use SSD chunked scan instead of associative scan (legacy, use scan_mode)
    scan_mode: str = 'associative'   # 'associative', 'ssd', or 'quadratic'
    ssd_chunk_size: int = 8          # Chunk size for SSD / chunked quadratic
    readout_norm: str = 'pre'        # 'pre' (norm→act→pool) or 'post' (act→norm→pool)
    # Spatial attention (SpatialAttentionCSSM) config
    num_heads: int = 4               # Number of attention heads for spatial_attn / spatiotemporal_attn
    mlp_ratio: float = 4.0           # MLP expansion ratio
    drop_path_rate: float = 0.0      # Stochastic depth rate
    # GatedDeltaNetCSSM (gdn) config
    short_conv_size: int = 4         # Temporal short conv kernel size (0=disabled)
    output_norm: str = 'rms'         # Output norm before gating ('rms', 'layer', 'none')
    use_input_gates: bool = True     # B_k, B_v spatial input gates
    output_gate_act: str = 'silu'    # Output gate activation ('silu' or 'sigmoid')
    use_spectral_l2_norm: bool = True  # L2-normalize q,k across freq bins (ablatable)
    qkv_conv_size: int = 1             # QKV spatial conv (1=Dense, 3/5=cross-freq one-shot)
    qkv_conv_separable: bool = True    # Depthwise-separable QKV conv (fewer params)
    cross_freq_conv_size: int = 0      # 1D conv across W_freq on output (0=disabled)
    delta_key_dim: int = 2             # Key dimension for GDN (2 or 3)
    no_spectral_clamp: bool = False    # Skip spectral magnitude squash on W_hat
    inter_mlp_ratio: float = 0.0      # Channel-mixing MLP expansion ratio between CSSM blocks (0=disabled)
    attn_kernel_size: int = 7          # Spatial kernel size for spectral InT attention (gdn_int)
    int_attn_dim: int = 16             # Hidden dim for Q·K attention projections (gdn_int_qk)
    # Mamba2SeqCSSM (mamba2_seq) config
    state_dim: int = 16                # SSM state dimension N
    expand_factor: int = 2             # Inner dimension expansion factor
    flatten_mode: str = 'temporal_spatial'  # 'temporal_spatial' or 'per_frame'

    def _get_act(self):
        """Get activation function."""
        if self.act_type == 'softplus':
            return nn.softplus
        elif self.act_type == 'gelu':
            return jax.nn.gelu
        elif self.act_type == 'relu':
            return jax.nn.relu
        else:
            return nn.softplus

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True,
                 return_spatial: bool = False,
                 return_features: bool = False,
                 injected_features: Optional[jnp.ndarray] = None,
                 injected_qkv_spatial: Optional[jnp.ndarray] = None,
                 injected_post_cssm: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input (B, T, H, W, C) or (B, H, W, C)
            return_spatial: If True, return per-pixel logits at all timesteps
                (B, T, H', W', num_classes) — skips frame selection, pool, norm_post.
            return_features: If True, return post-CSSM features (B, T, H', W', C)
                instead of logits.
            injected_features: If provided, replaces the stem output (pre-CSSM input)
                with this tensor. Stem still runs (for param tree compatibility),
                but its output is discarded. Enables clean gradients through CSSM+readout.
            injected_qkv_spatial: If provided, replaces the spatial Q/K/V inputs
                before gating+FFT inside the CSSM. Shape (B, T, H', W', C, n_states).
                Real-valued spatial domain — gradients are clean with no FFT artifacts.
            injected_post_cssm: If provided, replaces the post-CSSM features
                (after CSSM block + residual). Gradient flows through readout only —
                no FFT in the gradient path.
        Returns:
            Logits (B, num_classes), or features if return_features=True
        """
        act = self._get_act()

        # Resolve norm types (stem_norm/body_norm override norm_type if set)
        _stem_norm = self.stem_norm if self.stem_norm else self.norm_type
        _body_norm = self.body_norm if self.body_norm else self.norm_type

        def _apply_norm(x, norm_type, name):
            """Apply the specified normalization."""
            if norm_type == 'batch':
                return nn.BatchNorm(use_running_average=not training, name=name)(x)
            elif norm_type == 'instance':
                # GroupNorm with num_groups=C is instance norm (one group per channel)
                return nn.GroupNorm(num_groups=x.shape[-1], name=name)(x)
            elif norm_type == 'temporal_layer':
                # LayerNorm that pools over (T, C) for 5D tensors, giving
                # temporal-aware normalization. Falls back to standard LayerNorm
                # for lower-rank tensors (after frame selection / pooling).
                if x.ndim == 5:  # (B, T, H, W, C)
                    return nn.LayerNorm(reduction_axes=(1, 4), feature_axes=-1, name=name)(x)
                else:
                    return nn.LayerNorm(name=name)(x)
            elif norm_type == 'global_layer':
                # LayerNorm over (T, H, W, C) — one mean/sd per sample. Stats: (B,)
                # Falls back to standard LayerNorm for lower-rank tensors.
                if x.ndim == 5:  # (B, T, H, W, C)
                    return nn.LayerNorm(reduction_axes=(1, 2, 3, 4), feature_axes=-1, name=name)(x)
                else:
                    return nn.LayerNorm(name=name)(x)
            elif norm_type == 'global_instance':
                # Reduce over (T, H, W) — one mean/sd per sample per channel. Stats: (B, C)
                # Falls back to standard LayerNorm for lower-rank tensors.
                if x.ndim == 5:  # (B, T, H, W, C)
                    return nn.LayerNorm(reduction_axes=(1, 2, 3), feature_axes=-1, name=name)(x)
                else:
                    return nn.LayerNorm(name=name)(x)
            else:  # 'layer'
                return nn.LayerNorm(name=name)(x)

        # Handle single image input - repeat to create temporal sequence
        if x.ndim == 4:
            x = jnp.repeat(x[:, None, ...], self.seq_len, axis=1)  # (B, T, H, W, C)

        B, T, H, W, C = x.shape

        # === STEM ===
        # Reshape for 2D convs: (B*T, H, W, C)
        x = x.reshape(B * T, H, W, C)

        def _stem_norm_act(x, name):
            """Apply norm and activation in the configured order."""
            if self.stem_norm_order == 'pre':
                x = _apply_norm(x, _stem_norm, name)
                x = act(x)
            else:
                x = act(x)
                x = _apply_norm(x, _stem_norm, name)
            return x

        if self.stem_mode == 'pathtracker':
            # Minimal stem: 1x1 conv projection, no spatial downsampling
            if self.stem_norm_order == 'pre':
                x = _stem_norm_act(x, 'norm1')
                x = nn.Conv(self.embed_dim, kernel_size=(1, 1), name='conv1')(x)
            else:
                x = nn.Conv(self.embed_dim, kernel_size=(1, 1), name='conv1')(x)
                x = _stem_norm_act(x, 'norm1')
        else:
            # Default stem: N × (conv + norm/act + maxpool), each layer = 2x downsample
            for i in range(self.stem_layers):
                if self.stem_norm_order == 'pre':
                    x = _stem_norm_act(x, f'norm{i+1}')
                    x = nn.Conv(self.embed_dim, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name=f'conv{i+1}')(x)
                else:
                    x = nn.Conv(self.embed_dim, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name=f'conv{i+1}')(x)
                    x = _stem_norm_act(x, f'norm{i+1}')
                x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        # Reshape back: (B, T, H', W', embed_dim)
        _, H_new, W_new, _ = x.shape
        x = x.reshape(B, T, H_new, W_new, self.embed_dim)

        # === POSITION EMBEDDINGS ===
        if self.pos_embed == 'learnable':
            # Learnable 2D spatial embeddings (added, not rotated)
            pos = self.param('pos_embed', nn.initializers.normal(0.02),
                           (1, 1, H_new, W_new, self.embed_dim))
            x = x + pos
        elif self.pos_embed == 'separate':
            # Separate spatial RoPE + learned temporal
            # 1. Apply spatial-only RoPE
            x = apply_rope(x, mode='spatial_only')
            # 2. Apply learned temporal embedding (with interpolation support)
            temporal_embed = self.param('temporal_embed', nn.initializers.normal(0.02),
                                       (self.max_seq_len, self.embed_dim))
            x = apply_learned_temporal_encoding(x, temporal_embed)
        elif self.pos_embed == 'sinusoidal':
            # Separate spatial RoPE + sinusoidal temporal (no learned params)
            # 1. Apply spatial-only RoPE
            x = apply_rope(x, mode='spatial_only')
            # 2. Apply sinusoidal temporal encoding (naturally extrapolates to longer sequences)
            x = apply_sinusoidal_temporal_encoding(x)
        # Note: 'spatiotemporal', 'spatial_only', 'temporal' are applied INSIDE CSSM via rope_mode

        # === REGISTER TOKENS ===
        # Prepend learnable register tokens along temporal axis (scratch memory for the scan)
        if self.n_register_tokens > 0:
            reg = self.param('register_tokens', nn.initializers.normal(0.02),
                             (1, self.n_register_tokens, H_new, W_new, self.embed_dim))
            reg = jnp.broadcast_to(reg, (B, self.n_register_tokens, H_new, W_new, self.embed_dim))
            x = jnp.concatenate([reg, x], axis=1)  # (B, T+n_reg, H', W', embed_dim)

        # === FEATURE INJECTION (pre-CSSM) ===
        if injected_features is not None:
            x = injected_features
        self.sow('intermediates', 'pre_cssm', x)

        # === CSSM BLOCK(S) ===
        # Determine rope_mode to pass to CSSM
        # For 'separate' mode, spatial RoPE is already applied above, so pass 'none' to CSSM
        if self.pos_embed in ['spatiotemporal', 'spatial_only', 'temporal', 'mrope']:
            rope_mode = self.pos_embed
        else:
            rope_mode = 'none'

        CSSMClass = CSSM_REGISTRY.get(self.cssm_type, HGRUBilinearCSSM)
        for i in range(self.depth):
            # Build kwargs for CSSM
            cssm_kwargs = dict(
                channels=self.embed_dim,
                kernel_size=self.kernel_size,
                block_size=self.block_size,
                rope_mode=rope_mode,  # Spatiotemporal RoPE inside CSSM
                name=f'cssm_{i}'
            )
            # position_independent_gates applies to transformer variants
            if self.cssm_type in ['transformer', 'mult_transformer', 'g_transformer', 'mg_transformer'] and self.position_independent_gates:
                cssm_kwargs['position_independent_gates'] = True
            # use_goom applies to multiplicative/growing transformer variants
            if self.cssm_type in ['mult_transformer', 'g_transformer', 'mg_transformer']:
                cssm_kwargs['use_goom'] = self.use_goom
            # Ablation flags apply to growing transformer variants
            if self.cssm_type in ['g_transformer', 'mg_transformer']:
                cssm_kwargs['shared_kernel'] = self.shared_kernel
                cssm_kwargs['additive_kv'] = self.additive_kv
                cssm_kwargs['spectral_clip'] = self.spectral_clip
            if self.cssm_type == 'mg_transformer':
                cssm_kwargs['q_temporal'] = self.q_temporal
                cssm_kwargs['q_spatial'] = self.q_spatial
            # Flags for AdditiveCSSM variants
            if self.cssm_type in ['add_kqv', 'add_kqv_2', 'add_kqv_1', 'add_delta']:
                cssm_kwargs['use_goom'] = self.use_goom
                cssm_kwargs['gate_type'] = self.gate_type
                cssm_kwargs['learned_init'] = self.learned_init
                cssm_kwargs['use_complex32'] = self.use_complex32
                cssm_kwargs['use_complex16'] = self.use_complex16
                cssm_kwargs['use_ssd'] = self.use_ssd
                cssm_kwargs['scan_mode'] = self.scan_mode
                cssm_kwargs['ssd_chunk_size'] = self.ssd_chunk_size
                if self.cssm_type == 'add_kqv_2':
                    cssm_kwargs['no_k_state'] = True
                elif self.cssm_type == 'add_kqv_1':
                    cssm_kwargs['single_state'] = True
                elif self.cssm_type == 'add_delta':
                    cssm_kwargs['use_delta'] = True
                    cssm_kwargs['l2_norm_qk'] = True
            # Flags for DeltaNetCSSM
            if self.cssm_type == 'deltanet':
                cssm_kwargs['use_goom'] = self.use_goom
                cssm_kwargs['gate_type'] = self.gate_type
                cssm_kwargs['learned_init'] = self.learned_init
                cssm_kwargs['use_complex32'] = self.use_complex32
                cssm_kwargs['use_complex16'] = self.use_complex16
                cssm_kwargs['scan_mode'] = self.scan_mode
                cssm_kwargs['ssd_chunk_size'] = self.ssd_chunk_size
            # Flags for MatrixDeltaNetCSSM
            if self.cssm_type in ['deltanet_d2', 'deltanet_d3']:
                cssm_kwargs['use_goom'] = self.use_goom
                cssm_kwargs['gate_type'] = self.gate_type
                cssm_kwargs['learned_init'] = self.learned_init
                cssm_kwargs['use_complex32'] = self.use_complex32
                cssm_kwargs['use_complex16'] = self.use_complex16
                cssm_kwargs['scan_mode'] = self.scan_mode
                cssm_kwargs['ssd_chunk_size'] = self.ssd_chunk_size
                if self.cssm_type == 'deltanet_d2':
                    cssm_kwargs['delta_key_dim'] = 2
                else:
                    cssm_kwargs['delta_key_dim'] = 3
            # Flags for GatedDeltaNetCSSM
            if self.cssm_type in ['gdn', 'gdn_d2', 'gdn_d3', 'gdn_int', 'gdn_int_elem', 'gdn_int_qk']:
                cssm_kwargs['use_goom'] = self.use_goom
                cssm_kwargs['use_log'] = self.use_log
                cssm_kwargs['gate_type'] = self.gate_type
                cssm_kwargs['learned_init'] = self.learned_init
                cssm_kwargs['use_complex32'] = self.use_complex32
                cssm_kwargs['use_complex16'] = self.use_complex16
                cssm_kwargs['scan_mode'] = self.scan_mode
                cssm_kwargs['ssd_chunk_size'] = self.ssd_chunk_size
                cssm_kwargs['short_conv_size'] = self.short_conv_size
                cssm_kwargs['output_norm'] = self.output_norm
                cssm_kwargs['use_input_gates'] = self.use_input_gates
                cssm_kwargs['output_gate_act'] = self.output_gate_act
                cssm_kwargs['use_spectral_l2_norm'] = self.use_spectral_l2_norm
                cssm_kwargs['qkv_conv_size'] = self.qkv_conv_size
                cssm_kwargs['qkv_conv_separable'] = self.qkv_conv_separable
                cssm_kwargs['cross_freq_conv_size'] = self.cross_freq_conv_size
                cssm_kwargs['no_spectral_clamp'] = self.no_spectral_clamp
                if self.cssm_type == 'gdn_d3':
                    cssm_kwargs['delta_key_dim'] = 3
                elif self.cssm_type == 'gdn_d2':
                    cssm_kwargs['delta_key_dim'] = 2
                elif self.cssm_type == 'gdn_int':
                    cssm_kwargs['delta_key_dim'] = self.delta_key_dim
                    cssm_kwargs['int_attention_mode'] = 'spectral'
                    cssm_kwargs['attn_kernel_size'] = self.attn_kernel_size
                elif self.cssm_type == 'gdn_int_elem':
                    cssm_kwargs['delta_key_dim'] = self.delta_key_dim
                    cssm_kwargs['int_attention_mode'] = 'elementwise'
                elif self.cssm_type == 'gdn_int_qk':
                    cssm_kwargs['delta_key_dim'] = self.delta_key_dim
                    cssm_kwargs['int_attention_mode'] = 'qk'
                    cssm_kwargs['int_attn_dim'] = self.int_attn_dim
                else:
                    # 'gdn' — use explicit CLI value (default 2)
                    cssm_kwargs['delta_key_dim'] = self.delta_key_dim
            # Ablation flags for TransformerCSSM
            if self.cssm_type in ['transformer', 'kqv']:
                cssm_kwargs['asymmetric_qk'] = self.asymmetric_qk
                cssm_kwargs['no_feedback'] = self.no_feedback
                cssm_kwargs['no_spectral_clip'] = self.no_spectral_clip
                cssm_kwargs['use_layernorm'] = self.use_layernorm
                cssm_kwargs['no_k_state'] = self.no_k_state
                cssm_kwargs['readout_state'] = self.transformer_readout
            # SpatialAttentionCSSM config
            if self.cssm_type in ['spatial_attn', 'spatiotemporal_attn']:
                cssm_kwargs['num_heads'] = self.num_heads
                cssm_kwargs['mlp_ratio'] = self.mlp_ratio
                cssm_kwargs['drop_path_rate'] = self.drop_path_rate
                cssm_kwargs['use_temporal_attn'] = (self.cssm_type == 'spatiotemporal_attn')
            # Mamba2SeqCSSM config
            if self.cssm_type == 'mamba2_seq':
                cssm_kwargs['state_dim'] = self.state_dim
                cssm_kwargs['expand_factor'] = self.expand_factor
                cssm_kwargs['ssd_chunk_size'] = self.ssd_chunk_size
                cssm_kwargs['short_conv_size'] = self.short_conv_size
                cssm_kwargs['flatten_mode'] = self.flatten_mode
            # GDNSeqCSSM config
            if self.cssm_type == 'gdn_seq':
                cssm_kwargs['delta_key_dim'] = self.delta_key_dim
                cssm_kwargs['short_conv_size'] = self.short_conv_size
                cssm_kwargs['flatten_mode'] = self.flatten_mode
            # ConvSSMCSSM config (kernel_size already passed by default)
            if self.cssm_type == 'conv_ssm':
                pass  # kernel_size already in cssm_kwargs
            cssm = CSSMClass(**cssm_kwargs)
            x = x + cssm(x, injected_qkv_spatial=injected_qkv_spatial)  # Residual connection

            # Inter-block norm + act + channel-mixing MLP (skip after last block)
            # Attention blocks have internal pre-norm; extra inter-block norm is non-standard
            if self.depth > 1 and i < self.depth - 1:
                if self.cssm_type not in ('spatial_attn', 'spatiotemporal_attn'):
                    x = _apply_norm(x, _body_norm, f'body_norm_{i}')
                    x = act(x)
                # Channel-mixing MLP between blocks
                if self.inter_mlp_ratio > 0:
                    hidden = int(self.embed_dim * self.inter_mlp_ratio)
                    mlp_out = nn.Dense(hidden, name=f'mlp_{i}_up')(x)
                    mlp_out = act(mlp_out)
                    mlp_out = nn.Dense(self.embed_dim, name=f'mlp_{i}_down')(mlp_out)
                    x = x + mlp_out  # Residual

        # === POST-CSSM CAPTURE ===
        self.sow('intermediates', 'post_cssm', x)
        if injected_post_cssm is not None:
            x = injected_post_cssm
        if return_features:
            return x  # (B, T, H', W', C)

        # === READOUT ===
        # Strip register tokens before readout
        if self.n_register_tokens > 0:
            x = x[:, self.n_register_tokens:]  # (B, T, H', W', embed_dim)

        if return_spatial:
            # Per-pixel logits at all timesteps: norm_pre → act → head (as 1×1 conv)
            # Skip frame selection, pool, AND norm_post. norm_post is designed for
            # the post-pool vector (different statistics than per-pixel features).
            if self.readout_norm == 'pre':
                x = _apply_norm(x, _body_norm, 'norm_pre')
                x = act(x)
            else:
                x = act(x)
                x = _apply_norm(x, _body_norm, 'norm_pre')
            # Dense on last dim works as 1×1 conv: (B, T, H', W', C) → (B, T, H', W', num_classes)
            x = nn.Dense(self.num_classes, name='head')(x)
            return x

        # Frame selection
        if self.frame_readout == 'last':
            x = x[:, -1]  # (B, H', W', embed_dim)
        else:  # 'all' - keep temporal dimension for pooling
            pass  # x stays (B, T, H', W', embed_dim)

        # Pre-norm: norm → act | Post-norm: act → norm
        if self.readout_norm == 'pre':
            x = _apply_norm(x, _body_norm, 'norm_pre')
            x = act(x)
        else:  # post
            x = act(x)
            x = _apply_norm(x, _body_norm, 'norm_pre')

        # Pool over space (and time if frame_readout='all')
        if self.frame_readout == 'last':
            # (B, H', W', embed_dim) -> (B, embed_dim)
            if self.pool_type == 'max':
                x = x.max(axis=(1, 2))
            else:
                x = x.mean(axis=(1, 2))
        else:
            # (B, T, H', W', embed_dim) -> (B, embed_dim)
            if self.pool_type == 'max':
                x = x.max(axis=(1, 2, 3))
            else:
                x = x.mean(axis=(1, 2, 3))

        # Final norm
        x = _apply_norm(x, _body_norm, 'norm_post')

        # Head: 1x1 -> num_classes (as Dense since we pooled)
        x = nn.Dense(self.num_classes, name='head')(x)

        return x
