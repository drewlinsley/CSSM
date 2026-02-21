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

from .cssm import GatedCSSM, HGRUBilinearCSSM, TransformerCSSM, MultiplicativeTransformerCSSM, GrowingTransformerCSSM, MambaGrowingTransformerCSSM, SpectralTransformerCSSM, AdditiveCSSM, apply_rope, apply_learned_temporal_encoding, apply_sinusoidal_temporal_encoding


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
    use_goom: bool = True        # For mult_transformer: use complex log to handle negative values
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
    use_ssd: bool = False            # Use SSD chunked scan instead of associative scan
    ssd_chunk_size: int = 8          # Chunk size for SSD
    readout_norm: str = 'pre'        # 'pre' (norm→act→pool) or 'post' (act→norm→pool)

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
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input (B, T, H, W, C) or (B, H, W, C)
        Returns:
            Logits (B, num_classes)
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

        # === CSSM BLOCK(S) ===
        # Determine rope_mode to pass to CSSM
        # For 'separate' mode, spatial RoPE is already applied above, so pass 'none' to CSSM
        if self.pos_embed in ['spatiotemporal', 'spatial_only', 'temporal']:
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
            if self.cssm_type in ['add_kqv', 'add_kqv_2', 'add_kqv_1']:
                cssm_kwargs['use_goom'] = self.use_goom
                cssm_kwargs['gate_type'] = self.gate_type
                cssm_kwargs['learned_init'] = self.learned_init
                cssm_kwargs['use_complex32'] = self.use_complex32
                cssm_kwargs['use_ssd'] = self.use_ssd
                cssm_kwargs['ssd_chunk_size'] = self.ssd_chunk_size
                if self.cssm_type == 'add_kqv_2':
                    cssm_kwargs['no_k_state'] = True
                elif self.cssm_type == 'add_kqv_1':
                    cssm_kwargs['single_state'] = True
            # Ablation flags for TransformerCSSM
            if self.cssm_type in ['transformer', 'kqv']:
                cssm_kwargs['asymmetric_qk'] = self.asymmetric_qk
                cssm_kwargs['no_feedback'] = self.no_feedback
                cssm_kwargs['no_spectral_clip'] = self.no_spectral_clip
                cssm_kwargs['use_layernorm'] = self.use_layernorm
                cssm_kwargs['no_k_state'] = self.no_k_state
                cssm_kwargs['readout_state'] = self.transformer_readout
            cssm = CSSMClass(**cssm_kwargs)
            x = x + cssm(x)  # Residual connection

            # Inter-block norm + act (skip after last block — readout has its own)
            if self.depth > 1 and i < self.depth - 1:
                x = _apply_norm(x, _body_norm, f'body_norm_{i}')
                x = act(x)

        # === READOUT ===
        # Strip register tokens before readout
        if self.n_register_tokens > 0:
            x = x[:, self.n_register_tokens:]  # (B, T, H', W', embed_dim)

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
