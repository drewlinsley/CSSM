"""
JAX/Flax implementation of CoTracker3.

Provides both:
- CoTrackerThree: Faithful port with transformer attention
- CoTrackerThreeCSSM: CSSM variant replacing transformer

Usage:
    from src.cotracker.jax import CoTrackerThree, CoTrackerThreeCSSM, create_cotracker3

    # Transformer version
    model = CoTrackerThree()

    # CSSM version
    model = CoTrackerThreeCSSM(cssm_type='opponent')

    # Or use factory
    model = create_cotracker3(use_cssm=True, cssm_type='hgru_bi')
"""

from .cotracker3 import (
    CoTrackerThree,
    CoTrackerThreeCSSM,
    create_cotracker3,
    get_sinusoidal_encoding,
)
from .updateformer import EfficientUpdateFormer, CSSMUpdateFormer
from .encoder import BasicEncoder, ResidualBlock
from .correlation import EfficientCorrBlock, bilinear_sampler, bilinear_sample_simple
from .blocks import Attention, Mlp, AttnBlock, CrossAttnBlock

__all__ = [
    # Main models
    'CoTrackerThree',
    'CoTrackerThreeCSSM',
    'create_cotracker3',
    # Update blocks
    'EfficientUpdateFormer',
    'CSSMUpdateFormer',
    # Encoder
    'BasicEncoder',
    'ResidualBlock',
    # Correlation
    'EfficientCorrBlock',
    'bilinear_sampler',
    'bilinear_sample_simple',
    # Building blocks
    'Attention',
    'Mlp',
    'AttnBlock',
    'CrossAttnBlock',
    # Utils
    'get_sinusoidal_encoding',
]
