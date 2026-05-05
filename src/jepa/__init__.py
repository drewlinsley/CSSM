"""
CSSM-JEPA: Video Self-Supervised Learning with CSSM.

Joint Embedding Predictive Architecture for video representation learning
using CSSM-SHViT encoder with dual spectral + feature space prediction.

Two variants:
1. CSSMJEPA: Spatial masking (masks regions across all frames)
2. CSSMVJEPA: V-JEPA style causal future prediction (recommended)
"""

from .masking import TubeMasking, VJEPAMasking, VJEPAMaskInfo
from .loss import jepa_dual_loss, smooth_l1_loss, spectral_loss
from .predictor import JEPAPredictor
from .encoder import VideoEncoder
from .model import CSSMJEPA, CSSMVJEPA, create_cssm_jepa, create_cssm_vjepa

__all__ = [
    # Masking
    'TubeMasking',
    'VJEPAMasking',
    'VJEPAMaskInfo',
    # Loss
    'jepa_dual_loss',
    'smooth_l1_loss',
    'spectral_loss',
    # Components
    'JEPAPredictor',
    'VideoEncoder',
    # Models
    'CSSMJEPA',
    'CSSMVJEPA',
    'create_cssm_jepa',
    'create_cssm_vjepa',
]
