"""CSSM: Cepstral State Space Models for vision tasks."""

from .models import GatedCSSM, HGRUBilinearCSSM, TransformerCSSM

# Data loaders are optional (may have extra dependencies)
try:
    from .data import get_imagenette_video_loader
    _HAS_DATA = True
except ImportError:
    _HAS_DATA = False

__all__ = [
    "GatedCSSM",
    "HGRUBilinearCSSM",
    "TransformerCSSM",
]

if _HAS_DATA:
    __all__.append("get_imagenette_video_loader")
