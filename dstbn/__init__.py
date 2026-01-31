"""Ds-TBN: Dual-stage Tri-Branch Network for open-set fault diagnosis.

This package provides:
- Tri-branch feature extractor (CNN/ResNet, Transformer, GRU)
- Dynamic contrastive learning loss
- Dual-stage open-set inference (Weibull EVT + meta-recognition calibration)
- Reproducible training/evaluation utilities

The repository is designed to be easy to audit and reproduce.
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
