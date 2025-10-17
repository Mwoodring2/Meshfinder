"""
ML hooks for ModelFinder (local, pluggable)
"""
from .type_classifier import TypePrediction, predict_part_name, DEFAULT_THRESHOLD, CONTROLLED_PARTS

__all__ = [
    "TypePrediction",
    "predict_part_name",
    "DEFAULT_THRESHOLD",
    "CONTROLLED_PARTS",
]

