"""HYDRA pruning extensions for Ultralytics detection models.

Milestones 1–3 expose a minimal public API while keeping pruning logic
independent of Ultralytics model-building internals.
"""

from .hydra_convert import convert_to_hydra, iter_hydra_layers
from .hydra_layer import HydraConv2d, HydraMode
from .hydra_mask import freeze_hydra_masks, set_hydra_keep_ratio
from .hydra_score import initialize_hydra_scores

__all__ = (
    "HydraConv2d",
    "HydraMode",
    "convert_to_hydra",
    "iter_hydra_layers",
    "initialize_hydra_scores",
    "set_hydra_keep_ratio",
    "freeze_hydra_masks",
)
