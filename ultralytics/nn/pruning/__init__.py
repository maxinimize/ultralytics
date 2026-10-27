"""HYDRA pruning extensions for Ultralytics detection models.

Milestones 1–3 expose a minimal public API while keeping pruning logic
independent of Ultralytics model-building internals.
"""

from .hydra_convert import convert_to_hydra, iter_hydra_layers
from .hydra_layer import HydraConv2d, HydraMode, HydraPruningMode
from .hydra_mask import (
    HydraSparsityReport,
    calculate_model_sparsity,
    freeze_hydra_masks,
    get_hydra_pruning_mode,
    is_kernel_structured_mask,
    set_hydra_keep_ratio,
    set_hydra_pruning_mode,
)
from .hydra_score import initialize_hydra_scores

__all__ = (
    "HydraConv2d",
    "HydraMode",
    "HydraPruningMode",
    "HydraSparsityReport",
    "calculate_model_sparsity",
    "convert_to_hydra",
    "freeze_hydra_masks",
    "get_hydra_pruning_mode",
    "initialize_hydra_scores",
    "is_kernel_structured_mask",
    "iter_hydra_layers",
    "set_hydra_keep_ratio",
    "set_hydra_pruning_mode",
)
