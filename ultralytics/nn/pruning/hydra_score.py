"""HYDRA importance-score initialization.

Milestone 1 implements the scaled initialization used by HYDRA rather than
falling back to magnitude-only pruning.

For each convolutional layer:

    score = sqrt(6 / fan_in) * weight / max(abs(weight))

where fan_in is based on the actual Conv2d weight layout:
    weight.shape[1] * kernel_height * kernel_width

Using the stored weight shape is important for grouped/depthwise convolutions.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .hydra_convert import iter_hydra_layers


@torch.no_grad()
def initialize_layer_scaled_score(module: nn.Module, eps: float = 1e-12) -> None:
    """Initialize one HYDRA layer with scaled weight-proportional scores."""
    if not hasattr(module, "weight") or not hasattr(module, "popup_scores"):
        raise ValueError(f"Module {module} does not have weight and popup_scores attributes")

    weight = module.weight
    fan_in = weight.shape[1:].numel()
    scale = math.sqrt(6.0 / max(fan_in, 1))
    max_abs = weight.abs().max()
    denom = max_abs if max_abs > eps else torch.tensor(1.0, device=weight.device, dtype=weight.dtype)

    score = scale * (weight / denom)
    module.popup_scores.data.copy_(score)


@torch.no_grad()
def initialize_hydra_scores(model: nn.Module, method: str = "scaled") -> None:
    """Initialize all HYDRA score tensors."""
    if method != "scaled":
        raise ValueError(f"Unsupported HYDRA score initialization: {method}")

    for _, layer in iter_hydra_layers(model):
        initialize_layer_scaled_score(layer)
