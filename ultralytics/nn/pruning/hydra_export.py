"""Materialize a masked HYDRA model back to ordinary `nn.Conv2d` modules.

This is primarily required after Milestone 2/3 training so standard Ultralytics
validation, fusion, export, and deployment do not depend on dynamic HYDRA
classes or score tensors.

Materialization order:
    masked HYDRA model
        -> multiply dense weights by fixed masks
        -> replace HydraConv2d with plain nn.Conv2d
        -> verify output equivalence
        -> only then allow normal Ultralytics Conv-BN fusion/export
"""

from __future__ import annotations

import copy
import torch
from torch import nn

from .hydra_layer import HydraConv2d


def _replace_hydra_conv(module: nn.Module) -> None:
    """Recursively convert HydraConv2d submodules to standard nn.Conv2d."""
    for name, child in list(module.named_children()):
        if isinstance(child, HydraConv2d):
            new_conv = nn.Conv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            new_conv.to(device=child.weight.device, dtype=child.weight.dtype)

            # Materialized weight = weight * fixed_mask
            mat_weight = child.weight.detach() * child.fixed_mask.detach()
            new_conv.weight.data.copy_(mat_weight)
            new_conv.weight.requires_grad = child.weight.requires_grad

            if child.bias is not None and new_conv.bias is not None:
                new_conv.bias.data.copy_(child.bias.detach())
                new_conv.bias.requires_grad = child.bias.requires_grad

            setattr(module, name, new_conv)
        else:
            _replace_hydra_conv(child)


def materialize_hydra_model(model: nn.Module, *, inplace: bool = False) -> nn.Module:
    """Return an ordinary model with permanent pruned weights."""
    if not inplace:
        model = copy.deepcopy(model)

    if isinstance(model, HydraConv2d):
        new_conv = nn.Conv2d(
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            kernel_size=model.kernel_size,
            stride=model.stride,
            padding=model.padding,
            dilation=model.dilation,
            groups=model.groups,
            bias=model.bias is not None,
            padding_mode=model.padding_mode,
        )
        new_conv.to(device=model.weight.device, dtype=model.weight.dtype)
        mat_weight = model.weight.detach() * model.fixed_mask.detach()
        new_conv.weight.data.copy_(mat_weight)
        new_conv.weight.requires_grad = model.weight.requires_grad
        if model.bias is not None and new_conv.bias is not None:
            new_conv.bias.data.copy_(model.bias.detach())
            new_conv.bias.requires_grad = model.bias.requires_grad
        return new_conv

    _replace_hydra_conv(model)
    return model
