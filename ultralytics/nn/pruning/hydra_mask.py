"""HYDRA mask control, sparsity reporting, and training-mode transitions.

Milestone 2 responsibilities:
    - Set a common keep ratio across selected HYDRA layers.
    - Freeze dynamic score rankings into persistent masks.
    - Configure parameter trainability for score search vs fine-tuning.
    - Verify and report actual mask sparsity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .hydra_convert import iter_hydra_layers
from .hydra_layer import HydraConv2d, HydraMode


@dataclass
class HydraSparsityReport:
    """Summary of masked parameters across HYDRA layers."""

    total_mask_elements: int
    kept_mask_elements: int

    @property
    def sparsity(self) -> float:
        """Return the fraction of mask elements set to zero (prune_ratio = 1 - keep_ratio)."""
        if self.total_mask_elements == 0:
            return 0.0
        return 1.0 - self.kept_mask_elements / self.total_mask_elements


def set_hydra_keep_ratio(model: nn.Module, keep_ratio: float) -> None:
    """Apply a common layer-wise keep ratio to every HYDRA layer."""
    for _, layer in iter_hydra_layers(model):
        layer.set_keep_ratio(keep_ratio)


@torch.no_grad()
def freeze_hydra_masks(model: nn.Module) -> HydraSparsityReport:
    """Snapshot all dynamic score masks into fixed_mask buffers and return achieved sparsity."""
    total_elements = 0
    kept_elements = 0
    for _, layer in iter_hydra_layers(model):
        layer.freeze_current_mask()
        mask = layer.fixed_mask
        total_elements += mask.numel()
        kept_elements += int((mask > 0.5).sum().item())
    return HydraSparsityReport(
        total_mask_elements=total_elements,
        kept_mask_elements=kept_elements,
    )


def configure_score_search(model: nn.Module) -> None:
    """Freeze dense model parameters and train only HYDRA importance scores.

    BatchNorm Policy:
        Freeze BatchNorm affine parameters (weight and bias requires_grad=False)
        and keep BatchNorm modules in eval mode during score optimization.
        This ensures that the only learned variables during HYDRA score search
        are the importance scores (popup_scores).
    """
    for p in model.parameters():
        p.requires_grad = False

    for _, layer in iter_hydra_layers(model):
        layer.set_mode(HydraMode.SCORE)
        layer.popup_scores.requires_grad = True
        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False

    # Freeze BatchNorm affine parameters and keep modules in eval mode
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False


def configure_masked_finetune(model: nn.Module) -> None:
    """Freeze masks/scores and enable fine-tuning of retained dense parameters."""
    for name, p in model.named_parameters():
        if "popup_scores" in name:
            p.requires_grad = False
        else:
            p.requires_grad = True

    for _, layer in iter_hydra_layers(model):
        layer.set_mode(HydraMode.FINETUNE)
        layer.popup_scores.requires_grad = False
        layer.fixed_mask.requires_grad = False


@torch.no_grad()
def enforce_pruned_weights_zero(model: nn.Module) -> None:
    """Physically zero pruned Conv2d weights after optimizer steps.

    The masked forward already prevents pruned connections from contributing,
    but explicitly clamping pruned weights to zero makes checkpoint inspection
    and materialization unambiguous.
    """
    for _, layer in iter_hydra_layers(model):
        layer.weight.data.mul_(layer.fixed_mask)
