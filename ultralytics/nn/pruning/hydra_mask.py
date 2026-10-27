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
from .hydra_layer import HydraMode


@dataclass
class HydraSparsityReport:
    """Summary of masked parameters across HYDRA layers."""

    total_mask_elements: int
    kept_mask_elements: int
    pruning_mode: str = "unstructured"
    total_eligible_layers: int = 0
    total_eligible_kernels: int = 0
    kept_kernels: int = 0
    pruned_kernels: int = 0
    total_dense_params: int = 0
    effective_nonzero_params: int = 0

    @property
    def sparsity(self) -> float:
        """Return the fraction of mask elements set to zero (prune_ratio = 1 - keep_ratio)."""
        if self.total_mask_elements == 0:
            return 0.0
        return 1.0 - self.kept_mask_elements / self.total_mask_elements

    @property
    def element_sparsity(self) -> float:
        """Return weight-element level sparsity (alias for sparsity)."""
        return self.sparsity

    @property
    def kernel_sparsity(self) -> float:
        """Return 2D kernel level sparsity (fraction of kernels set to zero)."""
        if self.total_eligible_kernels == 0:
            return 0.0
        return 1.0 - self.kept_kernels / self.total_eligible_kernels

    @property
    def physical_compression_ratio(self) -> float:
        """Return physical model compression ratio (1.0 for mask-based Plan A)."""
        return 1.0

    def summary(self) -> str:
        """Return formatted summary of sparsity statistics."""
        lines = [
            f"Pruning Mode:              {self.pruning_mode}",
            f"Eligible Conv2d Layers:    {self.total_eligible_layers}",
            f"Total Eligible Kernels:    {self.total_eligible_kernels}",
            f"Retained Kernels:          {self.kept_kernels}",
            f"Pruned Kernels:            {self.pruned_kernels}",
            f"Actual Kernel Sparsity:    {self.kernel_sparsity:.4f} ({self.kernel_sparsity * 100:.2f}%)",
            f"Total Mask Elements:       {self.total_mask_elements}",
            f"Retained Mask Elements:    {self.kept_mask_elements}",
            f"Actual Element Sparsity:   {self.element_sparsity:.4f} ({self.element_sparsity * 100:.2f}%)",
            f"Total Dense Parameters:    {self.total_dense_params:,}",
            f"Effective Nonzero Params:  {self.effective_nonzero_params:,}",
            f"Physical Compression:      {self.physical_compression_ratio:.2f}x (mask-based, uncompressed tensor shapes)",
        ]
        return "\n".join(lines)


def set_hydra_keep_ratio(model: nn.Module, keep_ratio: float) -> None:
    """Apply a common layer-wise keep ratio to every HYDRA layer."""
    for _, layer in iter_hydra_layers(model):
        layer.set_keep_ratio(keep_ratio)


def set_hydra_pruning_mode(model: nn.Module, pruning_mode: str) -> None:
    """Apply a common pruning mode ('unstructured' or 'kernel') to every HYDRA layer."""
    for _, layer in iter_hydra_layers(model):
        layer.set_pruning_mode(pruning_mode)


def get_hydra_pruning_mode(model: nn.Module) -> str:
    """Return the pruning mode of the HYDRA layers in the model."""
    for _, layer in iter_hydra_layers(model):
        mode = getattr(layer, "pruning_mode", "unstructured")
        return mode.value if hasattr(mode, "value") else str(mode)
    return "unstructured"


def is_kernel_structured_mask(mask: torch.Tensor) -> bool:
    """Check if every 2D kernel in the mask is entirely 0 or entirely 1."""
    if mask.ndim != 4:
        return True
    kh, kw = mask.shape[2], mask.shape[3]
    if kh == 1 and kw == 1:
        return True
    spatial_mean = mask.float().mean(dim=(2, 3))
    return bool(((spatial_mean == 0.0) | (spatial_mean == 1.0)).all().item())


@torch.no_grad()
def calculate_model_sparsity(model: nn.Module) -> HydraSparsityReport:
    """Inspect all HYDRA layers and return comprehensive sparsity statistics."""
    total_elements = 0
    kept_elements = 0
    total_layers = 0
    total_kernels = 0
    kept_kernels = 0
    pruning_mode = "unstructured"

    for _, layer in iter_hydra_layers(model):
        total_layers += 1
        mode = getattr(layer, "pruning_mode", "unstructured")
        pruning_mode = mode.value if hasattr(mode, "value") else str(mode)
        mask = getattr(layer, "fixed_mask", None)
        if mask is None:
            mask = layer.dynamic_mask()
        total_elements += mask.numel()
        kept_elements += int((mask > 0.5).sum().item())

        k_count = mask.shape[0] * mask.shape[1]
        total_kernels += k_count
        k_active = (mask > 0.5).any(dim=-1).any(dim=-1)
        kept_kernels += int(k_active.sum().item())

    total_dense = sum(p.numel() for name, p in model.named_parameters() if "popup_scores" not in name)
    effective_nonzero = total_dense - (total_elements - kept_elements)

    return HydraSparsityReport(
        total_mask_elements=total_elements,
        kept_mask_elements=kept_elements,
        pruning_mode=pruning_mode,
        total_eligible_layers=total_layers,
        total_eligible_kernels=total_kernels,
        kept_kernels=kept_kernels,
        pruned_kernels=total_kernels - kept_kernels,
        total_dense_params=total_dense,
        effective_nonzero_params=effective_nonzero,
    )


@torch.no_grad()
def freeze_hydra_masks(model: nn.Module) -> HydraSparsityReport:
    """Snapshot all dynamic score masks into fixed_mask buffers and return achieved sparsity."""
    total_elements = 0
    kept_elements = 0
    total_layers = 0
    total_kernels = 0
    kept_kernels = 0
    pruning_mode = "unstructured"

    for _, layer in iter_hydra_layers(model):
        total_layers += 1
        mode = getattr(layer, "pruning_mode", "unstructured")
        pruning_mode = mode.value if hasattr(mode, "value") else str(mode)
        layer.freeze_current_mask()
        mask = layer.fixed_mask
        total_elements += mask.numel()
        kept_elements += int((mask > 0.5).sum().item())

        k_count = mask.shape[0] * mask.shape[1]
        total_kernels += k_count
        k_active = (mask > 0.5).any(dim=-1).any(dim=-1)
        kept_kernels += int(k_active.sum().item())

    total_dense = sum(p.numel() for name, p in model.named_parameters() if "popup_scores" not in name)
    effective_nonzero = total_dense - (total_elements - kept_elements)

    return HydraSparsityReport(
        total_mask_elements=total_elements,
        kept_mask_elements=kept_elements,
        pruning_mode=pruning_mode,
        total_eligible_layers=total_layers,
        total_eligible_kernels=total_kernels,
        kept_kernels=kept_kernels,
        pruned_kernels=total_kernels - kept_kernels,
        total_dense_params=total_dense,
        effective_nonzero_params=effective_nonzero,
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
