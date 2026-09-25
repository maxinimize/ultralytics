"""Core HYDRA convolution layer.

Milestone 1:
    - Implement a leaf-level replacement for `torch.nn.Conv2d`.
    - Add trainable importance scores (`popup_scores`).
    - Implement dynamic top-k masks with a Straight-Through Estimator (STE).
    - Preserve exact dense behavior when `keep_ratio == 1.0`.

Milestone 2:
    - Add fixed-mask fine-tuning mode.
    - Guarantee that pruned weights never contribute to the forward pass.

Do not place Ultralytics BatchNorm or activation logic in this module.
`HydraConv2d` replaces only the internal leaf `nn.Conv2d`.
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn


class HydraMode(str, Enum):
    """Execution mode for a HYDRA convolution."""

    DENSE = "dense"
    SCORE = "score"
    FINETUNE = "finetune"


class _GetSubnetSTE(torch.autograd.Function):
    """Build a binary layer-wise top-k mask while passing gradients to scores.

    Forward:
        Convert absolute importance scores to a binary mask that keeps exactly
        the configured fraction of entries for this layer.

    Backward:
        Apply the Straight-Through Estimator: pass the upstream gradient through
        to the score tensor and do not differentiate the keep ratio.
    """

    @staticmethod
    def forward(ctx, scores: torch.Tensor, keep_ratio: float) -> torch.Tensor:
        if keep_ratio >= 1.0:
            return torch.ones_like(scores)
        if keep_ratio <= 0.0:
            return torch.zeros_like(scores)

        numel = scores.numel()
        k = int(round(keep_ratio * numel))
        if k >= numel:
            return torch.ones_like(scores)
        if k <= 0:
            return torch.zeros_like(scores)

        flat_scores = scores.reshape(-1)
        _, indices = torch.topk(flat_scores, k, largest=True, sorted=False)

        mask = torch.zeros_like(flat_scores)
        mask[indices] = 1.0
        return mask.reshape_as(scores)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class HydraConv2d(nn.Conv2d):
    """`nn.Conv2d` augmented with HYDRA importance scores and masks.

    Design constraints:
        - Preserve all Conv2d geometry: stride, padding, dilation, groups, bias.
        - `popup_scores` has exactly the same shape as `weight`.
        - Dense weights are frozen during score search.
        - Score-search forward should use `self.weight.detach()` as an extra
          safeguard against external trainer code changing `requires_grad`.
        - Fine-tuning uses a fixed registered buffer mask.
    """

    def __init__(self, *args, keep_ratio: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.popup_scores = nn.Parameter(torch.ones_like(self.weight))
        self.keep_ratio = float(keep_ratio)
        self.hydra_mode = HydraMode.DENSE

        # Persistent buffer so masks follow `.to(device)`, DDP, and checkpoints.
        self.register_buffer("fixed_mask", torch.ones_like(self.weight))

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, keep_ratio: float = 1.0) -> HydraConv2d:
        """Create a HYDRA layer with the same geometry and parameters as `conv`."""
        hydra_layer = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            keep_ratio=keep_ratio,
        )
        hydra_layer.to(device=conv.weight.device, dtype=conv.weight.dtype)
        hydra_layer.weight.data.copy_(conv.weight.data)
        hydra_layer.weight.requires_grad = conv.weight.requires_grad
        if conv.bias is not None and hydra_layer.bias is not None:
            hydra_layer.bias.data.copy_(conv.bias.data)
            hydra_layer.bias.requires_grad = conv.bias.requires_grad
        return hydra_layer

    def set_mode(self, mode: HydraMode | str) -> None:
        """Switch layer execution mode without silently changing parameters."""
        self.hydra_mode = HydraMode(mode)

    def set_keep_ratio(self, keep_ratio: float) -> None:
        """Set the fraction of weights retained by dynamic score masks."""
        if not 0.0 <= keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in [0, 1], got {keep_ratio}")
        self.keep_ratio = float(keep_ratio)

    def dynamic_mask(self) -> torch.Tensor:
        """Return the current STE mask derived from `popup_scores`."""
        return _GetSubnetSTE.apply(self.popup_scores.abs(), self.keep_ratio)

    def build_fixed_mask(self) -> torch.Tensor:
        """Create a detached binary mask from current scores."""
        with torch.no_grad():
            return self.dynamic_mask().detach()

    @torch.no_grad()
    def freeze_current_mask(self) -> None:
        """Snapshot current score ranking into `fixed_mask`."""
        self.fixed_mask.copy_(self.build_fixed_mask())

    def effective_weight(self) -> torch.Tensor:
        """Return the weight tensor used by the current HYDRA mode."""
        if self.hydra_mode == HydraMode.DENSE:
            return self.weight

        if self.hydra_mode == HydraMode.SCORE:
            # Dense parameters must remain unchanged during importance search.
            if self.keep_ratio >= 1.0:
                return self.weight.detach()
            return self.weight.detach() * self.dynamic_mask()

        if self.hydra_mode == HydraMode.FINETUNE:
            if self.keep_ratio >= 1.0:
                return self.weight
            return self.weight * self.fixed_mask

        raise RuntimeError(f"Unsupported HYDRA mode: {self.hydra_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution using the effective HYDRA weight."""
        return F.conv2d(
            x,
            self.effective_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
