"""Recursive conversion utilities for attaching HYDRA to Ultralytics models.

Milestone 1 goal:
    Replace selected *leaf* `nn.Conv2d` instances with `HydraConv2d` while
    preserving the surrounding Ultralytics module graph.

Important:
    Do NOT replace Ultralytics `Conv` wrappers themselves. They own BatchNorm
    and activation behavior. Recursing to their `.conv` leaf is intentional.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from torch import nn

from .hydra_layer import HydraConv2d

ModuleSelector = Callable[[str, nn.Conv2d], bool]


def default_conv_selector(name: str, module: nn.Conv2d) -> bool:
    """Default Milestone-1 pruning scope.

    Prunes normal backbone/neck/head feature convolutions but excludes DFL
    fixed-projection modules and final detection head output predictors.
    """
    lower_name = name.lower()

    # Exclude DFL (Distribution Focal Loss) integral projection layers
    if "dfl" in lower_name:
        return False

    # Exclude final output projection heads in Detect module (e.g. cv2.0.2, cv3.0.2)
    parts = name.split(".")
    if len(parts) >= 3 and parts[-3] in {"cv2", "cv3", "one2one_cv2", "one2one_cv3"}:
        if parts[-1] == "2":
            return False

    return True


def iter_hydra_layers(model: nn.Module) -> Iterator[tuple[str, HydraConv2d]]:
    """Yield `(qualified_name, module)` for every HYDRA convolution."""
    for name, module in model.named_modules():
        if isinstance(module, HydraConv2d):
            yield name, module


def convert_to_hydra(
    model: nn.Module,
    *,
    keep_ratio: float = 1.0,
    selector: ModuleSelector | None = None,
) -> nn.Module:
    """Recursively replace selected `nn.Conv2d` leaves with `HydraConv2d`."""
    selector = selector or default_conv_selector

    if isinstance(model, nn.Conv2d) and not isinstance(model, HydraConv2d):
        if selector("", model):
            return HydraConv2d.from_conv(model, keep_ratio=keep_ratio)
        return model

    def _convert_rec(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Conv2d) and not isinstance(child, HydraConv2d):
                if selector(full_name, child):
                    hydra_layer = HydraConv2d.from_conv(child, keep_ratio=keep_ratio)
                    setattr(module, child_name, hydra_layer)
            else:
                _convert_rec(child, full_name)

    _convert_rec(model)
    return model
