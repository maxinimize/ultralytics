"""Feature-guided residual denoising with frozen YOLO P3/P4/P5 targets."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils import LOGGER
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import autocast


class ResidualDenoiser(nn.Module):
    """Small HGD-style U-Net predicting an additive RGB correction at arbitrary spatial sizes."""

    def __init__(self):
        """Build four encoder levels and three skip-connected decoder levels."""
        super().__init__()
        self.encoder = nn.ModuleList()
        for i, (cin, cout) in enumerate(((3, 32), (32, 64), (64, 128), (128, 256))):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(cin, cout, 3, stride=1 if i == 0 else 2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(cout, cout, 3, padding=1),
                    nn.ReLU(),
                )
            )
        self.decoder = nn.ModuleList(
            nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())
            for cin, cout in ((384, 128), (192, 64), (96, 32))
        )
        self.residual = nn.Conv2d(32, 3, 3, padding=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return clamp(images + residual, 0, 1), resizing decoder stages to skip tensors."""
        x, skips = images, []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return (images + self.residual(x)).clamp(0, 1)


class FGDModel(nn.Module):
    """Train one denoiser using equally weighted L1 distances of three frozen detection features.

    Tensor inputs return denoised images; paired batch dictionaries return the FGD objective and detached diagnostics.
    The teacher stays in eval mode even when the denoiser trains. Only its clean-target forward disables autograd.
    """

    def __init__(self, teacher_model: nn.Module):
        """Attach a frozen, three-scale YOLO teacher and an identity-initialized residual denoiser."""
        super().__init__()
        self.teacher_model = teacher_model.eval().requires_grad_(False)
        self.denoiser = ResidualDenoiser()
        self.stride = teacher_model.stride
        self.criterion = None
        self.group_sums, self.group_counts = {}, {}

    def train(self, mode: bool = True):
        """Keep guidance BatchNorm statistics fixed and reset epoch diagnostics on training entry."""
        super().train(mode)
        self.teacher_model.eval()
        if mode:
            self.group_sums, self.group_counts = {}, {}
        return self

    def features(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Read Detect input features from raw forward output without hooks or detaching gradients."""
        raw = self.teacher_model(images)[1]
        if "one2many" in raw:
            raw = raw["one2many"]
        feats = raw["feats"]
        if len(feats) != 3:
            raise ValueError("FGD requires three Detect input features: P3, P4, P5.")
        return feats

    def forward(self, batch: torch.Tensor | dict):
        """Compute batch-mean FGD loss, with a separate mean over each feature scale."""
        if torch.is_tensor(batch):
            return self.denoiser(batch)
        with torch.no_grad():
            clean_feats = [x.detach() for x in self.features(batch["clean_img"])]
        den_feats = self.features(self.denoiser(batch["img"]))
        if any(a.shape != b.shape for a, b in zip(den_feats, clean_feats)):
            raise ValueError("Clean and denoised P3/P4/P5 shapes must match.")
        per_image = torch.stack([(a - b).abs().flatten(1).mean(1) for a, b in zip(den_feats, clean_feats)], 1)
        if not torch.isfinite(per_image).all():
            raise FloatingPointError("Non-finite FGD feature loss.")
        loss = per_image.mean()
        for group in set(batch["fgd_type"]):
            indices = [i for i, name in enumerate(batch["fgd_type"]) if name == group]
            values = per_image[indices].detach().sum(0)
            self.group_sums[group] = self.group_sums.get(group, 0) + values
            self.group_counts[group] = self.group_counts.get(group, 0) + len(indices)
        items = dict(zip(("P3", "P4", "P5"), per_image.detach().mean(0)))
        return loss, {"fgd_loss": loss.detach(), **items}


def load_fgd_denoiser(fgd_weights: str | Path | None, device: torch.device) -> tuple[ResidualDenoiser | None, dict]:
    """Load ResidualDenoiser and metadata from FGD checkpoint."""
    if fgd_weights is None or str(fgd_weights).strip().lower() in {"", "none", "false"}:
        LOGGER.info("No FGD checkpoint specified; running baseline validation without denoiser.")
        return None, {}

    fgd_path = Path(fgd_weights).resolve()
    if not fgd_path.exists():
        raise FileNotFoundError(f"FGD checkpoint not found: {fgd_path}")

    LOGGER.info(f"Loading FGD denoiser from {fgd_path}...")
    ckpt = torch_load(fgd_path, map_location=device)
    denoiser = ResidualDenoiser().to(device)

    meta = {}
    if isinstance(ckpt, dict) and "denoiser" in ckpt:
        denoiser.load_state_dict(ckpt["denoiser"])
        meta = {
            "epoch": ckpt.get("epoch"),
            "guidance_model": ckpt.get("guidance_model"),
            "fgd_config": ckpt.get("fgd_config"),
        }
        LOGGER.info(
            f"FGD checkpoint loaded successfully (epoch {meta.get('epoch', 'unknown')}, "
            f"guidance: {meta.get('guidance_model', 'unknown')})."
        )
    elif isinstance(ckpt, dict):
        denoiser.load_state_dict(ckpt)
        LOGGER.info("Direct state_dict loaded into ResidualDenoiser.")
    else:
        raise ValueError(f"Unrecognized FGD checkpoint structure in {fgd_path}")

    denoiser.eval()
    denoiser.requires_grad_(False)
    return denoiser, meta


def apply_fgd(denoiser: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Apply the frozen Phase 1 denoiser in FP32 without disabling input autograd."""
    with autocast(False, device=images.device.type):
        return denoiser(images.float()).to(images.dtype)


class FGDDetectorWrapper(nn.Module):
    """Compose FGD with an attack detector while keeping FGD permanently frozen and in eval mode."""

    def __init__(self, denoiser: nn.Module, detector: nn.Module):
        """Reference the external denoiser and existing attack detector without copying their weights."""
        super().__init__()
        self.denoiser = denoiser.eval().requires_grad_(False)
        self.detector = detector
        self.training = detector.training

    @property
    def training(self):
        """Observe the detector's live mode even when the trainer switches it outside this wrapper."""
        detector = self._modules.get("detector")
        return detector.training if detector is not None else self._training

    @training.setter
    def training(self, mode):
        self._training = mode

    def train(self, mode: bool = True):
        """Allow existing attackers to switch detector mode while FGD always stays in eval mode."""
        super().train(mode)
        self.denoiser.eval()
        return self

    def forward(self, images: torch.Tensor, *args, **kwargs):
        """Preserve the graph from detection predictions through FGD to the raw attack input."""
        return self.detector(apply_fgd(self.denoiser, images), *args, **kwargs)

    def loss(self, batch, preds=None):
        """Reuse detector loss; existing ART forwards already preprocess when predictions are supplied."""
        if preds is None:
            batch = dict(batch)
            batch["img"] = apply_fgd(self.denoiser, batch["img"])
        return self.detector.loss(batch, preds)
