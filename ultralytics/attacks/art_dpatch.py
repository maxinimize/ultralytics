from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ultralytics.attacks.attacker import Attacker

try:
    from art.attacks.evasion import DPatch
except Exception as exc:  # pragma: no cover - depends on the runtime ART install
    DPatch = None
    _DPATCH_IMPORT_ERROR = exc
else:
    _DPATCH_IMPORT_ERROR = None

# Reuse the same YOLO -> ART object-detector wrapper that is already used by ARTPGD.
# This keeps DPatch on the same detector-style path as PGD/BIM/MIM instead of using
# the single-label classifier proxy used by classifier-style attacks.
try:
    from ultralytics.attacks.art_pgd import ARTDetector, YoloV5ForART
except Exception as exc:  # pragma: no cover - import path depends on package layout
    ARTDetector = None
    YoloV5ForART = None
    _ART_PGD_IMPORT_ERROR = exc
else:
    _ART_PGD_IMPORT_ERROR = None


class ARTDPatch(Attacker):
    """
    ART DPatch adapter with the same external interface as ARTPGD.

    External interface used by attack_bridge:
        forward(x, targets) -> patched_images

    Difference from PGD:
        - PGD.generate(...) directly returns adversarial images.
        - DPatch.generate(...) returns a learned patch.
        - DPatch.apply_patch(...) then applies that patch to the images.

    Args:
        model: YOLO detector used for the attack.
        epsilon: Kept only for compatibility with the generic Attacker base class.
        lr: DPatch learning rate. ART names this `learning_rate`.
        epoch: Number of DPatch optimization steps. ART names this `max_iter`.
        img_size: Input image size used by the ART object detector wrapper.
        patch_shape: ART DPatch shape in HWC order: (height, width, channels).
        batch_size: Internal ART attack batch size.
        random_location: Whether to paste the patch at a random valid location.
        target_label: Optional target label for targeted DPatch. Leave None for untargeted.
        per_image: If True, learn/apply one separate patch per image. This is much slower.
        reset_patch_each_batch: If True, rebuild the DPatch object for each forward call so
            patch optimization starts fresh for each batch, which is closer to PGD behavior.
    """

    attack_mode = "detector"
    requires_targets = True
    target_format = "detector_targets"

    def __init__(
        self,
        model,
        config=None,
        target=None,
        epsilon: float = 0.0,
        lr: float = 5.0,
        epoch: int = 20,
        img_size: int = 640,
        patch_shape: Tuple[int, int, int] = (40, 40, 3),
        batch_size: int = 16,
        random_location: bool = False,
        target_label: Optional[int | Sequence[int] | np.ndarray] = None,
        per_image: bool = False,
        reset_patch_each_batch: bool = True,
        verbose: bool = False,
    ):
        super().__init__(model, config, epsilon)

        if DPatch is None:
            raise ImportError(f"Could not import ART DPatch: {_DPATCH_IMPORT_ERROR}")
        if ARTDetector is None or YoloV5ForART is None:
            raise ImportError(f"Could not import ARTPGD wrapper utilities: {_ART_PGD_IMPORT_ERROR}")

        self.device = next(model.parameters()).device
        self.img_size = int(img_size)
        self.learning_rate = float(lr)
        self.max_iter = int(epoch)
        self.patch_shape = tuple(int(v) for v in patch_shape)
        self.batch_size = int(batch_size)
        self.random_location = bool(random_location)
        self.target_label = target_label if target_label is not None else target
        self.per_image = bool(per_image)
        self.reset_patch_each_batch = bool(reset_patch_each_batch)
        self.verbose = bool(verbose)

        # Keep the attack model gradient-enabled. DPatch optimizes the patch through
        # detector loss gradients, so the wrapper must be able to backpropagate.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        wrapped = YoloV5ForART(self.model, self.img_size)
        device_type = "gpu" if self.device.type != "cpu" else "cpu"

        self.estimator = ARTDetector(
            model=wrapped,
            input_shape=(3, self.img_size, self.img_size),
            clip_values=(0.0, 1.0),
            channels_first=True,
            device_type=device_type,
            attack_losses=("loss",),
        )

        if hasattr(self.estimator, "set_batchnorm"):
            self.estimator.set_batchnorm(False)
        if hasattr(self.estimator, "set_dropout"):
            self.estimator.set_dropout(False)

        self.attack = self._build_attack()

    def _build_attack(self):
        return DPatch(
            estimator=self.estimator,
            patch_shape=self.patch_shape,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (N, C, H, W), float in [0, 1].
            targets: Tensor of shape (M, 6): [img_idx, cls, x, y, w, h], normalized xywh.

        Returns:
            Tensor of patched images with the same shape/dtype/device as x.
        """
        n, _, h, w = x.shape
        y_art = self._to_art_labels(targets, h, w, n)
        x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)

        if self.per_image:
            patched = np.empty_like(x_np)
            for i in range(n):
                attack = self._build_attack() if self.reset_patch_each_batch else self.attack
                patch = self._generate_patch(attack, x_np[i : i + 1], [y_art[i]])
                patched[i : i + 1] = self._apply_patch(attack, x_np[i : i + 1], patch)
        else:
            if self.reset_patch_each_batch:
                self.attack = self._build_attack()
            patch = self._generate_patch(self.attack, x_np, y_art)
            patched = self._apply_patch(self.attack, x_np, patch)

        patched = np.clip(patched, 0.0, 1.0).astype(np.float32, copy=False)
        return torch.from_numpy(patched).to(self.device).type_as(x)

    def _generate_patch(self, attack, x_np: np.ndarray, y_art: List[Dict[str, np.ndarray]]) -> np.ndarray:
        kwargs = {}
        if self.target_label is not None:
            kwargs["target_label"] = self.target_label

        patch = attack.generate(x=x_np, y=y_art, **kwargs)

        # Some ART patch attacks return (patch, mask). DPatch returns patch only,
        # but this guard keeps the adapter tolerant to ART version differences.
        if isinstance(patch, tuple):
            patch = patch[0]
        return patch

    def _apply_patch(self, attack, x_np: np.ndarray, patch: np.ndarray) -> np.ndarray:
        try:
            return attack.apply_patch(
                x=x_np,
                patch_external=patch,
                random_location=self.random_location,
            )
        except TypeError:
            # Older ART versions may not expose random_location.
            return attack.apply_patch(x=x_np, patch_external=patch)

    @staticmethod
    def _to_art_labels(targets: torch.Tensor, H: int, W: int, batch_size: int) -> List[Dict[str, np.ndarray]]:
        """Convert YOLO normalized targets to ART object-detection label dictionaries."""
        out: List[Dict[str, np.ndarray]] = []

        if targets is None or targets.numel() == 0:
            for _ in range(batch_size):
                out.append(
                    {
                        "boxes": np.zeros((0, 4), dtype=np.float32),
                        "labels": np.zeros((0,), dtype=np.int64),
                        "scores": np.zeros((0,), dtype=np.float32),
                    }
                )
            return out

        t = targets.detach().cpu()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        elif t.dim() > 2:
            t = t.squeeze()
            if t.dim() == 1:
                t = t.unsqueeze(0)

        if t.size(1) < 6:
            raise ValueError(f"targets must have 6 columns, got {t.size(1)}")

        for i in range(batch_size):
            ti = t[t[:, 0] == i]
            if ti.size(0) == 0:
                out.append(
                    {
                        "boxes": np.zeros((0, 4), dtype=np.float32),
                        "labels": np.zeros((0,), dtype=np.int64),
                        "scores": np.zeros((0,), dtype=np.float32),
                    }
                )
                continue

            if ti.dim() == 1:
                ti = ti.unsqueeze(0)

            xywh = ti[:, 2:6].clone()
            xywh[:, 0] *= W
            xywh[:, 1] *= H
            xywh[:, 2] *= W
            xywh[:, 3] *= H

            xyxy = torch.zeros_like(xywh)
            xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
            xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
            xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
            xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, W)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, H)

            labels = ti[:, 1].to(torch.int64)
            scores = torch.ones((ti.size(0),), dtype=torch.float32)

            out.append(
                {
                    "boxes": xyxy.numpy().astype(np.float32),
                    "labels": labels.numpy().astype(np.int64),
                    "scores": scores.numpy().astype(np.float32),
                }
            )

        return out
