from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.attacks.attack_utils import BatchContainer
from ultralytics.attacks.attacker import Attacker
from ultralytics.utils.torch_utils import unwrap_model


def _slice_prediction_batch(preds: Any, index: int, batch_size: int) -> Any:
    """Recursively keep one sample from YOLO training predictions.

    Ultralytics detection heads can return tensors, lists/tuples of feature maps,
    or nested structures depending on the model/version.  Only tensors whose
    leading dimension equals the candidate batch size are sliced.
    """
    if torch.is_tensor(preds):
        if preds.ndim > 0 and preds.shape[0] == batch_size:
            return preds[index : index + 1]
        return preds

    if isinstance(preds, list):
        return [_slice_prediction_batch(v, index, batch_size) for v in preds]

    if isinstance(preds, tuple):
        return tuple(_slice_prediction_batch(v, index, batch_size) for v in preds)

    if isinstance(preds, dict):
        return {k: _slice_prediction_batch(v, index, batch_size) for k, v in preds.items()}

    return preds


class SpatialWorstOfK(Attacker):
    """Worst-of-k random spatial attack for Ultralytics object detection.

    For every input image, sample ``k`` random rotation/translation transforms,
    evaluate the detector loss of each transformed image with its *transformed
    ground-truth boxes*, and retain the candidate with the largest loss.

    This is a spatial attack, not an Lp pixel-noise attack.  The implementation
    therefore updates both image and detection targets.

    Parameters
    ----------
    k:
        Number of randomly sampled spatial candidates per image. Default: 10.
    degrees:
        Maximum absolute rotation in degrees. Angles are sampled uniformly from
        [-degrees, +degrees].
    translate:
        Maximum absolute translation as a fraction of image width/height.
        Example: 0.10 allows +/-10% horizontal and vertical shifts.
    fill:
        Constant fill value for pixels exposed by rotation/translation. Inputs
        are assumed to be in [0, 1]. 114/255 matches the common YOLO padding
        value reasonably well.
    min_box_size:
        Drop a transformed target if its clipped width or height is smaller than
        this many pixels.
    min_area_ratio:
        Drop a transformed target if its clipped area is less than this fraction
        of its original area.
    include_identity:
        If True, candidate 0 is the unmodified image/boxes and the remaining
        k-1 candidates are random spatial transforms. Default: False, matching
        the usual random Worst-of-k interpretation.
    seed:
        Optional private RNG seed for reproducible transformation sampling.
    """

    attack_mode = "detector"
    requires_targets = True
    target_format = "yolo_detection"
    batch_aware = True
    transforms_targets = True

    def __init__(
        self,
        model: nn.Module,
        config=None,
        target=None,
        epsilon: float = 0.0,
        *,
        k: int = 10,
        degrees: float = 30.0,
        translate: float = 0.10,
        img_size: int = 640,
        fill: float = 114.0 / 255.0,
        min_box_size: float = 2.0,
        min_area_ratio: float = 0.10,
        include_identity: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        # epsilon is kept only for compatibility with the common Attacker API.
        super().__init__(model, config, epsilon)

        if int(k) < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if float(degrees) < 0:
            raise ValueError(f"degrees must be >= 0, got {degrees}")
        if not 0.0 <= float(translate) <= 1.0:
            raise ValueError(f"translate must be in [0, 1], got {translate}")
        if not 0.0 <= float(fill) <= 1.0:
            raise ValueError(f"fill must be in [0, 1], got {fill}")
        if float(min_box_size) < 0:
            raise ValueError(f"min_box_size must be >= 0, got {min_box_size}")
        if not 0.0 <= float(min_area_ratio) <= 1.0:
            raise ValueError(f"min_area_ratio must be in [0, 1], got {min_area_ratio}")

        self.device = next(model.parameters()).device
        self.k = int(k)
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.img_size = int(img_size)
        self.fill = float(fill)
        self.min_box_size = float(min_box_size)
        self.min_area_ratio = float(min_area_ratio)
        self.include_identity = bool(include_identity)

        self._loss_fn = unwrap_model(self.model).loss

        # Use an isolated CPU RNG so reproducibility does not depend on CUDA RNG
        # state or the training dataloader's generator.
        self._generator = torch.Generator(device="cpu")
        if seed is None:
            self._generator.seed()
        else:
            self._generator.manual_seed(int(seed))

        # Candidate scoring does not need gradients. The model is temporarily
        # switched to train mode inside _score_candidates only because YOLO's
        # detection loss expects training-head outputs.
        self.model.eval()

        # Useful for logging/debugging after each call.
        self.last_scores: Optional[torch.Tensor] = None  # [B, k]
        self.last_best_indices: Optional[torch.Tensor] = None  # [B], values 0..k-1
        self.last_params: Optional[torch.Tensor] = None  # [B, k, 3] -> angle, tx, ty

    # ------------------------------------------------------------------
    # Public attack entry points
    # ------------------------------------------------------------------
    def forward_batch(self, batch: dict) -> torch.Tensor:
        """Attack one Ultralytics detection batch and update its boxes in-place.

        The method returns the adversarial image tensor for compatibility with
        existing training code that does:

            adv = run_attack_on_batch(attacker, batch)
            batch["img"] = adv

        ``batch_idx``, ``cls`` and ``bboxes`` are already synchronized here.
        """
        images = batch.get("img")
        batch_idx = batch.get("batch_idx")
        cls = batch.get("cls")
        bboxes = batch.get("bboxes")

        if images is None:
            raise ValueError("Detection batch is missing batch['img']")
        if batch_idx is None or cls is None or bboxes is None:
            raise ValueError("SpatialWorstOfK requires batch_idx, cls and bboxes")

        targets = self._pack_targets(batch_idx, cls, bboxes, device=images.device)
        adv_images, adv_targets = self._attack(images.float(), targets)

        # Preserve the original batch tensor dtypes/shapes expected by Ultralytics.
        batch["img"] = adv_images.to(dtype=images.dtype)
        self._write_targets_back(batch, adv_targets, batch_idx, cls, bboxes)
        return batch["img"]

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Image-only compatibility method.

        Direct use is discouraged for detection training because a spatial
        transform changes the ground-truth boxes. ``run_attack_on_batch`` should
        call ``forward_batch`` instead so labels remain synchronized.
        """
        adv_images, adv_targets = self._attack(x.float(), targets)
        self.last_transformed_targets = adv_targets
        return adv_images.to(dtype=x.dtype)

    # ------------------------------------------------------------------
    # Core Worst-of-k logic
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _attack(self, x: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected x in NCHW format, got shape {tuple(x.shape)}")

        batch_size, _, height, width = x.shape
        if batch_size == 0:
            return x, targets

        targets = self._validate_targets(targets, batch_size).to(device=x.device, dtype=torch.float32)

        # [B, k, 3] containing angle(deg), tx(px), ty(px)
        params = self._sample_params(batch_size, height, width, device=x.device, dtype=x.dtype)
        matrices = self._build_forward_matrices(params, height, width, dtype=x.dtype)  # [B,k,3,3]

        # Expand original images to [B*k,C,H,W], then warp all candidates in one GPU op.
        expanded = x[:, None].expand(-1, self.k, -1, -1, -1).reshape(
            batch_size * self.k, x.shape[1], height, width
        )
        flat_matrices = matrices.reshape(batch_size * self.k, 3, 3)
        candidates = self._warp_images(expanded, flat_matrices)

        candidate_targets = self._transform_targets(
            targets,
            matrices,
            batch_size=batch_size,
            height=height,
            width=width,
        )

        scores = self._score_candidates(candidates, candidate_targets)
        scores_2d = scores.view(batch_size, self.k)
        best_j = scores_2d.argmax(dim=1)
        best_flat = torch.arange(batch_size, device=x.device) * self.k + best_j

        adv_images = candidates.index_select(0, best_flat)
        adv_targets = self._select_targets(candidate_targets, best_flat, batch_size)

        self.last_scores = scores_2d.detach().cpu()
        self.last_best_indices = best_j.detach().cpu()
        self.last_params = params.detach().cpu()

        return adv_images, adv_targets

    def _sample_params(
        self,
        batch_size: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rand = torch.rand((batch_size, self.k, 3), generator=self._generator, dtype=torch.float32)

        angles = (rand[..., 0] * 2.0 - 1.0) * self.degrees
        tx = (rand[..., 1] * 2.0 - 1.0) * (self.translate * width)
        ty = (rand[..., 2] * 2.0 - 1.0) * (self.translate * height)

        params = torch.stack((angles, tx, ty), dim=-1)

        if self.include_identity:
            params[:, 0, :] = 0.0

        return params.to(device=device, dtype=dtype)

    @staticmethod
    def _build_forward_matrices(
        params: torch.Tensor,
        height: int,
        width: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return pixel-space forward transforms input -> output, shape [B,k,3,3]."""
        angles = params[..., 0] * (math.pi / 180.0)
        tx = params[..., 1]
        ty = params[..., 2]

        c = torch.cos(angles)
        s = torch.sin(angles)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        shape = params.shape[:-1] + (3, 3)
        M = torch.zeros(shape, device=params.device, dtype=dtype)
        M[..., 0, 0] = c
        M[..., 0, 1] = -s
        M[..., 1, 0] = s
        M[..., 1, 1] = c
        M[..., 0, 2] = cx + tx - c * cx + s * cy
        M[..., 1, 2] = cy + ty - s * cx - c * cy
        M[..., 2, 2] = 1.0
        return M

    def _warp_images(self, images: torch.Tensor, forward_matrices: torch.Tensor) -> torch.Tensor:
        """Warp NCHW images using pixel-space forward matrices and constant fill."""
        n, c, h, w = images.shape
        device, dtype = images.device, images.dtype

        # affine_grid uses normalized output -> input coordinates. Convert the
        # inverse pixel transform to normalized coordinates for align_corners=False.
        P = torch.tensor(
            [[w / 2.0, 0.0, (w - 1.0) / 2.0],
             [0.0, h / 2.0, (h - 1.0) / 2.0],
             [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        P_inv = torch.linalg.inv(P)
        M_inv = torch.linalg.inv(forward_matrices)
        theta_h = P_inv.unsqueeze(0) @ M_inv @ P.unsqueeze(0)
        theta = theta_h[:, :2, :]

        grid = F.affine_grid(theta, size=(n, c, h, w), align_corners=False)

        # grid_sample only has zero padding. Subtract/add fill to obtain an
        # arbitrary constant border value while preserving bilinear interpolation.
        centered = images - self.fill
        warped = F.grid_sample(
            centered,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ) + self.fill
        return warped.clamp_(0.0, 1.0)

    def _transform_targets(
        self,
        targets: torch.Tensor,
        matrices: torch.Tensor,
        *,
        batch_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Transform normalized YOLO xywh targets for all B*k candidates."""
        if targets.numel() == 0:
            return targets.new_zeros((0, 6))

        img_idx = targets[:, 0].long()
        cls = targets[:, 1]
        xywh = targets[:, 2:6]

        # Original normalized xywh -> pixel xyxy.
        xc = xywh[:, 0] * width
        yc = xywh[:, 1] * height
        bw = xywh[:, 2] * width
        bh = xywh[:, 3] * height

        x1 = xc - bw / 2.0
        y1 = yc - bh / 2.0
        x2 = xc + bw / 2.0
        y2 = yc + bh / 2.0

        ones = torch.ones_like(x1)
        corners = torch.stack(
            (
                torch.stack((x1, y1, ones), dim=-1),
                torch.stack((x2, y1, ones), dim=-1),
                torch.stack((x2, y2, ones), dim=-1),
                torch.stack((x1, y2, ones), dim=-1),
            ),
            dim=1,
        )  # [M,4,3]

        k = int(matrices.shape[1])

        # Pick each object's image-specific k transforms: [M,k,3,3].
        object_matrices = matrices.index_select(0, img_idx)
        transformed = torch.einsum("mkij,mpj->mkpi", object_matrices, corners)
        xy = transformed[..., :2]  # [M,k,4,2]

        new_x1 = xy[..., 0].amin(dim=2).clamp(0.0, float(width))
        new_y1 = xy[..., 1].amin(dim=2).clamp(0.0, float(height))
        new_x2 = xy[..., 0].amax(dim=2).clamp(0.0, float(width))
        new_y2 = xy[..., 1].amax(dim=2).clamp(0.0, float(height))

        new_w = (new_x2 - new_x1).clamp(min=0.0)
        new_h = (new_y2 - new_y1).clamp(min=0.0)
        new_area = new_w * new_h
        old_area = (bw * bh).clamp(min=1e-6).unsqueeze(1)

        valid = (
            (new_w >= self.min_box_size)
            & (new_h >= self.min_box_size)
            & ((new_area / old_area) >= self.min_area_ratio)
        )

        new_xc = (new_x1 + new_x2) / 2.0 / width
        new_yc = (new_y1 + new_y2) / 2.0 / height
        new_wn = new_w / width
        new_hn = new_h / height

        # Candidate batch index: original image i, candidate j -> i*k + j.
        candidate_j = torch.arange(k, device=targets.device).view(1, k)
        candidate_batch_idx = img_idx.view(-1, 1) * k + candidate_j
        classes = cls.view(-1, 1).expand(-1, k)

        rows = torch.stack(
            (
                candidate_batch_idx.to(targets.dtype),
                classes,
                new_xc,
                new_yc,
                new_wn,
                new_hn,
            ),
            dim=-1,
        )  # [M,k,6]

        return rows[valid]

    def _score_candidates(self, candidates: torch.Tensor, candidate_targets: torch.Tensor) -> torch.Tensor:
        """Compute one scalar YOLO detection loss for every candidate image."""
        total = candidates.shape[0]
        scores = candidates.new_empty((total,), dtype=torch.float32)

        original_training = self.model.training
        try:
            # Training mode is needed to obtain detection-head feature maps used
            # by Ultralytics' training loss. Keep BN/Dropout deterministic.
            self.model.train()
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)):
                    module.eval()

            preds = self.model(candidates, augment=False)

            for idx in range(total):
                mask = candidate_targets[:, 0].long() == idx
                t = candidate_targets[mask]

                # Re-index targets to zero because this is now a one-image loss batch.
                if t.numel() == 0:
                    single_batch_idx = torch.zeros((0,), device=candidates.device, dtype=torch.long)
                    single_cls = torch.zeros((0, 1), device=candidates.device, dtype=torch.float32)
                    single_boxes = torch.zeros((0, 4), device=candidates.device, dtype=torch.float32)
                else:
                    single_batch_idx = torch.zeros((t.shape[0],), device=candidates.device, dtype=torch.long)
                    single_cls = t[:, 1:2]
                    single_boxes = t[:, 2:6]

                batch = BatchContainer(
                    img=candidates[idx : idx + 1],
                    batch_idx=single_batch_idx,
                    cls=single_cls,
                    bboxes=single_boxes,
                )
                pred_i = _slice_prediction_batch(preds, idx, total)
                loss, _ = self._loss_fn(batch, pred_i)
                scores[idx] = loss.detach().float().sum()

        finally:
            self.model.train(original_training)

        return scores

    @staticmethod
    def _select_targets(
        candidate_targets: torch.Tensor,
        best_flat: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Keep targets belonging to each image's selected candidate and re-index 0..B-1."""
        selected = []
        candidate_idx = candidate_targets[:, 0].long() if candidate_targets.numel() else None

        for image_i in range(batch_size):
            flat_i = best_flat[image_i]
            if candidate_idx is None:
                continue
            rows = candidate_targets[candidate_idx == flat_i].clone()
            if rows.numel() == 0:
                continue
            rows[:, 0] = float(image_i)
            selected.append(rows)

        if not selected:
            return candidate_targets.new_zeros((0, 6))
        return torch.cat(selected, dim=0)

    # ------------------------------------------------------------------
    # Batch target conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_targets(batch_idx, cls, bboxes, *, device: torch.device) -> torch.Tensor:
        bi = batch_idx
        c = cls
        boxes = bboxes

        if bi.dim() > 1:
            bi = bi.reshape(-1)
        if c.dim() == 1:
            c = c.unsqueeze(1)
        elif c.dim() > 2:
            c = c.reshape(-1, 1)
        if boxes.dim() > 2:
            boxes = boxes.reshape(-1, 4)

        if not (bi.shape[0] == c.shape[0] == boxes.shape[0]):
            raise ValueError(
                "batch_idx, cls and bboxes must contain the same number of targets: "
                f"{bi.shape[0]}, {c.shape[0]}, {boxes.shape[0]}"
            )

        if bi.numel() == 0:
            return torch.zeros((0, 6), device=device, dtype=torch.float32)

        return torch.cat(
            (
                bi.to(device=device, dtype=torch.float32).unsqueeze(1),
                c.to(device=device, dtype=torch.float32),
                boxes.to(device=device, dtype=torch.float32),
            ),
            dim=1,
        )

    @staticmethod
    def _validate_targets(targets: torch.Tensor, batch_size: int) -> torch.Tensor:
        if targets is None:
            raise ValueError("targets must not be None for SpatialWorstOfK")
        if targets.numel() == 0:
            return targets.reshape(0, 6)
        if targets.ndim != 2 or targets.shape[1] < 6:
            raise ValueError(f"targets must have shape (M,6), got {tuple(targets.shape)}")

        out = targets[:, :6]
        idx = out[:, 0]
        if torch.any(idx < 0) or torch.any(idx >= batch_size):
            raise ValueError("targets contain image indices outside the current batch")
        return out

    @staticmethod
    def _write_targets_back(batch: dict, targets: torch.Tensor, old_batch_idx, old_cls, old_bboxes) -> None:
        device = old_bboxes.device

        new_bi = targets[:, 0].to(device=device, dtype=old_batch_idx.dtype)
        new_cls = targets[:, 1:2].to(device=device, dtype=old_cls.dtype)
        new_boxes = targets[:, 2:6].to(device=device, dtype=old_bboxes.dtype)

        # Preserve common Ultralytics shapes, e.g. batch_idx=[M], cls=[M,1].
        if old_batch_idx.dim() > 1:
            new_bi = new_bi.reshape(-1, *([1] * (old_batch_idx.dim() - 1)))
        if old_cls.dim() == 1:
            new_cls = new_cls.reshape(-1)
        elif old_cls.dim() > 2:
            new_cls = new_cls.reshape(-1, *old_cls.shape[1:])
        if old_bboxes.dim() > 2:
            new_boxes = new_boxes.reshape(-1, *old_bboxes.shape[1:])

        batch["batch_idx"] = new_bi
        batch["cls"] = new_cls
        batch["bboxes"] = new_boxes
