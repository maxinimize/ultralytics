from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.attacks.attack_utils import (
    BatchContainer,
    capture_module_training_states,
    restore_module_training_states,
)
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
        seed: int | None = None,
        eval_batch_size: int = 4,
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
        self.eval_batch_size = max(1, int(eval_batch_size))

        self._loss_fn = unwrap_model(self.model).loss

        # Use an isolated CPU RNG so reproducibility does not depend on CUDA RNG
        # state or the training dataloader's generator.
        self.seed = None if seed is None else int(seed)
        self._generator = torch.Generator(device="cpu")
        if self.seed is None:
            self._generator.seed()
        else:
            self._generator.manual_seed(self.seed)

        # Candidate scoring does not need gradients. The model is temporarily
        # switched to train mode inside _score_candidates only because YOLO's
        # detection loss expects training-head outputs.
        saved_states = capture_module_training_states(self.model)
        try:
            self.model.eval()
        finally:
            restore_module_training_states(saved_states)

        # Useful for logging/debugging after each call.
        self.last_scores: torch.Tensor | None = None  # [B, k]
        self.last_best_indices: torch.Tensor | None = None  # [B], values 0..k-1
        self.last_params: torch.Tensor | None = None  # [B, k, 3] -> angle, tx, ty
        self.last_selected_matrices: torch.Tensor | None = None  # [B, 3, 3]
        self.last_selected_params: torch.Tensor | None = None  # [B, 3]

    def state_dict(self) -> dict[str, Any]:
        """Return lightweight state dict containing generator state and seed."""
        return {
            "generator_state": self._generator.get_state().clone(),
            "seed": self.seed,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore generator state from state dict."""
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be a dict, got {type(state_dict)}")
        if "generator_state" in state_dict:
            gen_state = state_dict["generator_state"]
            if isinstance(gen_state, torch.Tensor):
                self._generator.set_state(gen_state.cpu())
            elif isinstance(gen_state, (bytes, bytearray)):
                self._generator.set_state(torch.tensor(list(gen_state), dtype=torch.uint8))
        if "seed" in state_dict:
            self.seed = state_dict["seed"]

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset internal generator to a known seed (e.g. at the start of validation)."""
        target_seed = self.seed if seed is None else seed
        if target_seed is None:
            self._generator.seed()
        else:
            self._generator.manual_seed(int(target_seed))

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
        saved_states = capture_module_training_states(self.model)
        try:
            images = batch.get("img")
            batch_idx = batch.get("batch_idx")
            cls = batch.get("cls")
            bboxes = batch.get("bboxes")

            if images is None:
                raise ValueError("Detection batch is missing batch['img']")
            if batch_idx is None or cls is None or bboxes is None:
                raise ValueError("SpatialWorstOfK requires batch_idx, cls and bboxes")

            # Safely preserve clean pre-attack annotations before any transform
            clean_img = images.clone()
            clean_bi = batch_idx.clone()
            clean_c = cls.clone()
            clean_b = bboxes.clone()
            batch["clean_img"] = clean_img
            batch["clean_batch_idx"] = clean_bi
            batch["clean_cls"] = clean_c
            batch["clean_bboxes"] = clean_b
            batch["pre_attack_batch_idx"] = clean_bi
            batch["pre_attack_cls"] = clean_c
            batch["pre_attack_bboxes"] = clean_b

            targets = self._pack_targets(batch_idx, cls, bboxes, device=images.device)
            adv_images, adv_targets = self._attack(images.float(), targets)

            # Preserve the original batch tensor dtypes/shapes expected by Ultralytics.
            batch["img"] = adv_images.to(dtype=images.dtype)
            batch["gt_matrix"] = getattr(self, "last_selected_matrices", None)
            batch["gt_params"] = getattr(self, "last_selected_params", None)
            self._write_targets_back(batch, adv_targets, batch_idx, cls, bboxes)
            batch["adv_bboxes"] = batch["bboxes"].clone()
            return batch["img"]
        finally:
            restore_module_training_states(saved_states)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Image-only compatibility method.

        Direct use is discouraged for detection training because a spatial
        transform changes the ground-truth boxes. ``run_attack_on_batch`` should
        call ``forward_batch`` instead so labels remain synchronized.
        """
        saved_states = capture_module_training_states(self.model)
        try:
            adv_images, adv_targets = self._attack(x.float(), targets)
            self.last_transformed_targets = adv_targets
            return adv_images.to(dtype=x.dtype)
        finally:
            restore_module_training_states(saved_states)

    # ------------------------------------------------------------------
    # Core Worst-of-k logic
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _attack(self, x: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        expanded = x[:, None].expand(-1, self.k, -1, -1, -1).reshape(batch_size * self.k, x.shape[1], height, width)
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

        batch_indices = torch.arange(batch_size, device=x.device)
        selected_matrices = matrices[batch_indices, best_j]
        selected_params = params[batch_indices, best_j]

        self.last_scores = scores_2d.detach().cpu()
        self.last_best_indices = best_j.detach().cpu()
        self.last_params = params.detach().cpu()
        self.last_selected_matrices = selected_matrices
        self.last_selected_params = selected_params

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
            [[w / 2.0, 0.0, (w - 1.0) / 2.0], [0.0, h / 2.0, (h - 1.0) / 2.0], [0.0, 0.0, 1.0]],
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
        warped = (
            F.grid_sample(
                centered,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            + self.fill
        )
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
        """Transform normalized YOLO xywh targets for all B*k candidates using Sutherland-Hodgman clipping."""
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
        m_boxes = targets.shape[0]

        # Pick each object's image-specific k transforms: [M,k,3,3].
        object_matrices = matrices.index_select(0, img_idx)
        transformed = torch.einsum("mkij,mpj->mkpi", object_matrices, corners)
        xy = transformed[..., :2]  # [M,k,4,2]

        total_candidates = m_boxes * k
        xy_flat = xy.reshape(total_candidates, 4, 2)
        device = targets.device
        dtype_fp = torch.float32
        xy_fp = xy_flat.to(dtype=dtype_fp)

        # 1. Shoelace area before clipping for the 4 transformed corners
        x_init = xy_fp[..., 0]
        y_init = xy_fp[..., 1]
        area_before = 0.5 * torch.abs(
            x_init[:, 0] * y_init[:, 1]
            - y_init[:, 0] * x_init[:, 1]
            + x_init[:, 1] * y_init[:, 2]
            - y_init[:, 1] * x_init[:, 2]
            + x_init[:, 2] * y_init[:, 3]
            - y_init[:, 2] * x_init[:, 3]
            + x_init[:, 3] * y_init[:, 0]
            - y_init[:, 3] * x_init[:, 0]
        )

        # 2. Sutherland-Hodgman polygon clipping against canvas [0, width] x [0, height]
        max_v = 12
        cur_v = torch.zeros((total_candidates, max_v, 2), dtype=dtype_fp, device=device)
        cur_v[:, :4, :] = xy_fp
        cur_n = torch.full((total_candidates,), 4, dtype=torch.long, device=device)

        planes = [
            (1.0, 0.0, 0.0),  # x >= 0
            (-1.0, 0.0, float(width)),  # x <= width
            (0.0, 1.0, 0.0),  # y >= 0
            (0.0, -1.0, float(height)),  # y <= height
        ]

        for a, b, c in planes:
            out_v = torch.zeros_like(cur_v)
            out_n = torch.zeros_like(cur_n)
            max_k = int(cur_n.max().item())
            if max_k == 0:
                cur_v, cur_n = out_v, out_n
                break

            for i in range(max_k):
                active = i < cur_n
                v1 = cur_v[:, i, :]
                next_i = torch.remainder(torch.full_like(cur_n, i + 1), torch.clamp(cur_n, min=1))
                v2 = torch.gather(cur_v, 1, next_i.view(total_candidates, 1, 1).expand(-1, 1, 2)).squeeze(1)

                d1 = a * v1[:, 0] + b * v1[:, 1] + c
                d2 = a * v2[:, 0] + b * v2[:, 1] + c

                in1 = (d1 >= -1e-6) & active
                in2 = (d2 >= -1e-6) & active

                crossing = (in1 != in2) & active
                diff = d1 - d2
                safe_diff = torch.where(diff != 0, diff, torch.ones_like(diff))
                t = torch.where(crossing, d1 / safe_diff, torch.zeros_like(d1))
                inter = v1 + t.unsqueeze(-1) * (v2 - v1)

                add_inter = crossing & active
                if add_inter.any():
                    idx = torch.clamp(out_n, max=max_v - 1)
                    mask = add_inter.view(total_candidates, 1, 1)
                    target_slot = idx.view(total_candidates, 1, 1).expand(-1, 1, 2)
                    curr_val = torch.gather(out_v, 1, target_slot)
                    new_val = torch.where(mask, inter.view(total_candidates, 1, 2), curr_val)
                    out_v.scatter_(1, target_slot, new_val)
                    out_n = out_n + add_inter.long()

                add_v2 = in2 & active
                if add_v2.any():
                    idx = torch.clamp(out_n, max=max_v - 1)
                    mask = add_v2.view(total_candidates, 1, 1)
                    target_slot = idx.view(total_candidates, 1, 1).expand(-1, 1, 2)
                    curr_val = torch.gather(out_v, 1, target_slot)
                    new_val = torch.where(mask, v2.view(total_candidates, 1, 2), curr_val)
                    out_v.scatter_(1, target_slot, new_val)
                    out_n = out_n + add_v2.long()

            cur_v = out_v
            cur_n = out_n

        # 3. Deduplicate consecutive duplicate vertices (including wrap-around edge)
        dedup_v = torch.zeros_like(cur_v)
        dedup_n = torch.zeros_like(cur_n)
        max_k = int(cur_n.max().item())
        if max_k > 0:
            for i in range(max_k):
                active = i < cur_n
                v_curr = cur_v[:, i, :]
                next_i = torch.remainder(torch.full_like(cur_n, i + 1), torch.clamp(cur_n, min=1))
                v_next = torch.gather(cur_v, 1, next_i.view(total_candidates, 1, 1).expand(-1, 1, 2)).squeeze(1)

                diff = torch.norm(v_curr - v_next, dim=-1)
                keep = (diff > 1e-5) & active

                if keep.any():
                    idx = torch.clamp(dedup_n, max=max_v - 1)
                    mask = keep.view(total_candidates, 1, 1)
                    target_slot = idx.view(total_candidates, 1, 1).expand(-1, 1, 2)
                    curr_val = torch.gather(dedup_v, 1, target_slot)
                    new_val = torch.where(mask, v_curr.view(total_candidates, 1, 2), curr_val)
                    dedup_v.scatter_(1, target_slot, new_val)
                    dedup_n = dedup_n + keep.long()
            cur_v, cur_n = dedup_v, dedup_n

        # 4. Shoelace area after clipping
        area_after = torch.zeros((total_candidates,), dtype=dtype_fp, device=device)
        max_k = int(cur_n.max().item())
        if max_k >= 3:
            for i in range(max_k):
                active = (i < cur_n) & (cur_n >= 3)
                v1 = cur_v[:, i, :]
                next_i = torch.remainder(torch.full_like(cur_n, i + 1), torch.clamp(cur_n, min=1))
                v2 = torch.gather(cur_v, 1, next_i.view(total_candidates, 1, 1).expand(-1, 1, 2)).squeeze(1)
                cross = v1[:, 0] * v2[:, 1] - v2[:, 0] * v1[:, 1]
                area_after = area_after + torch.where(active, cross, torch.zeros_like(cross))
            area_after = 0.5 * torch.abs(area_after)

        # 5. Tight bounding box over the clipped polygon
        mask = (torch.arange(max_v, device=device).unsqueeze(0) < cur_n.unsqueeze(1)) & (cur_n.unsqueeze(1) >= 3)
        x_pts = cur_v[..., 0]
        y_pts = cur_v[..., 1]
        x_for_min = torch.where(mask, x_pts, torch.full_like(x_pts, float("inf")))
        y_for_min = torch.where(mask, y_pts, torch.full_like(y_pts, float("inf")))
        x_for_max = torch.where(mask, x_pts, torch.full_like(x_pts, float("-inf")))
        y_for_max = torch.where(mask, y_pts, torch.full_like(y_pts, float("-inf")))

        min_x = x_for_min.min(dim=-1).values
        min_y = y_for_min.min(dim=-1).values
        max_x = x_for_max.max(dim=-1).values
        max_y = y_for_max.max(dim=-1).values

        new_x1 = min_x.clamp(0.0, float(width))
        new_y1 = min_y.clamp(0.0, float(height))
        new_x2 = max_x.clamp(0.0, float(width))
        new_y2 = max_y.clamp(0.0, float(height))

        new_w = (new_x2 - new_x1).clamp(min=0.0)
        new_h = (new_y2 - new_y1).clamp(min=0.0)

        safe_ab = torch.where(area_before > 1e-7, area_before, torch.ones_like(area_before))
        area_ratio = area_after / safe_ab

        valid_flat = (
            (area_before > 1e-7)
            & (cur_n >= 3)
            & (area_after > 1e-7)
            & (area_ratio >= self.min_area_ratio)
            & (new_w >= self.min_box_size)
            & (new_h >= self.min_box_size)
        )

        valid = valid_flat.view(m_boxes, k)
        new_x1 = new_x1.view(m_boxes, k)
        new_y1 = new_y1.view(m_boxes, k)
        new_x2 = new_x2.view(m_boxes, k)
        new_y2 = new_y2.view(m_boxes, k)
        new_w = new_w.view(m_boxes, k)
        new_h = new_h.view(m_boxes, k)

        new_xc = ((new_x1 + new_x2) / 2.0 / width).to(dtype=targets.dtype)
        new_yc = ((new_y1 + new_y2) / 2.0 / height).to(dtype=targets.dtype)
        new_wn = (new_w / width).to(dtype=targets.dtype)
        new_hn = (new_h / height).to(dtype=targets.dtype)

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

    def _score_candidates(
        self,
        candidates: torch.Tensor,
        candidate_targets: torch.Tensor,
        eval_batch_size: int | None = None,
    ) -> torch.Tensor:
        """Compute one scalar YOLO detection loss for every candidate image."""
        total = candidates.shape[0]
        scores = candidates.new_empty((total,), dtype=torch.float32)

        saved_states = capture_module_training_states(self.model)
        try:
            # Training mode is needed to obtain detection-head feature maps used
            # by Ultralytics' training loss. Keep BN/Dropout deterministic.
            self.model.train()
            for module in self.model.modules():
                if isinstance(
                    module,
                    (
                        nn.BatchNorm1d,
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                        nn.SyncBatchNorm,
                        nn.Dropout,
                    ),
                ):
                    module.eval()
            if hasattr(self.model, "pipeline") and self.model.pipeline is not None:
                self.model.pipeline.eval()

            chunk_size = eval_batch_size or getattr(self, "eval_batch_size", 4)
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                c_chunk = candidates[start:end]
                preds_chunk = self.model(c_chunk, augment=False)
                chunk_len = end - start

                for offset in range(chunk_len):
                    idx = start + offset
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
                    pred_i = _slice_prediction_batch(preds_chunk, offset, chunk_len)
                    loss, _ = self._loss_fn(batch, pred_i)
                    scores[idx] = loss.detach().float().sum()

        finally:
            restore_module_training_states(saved_states)

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
