# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Spatial Worst-of-k attack interface extension with structured metadata for STN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ultralytics.attacks.spatial_worst_of_k import SpatialWorstOfK
from ultralytics.defenses.spatial_transformer import build_forward_matrix, compute_validity_mask


@dataclass
class SpatialAttackMetadata:
    """Structured container holding attack parameters and tensor states for STN supervision."""

    clean_images: torch.Tensor
    adv_images: torch.Tensor
    selected_params: torch.Tensor  # [B, 3] -> angle_deg, tx_px, ty_px
    selected_matrices: torch.Tensor  # [B, 3, 3] -> forward matrix
    selected_indices: torch.Tensor  # [B] -> 0..k-1
    clean_targets: torch.Tensor  # [M_clean, 6]
    adv_targets: torch.Tensor  # [M_adv, 6]
    validity_masks: Optional[torch.Tensor] = None  # [B, 1, H, W]


class SpatialWorstOfKWithMetadata(SpatialWorstOfK):
    """SpatialWorstOfK attack with structured metadata support for STN training and evaluation.

    Extends SpatialWorstOfK to retain clean targets, attacked targets, selected transformation
    parameters, forward matrices, and validity masks without modifying the existing attacker
    calling contract.
    """

    def __init__(self, model: nn.Module, *args, **kwargs) -> None:
        """Initialize SpatialWorstOfKWithMetadata."""
        super().__init__(model, *args, **kwargs)
        self.last_metadata: Optional[SpatialAttackMetadata] = None
        self.last_selected_matrices: Optional[torch.Tensor] = None
        self.last_selected_params: Optional[torch.Tensor] = None
        self.last_clean_images: Optional[torch.Tensor] = None
        self.last_validity_masks: Optional[torch.Tensor] = None

    def forward_batch(self, batch: dict) -> torch.Tensor:
        """Attack detection batch while safely retaining clean copies of images and ground-truth boxes.

        Args:
            batch (dict): Batch dictionary containing 'img', 'batch_idx', 'cls', 'bboxes'.

        Returns:
            torch.Tensor: Adversarial image tensor matching original dtype.
        """
        images = batch.get("img")
        batch_idx = batch.get("batch_idx")
        cls = batch.get("cls")
        bboxes = batch.get("bboxes")

        if images is None:
            raise ValueError("Detection batch is missing batch['img']")
        if batch_idx is None or cls is None or bboxes is None:
            raise ValueError("SpatialWorstOfK requires batch_idx, cls and bboxes")

        # Safely preserve clean images and targets before any transformation or in-place update
        clean_img = images.clone()
        batch["clean_img"] = clean_img
        batch["clean_batch_idx"] = batch_idx.clone()
        batch["clean_cls"] = cls.clone()
        batch["clean_bboxes"] = bboxes.clone()

        targets = self._pack_targets(batch_idx, cls, bboxes, device=images.device)
        adv_images, adv_targets, metadata = self._attack_with_metadata(images.float(), targets)

        # Store metadata
        self.last_metadata = metadata
        self.last_selected_matrices = metadata.selected_matrices
        self.last_selected_params = metadata.selected_params
        self.last_clean_images = metadata.clean_images
        self.last_validity_masks = metadata.validity_masks

        # Inject into batch for downstream STN trainer/validator
        batch["attack_metadata"] = metadata
        batch["gt_matrix"] = metadata.selected_matrices
        batch["gt_params"] = metadata.selected_params
        if metadata.validity_masks is not None:
            batch["validity_mask"] = metadata.validity_masks

        # Update batch with attacked image and transformed targets
        batch["img"] = adv_images.to(dtype=images.dtype)
        self._write_targets_back(batch, adv_targets, batch_idx, cls, bboxes)
        batch["adv_bboxes"] = batch["bboxes"].clone()
        return batch["img"]

    def forward_with_metadata(self, batch: dict) -> Tuple[torch.Tensor, SpatialAttackMetadata]:
        """Convenience method returning both attacked images and structured metadata."""
        adv_img = self.forward_batch(batch)
        return adv_img, self.last_metadata

    @torch.no_grad()
    def _attack_with_metadata(
        self, x: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, SpatialAttackMetadata]:
        """Execute attack and retain forward transformation matrices on the active device."""
        batch_size, _, height, width = x.shape
        if batch_size == 0:
            empty_meta = SpatialAttackMetadata(
                clean_images=x,
                adv_images=x,
                selected_params=torch.zeros((0, 3), device=x.device, dtype=x.dtype),
                selected_matrices=torch.zeros((0, 3, 3), device=x.device, dtype=x.dtype),
                selected_indices=torch.zeros((0,), device=x.device, dtype=torch.long),
                clean_targets=targets,
                adv_targets=targets,
            )
            return x, targets, empty_meta

        targets = self._validate_targets(targets, batch_size).to(device=x.device, dtype=torch.float32)

        # [B, k, 3] containing angle(deg), tx(px), ty(px)
        params = self._sample_params(batch_size, height, width, device=x.device, dtype=x.dtype)
        matrices = self._build_forward_matrices(params, height, width, dtype=x.dtype)  # [B, k, 3, 3]

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
        best_j = scores_2d.argmax(dim=1)  # [B]
        best_flat = torch.arange(batch_size, device=x.device) * self.k + best_j

        adv_images = candidates.index_select(0, best_flat)
        adv_targets = self._select_targets(candidate_targets, best_flat, batch_size)

        batch_indices = torch.arange(batch_size, device=x.device)
        selected_matrices = matrices[batch_indices, best_j]  # [B, 3, 3]
        selected_params = params[batch_indices, best_j]  # [B, 3]

        # Compute validity mask for the selected attack transformation
        val_mask = compute_validity_mask(selected_matrices, selected_matrices, height=height, width=width)

        self.last_scores = scores_2d.detach().cpu()
        self.last_best_indices = best_j.detach().cpu()
        self.last_params = params.detach().cpu()

        metadata = SpatialAttackMetadata(
            clean_images=x.clone(),
            adv_images=adv_images.clone(),
            selected_params=selected_params,
            selected_matrices=selected_matrices,
            selected_indices=best_j,
            clean_targets=targets.clone(),
            adv_targets=adv_targets.clone(),
            validity_masks=val_mask,
        )

        return adv_images, adv_targets, metadata
