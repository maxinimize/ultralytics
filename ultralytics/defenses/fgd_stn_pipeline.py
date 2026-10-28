# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Unified FGD + STN Defense Pipeline combining feature-guided residual denoising and spatial rectification."""

from __future__ import annotations

from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ultralytics.defenses.hgd_fgd import ResidualDenoiser, apply_fgd, load_fgd_denoiser
from ultralytics.defenses.spatial_transformer import (
    SpatialTransformer,
    load_stn_checkpoint,
    transform_boxes,
)
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model


# Canonical defense preprocessing modes
VALID_MODES: Tuple[str, ...] = ("none", "fgd", "stn", "fgd_stn", "stn_fgd")

MODE_DESCRIPTIONS: Dict[str, str] = {
    "none": "E1: YOLO only (Baseline)",
    "fgd": "E2: FGD -> YOLO",
    "stn": "E3: STN -> YOLO",
    "fgd_stn": "E4: FGD -> STN -> YOLO",
    "stn_fgd": "E5: STN -> FGD -> YOLO",
}

MODE_ALIASES: Dict[str, str] = {
    "e1": "none",
    "baseline": "none",
    "raw": "none",
    "clean_baseline": "none",
    "yolo": "none",
    "e2": "fgd",
    "fgd_only": "fgd",
    "e3": "stn",
    "stn_only": "stn",
    "e4": "fgd_stn",
    "fgd-stn": "fgd_stn",
    "fgdstn": "fgd_stn",
    "e5": "stn_fgd",
    "stn-fgd": "stn_fgd",
    "stnfgd": "stn_fgd",
}


def canonicalize_mode(mode: str) -> str:
    """Normalize preprocessing mode string to one of the canonical 5 modes.

    Args:
        mode (str): Raw mode name or alias.

    Returns:
        str: Canonical mode name ('none', 'fgd', 'stn', 'fgd_stn', 'stn_fgd').

    Raises:
        ValueError: If mode is not recognized.
    """
    key = str(mode).strip().lower().replace("-", "_")
    if key in VALID_MODES:
        return key
    if key in MODE_ALIASES:
        return MODE_ALIASES[key]
    raise ValueError(f"Unknown preprocessing mode '{mode}'. Supported modes are: {list(VALID_MODES)}")


class FGDSTNPipeline(nn.Module):
    """Unified defense pipeline integrating FGD denoiser and STN spatial transformer.

    Supports five canonical preprocessing configurations:
    - none: YOLO only (raw image input, baseline)
    - fgd: FGD residual denoising followed by YOLO
    - stn: STN geometric rectification followed by YOLO
    - fgd_stn: FGD denoising -> STN rectification -> YOLO
    - stn_fgd: STN rectification -> FGD denoising -> YOLO
    """

    def __init__(
        self,
        denoiser: Optional[ResidualDenoiser] = None,
        stn: Optional[SpatialTransformer] = None,
    ) -> None:
        """Initialize pipeline with optional pretrained FGD and STN modules.

        Args:
            denoiser (ResidualDenoiser, optional): Trained FGD denoiser module.
            stn (SpatialTransformer, optional): Trained or identity STN module.
        """
        super().__init__()
        self.denoiser = denoiser
        self.stn = stn

        # Ensure defenses are permanently frozen in evaluation mode
        if self.denoiser is not None:
            self.denoiser.eval().requires_grad_(False)
        if self.stn is not None:
            self.stn.eval().requires_grad_(False)

    @property
    def has_denoiser(self) -> bool:
        """Return True if an FGD denoiser is available."""
        return self.denoiser is not None

    @property
    def has_stn(self) -> bool:
        """Return True if an STN module is available."""
        return self.stn is not None

    def forward(
        self,
        images: torch.Tensor,
        mode: str = "none",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Preprocess batch of images according to the selected mode.

        Args:
            images (torch.Tensor): Input batch tensor of shape [B, 3, H, W] in [0, 1].
            mode (str): One of ('none', 'fgd', 'stn', 'fgd_stn', 'stn_fgd').

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Preprocessed images ready for YOLO detector.
                - STN forward transformation matrices of shape [B, 3, 3] if STN was used,
                  or None if no spatial transformation was applied.
        """
        c_mode = canonicalize_mode(mode)

        if c_mode == "none":
            return images, None

        if c_mode == "fgd":
            if self.denoiser is None:
                raise RuntimeError("FGD denoiser module is required for mode 'fgd' but was not loaded.")
            denoised = apply_fgd(self.denoiser, images)
            return denoised, None

        if c_mode == "stn":
            if self.stn is None:
                raise RuntimeError("STN module is required for mode 'stn' but was not loaded.")
            rect_img, pred_m = self.stn(images)
            return rect_img, pred_m

        if c_mode == "fgd_stn":
            if self.denoiser is None:
                raise RuntimeError("FGD denoiser module is required for mode 'fgd_stn' but was not loaded.")
            if self.stn is None:
                raise RuntimeError("STN module is required for mode 'fgd_stn' but was not loaded.")
            denoised = apply_fgd(self.denoiser, images)
            rect_img, pred_m = self.stn(denoised)
            return rect_img, pred_m

        if c_mode == "stn_fgd":
            if self.stn is None:
                raise RuntimeError("STN module is required for mode 'stn_fgd' but was not loaded.")
            if self.denoiser is None:
                raise RuntimeError("FGD denoiser module is required for mode 'stn_fgd' but was not loaded.")
            rect_img, pred_m = self.stn(images)
            denoised = apply_fgd(self.denoiser, rect_img)
            return denoised, pred_m

        raise ValueError(f"Unhandled mode: {c_mode}")

    @staticmethod
    def unwarp_predictions(
        preds: Sequence[torch.Tensor],
        pred_matrices: Optional[torch.Tensor],
        height: int,
        width: int,
        clip: bool = True,
    ) -> List[torch.Tensor]:
        """Map predicted bounding boxes back to the original image canvas if STN was applied.

        Args:
            preds (Sequence[torch.Tensor]): List of detection predictions per image,
                each of shape [N, >=6] containing [x1, y1, x2, y2, conf, cls, ...].
            pred_matrices (torch.Tensor, optional): Forward transformation matrices of shape [B, 3, 3]
                predicted by STN, mapping rectified coordinates to original coordinates.
            height (int): Canvas height in pixels.
            width (int): Canvas width in pixels.
            clip (bool): Whether to clamp mapped coordinates to image bounds [0, width] x [0, height].

        Returns:
            List[torch.Tensor]: Prediction tensors with boxes mapped back to original canvas coordinates.
        """
        if pred_matrices is None:
            return [p.clone() for p in preds]

        mapped_preds: List[torch.Tensor] = []
        for si, pred_si in enumerate(preds):
            if pred_si.shape[0] == 0:
                mapped_preds.append(pred_si.clone())
                continue

            m_si = pred_matrices[si]
            mapped_boxes = transform_boxes(pred_si[:, :4], m_si, height=height, width=width, clip=clip)

            # Filter degenerate bounding boxes (non-positive width or height)
            bw = mapped_boxes[:, 2] - mapped_boxes[:, 0]
            bh = mapped_boxes[:, 3] - mapped_boxes[:, 1]
            valid = (bw > 0.0) & (bh > 0.0)

            new_pred = torch.cat((mapped_boxes, pred_si[:, 4:]), dim=-1)[valid]
            mapped_preds.append(new_pred)

        return mapped_preds


def build_fgd_stn_pipeline(
    fgd_weights: Optional[Union[str, Path]] = None,
    stn_weights: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    allow_identity_fallback: bool = True,
    max_degrees: float = 45.0,
    max_translate: float = 0.15,
) -> Tuple[FGDSTNPipeline, Dict[str, Any]]:
    """Build and initialize an FGDSTNPipeline loading checkpoints from disk.

    Args:
        fgd_weights (str | Path, optional): Path to trained FGD checkpoint.
        stn_weights (str | Path, optional): Path to trained STN checkpoint.
        device (torch.device, optional): Target device for defense modules.
        allow_identity_fallback (bool): If True, instantiates identity-initialized modules
            when weight paths are not provided.
        max_degrees (float): Max rotation degrees for fallback STN module.
        max_translate (float): Max translation ratio for fallback STN module.

    Returns:
        Tuple[FGDSTNPipeline, Dict[str, Any]]: Initialized pipeline and loaded metadata dictionary.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    metadata: Dict[str, Any] = {}

    # 1. Load or instantiate FGD denoiser
    denoiser: Optional[ResidualDenoiser] = None
    if fgd_weights and str(fgd_weights).strip().lower() not in {"", "none", "false"}:
        denoiser, fgd_meta = load_fgd_denoiser(fgd_weights, device=device)
        metadata["fgd_meta"] = fgd_meta
    elif allow_identity_fallback:
        LOGGER.info("No fgd_weights provided; initializing identity ResidualDenoiser.")
        denoiser = ResidualDenoiser().to(device)
        denoiser.eval().requires_grad_(False)
        metadata["fgd_meta"] = {"identity_fallback": True}

    # 2. Load or instantiate STN module
    stn: Optional[SpatialTransformer] = None
    if stn_weights and str(stn_weights).strip().lower() not in {"", "none", "false"}:
        stn, stn_meta = load_stn_checkpoint(stn_weights, device=device)
        metadata["stn_meta"] = stn_meta
    elif allow_identity_fallback:
        LOGGER.info("No stn_weights provided; initializing identity SpatialTransformer.")
        stn = SpatialTransformer(max_degrees=max_degrees, max_translate=max_translate).to(device)
        stn.eval().requires_grad_(False)
        metadata["stn_meta"] = {"identity_fallback": True}

    pipeline = FGDSTNPipeline(denoiser=denoiser, stn=stn)
    return pipeline, metadata


def transform_batch_bboxes_for_stn(
    batch: dict,
    pred_matrices: Optional[torch.Tensor],
    height: int,
    width: int,
    min_box_size: float = 1.0,
) -> dict:
    """Transform ground-truth bounding boxes into the STN rectified coordinate system.

    Given forward matrices M_pred of shape [B, 3, 3] mapping rectified coordinates
    to input coordinates (p_in = M_pred @ p_rect), ground-truth bounding boxes
    in input coordinates are mapped to rectified coordinates via M_rect = (M_pred)^(-1).

    Args:
        batch (dict): Batch dictionary containing 'bboxes' [M, 4] in normalized xywh format,
            'batch_idx' [M] or [M, 1], 'cls' [M] or [M, 1], etc.
        pred_matrices (torch.Tensor, optional): Forward transformation matrices of shape [B, 3, 3].
            If None, returns batch unmodified.
        height (int): Image canvas height in pixels.
        width (int): Image canvas width in pixels.
        min_box_size (float): Minimum width and height in pixels for valid bounding boxes.

    Returns:
        dict: Updated batch dictionary with 'bboxes' transformed into normalized xywh
            in the STN output coordinate system. Degenerate boxes are filtered.
    """
    if pred_matrices is None:
        return batch

    bboxes = batch.get("bboxes")
    batch_idx = batch.get("batch_idx")
    cls = batch.get("cls")

    if bboxes is None or bboxes.numel() == 0 or batch_idx is None or batch_idx.numel() == 0:
        return batch

    device = bboxes.device
    dtype = bboxes.dtype
    orig_bi_dim = batch_idx.dim()
    orig_cls_dim = cls.dim() if cls is not None else 1

    flat_bi = batch_idx.view(-1).long()
    num_boxes = bboxes.shape[0]

    # Compute inverse matrices mapping input -> rectified
    if pred_matrices.ndim == 2:
        m_rect = torch.linalg.inv(pred_matrices.float()).to(dtype=dtype).unsqueeze(0)
    else:
        m_rect = torch.linalg.inv(pred_matrices.float()).to(dtype=dtype)

    # Convert normalized xywh to pixel xyxy
    xc = bboxes[:, 0] * float(width)
    yc = bboxes[:, 1] * float(height)
    bw = bboxes[:, 2] * float(width)
    bh = bboxes[:, 3] * float(height)
    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0
    boxes_xyxy = torch.stack((x1, y1, x2, y2), dim=-1)

    # Pick matrix for each box based on batch_idx
    if m_rect.shape[0] == 1 and num_boxes > 1:
        m_boxes = m_rect.expand(num_boxes, 3, 3)
    else:
        clamped_bi = flat_bi.clamp(0, m_rect.shape[0] - 1)
        m_boxes = m_rect[clamped_bi]

    # Transform boxes and clamp to canvas bounds
    mapped_xyxy = transform_boxes(boxes_xyxy, m_boxes, height=height, width=width, clip=True)

    # Filter degenerate boxes
    nw = mapped_xyxy[:, 2] - mapped_xyxy[:, 0]
    nh = mapped_xyxy[:, 3] - mapped_xyxy[:, 1]
    valid = (nw >= min_box_size) & (nh >= min_box_size)

    if not valid.all():
        mapped_xyxy = mapped_xyxy[valid]
        flat_bi = flat_bi[valid]
        if cls is not None:
            cls = cls[valid]

        # Filter other optional keys if aligned with targets
        for k in ("masks", "keypoints", "segments", "obb"):
            if k in batch and isinstance(batch[k], torch.Tensor) and batch[k].shape[0] == num_boxes:
                batch[k] = batch[k][valid]

    if mapped_xyxy.shape[0] == 0:
        # Empty targets after filtering
        batch["batch_idx"] = torch.zeros(0, dtype=torch.long, device=device)
        batch["cls"] = torch.zeros((0, 1) if orig_cls_dim > 1 else (0,), dtype=torch.float32, device=device)
        batch["bboxes"] = torch.zeros((0, 4), dtype=dtype, device=device)
        return batch

    # Convert back to normalized xywh
    new_xc = ((mapped_xyxy[:, 0] + mapped_xyxy[:, 2]) / (2.0 * float(width))).clamp(0.0, 1.0)
    new_yc = ((mapped_xyxy[:, 1] + mapped_xyxy[:, 3]) / (2.0 * float(height))).clamp(0.0, 1.0)
    new_wn = ((mapped_xyxy[:, 2] - mapped_xyxy[:, 0]) / float(width)).clamp(0.0, 1.0)
    new_hn = ((mapped_xyxy[:, 3] - mapped_xyxy[:, 1]) / float(height)).clamp(0.0, 1.0)
    new_bboxes = torch.stack((new_xc, new_yc, new_wn, new_hn), dim=-1)

    # Sort targets by batch_idx to maintain strictly non-decreasing order required by Ultralytics loss
    sort_idx = torch.argsort(flat_bi)
    flat_bi = flat_bi[sort_idx]
    new_bboxes = new_bboxes[sort_idx]
    if cls is not None:
        cls = cls[sort_idx]
        if orig_cls_dim > 1 and cls.ndim == 1:
            cls = cls.unsqueeze(-1)
        elif orig_cls_dim == 1 and cls.ndim > 1:
            cls = cls.squeeze(-1)
        batch["cls"] = cls

    if orig_bi_dim > 1 and flat_bi.dim() == 1:
        batch["batch_idx"] = flat_bi.unsqueeze(-1)
    else:
        batch["batch_idx"] = flat_bi

    batch["bboxes"] = new_bboxes
    return batch


class PreprocessDetectorWrapper(nn.Module):
    """Integrated defense wrapper combining FGD+STN pipeline with a YOLO detector.

    Enables adversarial attack generation (PGD, BIM, MIM, Worst-of-K) to pass through
    the selected preprocessing pipeline with gradients flowing through to inputs,
    while keeping the defense pipeline permanently frozen and in eval mode.
    """

    def __init__(
        self,
        pipeline: FGDSTNPipeline,
        detector: nn.Module,
        mode: str = "none",
    ) -> None:
        """Initialize wrapper with defense pipeline and detector.

        Args:
            pipeline (FGDSTNPipeline): Defense pipeline.
            detector (nn.Module): Detection model (trainable or frozen).
            mode (str): Preprocessing mode ('none', 'fgd', 'stn', 'fgd_stn', 'stn_fgd').
        """
        super().__init__()
        self.pipeline = pipeline
        self.detector = detector
        self.mode = canonicalize_mode(mode)
        self._training = detector.training
        self._last_pred_matrix: Optional[torch.Tensor] = None
        self._score_idx: int = 0
        self.current_paths: list = []

        # Keep pipeline permanently frozen and in eval mode
        self.pipeline.eval().requires_grad_(False)

        # Delegate common attributes for compatibility with attackers and validators
        for attr in ("names", "nc", "stride", "pt_path", "yaml", "args"):
            if hasattr(detector, attr):
                setattr(self, attr, getattr(detector, attr))

    @property
    def training(self) -> bool:
        detector = self._modules.get("detector")
        return detector.training if detector is not None else self._training

    @training.setter
    def training(self, mode: bool) -> None:
        self._training = mode

    def train(self, mode: bool = True):
        """Allow existing attackers to switch detector mode while pipeline stays in eval."""
        super().train(mode)
        self.detector.train(mode)
        self.pipeline.eval()
        return self

    def forward(self, images: torch.Tensor, *args, **kwargs) -> Any:
        """Preprocess images through pipeline and run detector forward."""
        processed_img, pred_matrix = self.pipeline.forward(images, mode=self.mode)
        self._last_pred_matrix = pred_matrix
        self._score_idx = 0
        return self.detector(processed_img, *args, **kwargs)

    def loss(self, batch: dict, preds: Any = None) -> Any:
        """Compute detector loss with ground truth transformed into the preprocessed coordinate system."""
        unwrapped = unwrap_model(self.detector)
        if preds is None:
            batch_copy = {k: v.clone() if torch.is_tensor(v) else copy(v) for k, v in batch.items()}
            processed_img, pred_matrix = self.pipeline.forward(batch_copy["img"], mode=self.mode)
            batch_copy["img"] = processed_img
            _, _, h, w = processed_img.shape
            if pred_matrix is not None:
                batch_copy = transform_batch_bboxes_for_stn(batch_copy, pred_matrix, height=h, width=w)
            return unwrapped.loss(batch_copy, preds)

        # preds is already computed from forward(candidates)
        if self._last_pred_matrix is not None:
            _, _, h, w = batch["img"].shape
            b_size = batch["img"].shape[0]
            if self._last_pred_matrix.shape[0] >= self._score_idx + b_size:
                pred_m = self._last_pred_matrix[self._score_idx : self._score_idx + b_size]
                self._score_idx += b_size
            else:
                pred_m = self._last_pred_matrix[:b_size]
                self._score_idx = 0

            batch_copy = {k: v.clone() if torch.is_tensor(v) else copy(v) for k, v in batch.items()}
            batch_copy = transform_batch_bboxes_for_stn(batch_copy, pred_m, height=h, width=w)
            return unwrapped.loss(batch_copy, preds)

        return unwrapped.loss(batch, preds)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "detector" in self.__dict__:
                return getattr(self.detector, name)
            raise


__all__ = [
    "FGDSTNPipeline",
    "VALID_MODES",
    "MODE_DESCRIPTIONS",
    "MODE_ALIASES",
    "build_fgd_stn_pipeline",
    "canonicalize_mode",
    "transform_batch_bboxes_for_stn",
    "PreprocessDetectorWrapper",
]

