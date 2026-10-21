# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Independent Validator for Spatial Transformer Network (STN) Defense."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ultralytics.attacks.spatial_worst_of_k_stn import SpatialWorstOfKWithMetadata
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.defenses.spatial_transformer import (
    SpatialTransformer,
    compute_control_points_loss,
    compute_masked_recon_loss,
    compute_validity_mask,
    load_stn_checkpoint,
    transform_boxes,
)
from ultralytics.engine.validator import BaseValidator
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr, ops
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.metrics import DetMetrics, ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import select_device, unwrap_model


class STNDetectionValidator(BaseValidator):
    """Validation pipeline for evaluating STN defense against Worst-of-k attacks.

    Supports four canonical evaluation modes:
    1. clean_baseline: Clean images evaluated on frozen YOLO (no STN)
    2. worstk_baseline: Worst-of-k attacked images on frozen YOLO (no STN)
    3. clean_stn: Clean images evaluated through STN + frozen YOLO
    4. worstk_stn: Worst-of-k attacked images through STN + frozen YOLO
    """

    def __init__(
        self,
        dataloader=None,
        save_dir: Optional[Path] = None,
        args: Optional[dict] = None,
        _callbacks: Optional[dict] = None,
    ) -> None:
        """Initialize STNDetectionValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # mAP@0.5:0.95 thresholds
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()

        self.stn: Optional[SpatialTransformer] = None
        self.guidance_model: Optional[nn.Module] = None
        self.attacker: Optional[SpatialWorstOfKWithMetadata] = None

    def postprocess(self, preds: Any) -> List[torch.Tensor]:
        """Perform Non-Maximum Suppression (NMS) on detector predictions."""
        conf_thres = getattr(self.args, "conf", 0.001)
        iou_thres = getattr(self.args, "iou", 0.6)
        classes = getattr(self.args, "classes", None)
        return non_max_suppression(
            preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            multi_label=True,
            max_det=300,
        )

    def update_metrics(self, preds: List[torch.Tensor], batch: dict) -> None:
        """Process predicted boxes and ground-truth targets to update evaluation metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )

            # Ground truth targets for this image
            idx = batch["batch_idx"] == si
            cls = batch["cls"][idx]
            bbox = batch["bboxes"][idx]
            nl = len(cls)
            stat["target_cls"] = cls.squeeze(-1) if cls.ndim > 1 else cls

            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                continue

            # Targets are in normalized xywh format; convert to pixel xyxy
            _, _, h, w = batch["img"].shape
            if nl:
                tbox = ops.xywh2xyxy(bbox)
                scale = torch.tensor([w, h, w, h], device=self.device, dtype=bbox.dtype)
                tbox = tbox * scale
            else:
                tbox = torch.zeros((0, 4), device=self.device)

            predn = pred.clone()
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate IoUs if ground truth targets exist
            if nl:
                stat["tp"] = self._process_batch(predn, tbox, stat["target_cls"])

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

    def _process_batch(self, detections: torch.Tensor, gt_bboxes: torch.Tensor, gt_cls: torch.Tensor) -> torch.Tensor:
        """Compute match matrix of detections against ground-truth boxes across IoU thresholds."""
        correct = torch.zeros(detections.shape[0], self.niou, dtype=torch.bool, device=self.device)
        iou = box_iou(gt_bboxes, detections[:, :4])
        correct_class = gt_cls[:, None] == detections[:, 5]
        for i, threshold in enumerate(self.iouv.to(self.device)):
            matches = torch.nonzero((iou >= threshold) & correct_class)
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort(descending=True)]
                    matches = matches[torch.unique(matches[:, 1], return_inverse=True)[1]]
                    matches = matches[torch.unique(matches[:, 0], return_inverse=True)[1]]
                correct[matches[:, 1].long(), i] = True
        return correct

    def run_single_eval(
        self,
        mode: str,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Execute one complete validation pass for the given mode.

        Args:
            mode (str): One of 'clean_baseline', 'worstk_baseline', 'clean_stn', 'worstk_stn'.
            seed (int): Reproducibility seed for consistent attack sampling.

        Returns:
            Dict[str, float]: Metric dictionary including mAP50 and mAP50-95.
        """
        use_attack = "worstk" in mode
        use_stn = "stn" in mode

        # Reset statistics
        self.seen = 0
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
        self.metrics = DetMetrics()
        geom_errors = []
        recon_losses = []

        # Ensure reproducible attack sampling across baseline and defended evaluations
        if use_attack and self.attacker is not None:
            self.attacker._generator.manual_seed(int(seed))

        bar = TQDM(self.dataloader, desc=f"Evaluating {mode}", total=len(self.dataloader))

        for batch_i, raw_batch in enumerate(bar):
            # Move tensors to device
            batch = {}
            for k, v in raw_batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
                else:
                    batch[k] = v

            # Preprocess to float in [0, 1]
            batch["img"] = batch["img"].float() / 255.0

            clean_img = batch["img"].clone()
            batch["clean_img"] = clean_img

            # Apply Worst-of-k attack if enabled
            if use_attack and self.attacker is not None:
                self.attacker.forward_batch(batch)
                input_img = batch["img"]
                gt_matrix = batch.get("gt_matrix")
            else:
                input_img = batch["img"]
                gt_matrix = None

            _, _, h, w = input_img.shape

            # STN Rectification step (autonomous: STN never accesses gt_matrix or attack params)
            if use_stn and self.stn is not None:
                with torch.no_grad():
                    rect_img, pred_matrix = self.stn(input_img)
                    model_in = rect_img

                    # Compute diagnostic geometry error and masked reconstruction loss if attacked
                    if gt_matrix is not None:
                        g_err = compute_control_points_loss(pred_matrix, gt_matrix, height=h, width=w)
                        geom_errors.append(g_err.item())

                        mask = compute_validity_mask(gt_matrix, pred_matrix, height=h, width=w)
                        r_loss = compute_masked_recon_loss(rect_img, clean_img, mask)
                        recon_losses.append(r_loss.item())
            else:
                model_in = input_img
                pred_matrix = None

            # Frozen detector inference
            with torch.no_grad():
                preds_raw = self.guidance_model(model_in, augment=False)
                preds = self.postprocess(preds_raw)

            # Map predicted bounding boxes back to attacked image coordinates
            if use_stn and pred_matrix is not None:
                mapped_preds = []
                for si, pred_si in enumerate(preds):
                    if pred_si.shape[0] == 0:
                        mapped_preds.append(pred_si)
                        continue
                    m_si = pred_matrix[si]
                    mapped_boxes = transform_boxes(pred_si[:, :4], m_si, height=h, width=w, clip=True)

                    # Filter degenerate boxes
                    bw = mapped_boxes[:, 2] - mapped_boxes[:, 0]
                    bh = mapped_boxes[:, 3] - mapped_boxes[:, 1]
                    valid = (bw > 0.0) & (bh > 0.0)

                    new_pred = torch.cat((mapped_boxes, pred_si[:, 4:]), dim=-1)[valid]
                    mapped_preds.append(new_pred)
                preds = mapped_preds

            # Update metrics comparing predictions with ground truth boxes
            self.update_metrics(preds, batch)

        # Compute detection mAP
        stats = {
            k: (torch.cat(v, 0).cpu().numpy() if len(v) else (np.zeros((0, self.niou)) if k == "tp" else np.zeros(0)))
            for k, v in self.stats.items()
        }
        names = getattr(self.guidance_model, "names", getattr(self, "names", dict()))
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))

        if len(stats) and stats["tp"].shape[0] and stats["target_cls"].shape[0]:
            results = ap_per_class(
                stats["tp"],
                stats["conf"],
                stats["pred_cls"],
                stats["target_cls"],
                names=names,
            )[2:]
            self.metrics.box.nc = len(names)
            self.metrics.box.update(results)
            mp = self.metrics.mean_results()
            map50 = float(mp[2])
            map50_95 = float(mp[3])
        else:
            map50, map50_95 = 0.0, 0.0

        out = {
            f"{mode}/mAP50": map50,
            f"{mode}/mAP50-95": map50_95,
        }
        if geom_errors:
            out[f"{mode}/geom_error"] = float(sum(geom_errors) / len(geom_errors))
        if recon_losses:
            out[f"{mode}/masked_recon"] = float(sum(recon_losses) / len(recon_losses))

        return out

    def __call__(
        self,
        trainer=None,
        modes: Optional[List[str]] = None,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Execute evaluation across selected modes and return consolidated results."""
        if trainer is not None:
            self.device = trainer.device
            self.stn = trainer.stn
            self.guidance_model = trainer.guidance_model
            self.attacker = trainer.attacker
            self.dataloader = trainer.test_loader
            if modes is None:
                # Fast epoch validation during training
                modes = ["clean_stn", "worstk_stn"]

        if modes is None:
            modes = ["clean_baseline", "worstk_baseline", "clean_stn", "worstk_stn"]

        self.guidance_model.eval().requires_grad_(False)
        if self.stn is not None:
            self.stn.eval()

        all_results = {}
        for m in modes:
            LOGGER.info(f"--- Running STN Validator: {m} ---")
            res = self.run_single_eval(m, seed=seed)
            all_results.update(res)
            LOGGER.info(
                f"{m} -> mAP50: {res.get(f'{m}/mAP50', 0.0):.4f}, "
                f"mAP50-95: {res.get(f'{m}/mAP50-95', 0.0):.4f}"
            )

        # Standardize main summary metrics
        if "worstk_stn/mAP50-95" in all_results:
            all_results["worstk/mAP50-95"] = all_results["worstk_stn/mAP50-95"]
            all_results["worstk/mAP50"] = all_results["worstk_stn/mAP50"]
        if "clean_stn/mAP50-95" in all_results:
            all_results["clean/mAP50-95"] = all_results["clean_stn/mAP50-95"]
            all_results["clean/mAP50"] = all_results["clean_stn/mAP50"]

        return all_results
