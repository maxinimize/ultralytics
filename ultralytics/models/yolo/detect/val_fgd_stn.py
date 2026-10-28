# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Unified Validator for combined FGD and STN Defenses across E1-E5 preprocessing modes."""

from __future__ import annotations

import csv
import json
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch
from ultralytics.attacks.spatial_worst_of_k_stn import SpatialWorstOfKWithMetadata
from ultralytics.defenses.fgd_stn_pipeline import (
    MODE_DESCRIPTIONS,
    VALID_MODES,
    FGDSTNPipeline,
    canonicalize_mode,
)
from ultralytics.defenses.spatial_transformer import (
    compute_control_points_loss,
    compute_masked_recon_loss,
    compute_validity_mask,
)
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, TQDM, ops
from ultralytics.utils.metrics import DetMetrics, ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression

SUPPORTED_ATTACKS: Tuple[str, ...] = ("clean", "pgd", "bim", "mim", "worstk")

ATTACK_ALIASES: Dict[str, str] = {
    "raw": "clean",
    "none": "clean",
    "clean_baseline": "clean",
    "worst_of_k": "worstk",
    "worst-of-k": "worstk",
    "worst_k": "worstk",
}


def canonicalize_attack(attack: str) -> str:
    """Normalize attack name to canonical attack identifier.

    Args:
        attack (str): Attack name or alias.

    Returns:
        str: Canonical attack ('clean', 'pgd', 'bim', 'mim', 'worstk').
    """
    key = str(attack).strip().lower().replace("-", "_")
    if key in SUPPORTED_ATTACKS:
        return key
    if key in ATTACK_ALIASES:
        return ATTACK_ALIASES[key]
    raise ValueError(f"Unknown attack '{attack}'. Supported attacks are: {list(SUPPORTED_ATTACKS)}")


class UnifiedFGDSTNValidator(BaseValidator):
    """Validation pipeline for evaluating Unified FGD + STN defenses across E1-E5 modes.

    Five Preprocessing Modes (E1-E5):
    - E1 ('none'): YOLO only (Baseline)
    - E2 ('fgd'): FGD denoiser -> YOLO
    - E3 ('stn'): STN spatial transformer -> YOLO -> unwarp predicted boxes
    - E4 ('fgd_stn'): FGD -> STN -> YOLO -> unwarp predicted boxes
    - E5 ('stn_fgd'): STN -> FGD -> YOLO -> unwarp predicted boxes

    Five Supported Attacks:
    - Clean: Non-attacked original validation images
    - PGD: Projected Gradient Descent evasion attack
    - BIM: Basic Iterative Method evasion attack
    - MIM: Momentum Iterative Method evasion attack
    - Worst-of-K: Spatial rotation and translation worst-of-k attack

    Key Features:
    - Generates or caches adversarial samples once per batch and reuses them across modes.
    - Attack generation is independent of the evaluated defense mode.
    - Handles STN bounding box coordinate transformations correctly.
    - Supports black-box transfer attacks with a separately specified attack model.
    - Exports comparison table, CSV, and JSON results.
    """

    def __init__(
        self,
        dataloader=None,
        save_dir: Optional[Path] = None,
        args: Optional[dict] = None,
        _callbacks: Optional[dict] = None,
    ) -> None:
        """Initialize UnifiedFGDSTNValidator.

        Args:
            dataloader: DataLoader to evaluate.
            save_dir (Path, optional): Directory to save outputs.
            args (dict, optional): Validator configuration arguments.
            _callbacks (dict, optional): Callbacks dictionary.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()

        self.guidance_model: Optional[nn.Module] = None
        self.attack_model: Optional[nn.Module] = None
        self.pipeline: Optional[FGDSTNPipeline] = None
        self.attackers: Dict[str, Any] = {}

        # Internal metric tracking structures
        self.mode_stats: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        self.mode_seen: Dict[str, int] = {}
        self.mode_diagnostics: Dict[str, Dict[str, List[float]]] = {}

    def postprocess(self, preds: Any) -> List[torch.Tensor]:
        """Perform Non-Maximum Suppression (NMS) on detector predictions.

        Args:
            preds (Any): Raw output predictions from YOLO detector.

        Returns:
            List[torch.Tensor]: List of post-processed detections per image.
        """
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

    def _process_batch(
        self,
        detections: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
    ) -> torch.Tensor:
        """Match detections against ground truth boxes across IoU thresholds.

        Args:
            detections (torch.Tensor): Detections tensor [N, 6] (x1, y1, x2, y2, conf, cls).
            gt_bboxes (torch.Tensor): Ground truth boxes [M, 4] in pixel xyxy format.
            gt_cls (torch.Tensor): Ground truth class labels [M].

        Returns:
            torch.Tensor: Boolean match tensor of shape [N, 10].
        """
        if detections.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            return torch.zeros(detections.shape[0], self.niou, dtype=torch.bool, device=self.device)
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou).to(self.device)

    def _reset_mode_stats(self, modes: Sequence[str]) -> None:
        """Reset accumulated statistics and counters for selected modes."""
        self.mode_stats = {m: {"tp": [], "conf": [], "pred_cls": [], "target_cls": []} for m in modes}
        self.mode_seen = {m: 0 for m in modes}
        self.mode_diagnostics = {m: {"geom_error": [], "masked_recon": []} for m in modes}

    def update_mode_metrics(self, mode: str, preds: List[torch.Tensor], batch: dict) -> None:
        """Update detection statistics for a specific preprocessing mode.

        Args:
            mode (str): Preprocessing mode key.
            preds (List[torch.Tensor]): List of detection predictions per image.
            batch (dict): Current batch dictionary containing ground truth.
        """
        stats = self.mode_stats[mode]
        _, _, h, w = batch["img"].shape

        for si, pred in enumerate(preds):
            self.mode_seen[mode] += 1
            npr = len(pred)
            stat = {
                "conf": torch.zeros(0, device=self.device),
                "pred_cls": torch.zeros(0, device=self.device),
                "tp": torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            }

            # Retrieve ground truth targets for this image
            if (
                "cls" in batch
                and batch["cls"].shape[0] > 0
                and "batch_idx" in batch
                and batch["cls"].shape[0] == batch["batch_idx"].shape[0]
            ):
                idx = batch["batch_idx"] == si
                cls = batch["cls"][idx]
                bbox = batch["bboxes"][idx]
            else:
                cls = torch.zeros(0, device=self.device)
                bbox = torch.zeros((0, 4), device=self.device)

            nl = len(cls)
            stat["target_cls"] = cls.squeeze(-1) if cls.ndim > 1 else cls

            if npr == 0:
                if nl:
                    for k in stats:
                        stats[k].append(stat[k])
                continue

            # Targets are in normalized xywh format; convert to pixel xyxy
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

            for k in stats:
                stats[k].append(stat[k])

    def generate_attacked_batch(self, attack: str, batch: dict) -> dict:
        """Generate adversarial sample batch once, independent of defense preprocessing.

        Preserves existing Worst-of-K ground-truth transformations while ensuring
        identical adversarial samples are delivered to all downstream modes.

        Args:
            attack (str): Canonical attack name.
            batch (dict): Preprocessed batch dictionary with images in [0, 1].

        Returns:
            dict: Attacked batch dictionary.
        """
        # Shallow copy batch with cloned tensors to isolate downstream pipelines
        att_batch = {k: v.clone() if torch.is_tensor(v) else copy(v) for k, v in batch.items()}

        if attack == "clean":
            return att_batch

        if attack == "worstk":
            attacker = self.attackers.get("worstk")
            if attacker is not None and callable(getattr(attacker, "forward_batch", None)):
                attacker.forward_batch(att_batch)
            return att_batch

        if attack in {"pgd", "bim", "mim"}:
            attacker = self.attackers.get(attack)
            if attacker is not None:
                adv_batch_size = int(getattr(self.args, "adv_val_batch", 16))
                imgs_adv = run_attack_on_batch(
                    attacker,
                    att_batch,
                    label_policy="largest_box",
                    max_batch_size=adv_batch_size,
                )
                if imgs_adv is not None:
                    att_batch["img"] = imgs_adv.to(att_batch["img"].dtype).to(self.device)
            return att_batch

        return att_batch

    def compute_mode_metrics(self, mode: str) -> Dict[str, float]:
        """Compute final mAP50 and mAP50-95 for a specific mode from accumulated statistics.

        Args:
            mode (str): Preprocessing mode key.

        Returns:
            Dict[str, float]: Metric dictionary including mAP50 and mAP50-95.
        """
        stats = {
            k: (
                torch.cat(v, 0).cpu().numpy()
                if len(v)
                else (np.zeros((0, self.niou)) if k == "tp" else np.zeros(0))
            )
            for k, v in self.mode_stats[mode].items()
        }

        names = getattr(self.guidance_model, "names", None)
        if not names and isinstance(getattr(self, "data", None), dict):
            names = self.data.get("names")
        if not names:
            names = getattr(self, "names", {})
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
            metrics = DetMetrics()
            metrics.box.nc = len(names)
            metrics.box.update(results)
            mp = metrics.mean_results()
            map50 = float(mp[2])
            map50_95 = float(mp[3])
        else:
            map50, map50_95 = 0.0, 0.0

        out = {
            "mAP50": map50,
            "mAP50-95": map50_95,
        }

        # Optional geometry and reconstruction diagnostics for STN modes
        diag = self.mode_diagnostics.get(mode, {})
        g_errs = diag.get("geom_error", [])
        if g_errs:
            out["geom_error"] = float(sum(g_errs) / len(g_errs))
        r_losses = diag.get("masked_recon", [])
        if r_losses:
            out["masked_recon"] = float(sum(r_losses) / len(r_losses))

        return out

    def run_eval_for_attack(
        self,
        attack: str,
        modes: Sequence[str],
        seed: int = 0,
    ) -> Dict[str, Dict[str, float]]:
        """Run single-pass dataloader evaluation across all selected modes for one attack.

        Generates or samples attacked batches once, evaluating all modes on the exact same inputs.

        Args:
            attack (str): Canonical attack name.
            modes (Sequence[str]): Canonical mode names to evaluate.
            seed (int): Reproducibility seed.

        Returns:
            Dict[str, Dict[str, float]]: Results mapped by mode name.
        """
        # Ensure reproducibility
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        if attack == "worstk" and "worstk" in self.attackers:
            wk_att = self.attackers["worstk"]
            if hasattr(wk_att, "_generator"):
                wk_att._generator.manual_seed(int(seed))

        self._reset_mode_stats(modes)
        desc = f"Evaluating [{attack.upper()}] across modes ({', '.join(modes)})"
        bar = TQDM(self.dataloader, desc=desc, total=len(self.dataloader))

        for batch_i, raw_batch in enumerate(bar):
            # Move batch tensors to device
            batch: Dict[str, Any] = {}
            for k, v in raw_batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
                else:
                    batch[k] = v

            # Preprocess to float in [0, 1]
            batch["img"] = batch["img"].float() / 255.0
            batch["clean_img"] = batch["img"].clone()

            # 1. Generate attacked batch ONCE per batch (independent of defense mode)
            att_batch = self.generate_attacked_batch(attack, batch)
            _, _, h, w = att_batch["img"].shape
            gt_matrix = att_batch.get("gt_matrix")

            # 2. Sequential evaluation across all selected defense modes
            for mode in modes:
                # Clone attacked image tensor to isolate pipeline modifications
                img_in = att_batch["img"].clone()

                with torch.no_grad():
                    processed_img, pred_matrix = self.pipeline.forward(img_in, mode=mode)
                    preds_raw = self.guidance_model(processed_img, augment=False)
                    preds = self.postprocess(preds_raw)

                    # Unwarp bounding boxes if STN rectification was applied
                    if pred_matrix is not None:
                        preds = self.pipeline.unwarp_predictions(
                            preds, pred_matrix, height=h, width=w, clip=True
                        )

                        # Record STN geometry diagnostics if Worst-of-K ground-truth matrix is present
                        if gt_matrix is not None:
                            g_err = compute_control_points_loss(pred_matrix, gt_matrix, height=h, width=w)
                            self.mode_diagnostics[mode]["geom_error"].append(g_err.item())

                            mask = compute_validity_mask(gt_matrix, pred_matrix, height=h, width=w)
                            r_loss = compute_masked_recon_loss(processed_img, batch["clean_img"], mask)
                            self.mode_diagnostics[mode]["masked_recon"].append(r_loss.item())

                # Update metrics comparing predictions with ground truth boxes
                self.update_mode_metrics(mode, preds, att_batch)

        # Compute detection mAP for each mode
        attack_results: Dict[str, Dict[str, float]] = {}
        for mode in modes:
            m_res = self.compute_mode_metrics(mode)
            attack_results[mode] = m_res
            LOGGER.info(
                f"[{attack.upper()}] Mode {mode:<10} ({MODE_DESCRIPTIONS.get(mode, mode)}) -> "
                f"mAP50: {m_res['mAP50']:.4f}, mAP50-95: {m_res['mAP50-95']:.4f}"
            )

        return attack_results

    def evaluate_all(
        self,
        modes: Sequence[str] | str = "all",
        attacks: Sequence[str] | str = "all",
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Orchestrate evaluation across specified preprocessing modes and attacks.

        Args:
            modes (Sequence[str] | str): Modes to evaluate (or 'all' for E1-E5).
            attacks (Sequence[str] | str): Attacks to evaluate (or 'all' for all 5 attacks).
            seed (int): Reproducibility random seed.

        Returns:
            Dict[str, Any]: Consolidated results containing per-attack and per-mode metrics,
                comparison tables, and export summaries.
        """
        # Resolve modes
        if isinstance(modes, str):
            if modes.strip().lower() == "all":
                sel_modes = list(VALID_MODES)
            else:
                sel_modes = [canonicalize_mode(modes)]
        else:
            sel_modes = []
            for m in modes:
                if str(m).strip().lower() == "all":
                    sel_modes = list(VALID_MODES)
                    break
                sel_modes.append(canonicalize_mode(m))
            # Preserve order while removing duplicates
            sel_modes = list(dict.fromkeys(sel_modes))

        # Resolve attacks
        if isinstance(attacks, str):
            if attacks.strip().lower() == "all":
                sel_attacks = list(SUPPORTED_ATTACKS)
            else:
                sel_attacks = [canonicalize_attack(attacks)]
        else:
            sel_attacks = []
            for a in attacks:
                if str(a).strip().lower() == "all":
                    sel_attacks = list(SUPPORTED_ATTACKS)
                    break
                sel_attacks.append(canonicalize_attack(a))
            sel_attacks = list(dict.fromkeys(sel_attacks))

        LOGGER.info(f"Starting Unified FGD + STN Defense Evaluation")
        LOGGER.info(f"Selected Preprocessing Modes: {sel_modes}")
        LOGGER.info(f"Selected Attacks: {sel_attacks}")
        LOGGER.info(f"Evaluation Seed: {seed}")

        # Ensure guidance model and defenses are in eval mode and frozen
        self.guidance_model.eval().requires_grad_(False)
        if self.pipeline.denoiser is not None:
            self.pipeline.denoiser.eval().requires_grad_(False)
        if self.pipeline.stn is not None:
            self.pipeline.stn.eval().requires_grad_(False)

        raw_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for attack in sel_attacks:
            LOGGER.info(f"\n{'='*70}\nRunning Attack Suite: {attack.upper()}\n{'='*70}")
            att_res = self.run_eval_for_attack(attack, sel_modes, seed=seed)
            raw_results[attack] = att_res

        # Format comparison summaries
        summary = self.build_summary(raw_results, sel_modes, sel_attacks)

        # Save results to CSV and JSON
        if self.save_dir:
            self.save_dir = Path(self.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.export_results(summary)

        self.print_summary_table(summary)
        return summary

    def build_summary(
        self,
        raw_results: Dict[str, Dict[str, Dict[str, float]]],
        modes: List[str],
        attacks: List[str],
    ) -> Dict[str, Any]:
        """Consolidate nested results into structured summary and comparison tables.

        Args:
            raw_results: Dict[attack][mode] -> metric_dict.
            modes: Evaluated modes.
            attacks: Evaluated attacks.

        Returns:
            Dict[str, Any]: Comprehensive summary dictionary.
        """
        flat_results: Dict[str, float] = {}
        mode_table: List[Dict[str, Any]] = []

        for m in modes:
            row: Dict[str, Any] = {
                "mode": m,
                "description": MODE_DESCRIPTIONS.get(m, m),
            }
            for a in attacks:
                m_res = raw_results.get(a, {}).get(m, {})
                m50 = m_res.get("mAP50", 0.0)
                m95 = m_res.get("mAP50-95", 0.0)
                row[f"{a}_mAP50"] = m50
                row[f"{a}_mAP50_95"] = m95
                flat_results[f"{a}/{m}/mAP50"] = m50
                flat_results[f"{a}/{m}/mAP50-95"] = m95
                for extra_k in ("geom_error", "masked_recon"):
                    if extra_k in m_res:
                        row[f"{a}_{extra_k}"] = m_res[extra_k]
                        flat_results[f"{a}/{m}/{extra_k}"] = m_res[extra_k]
            mode_table.append(row)

        return {
            "modes": modes,
            "attacks": attacks,
            "raw_results": raw_results,
            "mode_table": mode_table,
            "flat_results": flat_results,
        }

    def print_summary_table(self, summary: Dict[str, Any]) -> None:
        """Format and log an ASCII comparison table of E1-E5 across evaluated attacks.

        Args:
            summary (Dict[str, Any]): Summary dictionary from build_summary.
        """
        modes = summary["modes"]
        attacks = summary["attacks"]
        raw = summary["raw_results"]

        LOGGER.info("\n" + "=" * 95)
        LOGGER.info("UNIFIED FGD + STN DEFENSE VALIDATION RESULTS (mAP50 / mAP50-95)")
        LOGGER.info("=" * 95)

        # Header
        col_w = 20
        header = f"{'Mode':<10} | {'Description':<28} | " + " | ".join(f"{a.upper():^{col_w}}" for a in attacks)
        LOGGER.info(header)
        LOGGER.info("-" * len(header))

        for m in modes:
            desc = MODE_DESCRIPTIONS.get(m, m)
            cells = []
            for a in attacks:
                res = raw.get(a, {}).get(m, {})
                m50 = res.get("mAP50", 0.0)
                m95 = res.get("mAP50-95", 0.0)
                val_str = f"{m50:.4f} / {m95:.4f}"
                cells.append(f"{val_str:^{col_w}}")
            row_str = f"{m:<10} | {desc:<28} | " + " | ".join(cells)
            LOGGER.info(row_str)

        LOGGER.info("=" * 95 + "\n")

    def export_results(self, summary: Dict[str, Any]) -> None:
        """Save results to CSV and JSON formats in self.save_dir.

        Args:
            summary (Dict[str, Any]): Summary dictionary from build_summary.
        """
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save detailed JSON
        json_path = save_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        LOGGER.info(f"Saved evaluation results JSON to {json_path}")

        # 2. Save comparison table CSV
        csv_path = save_dir / "results.csv"
        mode_table = summary.get("mode_table", [])
        if mode_table:
            fieldnames = []
            for row in mode_table:
                for k in row.keys():
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
                writer.writeheader()
                for row in mode_table:
                    writer.writerow(row)
            LOGGER.info(f"Saved evaluation comparison table CSV to {csv_path}")

        # 3. If single mode is evaluated, also save dedicated individual files
        modes = summary.get("modes", [])
        if len(modes) == 1:
            single_m = modes[0]
            single_json = save_dir / f"result_{single_m}.json"
            with open(single_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mode": single_m,
                        "description": MODE_DESCRIPTIONS.get(single_m, single_m),
                        "attacks": summary["attacks"],
                        "results": {a: summary["raw_results"][a][single_m] for a in summary["attacks"]},
                    },
                    f,
                    indent=2,
                )
            LOGGER.info(f"Saved individual mode result to {single_json}")

    def __call__(
        self,
        trainer=None,
        modes: Sequence[str] | str = "all",
        attacks: Sequence[str] | str = "all",
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Callable interface matching Ultralytics validator calling convention.

        Args:
            trainer (optional): Trainer instance if called within training loop.
            modes: Preprocessing mode(s) to evaluate.
            attacks: Attack(s) to evaluate.
            seed: Reproducibility random seed.

        Returns:
            Dict[str, Any]: Consolidated results summary.
        """
        if trainer is not None:
            self.device = trainer.device
            self.guidance_model = getattr(trainer, "guidance_model", trainer.model)
            self.dataloader = trainer.test_loader
            if hasattr(trainer, "data") and trainer.data:
                self.data = trainer.data
        return self.evaluate_all(modes=modes, attacks=attacks, seed=seed)
