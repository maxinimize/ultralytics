# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch
from ultralytics.attacks.attack_utils import extract_detection_model, setup_attack_model
from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.defenses.fgd_stn_pipeline import PreprocessDetectorWrapper
from ultralytics.engine.validator import BaseValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    TQDM,
    IterableSimpleNamespace,
    callbacks,
    colorstr,
    nms,
    ops,
)
from ultralytics.utils.checks import check_imgsz, check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import attempt_compile, autocast, select_device, unwrap_model


def parse_list_arg(val, type_conv=str):
    if val is None:
        return []
    if isinstance(val, (int, float)):
        return [type_conv(val)]
    if isinstance(val, (list, tuple)):
        res = []
        for x in val:
            res.extend(parse_list_arg(x, type_conv))
        return res
    # assume string
    val = str(val).strip()
    if not val:
        return []
    # If it contains commas, split by comma; otherwise split by whitespace
    if "," in val:
        parts = val.split(",")
    else:
        parts = val.split()
    return [type_conv(p.strip()) for p in parts if p.strip()]


class DetectionValidator(BaseValidator):
    """A class extending the BaseValidator class for validation based on a detection model.

    This class implements validation functionality specific to object detection tasks, including metrics calculation,
    prediction processing, and visualization of results.

    Attributes:
        is_coco (bool): Whether the dataset is COCO.
        is_lvis (bool): Whether the dataset is LVIS.
        class_map (list[int]): Mapping from model class indices to dataset class indices.
        metrics (DetMetrics): Object detection metrics calculator.
        iouv (torch.Tensor): IoU thresholds for mAP calculation.
        niou (int): Number of IoU thresholds.
        lb (list[Any]): List for storing ground truth labels for hybrid saving.
        jdict (list[dict[str, Any]]): List for storing JSON detection results.
        stats (dict[str, list[torch.Tensor]]): Dictionary for storing statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionValidator
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml")
        >>> validator = DetectionValidator(args=args)
        >>> validator()
    """

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None,
        attack_weights: str | Path | None = None,
    ) -> None:
        """Initialize detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict[str, Any], optional): Arguments for the validator.
            _callbacks (list[Any], optional): List of callback functions.
            attack_weights (str | Path | None, optional): Surrogate weights source ('current' or checkpoint path).
        """
        attack_names = []
        attack_ratios = []
        eval_weights = None
        raw_attack_weights = attack_weights
        if isinstance(args, dict):
            raw_attack_name = args.pop("attack_name", None)
            raw_ratio = args.pop("ratio", None)
            raw_eval_w = args.pop("eval_weights", None)
            raw_attack_weights = args.pop("attack_weights", raw_attack_weights)
            if raw_attack_name is not None:
                attack_names = parse_list_arg(raw_attack_name, str)
            if raw_ratio is not None:
                attack_ratios = parse_list_arg(raw_ratio, float)
            if raw_eval_w is not None:
                eval_weights = raw_eval_w
        elif args is not None:
            for k in ("attack_name", "ratio", "eval_weights", "attack_weights"):
                if hasattr(args, k):
                    val = getattr(args, k)
                    if k == "attack_name" and val is not None:
                        attack_names = parse_list_arg(val, str)
                    elif k == "ratio" and val is not None:
                        attack_ratios = parse_list_arg(val, float)
                    elif k == "eval_weights" and val is not None:
                        eval_weights = val
                    elif k == "attack_weights" and val is not None:
                        raw_attack_weights = val
                    try:
                        delattr(args, k)
                    except AttributeError:
                        pass

        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()
        self.attack_names = attack_names
        self.attack_ratios = attack_ratios
        self.eval_weights = eval_weights
        self.attack_weights = raw_attack_weights
        self._fixed_surrogate = None
        self._fixed_surrogate_source = None
        self.last_attack_maps = {}
        self.attack_failures = set()

    def __call__(self, trainer=None, model=None):
        """Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        if trainer is not None and getattr(self, "attack_weights", None) is None:
            self.attack_weights = (
                getattr(trainer, "attack_weights", None)
                or getattr(trainer, "_attack_weights_arg", None)
                or getattr(getattr(trainer, "args", None), "attack_weights", None)
            )
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.quantize = 16 if (self.device.type != "cpu" and getattr(trainer, "amp", False)) else None
            model = (
                (getattr(trainer, "ema", None) and getattr(trainer.ema, "ema", None))
                or getattr(trainer, "model", None)
                or model
            )
            if getattr(trainer.args, "compile", False) and hasattr(model, "_orig_mod"):
                model = model._orig_mod  # validate non-compiled original model to avoid issues
            # Keep training validation read-only: inputs may be fp16, but EMA/model weights stay fp32 under autocast.
            if model is not None:
                model = model.float()
            loss_items = getattr(trainer, "loss_items", None)
            if loss_items:
                self.loss = {k: torch.zeros_like(v) for k, v in loss_items.items()}
            else:
                self.loss = {}
            possible_stop = getattr(getattr(trainer, "stopper", None), "possible_stop", False)
            trainer_epoch = getattr(trainer, "epoch", 0)
            trainer_epochs = getattr(trainer, "epochs", 1)
            self.args.plots &= possible_stop or (trainer_epoch == trainer_epochs - 1)
            if model is not None:
                model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device) if RANK == -1 else torch.device("cuda", RANK),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.quantize == 16,
            )
            self.device = model.device  # update device
            self.args.quantize = 16 if model.fp16 else None  # record actual inference precision
            stride, fmt = model.stride, model.format
            pt = fmt == "pt"
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if fmt not in {"pt", "torchscript"} and not getattr(model, "dynamic", False):
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if isinstance(getattr(self, "data", None), dict):
                pass
            elif str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(f"Dataset '{self.args.data}' for task={self.args.task} not found")

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not (pt or (getattr(model, "dynamic", False) and fmt != "imx")):
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            if self.args.compile:
                model = attempt_compile(model, device=self.device)
            ch = self.data.get("channels", 3) if isinstance(self.data, dict) else 3
            model.warmup(imgsz=(1 if pt else self.args.batch, ch, imgsz, imgsz))  # warmup

        # Determine the runs to execute: Always evaluate clean (raw) + specified attacks
        raw_att = (
            getattr(self, "attack_names", None)
            or getattr(self, "attack_name", None)
            or getattr(self.args, "attack_names", None)
            or getattr(self.args, "attack_name", None)
        )
        parsed_attacks = parse_list_arg(raw_att, str)
        if not parsed_attacks:
            parsed_attacks = ["pgd", "bim", "mim", "worstk"]

        # Ensure raw is always evaluated first, followed by each attack
        unique_runs = ["raw"]
        for att in parsed_attacks:
            if att.lower().strip() not in {"raw", "clean", "none"} and att not in unique_runs:
                unique_runs.append(att)

        # Non-clean attack keys
        eval_attacks = [a for a in unique_runs if a != "raw"]

        # Parse fixed evaluation weights: spec requires PGD=.25 BIM=.15 MIM=.15 worstk=.20, Clean=0.0
        raw_eval_w = getattr(self, "eval_weights", None) or getattr(self.args, "eval_weights", None)
        eval_weights_dict = {}
        if isinstance(raw_eval_w, dict):
            eval_weights_dict = {str(k).lower().strip(): float(v) for k, v in raw_eval_w.items()}
        elif raw_eval_w is not None:
            parsed_eval_w = parse_list_arg(raw_eval_w, float)
            for k, w in zip(eval_attacks, parsed_eval_w):
                eval_weights_dict[k] = float(w)

        default_fixed_weights = {"pgd": 0.25, "bim": 0.15, "mim": 0.15, "worstk": 0.20}
        for k, v in default_fixed_weights.items():
            if k not in eval_weights_dict:
                eval_weights_dict[k] = v

        # Normalize weights over non-clean attacks; Clean weight is strictly 0.0
        w_sum = sum(eval_weights_dict.get(a, 0.0) for a in eval_attacks)
        if w_sum > 0:
            norm_eval_weights = {a: eval_weights_dict.get(a, 0.0) / w_sum for a in eval_attacks}
        else:
            norm_eval_weights = {a: 1.0 / len(eval_attacks) for a in eval_attacks} if eval_attacks else {}
        norm_eval_weights["raw"] = 0.0

        all_results = {}
        all_losses = {}
        all_speeds = {}
        metric_arrays = {}
        self.attack_failures = set()

        self.run_callbacks("on_val_start")

        self.setup_validation_attackers(eval_attacks, model=model, trainer=trainer)

        for run in unique_runs:
            if run != "raw":
                LOGGER.info(
                    f"Running validation for attack: {run} (Eval Weight: {norm_eval_weights.get(run, 0.0):.4f})"
                )
            else:
                LOGGER.info("Running validation for raw images (Eval Weight: 0.0000 [Clean])")

            # Set dataset loading method
            self.current_attack = run

            # Reset dataloader workers to ensure the updated dataset load method is copied
            if hasattr(self.dataloader, "reset"):
                self.dataloader.reset()

            # Reset validation stats and metrics
            if self.training:
                self.loss = {k: torch.zeros_like(v) for k, v in trainer.loss_items.items()}
            self.init_metrics(unwrap_model(model))

            dt = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            bar = TQDM(self.dataloader, desc=self.get_desc() + f" ({run})", total=len(self.dataloader))
            for batch_i, batch in enumerate(bar):
                self.run_callbacks("on_val_batch_start")
                self.batch_i = batch_i
                # Preprocess
                with dt[0]:
                    batch = self.preprocess(batch)

                with autocast(self.training and self.args.quantize == 16, device=self.device.type):
                    # Inference
                    with dt[1]:
                        model.eval()
                        with torch.no_grad():
                            preds = model(batch["img"], augment=augment)

                    # Loss
                    with dt[2]:
                        if self.training and hasattr(self, "loss") and self.loss is not None:
                            for k, v in self.compute_validation_loss(batch, preds, model)[1].items():
                                if k in self.loss:
                                    self.loss[k] += v

                # Postprocess
                with dt[3]:
                    preds = self.postprocess(preds)

                self.update_metrics(preds, batch)
                if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                    self.plot_val_samples(batch, batch_i)
                    self.plot_predictions(batch, preds, batch_i)

                self.run_callbacks("on_val_batch_end")

            self.gather_stats()
            if RANK in {-1, 0}:
                stats = self.get_stats()
                run_speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
                for k, v in run_speed.items():
                    all_speeds[k] = all_speeds.get(k, 0.0) + v

                # Print results for current run
                self.print_results()

                # Save results and metric arrays for this run
                all_results[run] = stats
                metric_arrays[run] = {
                    "p": self.metrics.box.p,
                    "r": self.metrics.box.r,
                    "f1": self.metrics.box.f1,
                    "all_ap": self.metrics.box.all_ap,
                    "ap_class_index": self.metrics.box.ap_class_index,
                }

                if self.training and hasattr(self, "loss") and self.loss is not None:
                    loss = {k: v.clone().detach() for k, v in self.loss.items()}
                    if getattr(trainer, "world_size", 1) > 1:
                        for v in loss.values():
                            dist.reduce(v, dst=0, op=dist.ReduceOp.AVG)
                    all_losses[run] = {k: v.cpu() / len(self.dataloader) for k, v in loss.items()}

        if RANK > 0:
            return

        if RANK in {-1, 0}:
            # Compute weighted metrics/arrays using normalized fixed evaluation weights
            ref_run = eval_attacks[0] if eval_attacks else unique_runs[0]
            weighted_p = np.zeros_like(metric_arrays[ref_run]["p"])
            weighted_r = np.zeros_like(metric_arrays[ref_run]["r"])
            weighted_f1 = np.zeros_like(metric_arrays[ref_run]["f1"])
            weighted_all_ap = np.zeros_like(metric_arrays[ref_run]["all_ap"])

            for a in eval_attacks:
                w = norm_eval_weights.get(a, 0.0)
                weighted_p += w * metric_arrays[a]["p"]
                weighted_r += w * metric_arrays[a]["r"]
                weighted_f1 += w * metric_arrays[a]["f1"]
                weighted_all_ap += w * metric_arrays[a]["all_ap"]

            self.metrics.box.p = weighted_p
            self.metrics.box.r = weighted_r
            self.metrics.box.f1 = weighted_f1
            self.metrics.box.all_ap = weighted_all_ap
            self.metrics.box.ap_class_index = metric_arrays[ref_run]["ap_class_index"]
            self.metrics.box.nc = len(self.names)

            self.speed = {k: v / len(unique_runs) for k, v in all_speeds.items()}
            self.finalize_metrics()

            LOGGER.info("\nValidation Results Summary (Clean + Attacks):")
            for run in unique_runs:
                m50_95 = all_results.get(run, {}).get("metrics/mAP50-95(B)", 0.0)
                m50 = all_results.get(run, {}).get("metrics/mAP50(B)", 0.0)
                w_str = (
                    f"(Eval Weight: {norm_eval_weights.get(run, 0.0):.4f})"
                    if run != "raw"
                    else "(Eval Weight: 0.0000 [Clean])"
                )
                LOGGER.info(f"  {run:10s} -> mAP50-95: {m50_95:.5f}, mAP50: {m50:.5f}  {w_str}")

            self.run_callbacks("on_val_end")

            # Extract final metrics dict
            stats = self.metrics.results_dict

            # Fitness is computed strictly from fixed evaluation weights on attack mAP50-95 (Clean weight=0)
            fitness = sum(
                norm_eval_weights.get(a, 0.0) * all_results.get(a, {}).get("metrics/mAP50-95(B)", 0.0)
                for a in eval_attacks
            )
            stats["fitness"] = float(fitness)
            LOGGER.info(f"fitness (fixed weighted attack mAP50-95, Clean weight=0): {stats['fitness']:.5f}\n")

            # Record individual attack mAPs into stats
            self.last_attack_maps = {}
            for run in unique_runs:
                m50_95 = all_results.get(run, {}).get("metrics/mAP50-95(B)", 0.0)
                self.last_attack_maps[run] = float(m50_95)
                stats[f"metrics/mAP50-95(B)_{run}"] = float(m50_95)
                stats[f"metrics/mAP50(B)_{run}"] = float(all_results.get(run, {}).get("metrics/mAP50(B)", 0.0))

            if self.training:
                # Compute weighted validation loss across attacks
                weighted_loss = {k: torch.zeros_like(v) for k, v in all_losses[ref_run].items()}
                for a in eval_attacks:
                    w = norm_eval_weights.get(a, 0.0)
                    for k, v in all_losses.get(a, {}).items():
                        weighted_loss[k] += w * v

                results = {**stats, **trainer.label_loss_items(weighted_loss, prefix="val")}
                for run in unique_runs:
                    if run in all_losses:
                        for k, v in all_losses[run].items():
                            results[f"val/{k}_{run}"] = float(v)
                return {k: round(float(v), 5) for k, v in results.items()}
            else:
                if self.args.save_json and self.jdict:
                    with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                        LOGGER.info(f"Saving {f.name}...")
                        json.dump(self.jdict, f)
                    stats = self.eval_json(stats)

                if self.args.plots or self.args.save_json:
                    LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
                return stats

    def compute_validation_loss(
        self,
        batch: dict[str, Any],
        preds: Any,
        model: torch.nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute validation loss for the current batch and predictions.

        Args:
            batch (dict): Batch dictionary containing input images and ground truth annotations.
            preds (Any): Model forward predictions.
            model (nn.Module, optional): Model instance to compute loss with. Defaults to self.model.

        Returns:
            (tuple[torch.Tensor, dict[str, torch.Tensor]]): Tuple of (loss, loss_items) as returned by model.loss.
        """
        target = model if model is not None else getattr(self, "model", None)
        if target is None:
            raise ValueError("No model provided or available on validator to compute validation loss.")
        return target.loss(batch, preds)

    def setup_validation_attackers(
        self,
        eval_attacks: list[str],
        model: Any,
        trainer: Any = None,
    ) -> None:
        """Setup validation attackers with unified source model resolution.

        Supports training-time validation (targeting EMA), final evaluation (targeting best.pt),
        and standalone validation (targeting evaluated model), or fixed surrogate weights.

        Args:
            eval_attacks (list[str]): List of attack names to evaluate.
            model (Any): The model being validated (AutoBackend, DetectionModel, or checkpoint).
            trainer (Any, optional): Trainer instance if called during training.
        """
        if getattr(self, "stage", None) == 1 or not eval_attacks:
            self.attackers = {}
            self.attacker = None
            return

        # Determine attack_weights source semantics
        if getattr(self, "attack_weights", None) is None and trainer is not None:
            self.attack_weights = (
                getattr(trainer, "attack_weights", None)
                or getattr(trainer, "_attack_weights_arg", None)
                or getattr(getattr(trainer, "args", None), "attack_weights", None)
            )
        if getattr(self, "attack_weights", None) is None and hasattr(self, "args"):
            self.attack_weights = getattr(self.args, "attack_weights", None)

        att_w = self.attack_weights
        is_current = (att_w is None) or (isinstance(att_w, str) and att_w.strip().lower() in {"", "none", "current"})

        imgsz = getattr(self.args, "imgsz", 640)
        target_device = getattr(self, "device", None)
        if target_device is None:
            target_device = getattr(trainer, "device", None) or torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
            self.device = target_device

        if is_current:
            if trainer is not None:
                source_model = (
                    (getattr(trainer, "ema", None) and getattr(trainer.ema, "ema", None))
                    or getattr(trainer, "model", None)
                    or model
                )
                epoch_val = getattr(trainer, "epoch", -1)
                source_desc = f"trainer.ema/model (epoch {epoch_val})"
            else:
                source_model = model
                model_name = (
                    getattr(model, "pt_path", None)
                    or getattr(getattr(model, "model", None), "yaml_file", None)
                    or type(model).__name__
                )
                source_desc = f"evaluated model ({model_name})"

            raw_detector = extract_detection_model(source_model)
            eval_surrogate = deepcopy(raw_detector).float()
            if hasattr(eval_surrogate, "args") and isinstance(eval_surrogate.args, dict):
                eval_surrogate.args = IterableSimpleNamespace(**eval_surrogate.args)
            elif not hasattr(eval_surrogate, "args") or eval_surrogate.args is None:
                eval_surrogate.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
            for p in eval_surrogate.parameters():
                p.requires_grad_(False)
            eval_surrogate.eval()
            eval_surrogate.to(self.device)
            if hasattr(eval_surrogate, "init_criterion") and getattr(eval_surrogate, "criterion", None) is None:
                try:
                    eval_surrogate.criterion = eval_surrogate.init_criterion()
                except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                    LOGGER.warning(f"Could not pre-initialize criterion on surrogate: {e}")
            target_surrogate = eval_surrogate
            self._fixed_surrogate = None
            self._fixed_surrogate_source = None
            unwrapped_info = f"unwrapped from {type(source_model).__name__} to {type(raw_detector).__name__}"
        else:
            # Fixed surrogate mode: must evaluate on fixed surrogate, cannot fallback to current
            if getattr(self, "_fixed_surrogate", None) is not None and getattr(
                self, "_fixed_surrogate_source", None
            ) == str(att_w):
                target_surrogate = self._fixed_surrogate
                target_surrogate.to(self.device)
                source_desc = f"cached fixed weights ({att_w})"
            else:
                nc = 80
                if isinstance(getattr(self, "data", None), dict):
                    nc = self.data.get("nc", 80)
                target_surrogate = setup_attack_model(
                    att_w,
                    device=self.device,
                    nc=nc,
                    training=False,
                    imgsz=imgsz,
                )
                self._fixed_surrogate = target_surrogate
                self._fixed_surrogate_source = str(att_w)
                source_desc = f"fixed weights ({att_w})"
            unwrapped_info = "fixed_surrogate (no unwrapping)"

        if (
            getattr(self, "attack_through_preprocess", False)
            and getattr(self, "preprocess_mode", "none") != "none"
            and getattr(self, "pipeline", None) is not None
        ):
            LOGGER.info(
                f"Validation attacks configured to pass through defense preprocessing (mode='{self.preprocess_mode}')."
            )
            val_attack_target = PreprocessDetectorWrapper(
                pipeline=self.pipeline,
                detector=target_surrogate,
                mode=self.preprocess_mode,
            )
            target_desc = f"PreprocessDetectorWrapper(mode={self.preprocess_mode}) -> YOLO"
        else:
            LOGGER.info("Validation attacks target YOLO detector directly (defense applied only during inference).")
            val_attack_target = target_surrogate
            target_desc = "pure YOLO detector directly"

        weight_checksum = sum(p.sum().item() for p in target_surrogate.parameters() if p.numel() > 0)
        mode_str = "current (evaluated YOLO)" if is_current else f"fixed surrogate ({att_w})"
        LOGGER.info(
            f"Validation attack source setup: mode={mode_str} | source={source_desc} | "
            f"{unwrapped_info} | target={target_desc} | weight_checksum={weight_checksum:.6f}"
        )

        val_attackers = {}
        for att in eval_attacks:
            val_attackers[att] = build_attacker(att, model=val_attack_target, img_size=imgsz)
        self.attackers = val_attackers
        self.attacker = val_attackers.get(eval_attacks[0], None)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch of images for YOLO validation."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = (batch["img"].half() if self.args.quantize == 16 else batch["img"].float()) / 255

        attack_name = getattr(self, "current_attack", None) or getattr(self, "attack_name", None)
        if attack_name and attack_name.lower().strip() not in {"raw", "none", "clean"}:
            attacker = None
            if hasattr(self, "attackers") and isinstance(self.attackers, dict):
                attacker = self.attackers.get(attack_name, None)
            if attacker is None:
                attacker = getattr(self, "attacker", None)

            if attacker:
                try:
                    im_files = batch.get("im_file", [])
                    if hasattr(attacker, "proxy_model"):
                        attacker.proxy_model.current_paths = im_files
                    if hasattr(attacker, "estimator") and hasattr(attacker.estimator, "model"):
                        attacker.estimator.model.current_paths = im_files

                    if self.device.type == "cuda" and getattr(self.args, "amp", None) == "bf16":
                        amp_ctx = autocast(torch.bfloat16, device=self.device.type)
                    elif self.device.type == "cuda" and (
                        getattr(self.args, "amp", False)
                        or getattr(self.args, "half", False)
                        or getattr(self.args, "quantize", None) == 16
                    ):
                        amp_ctx = autocast(True, device=self.device.type)
                    else:
                        amp_ctx = autocast(False, device=self.device.type)

                    adv_batch_size = int(getattr(self.args, "adv_val_batch", 16))
                    with amp_ctx:
                        imgs_adv = run_attack_on_batch(
                            attacker, batch, label_policy="largest_box", max_batch_size=adv_batch_size
                        )
                    if imgs_adv is not None:
                        imgs_adv_cast = imgs_adv.to(batch["img"].dtype).to(self.device)
                        batch["img"] = imgs_adv_cast
                        del imgs_adv, imgs_adv_cast
                        torch.cuda.empty_cache()
                except Exception as e:
                    self.attack_failures = getattr(self, "attack_failures", set())
                    self.attack_failures.add(attack_name)
                    LOGGER.error(
                        f"Adversarial generation failed for {attack_name}: {e}. Skipping attack for this batch."
                    )
                    if getattr(self.args, "strict_attacks", False):
                        raise

        denoiser = getattr(self, "denoiser", None)
        if denoiser is not None:
            with torch.no_grad():
                den_dtype = next(denoiser.parameters()).dtype
                batch["img"] = denoiser(batch["img"].to(den_dtype)).to(batch["img"].dtype)

        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.seen = 0
        self.jdict = []
        self.metrics.names = model.names
        self.confusion_matrix = ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)

    def get_desc(self) -> str:
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict contains 'bboxes', 'conf',
                'cls', and 'extra' tensors.
        """
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch of images and annotations for validation.

        Args:
            si (int): Batch index.
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prepare predictions for evaluation against ground truth.

        Args:
            pred (dict[str, torch.Tensor]): Post-processed predictions from the model.

        Returns:
            (dict[str, torch.Tensor]): Prepared predictions in native space.
        """
        if self.args.single_cls:
            pred["cls"] *= 0
        return pred

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update metrics with new predictions and ground truth.

        Args:
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            batch (dict[str, Any]): Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    "im_name": Path(pbatch["im_file"]).name,
                }
            )
            # Evaluate
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            if no_pred:
                continue

            # Save
            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def finalize_metrics(self) -> None:
        """Set final values for metrics speed and confusion matrix."""
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def gather_stats(self) -> None:
        """Gather stats from all GPUs."""
        if RANK == 0:
            gathered_stats = [None] * dist.get_world_size()
            dist.gather_object(self.metrics.stats, gathered_stats, dst=0)
            merged_stats = {key: [] for key in self.metrics.stats.keys()}
            for stats_dict in gathered_stats:
                for key in merged_stats:
                    merged_stats[key].extend(stats_dict[key])
            gathered_jdict = [None] * dist.get_world_size()
            dist.gather_object(self.jdict, gathered_jdict, dst=0)
            self.jdict = []
            for jdict in gathered_jdict:
                self.jdict.extend(jdict)
            self.metrics.stats = merged_stats
            self.seen = len(self.dataloader.dataset)  # total image count from dataset
        elif RANK > 0:
            dist.gather_object(self.metrics.stats, None, dst=0)
            dist.gather_object(self.jdict, None, dst=0)
            self.jdict = []
            self.metrics.clear_stats()

    def get_stats(self) -> dict[str, Any]:
        """Calculate and return metrics statistics.

        Returns:
            (dict[str, Any]): Dictionary containing metrics results.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, cannot compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return correct prediction matrix.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing prediction data with 'bboxes' and 'cls' keys.
            batch (dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' and 'cls' keys.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing 'tp' key with correct prediction matrix of shape (N, 10) for
                10 IoU levels.
        """
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """Construct and return dataloader.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): DataLoader for validation.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=self.args.compile,
            pin_memory=self.training,
        )

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot validation image samples.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            ni (int): Batch index.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int, max_det: int | None = None
    ) -> None:
        """Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
            max_det (Optional[int]): Maximum number of detections to plot.
        """
        if not preds:
            return
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # add batch index to predictions
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])  # convert to xywh format
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', and 'cls'.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Serialize YOLO predictions to COCO json format.

        Args:
            predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys with
                bounding box coordinates, confidence scores, and class predictions.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Examples:
             >>> result = {
             ...     "image_id": 42,
             ...     "file_name": "42.jpg",
             ...     "category_id": 18,
             ...     "bbox": [258.15, 41.29, 348.26, 243.78],
             ...     "score": 0.236,
             ... }
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"],
                predn["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Evaluate YOLO output in JSON format and return performance statistics.

        Args:
            stats (dict[str, Any]): Current statistics dictionary.

        Returns:
            (dict[str, Any]): Updated statistics dictionary with COCO/LVIS evaluation results.
        """
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return self.coco_evaluate(stats, pred_json, anno_json)

    def coco_evaluate(
        self,
        stats: dict[str, Any],
        pred_json: str,
        anno_json: str,
        iou_types: str | list[str] = "bbox",
        suffix: str | list[str] = "Box",
    ) -> dict[str, Any]:
        """Evaluate COCO/LVIS metrics using faster-coco-eval library.

        Performs evaluation using the faster-coco-eval library to compute mAP metrics for object detection. Updates the
        provided stats dictionary with computed metrics including mAP50, mAP50-95, and LVIS-specific metrics if
        applicable.

        Args:
            stats (dict[str, Any]): Dictionary to store computed metrics and statistics.
            pred_json (str | Path): Path to JSON file containing predictions in COCO format.
            anno_json (str | Path): Path to JSON file containing ground truth annotations in COCO format.
            iou_types (str | list[str]): IoU type(s) for evaluation. Can be single string or list of strings. Common
                values include "bbox", "segm", "keypoints". Defaults to "bbox".
            suffix (str | list[str]): Suffix to append to metric names in stats dictionary. Should correspond to
                iou_types if multiple types provided. Defaults to "Box".

        Returns:
            (dict[str, Any]): Updated stats dictionary containing the computed COCO/LVIS evaluation metrics.
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
            try:
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
                suffix = [suffix] if isinstance(suffix, str) else suffix
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster

                anno = COCO(anno_json)
                pred = anno.loadRes(pred_json)
                for i, iou_type in enumerate(iou_types):
                    val = COCOeval_faster(
                        anno, pred, iouType=iou_type, lvis_style=self.is_lvis, print_function=LOGGER.info
                    )
                    val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                    # update mAP50-95 and mAP50
                    stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]

                    if self.is_lvis:
                        stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                        stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                        stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]  # always use box mAP50-95 for fitness
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval unable to run: {e}")
        return stats
