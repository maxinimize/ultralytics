# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Staged Adversarial Validation Pipeline with Frozen FGD and STN Preprocessing."""

from __future__ import annotations

from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.defenses.fgd_stn_pipeline import (
    MODE_DESCRIPTIONS,
    FGDSTNPipeline,
    PreprocessDetectorWrapper,
    build_fgd_stn_pipeline,
    canonicalize_mode,
    transform_batch_bboxes_for_stn,
)
from ultralytics.defenses.spatial_transformer import transform_boxes
from ultralytics.models.yolo.detect.val_adv_adaptive import (
    parse_list_arg,
)
from ultralytics.models.yolo.detect.val_adv_fgd_stn_adaptive import (
    DetectionValidatorAdvFGDSTNAdaptive,
)
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, RANK, TQDM, IterableSimpleNamespace, callbacks, ops
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import attempt_compile, autocast, select_device, unwrap_model


class DetectionValidatorAdvFGDSTNStaged(DetectionValidatorAdvFGDSTNAdaptive):
    """Detection validator for Staged Adaptive Adversarial Training with frozen FGD and STN.

    Stage 1:
        Validates Clean images only. Fitness is Clean mAP50-95.
        No adversarial attackers are constructed or evaluated.

    Stage 2:
        Validates Clean and all active attacks (PGD, BIM, MIM, Worst-of-K).
        Fitness is a configurable weighted sum including Clean mAP50-95:
        fitness = 0.20 * Clean + 0.20 * PGD + 0.15 * BIM + 0.15 * MIM + 0.30 * WorstK.
        Attacks pass through the full STN -> FGD -> YOLO pipeline when attack_through_preprocess=True.
        STN bounding boxes are unwarped back to input canvas coordinates before computing metrics.
        Failed attacks raise explicit errors and are never silently replaced with Clean samples.
    """

    def __init__(
        self,
        dataloader=None,
        save_dir: Path | None = None,
        args: dict | Any | None = None,
        _callbacks: dict | None = None,
        pipeline: FGDSTNPipeline | None = None,
        preprocess_mode: str = "stn_fgd",
        fgd_weights: str | Path | None = None,
        stn_weights: str | Path | None = None,
        attack_through_preprocess: bool = True,
        stage: int = 1,
        eval_weights: Sequence[float] | dict[str, float] | None = None,
        strict_attacks: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize staged detection validator with defense pipeline and stage options."""
        raw_mode = preprocess_mode
        raw_fgd_w = fgd_weights
        raw_stn_w = stn_weights
        raw_att_through = attack_through_preprocess
        raw_stage = stage
        raw_strict = strict_attacks
        raw_eval_w = eval_weights

        if isinstance(args, dict):
            raw_mode = args.pop("preprocess_mode", raw_mode)
            raw_fgd_w = args.pop("fgd_weights", raw_fgd_w)
            raw_stn_w = args.pop("stn_weights", raw_stn_w)
            raw_att_through = args.pop("attack_through_preprocess", raw_att_through)
            raw_stage = args.pop("stage", raw_stage)
            raw_strict = args.pop("strict_attacks", raw_strict)
            raw_eval_w = args.pop("eval_weights", raw_eval_w)
        elif args is not None:
            for k in (
                "preprocess_mode",
                "fgd_weights",
                "stn_weights",
                "attack_through_preprocess",
                "stage",
                "strict_attacks",
                "eval_weights",
            ):
                if hasattr(args, k):
                    val = getattr(args, k)
                    if k == "preprocess_mode":
                        raw_mode = val
                    elif k == "fgd_weights":
                        raw_fgd_w = val
                    elif k == "stn_weights":
                        raw_stn_w = val
                    elif k == "attack_through_preprocess":
                        raw_att_through = val
                    elif k == "stage":
                        raw_stage = val
                    elif k == "strict_attacks":
                        raw_strict = val
                    elif k == "eval_weights":
                        raw_eval_w = val
                    try:
                        delattr(args, k)
                    except AttributeError:
                        pass

        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks, **kwargs)

        self.preprocess_mode = canonicalize_mode(raw_mode or "stn_fgd")
        self.fgd_weights = raw_fgd_w
        self.stn_weights = raw_stn_w
        self.attack_through_preprocess = (
            raw_att_through
            if isinstance(raw_att_through, bool)
            else str(raw_att_through).lower() in ("true", "1", "yes")
        )
        self.stage = int(raw_stage) if raw_stage is not None else 1
        self.strict_attacks = (
            raw_strict if isinstance(raw_strict, bool) else str(raw_strict).lower() in ("true", "1", "yes")
        )

        self.eval_weights = raw_eval_w
        self.pipeline: FGDSTNPipeline | None = pipeline
        self.current_pred_matrix: torch.Tensor | None = None
        self._current_hw: tuple[int, int] = (640, 640)
        self.attack_failures: set[str] = set()
        self.last_attack_maps: dict[str, float] = {}

    def _setup_pipeline(self, device: torch.device):
        """Ensure defense pipeline is loaded, verified without fallback, and permanently frozen."""
        if self.pipeline is None:
            # Checkpoint verification: Do not silently fall back to identity preprocessing
            if self.preprocess_mode in {"fgd", "fgd_stn", "stn_fgd"} and (
                not self.fgd_weights or not Path(self.fgd_weights).exists()
            ):
                raise FileNotFoundError(
                    f"FGD checkpoint is required for mode '{self.preprocess_mode}' but was not found at: "
                    f"{self.fgd_weights}. Silent fallback to identity is prohibited."
                )
            if self.preprocess_mode in {"stn", "fgd_stn", "stn_fgd"} and (
                not self.stn_weights or not Path(self.stn_weights).exists()
            ):
                raise FileNotFoundError(
                    f"STN checkpoint is required for mode '{self.preprocess_mode}' but was not found at: "
                    f"{self.stn_weights}. Silent fallback to identity is prohibited."
                )

            self.pipeline, _pipeline_meta = build_fgd_stn_pipeline(
                fgd_weights=self.fgd_weights,
                stn_weights=self.stn_weights,
                device=device,
                allow_identity_fallback=False,
            )

            if self.preprocess_mode in {"fgd", "fgd_stn", "stn_fgd"} and not self.pipeline.has_denoiser:
                raise RuntimeError("FGD denoiser module failed to load from checkpoint!")
            if self.preprocess_mode in {"stn", "fgd_stn", "stn_fgd"} and not self.pipeline.has_stn:
                raise RuntimeError("STN module failed to load from checkpoint!")

        self.pipeline.to(device)
        self.pipeline.eval().requires_grad_(False)
        for p in self.pipeline.parameters():
            p.requires_grad = False

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch: move to device, generate online attack, and apply defense pipeline."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = (batch["img"].half() if self.args.quantize == 16 else batch["img"].float()) / 255.0

        attack_name = getattr(self, "current_attack", None) or getattr(self, "attack_name", None)
        if attack_name and attack_name.lower().strip() not in {"raw", "none", "clean"}:
            attacker = None
            if hasattr(self, "attackers") and isinstance(self.attackers, dict):
                attacker = self.attackers.get(attack_name, None)
            if attacker is None:
                attacker = getattr(self, "attacker", None)

            if attacker is not None:
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
                    has_targets = "bboxes" in batch and batch["bboxes"] is not None and batch["bboxes"].numel() > 0
                    with amp_ctx:
                        if attack_name == "worstk" and callable(getattr(attacker, "forward_batch", None)):
                            attacker.forward_batch(batch)
                            imgs_adv = batch["img"]
                        elif not has_targets:
                            imgs_adv = None
                        else:
                            imgs_adv = run_attack_on_batch(
                                attacker, batch, label_policy="largest_box", max_batch_size=adv_batch_size
                            )

                    if imgs_adv is not None:
                        batch["img"] = imgs_adv.to(batch["img"].dtype).to(self.device)
                    elif has_targets:
                        self.attack_failures.add(attack_name)
                        msg = f"Adversarial generation returned None for attack '{attack_name}' on batch with targets."
                        LOGGER.error(msg)
                        if self.strict_attacks:
                            raise RuntimeError(msg)
                except Exception as e:
                    self.attack_failures.add(attack_name)
                    LOGGER.error(f"Adversarial generation failed for attack '{attack_name}': {e}")
                    if self.strict_attacks:
                        raise
            else:
                self.attack_failures.add(attack_name)
                msg = f"Attacker for '{attack_name}' was not initialized. Cannot substitute clean data."
                LOGGER.error(msg)
                if self.strict_attacks:
                    raise RuntimeError(msg)

        _, _, h, w = batch["img"].shape
        self._current_hw = (h, w)

        # Apply defense preprocessing pipeline according to configured mode
        if self.pipeline is not None:
            with torch.no_grad():
                processed_img, pred_matrix = self.pipeline.forward(batch["img"], mode=self.preprocess_mode)
                batch["img"] = processed_img
                self.current_pred_matrix = pred_matrix
        else:
            self.current_pred_matrix = None

        return batch

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Perform NMS and unwarp bounding boxes back to canvas coordinates if STN was applied."""
        if isinstance(preds, (list, tuple)) and len(preds) > 0 and isinstance(preds[0], dict):
            processed_preds = preds
        else:
            processed_preds = super().postprocess(preds)

        if self.current_pred_matrix is not None and self.pipeline is not None:
            h, w = self._current_hw
            unwarped_preds = []
            for si, pred_dict in enumerate(processed_preds):
                boxes = pred_dict.get("bboxes")
                if boxes is None or boxes.numel() == 0:
                    unwarped_preds.append(pred_dict)
                    continue

                m_si = self.current_pred_matrix[si]
                mapped_boxes = transform_boxes(boxes, m_si, height=h, width=w, clip=True)

                bw = mapped_boxes[:, 2] - mapped_boxes[:, 0]
                bh = mapped_boxes[:, 3] - mapped_boxes[:, 1]
                valid = (bw > 0.0) & (bh > 0.0)

                new_dict = {
                    "bboxes": mapped_boxes[valid],
                    "conf": pred_dict["conf"][valid],
                    "cls": pred_dict["cls"][valid],
                }
                if "extra" in pred_dict and pred_dict["extra"] is not None:
                    new_dict["extra"] = pred_dict["extra"][valid]
                unwarped_preds.append(new_dict)

            return unwarped_preds

        return processed_preds

    def __call__(self, trainer=None, model=None):
        """Execute staged validation process with defense preprocessing and proper fitness computation."""
        if trainer is not None:
            if hasattr(trainer, "stage"):
                self.stage = int(trainer.stage)
            if hasattr(trainer, "preprocess_mode"):
                self.preprocess_mode = canonicalize_mode(trainer.preprocess_mode)
            if hasattr(trainer, "fgd_weights"):
                self.fgd_weights = trainer.fgd_weights
            if hasattr(trainer, "stn_weights"):
                self.stn_weights = trainer.stn_weights
            if hasattr(trainer, "attack_through_preprocess"):
                self.attack_through_preprocess = trainer.attack_through_preprocess
            if hasattr(trainer, "pipeline") and trainer.pipeline is not None:
                self.pipeline = trainer.pipeline
            if hasattr(trainer, "strict_attacks"):
                self.strict_attacks = bool(trainer.strict_attacks)
            if hasattr(trainer, "eval_weights") and trainer.eval_weights is not None:
                self.eval_weights = trainer.eval_weights

        target_device = trainer.device if trainer is not None else getattr(self, "device", None)
        if target_device is None:
            target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._setup_pipeline(target_device)

        LOGGER.info(
            f"Staged FGD+STN Validator (Stage {self.stage}) | Mode: '{self.preprocess_mode}' "
            f"({MODE_DESCRIPTIONS.get(self.preprocess_mode, self.preprocess_mode)}) | "
            f"attack_through_preprocess={self.attack_through_preprocess}"
        )

        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.quantize = 16 if (self.device.type != "cpu" and trainer.amp) else None
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod
            model = model.float()
            self.loss = {k: torch.zeros_like(v) for k, v in trainer.loss_items.items()}
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("Validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device) if RANK == -1 else torch.device("cuda", RANK),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.quantize == 16,
            )
            self.device = model.device
            self.args.quantize = 16 if model.fp16 else None
            stride, fmt = model.stride, model.format
            pt = fmt == "pt"
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if fmt not in {"pt", "torchscript"} and not getattr(model, "dynamic", False):
                self.args.batch = model.metadata.get("batch", 1)

            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(f"Dataset '{self.args.data}' for task={self.args.task} not found")

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0
            if not (pt or (getattr(model, "dynamic", False) and fmt != "imx")):
                self.args.rect = False
            self.stride = model.stride
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
            model.eval()
            if self.args.compile:
                model = attempt_compile(model, device=self.device)
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))

        # Determine validation runs based on stage
        if self.stage == 1:
            # Stage 1: Clean images only
            unique_runs = ["raw"]
            eval_attacks: list[str] = []
        else:
            # Stage 2: Clean + Attacks
            raw_att = (
                getattr(self, "attack_names", None)
                or getattr(self, "attack_name", None)
                or getattr(self.args, "attack_names", None)
                or getattr(self.args, "attack_name", None)
            )
            parsed_attacks = parse_list_arg(raw_att, str)
            if not parsed_attacks:
                parsed_attacks = ["pgd", "bim", "mim", "worstk"]

            unique_runs = ["raw"]
            for att in parsed_attacks:
                att_clean = att.lower().strip()
                if att_clean not in {"raw", "clean", "none"} and att_clean not in unique_runs:
                    unique_runs.append(att_clean)
            eval_attacks = [a for a in unique_runs if a != "raw"]

        # Parse evaluation weights for checkpoint selection
        # Default suggested weights: Clean=0.20, PGD=0.20, BIM=0.15, MIM=0.15, WorstK=0.30
        weights_dict: dict[str, float] = {}
        if self.stage == 1:
            weights_dict["raw"] = 1.0
        else:
            default_weights = {"raw": 0.20, "pgd": 0.20, "bim": 0.15, "mim": 0.15, "worstk": 0.30}
            raw_eval_w = self.eval_weights
            if isinstance(raw_eval_w, dict):
                weights_dict = {str(k).lower().strip(): float(v) for k, v in raw_eval_w.items()}
                if "clean" in weights_dict and "raw" not in weights_dict:
                    weights_dict["raw"] = weights_dict.pop("clean")
            elif raw_eval_w is not None:
                parsed_w = [float(x) for x in parse_list_arg(raw_eval_w, float)]
                if len(parsed_w) == len(unique_runs):
                    # Direct match: [clean, att1, att2, ...]
                    for r, w in zip(unique_runs, parsed_w):
                        weights_dict[r] = w
                elif len(parsed_w) == len(eval_attacks):
                    # Attack-only weights provided: Clean uses default (0.20)
                    weights_dict["raw"] = 0.20
                    for a, w in zip(eval_attacks, parsed_w):
                        weights_dict[a] = w
                else:
                    LOGGER.warning(
                        f"eval_weights length ({len(parsed_w)}) does not match runs ({len(unique_runs)}). Using defaults."
                    )
            for k, v in default_weights.items():
                if k not in weights_dict:
                    weights_dict[k] = v

        # Normalize weights over the runs being evaluated
        w_sum = sum(weights_dict.get(r, 0.0) for r in unique_runs)
        if w_sum > 0:
            norm_eval_weights = {r: weights_dict.get(r, 0.0) / w_sum for r in unique_runs}
        else:
            norm_eval_weights = {r: 1.0 / len(unique_runs) for r in unique_runs}

        # Build validation attackers if needed
        self.attack_failures = set()
        if eval_attacks:
            imgsz = getattr(self.args, "imgsz", 640)
            target_surrogate = None
            if trainer is not None and getattr(trainer, "attack_model", None) is not None:
                target_surrogate = trainer.attack_model
            else:
                # White-box AT / final evaluation: Attack the evaluated model (live/EMA during training, or best.pt during final_eval)
                raw_detector = (
                    model.model
                    if hasattr(model, "model") and isinstance(model.model, nn.Module)
                    else unwrap_model(model)
                )
                eval_surrogate = deepcopy(raw_detector).float()
                if hasattr(eval_surrogate, "args") and isinstance(eval_surrogate.args, dict):
                    eval_surrogate.args = IterableSimpleNamespace(**eval_surrogate.args)
                for p in eval_surrogate.parameters():
                    p.requires_grad_(False)
                eval_surrogate.eval()
                target_surrogate = eval_surrogate

            if self.attack_through_preprocess and self.preprocess_mode != "none":
                val_attack_target = PreprocessDetectorWrapper(
                    pipeline=self.pipeline,
                    detector=target_surrogate,
                    mode=self.preprocess_mode,
                )
            else:
                val_attack_target = target_surrogate

            val_attackers = {}
            for att in eval_attacks:
                val_attackers[att] = build_attacker(att, model=val_attack_target, img_size=imgsz)
            self.attackers = val_attackers
            self.attacker = val_attackers.get(eval_attacks[0], None)
        else:
            self.attackers = {}
            self.attacker = None

        target_model = (trainer.ema.ema or trainer.model) if trainer is not None else model
        orig_loss = None
        if trainer is not None and target_model is not None and hasattr(target_model, "loss"):
            orig_loss = target_model.loss

            def loss_with_stn(batch_in, preds_in=None):
                if self.current_pred_matrix is not None:
                    h, w = self._current_hw
                    batch_loss = {k: v.clone() if torch.is_tensor(v) else copy(v) for k, v in batch_in.items()}
                    batch_loss = transform_batch_bboxes_for_stn(batch_loss, self.current_pred_matrix, height=h, width=w)
                    return orig_loss(batch_loss, preds_in)
                return orig_loss(batch_in, preds_in)

            target_model.loss = loss_with_stn

        all_results = {}
        all_losses = {}
        all_speeds = {}
        metric_arrays = {}

        self.run_callbacks("on_val_start")
        try:
            for run in unique_runs:
                w_disp = norm_eval_weights.get(run, 0.0)
                tag = f"(Eval Weight: {w_disp:.4f})"
                if run == "raw":
                    LOGGER.info(f"Running validation for Clean images {tag}")
                else:
                    LOGGER.info(f"Running validation for attack: '{run}' {tag}")

                self.current_attack = run
                if hasattr(self.dataloader, "reset"):
                    self.dataloader.reset()

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
                    with dt[0]:
                        batch = self.preprocess(batch)

                    with autocast(self.training and self.args.quantize == 16, device=self.device.type):
                        with dt[1]:
                            model.eval()
                            with torch.no_grad():
                                preds = model(batch["img"], augment=augment)

                        with dt[2]:
                            if self.training:
                                for k, v in model.loss(batch, preds)[1].items():
                                    self.loss[k] += v

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

                    self.print_results()
                    all_results[run] = stats
                    metric_arrays[run] = {
                        "p": self.metrics.box.p,
                        "r": self.metrics.box.r,
                        "f1": self.metrics.box.f1,
                        "all_ap": self.metrics.box.all_ap,
                        "ap_class_index": self.metrics.box.ap_class_index,
                    }

                    if self.training:
                        loss = {k: v.clone().detach() for k, v in self.loss.items()}
                        if trainer.world_size > 1:
                            for v in loss.values():
                                dist.reduce(v, dst=0, op=dist.ReduceOp.AVG)
                        all_losses[run] = {k: v.cpu() / len(self.dataloader) for k, v in loss.items()}

        finally:
            if orig_loss is not None and target_model is not None:
                target_model.loss = orig_loss

        if RANK > 0:
            return {}

        # Calculate final fitness and aggregate metrics
        ref_run = unique_runs[0]
        weighted_p = np.zeros_like(metric_arrays[ref_run]["p"])
        weighted_r = np.zeros_like(metric_arrays[ref_run]["r"])
        weighted_f1 = np.zeros_like(metric_arrays[ref_run]["f1"])
        weighted_all_ap = np.zeros_like(metric_arrays[ref_run]["all_ap"])

        for r in unique_runs:
            w = norm_eval_weights.get(r, 0.0)
            weighted_p += w * metric_arrays[r]["p"]
            weighted_r += w * metric_arrays[r]["r"]
            weighted_f1 += w * metric_arrays[r]["f1"]
            weighted_all_ap += w * metric_arrays[r]["all_ap"]

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
            w_str = f"(Weight: {norm_eval_weights.get(run, 0.0):.4f})"
            LOGGER.info(f"  {run:10s} -> mAP50-95: {m50_95:.5f}, mAP50: {m50:.5f}  {w_str}")

        self.run_callbacks("on_val_end")
        stats = self.metrics.results_dict

        # Compute fitness
        if self.stage == 1:
            fitness = float(all_results.get("raw", {}).get("metrics/mAP50-95(B)", 0.0))
            LOGGER.info(f"Stage 1 fitness (Clean mAP50-95): {fitness:.5f}\n")
        else:
            fitness = sum(
                norm_eval_weights.get(r, 0.0) * float(all_results.get(r, {}).get("metrics/mAP50-95(B)", 0.0))
                for r in unique_runs
            )
            LOGGER.info(f"Stage 2 fitness (Weighted Clean + Attacks mAP50-95): {fitness:.5f}\n")

        stats["fitness"] = float(fitness)

        # Record every individual metric
        self.last_attack_maps = {}
        for run in unique_runs:
            m50_95 = float(all_results.get(run, {}).get("metrics/mAP50-95(B)", 0.0))
            m50 = float(all_results.get(run, {}).get("metrics/mAP50(B)", 0.0))
            self.last_attack_maps[run] = m50_95
            stats[f"metrics/mAP50-95(B)_{run}"] = m50_95
            stats[f"metrics/mAP50(B)_{run}"] = m50

        # Check for attack failures
        if self.attack_failures and self.strict_attacks:
            raise RuntimeError(
                f"Adversarial generation failed during validation for attacks: {self.attack_failures}. "
                "Silent fallback is prohibited."
            )

        return stats


DetectionValidator = DetectionValidatorAdvFGDSTNStaged

__all__ = [
    "DetectionValidator",
    "DetectionValidatorAdvFGDSTNStaged",
]
