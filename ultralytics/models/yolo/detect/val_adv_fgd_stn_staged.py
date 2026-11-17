# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Staged Adversarial Validation Pipeline with Frozen FGD and STN Preprocessing."""

from __future__ import annotations

import json
from copy import copy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.attacks.attack_bridge import run_attack_on_batch
from ultralytics.attacks.attack_utils import isolate_rng
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.defenses.fgd_stn_pipeline import (
    MODE_DESCRIPTIONS,
    FGDSTNPipeline,
    build_fgd_stn_pipeline,
    canonicalize_mode,
)
from ultralytics.models.yolo.detect.val_adv_adaptive import (
    parse_list_arg,
)
from ultralytics.models.yolo.detect.val_adv_fgd_stn_adaptive import (
    DetectionValidatorAdvFGDSTNAdaptive,
)
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, RANK, TQDM, callbacks, ops
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
        attack_through_preprocess: bool = False,
        stage: int = 1,
        eval_weights: Sequence[float] | dict[str, float] | None = None,
        strict_attacks: bool = True,
        attack_weights: str | Path | None = None,
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
        raw_attack_weights = attack_weights

        if isinstance(args, dict):
            args = dict(args)
            raw_mode = args.pop("preprocess_mode", raw_mode)
            raw_fgd_w = args.pop("fgd_weights", raw_fgd_w)
            raw_stn_w = args.pop("stn_weights", raw_stn_w)
            raw_att_through = args.pop("attack_through_preprocess", raw_att_through)
            raw_stage = args.pop("stage", raw_stage)
            raw_strict = args.pop("strict_attacks", raw_strict)
            arg_eval_w = args.pop("eval_weights", None)
            if raw_eval_w is None:
                raw_eval_w = arg_eval_w
            raw_attack_weights = args.pop("attack_weights", raw_attack_weights)
            for k in (
                "eval_only",
                "eval_epoch0",
                "freeze_all_bn",
                "freeze_bn",
                "train_head_only",
                "stage1_weights",
                "adaptive_ratio",
                "attack_ratio",
                "attack_num",
                "p_clean",
                "tau",
                "update_interval",
            ):
                args.pop(k, None)
        elif args is not None:
            args = copy(args)
            for k in (
                "preprocess_mode",
                "fgd_weights",
                "stn_weights",
                "attack_through_preprocess",
                "stage",
                "strict_attacks",
                "eval_weights",
                "eval_only",
                "eval_epoch0",
                "freeze_all_bn",
                "freeze_bn",
                "train_head_only",
                "stage1_weights",
                "adaptive_ratio",
                "attack_weights",
                "attack_ratio",
                "attack_num",
                "p_clean",
                "tau",
                "update_interval",
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
                    elif k == "eval_weights" and raw_eval_w is None:
                        raw_eval_w = val
                    elif k == "attack_weights":
                        raw_attack_weights = val
                    try:
                        delattr(args, k)
                    except AttributeError:
                        pass

        super().__init__(
            dataloader=dataloader,
            save_dir=save_dir,
            args=args,
            _callbacks=_callbacks,
            attack_weights=raw_attack_weights,
            strict_attacks=raw_strict,
            eval_weights=raw_eval_w,
            **kwargs,
        )

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
        self.current_pred_matrix = None
        self._current_hw = None
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

                    batch_i = getattr(self, "batch_i", 0)
                    if imgs_adv is not None:
                        if not torch.isfinite(imgs_adv).all():
                            raise RuntimeError(
                                f"Adversarial generation produced NaN or Inf for attack '{attack_name}' at batch {batch_i}."
                            )
                        if imgs_adv.shape != batch["img"].shape:
                            raise RuntimeError(
                                f"Adversarial generation output shape {imgs_adv.shape} mismatch with input {batch['img'].shape} for attack '{attack_name}' at batch {batch_i}."
                            )
                        batch["img"] = imgs_adv.to(batch["img"].dtype).to(self.device)
                    elif has_targets:
                        self.attack_failures.add(attack_name)
                        msg = f"Adversarial generation returned None for attack '{attack_name}' at batch {batch_i} on batch with targets."
                        LOGGER.error(msg)
                        if self.strict_attacks:
                            raise RuntimeError(msg)
                except Exception as e:
                    self.attack_failures.add(attack_name)
                    batch_i = getattr(self, "batch_i", 0)
                    msg = f"Adversarial generation failed for attack '{attack_name}' at batch {batch_i}: {e}"
                    LOGGER.error(msg)
                    if self.strict_attacks:
                        raise RuntimeError(msg) from e
            else:
                batch_i = getattr(self, "batch_i", 0)
                self.attack_failures.add(attack_name)
                msg = f"Attacker for '{attack_name}' was not initialized at batch {batch_i}. Cannot substitute clean data."
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

    def __call__(self, trainer=None, model=None):
        """Execute staged validation process with defense preprocessing and proper fitness computation."""
        if (getattr(self, "stage", None) == 1 or getattr(trainer, "stage", None) == 1) and hasattr(self, "args"):
            self.args.quantize = None
            if hasattr(self.args, "half"):
                self.args.half = False

        val_seed = getattr(self.args, "seed", None)
        if val_seed is None and trainer is not None:
            val_seed = getattr(getattr(trainer, "args", None), "seed", None)
        val_seed = int(val_seed) if val_seed is not None else 0

        active_attackers = []
        if getattr(self, "attacker", None) is not None:
            active_attackers.append(self.attacker)
        if getattr(self, "attackers", None) and isinstance(self.attackers, dict):
            active_attackers.extend(self.attackers.values())
        if trainer is not None:
            if getattr(trainer, "attacker", None) is not None:
                active_attackers.append(trainer.attacker)
            if getattr(trainer, "attackers", None) and isinstance(trainer.attackers, dict):
                active_attackers.extend(trainer.attackers.values())

        dl = getattr(trainer, "train_loader", None) if trainer else None

        with isolate_rng(seed=val_seed, attackers=active_attackers, dataloader=dl):
            return self._execute_validation(trainer=trainer, model=model)

    def record_validation_context(
        self,
        trainer=None,
        model_source: str | None = None,
        requested_model: str | None = None,
    ) -> dict[str, Any]:
        """Record and log validation configuration context and actual runtime parameters."""
        req_m = str(requested_model or getattr(self.args, "model", "AutoBackend"))
        if model_source is None:
            is_training = getattr(self, "training", False) or (trainer is not None)
            if is_training:
                if getattr(trainer, "ema", None) and getattr(trainer.ema, "ema", None):
                    model_source = "trainer.ema"
                elif getattr(trainer, "model", None):
                    model_source = "trainer.model"
                else:
                    model_source = "trainer"
            else:
                model_source = req_m

        # Determine actual precision
        val_precision = "FP32" if self.stage == 1 or getattr(self.args, "quantize", None) != 16 else "FP16"

        stride_val = getattr(self, "stride", None)
        if stride_val is None and trainer is not None:
            stride_val = getattr(trainer, "stride", None)
        if stride_val is None:
            stride_val = 32
        if isinstance(stride_val, (torch.Tensor, np.ndarray, list, tuple)):
            stride_max = int(max(stride_val))
        else:
            stride_max = int(stride_val)
        effective_imgsz = check_imgsz(getattr(self.args, "imgsz", 640), stride=stride_max)
        if isinstance(effective_imgsz, (list, tuple)):
            effective_imgsz = list(effective_imgsz)

        actual_batch = (
            getattr(self.dataloader, "batch_size", None)
            or getattr(getattr(trainer, "test_loader", None), "batch_size", None)
            or getattr(self.args, "batch", 16)
            or 16
        )
        if hasattr(self.dataloader, "dataset"):
            try:
                ds_len = len(self.dataloader.dataset)
                if 0 < ds_len < actual_batch:
                    actual_batch = ds_len
            except (TypeError, AttributeError):
                pass

        data_obj = getattr(self, "data", None)
        if data_obj is None and trainer is not None:
            data_obj = getattr(trainer, "data", None)
        dataset_yaml = (
            data_obj.get("yaml_file", getattr(self.args, "data", "unknown"))
            if isinstance(data_obj, dict)
            else getattr(self.args, "data", "unknown")
        )

        val_seed = getattr(self.args, "seed", None)
        if val_seed is None and trainer is not None:
            val_seed = getattr(getattr(trainer, "args", None), "seed", None)
        val_seed = int(val_seed) if val_seed is not None else 0

        self.last_validation_context = {
            "epoch": getattr(trainer, "epoch", 0) if trainer else 0,
            "stage": self.stage,
            "model_source": str(model_source),
            "requested_model": req_m,
            "dataset": str(dataset_yaml),
            "split": str(self.args.split),
            "requested_imgsz": self.args.imgsz,
            "effective_imgsz": effective_imgsz,
            "requested_batch": getattr(self.args, "batch", 16),
            "actual_batch": int(actual_batch),
            "seed": val_seed,
            "precision": val_precision,
            "preprocess_mode": self.preprocess_mode,
            "attack_through_preprocess": self.attack_through_preprocess,
        }

        LOGGER.info(
            f"Validation Context: Stage={self.stage} | Model={model_source} | Precision={val_precision} | "
            f"Dataset={dataset_yaml} (split='{self.args.split}') | "
            f"Batch: requested={getattr(self.args, 'batch', 16)}, actual={actual_batch} | "
            f"Imgsz: requested={self.args.imgsz}, effective={effective_imgsz} | Seed={val_seed}"
        )

        if getattr(self, "save_dir", None):
            try:
                context_file = Path(self.save_dir) / "validation_context.jsonl"
                with open(context_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(self.last_validation_context) + "\n")
            except (OSError, TypeError, ValueError) as e:
                LOGGER.debug(f"Could not append to validation_context.jsonl: {e}")

        return self.last_validation_context

    def _execute_validation(self, trainer=None, model=None):
        if trainer is not None:
            if hasattr(trainer, "stage"):
                self.stage = int(trainer.stage)
            if hasattr(trainer, "preprocess_mode"):
                self.preprocess_mode = canonicalize_mode(trainer.preprocess_mode)
            if hasattr(trainer, "fgd_weights"):
                self.fgd_weights = trainer.fgd_weights
            if hasattr(trainer, "stn_weights"):
                self.stn_weights = trainer.stn_weights
            if hasattr(trainer, "pipeline") and trainer.pipeline is not None:
                self.pipeline = trainer.pipeline
            if hasattr(trainer, "strict_attacks"):
                self.strict_attacks = bool(trainer.strict_attacks)
            if hasattr(trainer, "eval_weights") and trainer.eval_weights is not None:
                self.eval_weights = trainer.eval_weights

        # Determine actual model source description before model reference is replaced
        requested_model = getattr(self.args, "model", "AutoBackend")
        if trainer is not None:
            if getattr(trainer, "ema", None) and getattr(trainer.ema, "ema", None):
                actual_model_source = "trainer.ema"
            elif getattr(trainer, "model", None):
                actual_model_source = "trainer.model"
            else:
                actual_model_source = "trainer"
        elif model is not None:
            if isinstance(model, (str, Path)):
                actual_model_source = str(Path(model))
            elif hasattr(model, "pt_path") and model.pt_path:
                actual_model_source = str(Path(model.pt_path))
            elif hasattr(model, "yaml_file") and model.yaml_file:
                actual_model_source = str(Path(model.yaml_file))
            elif isinstance(model, torch.nn.Module):
                actual_model_source = model.__class__.__name__
            else:
                actual_model_source = str(model)
        else:
            actual_model_source = str(requested_model)

        target_device = trainer.device if trainer is not None else getattr(self, "device", None)
        if target_device is None and hasattr(self, "args") and getattr(self.args, "device", None) is not None:
            target_device = select_device(self.args.device)
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
            if self.stage == 1:
                # Stage 1: Explicitly unify validation precision to FP32 across all paths
                self.args.quantize = None
            else:
                self.args.quantize = 16 if (self.device.type != "cpu" and getattr(trainer, "amp", False)) else None
            model = (getattr(trainer, "ema", None) and getattr(trainer.ema, "ema", None)) or trainer.model
            if getattr(trainer.args, "compile", False) and hasattr(model, "_orig_mod"):
                model = model._orig_mod
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
            if self.dataloader is None:
                if getattr(trainer, "test_loader", None) is not None:
                    self.dataloader = trainer.test_loader
                else:
                    self.dataloader = self.get_dataloader(self.data.get(self.args.split), self.args.batch)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("Validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            use_fp16 = False if self.stage == 1 else (self.args.quantize == 16)
            model = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device) if RANK == -1 else torch.device("cuda", RANK),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=use_fp16,
            )
            self.device = model.device
            if self.pipeline is not None:
                self.pipeline.to(self.device)
            if self.stage == 1:
                self.args.quantize = None
            else:
                self.args.quantize = 16 if model.fp16 else None
            stride, fmt = model.stride, model.format
            pt = fmt == "pt"
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if fmt not in {"pt", "torchscript"} and not getattr(model, "dynamic", False):
                self.args.batch = model.metadata.get("batch", 1)

            if isinstance(getattr(self, "data", None), dict):
                pass
            elif str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
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
            ch = self.data.get("channels", 3) if isinstance(self.data, dict) else 3
            model.warmup(imgsz=(1 if pt else self.args.batch, ch, imgsz, imgsz))

        self.record_validation_context(
            trainer=trainer,
            model_source=actual_model_source,
            requested_model=requested_model,
        )

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
        self.setup_validation_attackers(eval_attacks, model=model, trainer=trainer)

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
                            if self.training and hasattr(self, "loss") and self.loss is not None:
                                for k, v in self.compute_validation_loss(batch, preds, model)[1].items():
                                    if k in self.loss:
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
                    try:
                        n_samples = len(self.dataloader.dataset)
                    except (TypeError, AttributeError):
                        try:
                            n_samples = len(self.dataloader)
                        except (TypeError, AttributeError):
                            n_samples = 1
                    n_samples = n_samples or 1
                    run_speed = dict(zip(self.speed.keys(), (x.t / n_samples * 1e3 for x in dt)))
                    for k, v in run_speed.items():
                        all_speeds[k] = all_speeds.get(k, 0.0) + v

                    self.print_results()
                    all_results[run] = stats
                    metric_arrays[run] = self._snapshot_metric_arrays()

                    if self.training:
                        loss = {k: v.clone().detach() for k, v in self.loss.items()}
                        if getattr(trainer, "world_size", 1) > 1:
                            for v in loss.values():
                                dist.reduce(v, dst=0, op=dist.ReduceOp.AVG)
                        all_losses[run] = {k: v.cpu() / len(self.dataloader) for k, v in loss.items()}

        finally:
            self.current_pred_matrix = None
            self._current_hw = None

        if RANK > 0:
            return {}

        # Check for attack failures
        if self.attack_failures:
            if self.strict_attacks:
                raise RuntimeError(
                    f"Adversarial generation failed during validation for attacks: {self.attack_failures}. "
                    "Silent fallback is prohibited."
                )
            for f_att in self.attack_failures:
                if f_att in all_results:
                    all_results[f_att]["metrics/mAP50-95(B)"] = 0.0
                    all_results[f_att]["metrics/mAP50(B)"] = 0.0
                    all_results[f_att]["metrics/precision(B)"] = 0.0
                    all_results[f_att]["metrics/recall(B)"] = 0.0

        eval_runs = ["raw"] if self.stage == 1 else unique_runs
        stats = self.aggregate_multirun_metrics(
            metric_arrays=metric_arrays,
            all_results=all_results,
            norm_eval_weights=norm_eval_weights,
            eval_runs=eval_runs,
            failures=self.attack_failures,
        )

        self.speed = {k: v / len(unique_runs) for k, v in all_speeds.items()}
        self.finalize_metrics()

        LOGGER.info("\nValidation Results Summary (Clean + Attacks):")
        for run in unique_runs:
            m50_95 = all_results.get(run, {}).get("metrics/mAP50-95(B)", 0.0)
            m50 = all_results.get(run, {}).get("metrics/mAP50(B)", 0.0)
            w_str = f"(Weight: {norm_eval_weights.get(run, 0.0):.4f})"
            LOGGER.info(f"  {run:10s} -> mAP50-95: {m50_95:.5f}, mAP50: {m50:.5f}  {w_str}")

        self.run_callbacks("on_val_end")

        if self.stage == 1:
            LOGGER.info(f"Stage 1 fitness (Clean mAP50-95): {stats['fitness']:.5f}\n")
        else:
            LOGGER.info(f"Stage 2 fitness (Weighted Clean + Attacks mAP50-95): {stats['fitness']:.5f}\n")

        # Record every individual metric
        self.last_attack_maps = {}
        for run in unique_runs:
            if run in self.attack_failures:
                m50_95 = 0.0
                m50 = 0.0
                stats[f"metrics/attack_failed_{run}"] = 1.0
            else:
                m50_95 = float(all_results.get(run, {}).get("metrics/mAP50-95(B)", 0.0))
                m50 = float(all_results.get(run, {}).get("metrics/mAP50(B)", 0.0))
            self.last_attack_maps[run] = m50_95
            stats[f"metrics/mAP50-95(B)_{run}"] = m50_95
            stats[f"metrics/mAP50(B)_{run}"] = m50

        if hasattr(self.metrics, "_overall_results"):
            self.metrics._overall_results.update(stats)

        return stats


DetectionValidator = DetectionValidatorAdvFGDSTNStaged

__all__ = [
    "DetectionValidator",
    "DetectionValidatorAdvFGDSTNStaged",
]
