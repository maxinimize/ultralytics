# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Staged Adaptive Adversarial Training with Frozen FGD and STN for YOLO."""

from __future__ import annotations

import io
import math
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from ultralytics import __version__
from ultralytics.attacks.attack_bridge import run_attack_on_batch
from ultralytics.attacks.attack_utils import setup_attack_model
from ultralytics.defenses.fgd_stn_pipeline import (
    MODE_DESCRIPTIONS,
    FGDSTNPipeline,
    PreprocessDetectorWrapper,
    build_fgd_stn_pipeline,
    transform_batch_bboxes_for_stn,
)
from ultralytics.models.yolo.detect.train_adv_fgd_stn_adaptive import (
    AdaptiveAttackRatioController,
    DetectionTrainerAdvFGDSTNAdaptive,
)
from ultralytics.models.yolo.detect.val_adv_fgd_stn_staged import (
    DetectionValidatorAdvFGDSTNStaged,
)
from ultralytics.nn.modules.head import OBB, Detect, Pose, Segment
from ultralytics.utils import DEFAULT_CFG, GIT, LOGGER, RANK, TQDM, YAML, colorstr
from ultralytics.utils.torch_utils import (
    autocast,
    convert_optimizer_state_dict_to_fp16,
    qat_state,
    strip_qat,
    unset_deterministic,
    unwrap_model,
)

HEAD_CLASSES = (Detect, Segment, Pose, OBB)


def find_detection_head(model: nn.Module) -> tuple[nn.Module | None, int | None]:
    """Identify the Detection Head submodule dynamically without assuming fixed layer indices.

    Args:
        model (nn.Module): The detector model (can be unwrapped or wrapped).

    Returns:
        tuple[Optional[nn.Module], Optional[int]]: The head module and its sequential index if found.
    """
    unwrapped = unwrap_model(model)
    head_module = None
    head_idx = None

    if hasattr(unwrapped, "model") and isinstance(unwrapped.model, (nn.Sequential, nn.ModuleList)):
        for idx, layer in enumerate(unwrapped.model):
            if isinstance(layer, HEAD_CLASSES):
                head_module = layer
                head_idx = idx
                break

    if head_module is None:
        for m in unwrapped.modules():
            if isinstance(m, HEAD_CLASSES):
                head_module = m
                break

    return head_module, head_idx


class DetectionTrainerAdvFGDSTNStaged(DetectionTrainerAdvFGDSTNAdaptive):
    """Two-Stage Training Pipeline for YOLO with Pretrained Frozen FGD and STN.

    Stage 1 (Clean Fine-tuning):
        - Train only Detection Head on Clean images with frozen Backbone, Neck, FGD, and STN.
        - Evaluates Clean mAP50 and mAP50-95 after each epoch.
        - Selects best.pt using Clean mAP50-95.
        - No adversarial samples or attackers are constructed.

    Stage 2 (Adaptive Adversarial Training):
        - Initialized from Stage 1 best.pt as a fresh training run (fresh optimizer and scheduler).
        - Clean ratio fixed at 0.65; adversarial ratios (PGD 0.12, BIM 0.05, MIM 0.10, WorstK 0.08)
          dynamically adjust every 5 epochs using softmax degradation summing to 0.35.
        - Evaluates Clean, PGD, BIM, MIM, and WorstK after each epoch.
        - Selects best.pt using weighted validation fitness:
          fitness = 0.20 * Clean + 0.20 * PGD + 0.15 * BIM + 0.15 * MIM + 0.30 * WorstK.
        - Supports attack_through_preprocess=True/False.
    """

    pipeline_defaults: ClassVar[dict] = {
        "stage": 1,
        "stage1_weights": None,
        "preprocess_mode": "stn_fgd",
        "fgd_weights": "fgd_yolo12l.pt",
        "stn_weights": "runs/detect/runs/train_stn/stn_train_yolo12l_coco_train_traffic5k_ep50_b8_k10_deg30.0_tr0.10/weights/best.pt",
        "train_head_only": True,
        "freeze_bn": True,
        "adaptive_ratio": True,
        "attack_through_preprocess": True,
        "p_clean": 1.0,
        "tau": 5.0,
        "update_interval": 5,
        "eval_weights": None,
        "attack_name": None,
        "attack_ratio": None,
        "attack_weights": "current",
        "strict_attacks": True,
    }

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
        attack_weights="",
    ):
        """Initialize DetectionTrainerAdvFGDSTNStaged with stage-specific defaults."""
        if overrides is None:
            overrides = {}

        # Parse saved args if resuming
        saved_args = {}
        resume = overrides.get("resume", False)
        if resume and isinstance(resume, (str, Path)) and str(resume).endswith(".pt"):
            args_yaml = Path(resume).resolve().parent.parent / "args.yaml"
            if args_yaml.exists():
                try:
                    import yaml

                    with open(args_yaml, "r", encoding="utf-8") as f:
                        saved_args = yaml.safe_load(f) or {}
                except (OSError, yaml.YAMLError) as e:
                    LOGGER.warning(f"Could not load saved configuration from {args_yaml}: {e}")

        # Stage configuration
        raw_stage = overrides.pop("stage", saved_args.get("stage", 1))
        self.stage = int(raw_stage)

        raw_stage1_w = overrides.pop("stage1_weights", saved_args.get("stage1_weights", None))
        self.stage1_weights = str(raw_stage1_w) if raw_stage1_w else None

        raw_train_head_only = overrides.pop("train_head_only", saved_args.get("train_head_only", True))
        self.train_head_only = (
            raw_train_head_only
            if isinstance(raw_train_head_only, bool)
            else str(raw_train_head_only).lower() in ("true", "1", "yes")
        )

        raw_freeze_bn = overrides.pop("freeze_bn", saved_args.get("freeze_bn", True))
        self.freeze_bn = (
            raw_freeze_bn if isinstance(raw_freeze_bn, bool) else str(raw_freeze_bn).lower() in ("true", "1", "yes")
        )

        raw_strict = overrides.pop("strict_attacks", saved_args.get("strict_attacks", True))
        self.strict_attacks = (
            raw_strict if isinstance(raw_strict, bool) else str(raw_strict).lower() in ("true", "1", "yes")
        )

        # Stage-specific defaults
        if self.stage == 1:
            stage_p_clean = 1.0
            stage_att_names: list[str] = []
            stage_att_ratios: list[float] = []
            stage_adaptive = False
            stage_eval_w = [1.0]
            stage_epochs = 5
        else:
            stage_p_clean = 0.65
            stage_att_names = ["pgd", "bim", "mim", "worstk"]
            stage_att_ratios = [0.12, 0.05, 0.10, 0.08]
            stage_adaptive = True
            stage_eval_w = [0.20, 0.20, 0.15, 0.15, 0.30]
            stage_epochs = 20

            # If stage1_weights is provided and resume is not set, use stage1_weights as base model
            if self.stage1_weights and not overrides.get("resume"):
                overrides["model"] = self.stage1_weights

        overrides.setdefault("preprocess_mode", saved_args.get("preprocess_mode", "stn_fgd"))
        overrides.setdefault("fgd_weights", saved_args.get("fgd_weights", "fgd_yolo12l.pt"))
        overrides.setdefault("stn_weights", saved_args.get("stn_weights", None))
        overrides.setdefault("adaptive_ratio", saved_args.get("adaptive_ratio", stage_adaptive))
        overrides.setdefault("attack_through_preprocess", saved_args.get("attack_through_preprocess", True))
        overrides.setdefault("p_clean", saved_args.get("p_clean", stage_p_clean))
        overrides.setdefault("tau", saved_args.get("tau", 5.0))
        overrides.setdefault("update_interval", saved_args.get("update_interval", 5))
        overrides.setdefault("eval_weights", saved_args.get("eval_weights", stage_eval_w))
        overrides.setdefault("attack_name", saved_args.get("attack_name", stage_att_names))
        overrides.setdefault("attack_ratio", saved_args.get("attack_ratio", stage_att_ratios))
        overrides.setdefault("epochs", saved_args.get("epochs", stage_epochs))
        overrides.setdefault("lr0", saved_args.get("lr0", 1e-5))
        overrides.setdefault("optimizer", saved_args.get("optimizer", "AdamW"))
        overrides.setdefault("warmup_bias_lr", saved_args.get("warmup_bias_lr", 0.0))

        # Attack source
        raw_att_w = overrides.get("attack_weights", attack_weights)
        if raw_att_w is not None and str(raw_att_w).strip().lower() in {"", "none", "current"}:
            overrides["attack_weights"] = "current"
            effective_attack_weights = "current"
        else:
            effective_attack_weights = raw_att_w or "current"

        super().__init__(cfg, overrides, _callbacks, attack_weights=effective_attack_weights)

        # Store staged attributes
        self.args.stage = self.stage
        self.args.stage1_weights = self.stage1_weights
        self.args.train_head_only = self.train_head_only
        self.args.freeze_bn = self.freeze_bn
        self.args.strict_attacks = self.strict_attacks

        # Re-ensure stage-specific controller configuration
        if self.stage == 1:
            self.attack_names = []
            self.attack_ratios = []
            self.p_clean = 1.0
            self.adaptive_ratio = False
            self.adaptive_controller = None
            self.args.attack_name = []
            self.args.attack_ratio = []
            self.args.p_clean = 1.0
            self.args.adaptive_ratio = False
        self.args.eval_weights = self.eval_weights
        self.args.attack_name = self.attack_names
        self.args.attack_ratio = self.attack_ratios
        self.args.strict_attacks = self.strict_attacks

        # Configure controller
        if self.stage == 2 and self.attack_names:
            self.adaptive_controller = AdaptiveAttackRatioController(
                attack_names=self.attack_names,
                initial_attack_ratios=self.attack_ratios,
                p_clean=self.p_clean,
                tau=self.tau,
                update_interval=self.update_interval,
                eval_weights=self.eval_weights,
                save_dir=self.save_dir,
                adaptive_ratio=self.adaptive_ratio,
            )
        else:
            self.adaptive_controller = None

        self.pipeline: FGDSTNPipeline | None = None
        self.pipeline_meta: dict[str, Any] = {}

    def set_model_attributes(self):
        """Build frozen defense pipeline and configure stage-specific attack generator."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

        # 1. Build and verify Unified Defense Pipeline (Never silently fall back to identity)
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

        self.pipeline, self.pipeline_meta = build_fgd_stn_pipeline(
            fgd_weights=self.fgd_weights,
            stn_weights=self.stn_weights,
            device=self.device,
            allow_identity_fallback=False,
        )

        if self.preprocess_mode in {"fgd", "fgd_stn", "stn_fgd"} and not self.pipeline.has_denoiser:
            raise RuntimeError(f"FGD denoiser failed to load from '{self.fgd_weights}'!")
        if self.preprocess_mode in {"stn", "fgd_stn", "stn_fgd"} and not self.pipeline.has_stn:
            raise RuntimeError(f"STN module failed to load from '{self.stn_weights}'!")

        self.pipeline.to(self.device)
        self.pipeline.eval().requires_grad_(False)
        for p in self.pipeline.parameters():
            p.requires_grad = False

        LOGGER.info(
            f"Defense Pipeline Initialized (Stage {self.stage}): Mode='{self.preprocess_mode}' "
            f"({MODE_DESCRIPTIONS.get(self.preprocess_mode, self.preprocess_mode)}) | "
            f"FGD: {'Loaded' if self.pipeline.has_denoiser else 'None'}, "
            f"STN: {'Loaded' if self.pipeline.has_stn else 'None'}"
        )

        # 2. Configure layer freezing for train_head_only
        if getattr(self, "train_head_only", True):
            unwrapped = unwrap_model(self.model)
            head_module, _head_idx = find_detection_head(unwrapped)
            if head_module is None:
                raise RuntimeError("Could not dynamically identify Detection Head in model structure!")
            if hasattr(unwrapped, "model") and isinstance(unwrapped.model, (nn.Sequential, nn.ModuleList)):
                self.args.freeze = [i for i, layer in enumerate(unwrapped.model) if not isinstance(layer, HEAD_CLASSES)]

        # 3. Stage-specific attacker configuration
        if self.stage == 1:
            LOGGER.info("Stage 1 Clean Fine-tuning: Skipping adversarial attacker initialization.")
            self.attack_model = None
            self.attackers = {}
            self.attacker = None
            return

        # Stage 2: Setup attackers
        attack_weights = getattr(self.args, "attack_weights", "current")
        use_current_model = (attack_weights is None) or (str(attack_weights).strip().lower() in {"", "none", "current"})

        imgsz = getattr(self.args, "imgsz", 640)
        if use_current_model:
            LOGGER.info("Stage 2 AT: Using CURRENT training model as attack generator (white-box AT).")
            self.attack_model = None
            target_detector = unwrap_model(self.model)
        else:
            LOGGER.info(f"Stage 2 AT: Loading fixed surrogate attack model from: {attack_weights}")
            self.attack_model = setup_attack_model(
                attack_weights,
                device=self.device,
                nc=self.data["nc"],
                training=False,
                imgsz=imgsz,
            )
            self.attack_model.eval()
            target_detector = self.attack_model

        if self.attack_through_preprocess and self.preprocess_mode != "none":
            LOGGER.info(f"Wrapping attack detector with PreprocessDetectorWrapper (mode='{self.preprocess_mode}')")
            target_attack_model = PreprocessDetectorWrapper(
                pipeline=self.pipeline,
                detector=target_detector,
                mode=self.preprocess_mode,
            )
        else:
            LOGGER.info("Attacks generate directly on detector without preprocessing wrapper.")
            target_attack_model = target_detector

        self.attackers = {}
        for att in self.attack_names:
            if att.lower().strip() not in {"raw", "none", "clean"}:
                self.attackers[att] = self.build_attacker(att, target_attack_model, imgsz)
                LOGGER.info(f"Attacker initialized: {att}")

        self.attacker = self.attackers.get(self.attack_names[0], None) if self.attackers else None

    def _setup_train(self):
        """Freeze non-head parameters and verify parameter isolation before training."""
        super()._setup_train()

        # Re-verify parameter freezing after _setup_train
        if self.train_head_only:
            unwrapped = unwrap_model(self.model)
            head_module, _ = find_detection_head(unwrapped)
            head_param_ids = {id(p) for p in head_module.parameters()} if head_module is not None else set()
            for k, v in self.model.named_parameters():
                if id(v) not in head_param_ids or ".dfl." in k:
                    v.requires_grad = False
                else:
                    v.requires_grad = True

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            assert len(trainable_params) > 0, "Detection Head has no trainable parameters! Check layer identification."
            assert all(id(p) in head_param_ids for p in trainable_params), (
                "Trainable parameters leak outside Detection Head!"
            )

            LOGGER.info(
                f"Model parameter freezing verified: Train Head Only = {self.train_head_only}. "
                f"Frozen params: {len(list(self.model.parameters())) - len(trainable_params)}, "
                f"Trainable head params: {len(trainable_params)}."
            )

        # Defense pipeline parameter isolation check
        if self.pipeline is not None:
            assert not any(p.requires_grad for p in self.pipeline.parameters()), (
                "Defense pipeline parameters must not require gradients!"
            )
            assert not self.pipeline.training, "Defense pipeline must be in eval mode!"

            pipeline_ids = {id(p) for p in self.pipeline.parameters()}
            assert not any(id(p) in pipeline_ids for group in self.optimizer.param_groups for p in group["params"]), (
                "Defense pipeline parameters must NOT be present in optimizer param groups!"
            )

        # Optimizer group check
        for group in self.optimizer.param_groups:
            assert all(p.requires_grad for p in group["params"]), (
                "Frozen parameters must NOT be present in optimizer param groups!"
            )
            if self.train_head_only:
                assert all(id(p) in head_param_ids for p in group["params"]), (
                    "Optimizer contains parameters outside Detection Head!"
                )

        config_record = {
            "stage": self.stage,
            "stage1_weights": str(self.stage1_weights) if self.stage1_weights else None,
            "train_head_only": self.train_head_only,
            "freeze_bn": self.freeze_bn,
            "preprocess_mode": self.preprocess_mode,
            "preprocess_mode_description": MODE_DESCRIPTIONS.get(self.preprocess_mode, self.preprocess_mode),
            "fgd_weights": str(self.fgd_weights) if self.fgd_weights else None,
            "stn_weights": str(self.stn_weights) if self.stn_weights else None,
            "adaptive_ratio": self.adaptive_ratio,
            "attack_through_preprocess": self.attack_through_preprocess,
            "p_clean": self.p_clean,
            "tau": self.tau,
            "update_interval": self.update_interval,
            "eval_weights": self.eval_weights,
            "attack_names": self.attack_names,
            "attack_ratios": self.attack_ratios,
            "yolo_model": str(self.args.model),
            "imgsz": self.args.imgsz,
            "epochs": self.args.epochs,
            "batch": self.args.batch,
            "lr0": self.args.lr0,
            "optimizer": self.args.optimizer,
            "amp": self.args.amp,
        }

        if RANK in {-1, 0}:
            YAML.save(self.save_dir / "staged_config.yaml", config_record)
            YAML.save(self.save_dir / "args.yaml", vars(self.args))
            LOGGER.info(f"Staged Training Setup Summary (Stage {self.stage}):")
            for k, v in config_record.items():
                LOGGER.info(f"  {k}: {v}")

    def build_optimizer(self, model, **kwargs):
        """Ensure optimizer groups only contain trainable YOLO parameters, excluding frozen defenses."""
        optimizer = super().build_optimizer(model, **kwargs)
        pipeline_param_ids = {id(p) for p in self.pipeline.parameters()} if self.pipeline is not None else set()

        unwrapped = unwrap_model(model)
        head_module, _ = find_detection_head(unwrapped)
        head_param_ids = {id(p) for p in head_module.parameters()} if head_module is not None else set()

        for group in optimizer.param_groups:
            filtered = []
            for p in group["params"]:
                if id(p) in pipeline_param_ids:
                    continue
                if not p.requires_grad:
                    continue
                if self.train_head_only and id(p) not in head_param_ids:
                    continue
                filtered.append(p)
            group["params"] = filtered

        return optimizer

    def _model_train(self):
        """Set model in training mode while keeping frozen layers and frozen BN in eval mode."""
        self.model.train()
        if getattr(self, "freeze_bn", True):
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                    mod_params = list(module.parameters())
                    if (not mod_params or not any(p.requires_grad for p in mod_params)) or any(
                        f in name for f in getattr(self, "freeze_layer_names", [])
                    ):
                        module.eval()

        if self.pipeline is not None:
            self.pipeline.eval().requires_grad_(False)

    def get_validator(self):
        """Return DetectionValidatorAdvFGDSTNStaged configured for active stage."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"

        validator_args = copy(self.args)
        for arg in [
            "attack_weights",
            "attack_name",
            "attack_ratio",
            "attack_num",
            "p_clean",
            "tau",
            "update_interval",
            "eval_weights",
            "preprocess_mode",
            "fgd_weights",
            "stn_weights",
            "adaptive_ratio",
            "attack_through_preprocess",
            "stage",
            "stage1_weights",
            "train_head_only",
            "freeze_bn",
            "strict_attacks",
        ]:
            if hasattr(validator_args, arg):
                delattr(validator_args, arg)

        validator = DetectionValidatorAdvFGDSTNStaged(
            getattr(self, "test_loader", None),
            save_dir=self.save_dir,
            args=validator_args,
            _callbacks=self.callbacks,
            pipeline=self.pipeline,
            preprocess_mode=self.preprocess_mode,
            fgd_weights=self.fgd_weights,
            stn_weights=self.stn_weights,
            attack_through_preprocess=self.attack_through_preprocess,
            stage=self.stage,
            eval_weights=self.eval_weights,
            strict_attacks=self.strict_attacks,
        )
        validator.attacker = getattr(self, "attacker", None)
        validator.attackers = getattr(self, "attackers", {})
        validator.attack_names = self.attack_names
        validator.attack_ratios = self.attack_ratios
        validator.eval_weights = self.eval_weights
        return validator

    def _load_checkpoint_state(self, ckpt: dict[str, Any]):
        """Load optimizer, scaler, EMA, and stage metadata from checkpoint."""
        super()._load_checkpoint_state(ckpt)
        if ckpt:
            if "stage" in ckpt:
                self.stage = int(ckpt["stage"])
                self.args.stage = self.stage
            if "stage1_weights" in ckpt:
                self.stage1_weights = ckpt["stage1_weights"]
                self.args.stage1_weights = self.stage1_weights
            if "train_head_only" in ckpt:
                self.train_head_only = bool(ckpt["train_head_only"])
                self.args.train_head_only = self.train_head_only
            if "freeze_bn" in ckpt:
                self.freeze_bn = bool(ckpt["freeze_bn"])
                self.args.freeze_bn = self.freeze_bn
            if "eval_weights" in ckpt:
                self.eval_weights = ckpt["eval_weights"]
                self.args.eval_weights = self.eval_weights

    def save_model(self):
        """Save model training checkpoints preserving staged metadata and controller state."""
        ema = self.ema.ema
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            model_sd = unwrap_model(self.model).state_dict()
            for k, v in ema.state_dict().items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all() and torch.isfinite(model_sd[k]).all():
                    v.copy_(model_sd[k])

        modelopt = qat_state(ema)
        ema = deepcopy(ema).half().to(memory_format=torch.contiguous_format)
        if hasattr(ema, "criterion"):
            ema.criterion = None
        for v in ema.state_dict().values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                torch.nan_to_num_(v)
        strip_qat(ema)

        live_model = deepcopy(unwrap_model(self.model)).half().to(memory_format=torch.contiguous_format)
        if hasattr(live_model, "criterion"):
            live_model.criterion = None
        for v in live_model.state_dict().values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                torch.nan_to_num_(v)
        strip_qat(live_model)

        ckpt_data = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": live_model,
            "ema": ema,
            "updates": self.ema.updates,
            "modelopt": modelopt,
            "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            "scaler": self.scaler.state_dict(),
            "train_args": vars(self.args),
            "train_metrics": {**self.metrics, "fitness": self.fitness},
            "train_results": self.read_results_csv(),
            "stage": self.stage,
            "stage1_weights": str(self.stage1_weights) if self.stage1_weights else None,
            "train_head_only": self.train_head_only,
            "freeze_bn": self.freeze_bn,
            "preprocess_mode": self.preprocess_mode,
            "fgd_weights": str(self.fgd_weights) if self.fgd_weights else None,
            "stn_weights": str(self.stn_weights) if self.stn_weights else None,
            "adaptive_ratio": self.adaptive_ratio,
            "attack_through_preprocess": self.attack_through_preprocess,
            "eval_weights": self.eval_weights,
            "date": datetime.now().astimezone().isoformat(),
            "version": __version__,
            "git": {
                "root": str(GIT.root),
                "branch": GIT.branch,
                "commit": GIT.commit,
                "message": GIT.message,
                "origin": GIT.origin,
            },
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }

        if self.adaptive_controller is not None:
            ckpt_data["adaptive_state"] = self.adaptive_controller.state_dict()

        buffer = io.BytesIO()
        torch.save(ckpt_data, buffer)
        serialized_ckpt = buffer.getvalue()

        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)
        return True

    def _do_train(self):
        """Train the model with defense preprocessing, supporting Stage 1 and Stage 2."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting Stage {self.stage} Training for {self.epochs} epochs "
            f"(mode='{self.preprocess_mode}', head_only={self.train_head_only})..."
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()

        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self._model_train()

            # Dynamic attack ratio allocation update for Stage 2
            base_seed = getattr(self.args, "seed", 0) or 42
            epoch_seed = base_seed + epoch
            needs_reset = False
            if (
                self.stage == 2
                and self.adaptive_controller is not None
                and ((epoch == self.start_epoch and self.resume) or epoch > self.start_epoch)
                and hasattr(self.train_loader.dataset, "update_virtual_samples")
            ):
                self.train_loader.dataset.update_virtual_samples(
                    attack_ratios=self.adaptive_controller.current_train_ratios,
                    seed=epoch_seed,
                )
                needs_reset = True

            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                needs_reset = True

            if needs_reset and hasattr(self.train_loader, "reset"):
                self.train_loader.reset()

            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None

            self.group_tloss = {"raw": None}
            self.group_batch_count = {"raw": 0}
            self.epoch_counts = {"raw": 0}
            for att in self.attack_names:
                self.group_tloss[att] = None
                self.group_batch_count[att] = 0
                self.epoch_counts[att] = 0

            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        x["lr"] = float(
                            np.interp(
                                ni,
                                xi,
                                [
                                    self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                    x["initial_lr"] * self.lf(epoch),
                                ],
                            )
                        )
                        if "momentum" in x:
                            x["momentum"] = float(np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum]))

                # Forward
                amp_dtype = torch.bfloat16 if getattr(self.args, "amp", None) == "bf16" else self.amp
                with autocast(amp_dtype, device=self.device.type):
                    batch = self.preprocess_batch(batch)

                    for st in batch.get("sample_type", []):
                        if st in self.epoch_counts:
                            self.epoch_counts[st] += 1

                    sub_batches = self.split_batch_by_type(batch)

                    # 1. Generate adversarial images on-the-fly for Stage 2
                    if getattr(self, "attackers", None) and self.stage == 2:
                        is_current_model = getattr(self, "attack_model", None) is None
                        if is_current_model:
                            param_grad_states = [p.requires_grad for p in self.model.parameters()]
                            for p in self.model.parameters():
                                p.requires_grad = False

                        try:
                            for att, attacker in self.attackers.items():
                                if att in sub_batches:
                                    sub_batch = sub_batches[att]
                                    has_targets = (
                                        "bboxes" in sub_batch
                                        and sub_batch["bboxes"] is not None
                                        and sub_batch["bboxes"].numel() > 0
                                    )
                                    if not has_targets and att != "worstk":
                                        continue

                                    try:
                                        if hasattr(attacker, "proxy_model"):
                                            attacker.proxy_model.current_paths = sub_batch.get("im_file", [])
                                        if hasattr(attacker, "estimator") and hasattr(attacker.estimator, "model"):
                                            attacker.estimator.model.current_paths = sub_batch.get("im_file", [])
                                        if self.attack_model is not None and hasattr(
                                            self.attack_model, "current_paths"
                                        ):
                                            self.attack_model.current_paths = sub_batch.get("im_file", [])

                                        if getattr(self.args, "amp", None) == "bf16":
                                            with autocast(torch.bfloat16, device=self.device.type):
                                                if att == "worstk" and callable(
                                                    getattr(attacker, "forward_batch", None)
                                                ):
                                                    attacker.forward_batch(sub_batch)
                                                    imgs_adv = sub_batch["img"]
                                                else:
                                                    imgs_adv = run_attack_on_batch(
                                                        attacker, sub_batch, label_policy="largest_box"
                                                    )
                                        else:
                                            with torch.amp.autocast(device_type="cuda", enabled=False):
                                                if att == "worstk" and callable(
                                                    getattr(attacker, "forward_batch", None)
                                                ):
                                                    attacker.forward_batch(sub_batch)
                                                    imgs_adv = sub_batch["img"]
                                                else:
                                                    imgs_adv = run_attack_on_batch(
                                                        attacker, sub_batch, label_policy="largest_box"
                                                    )

                                        if imgs_adv is not None:
                                            sub_batch["img"] = (
                                                imgs_adv.detach().to(sub_batch["img"].dtype).to(self.device)
                                            )
                                            del imgs_adv
                                        elif has_targets:
                                            msg = f"Adversarial generation returned None for {att} on a batch with targets."
                                            if self.strict_attacks:
                                                raise RuntimeError(msg)
                                            LOGGER.warning(msg)
                                    except Exception as e:
                                        if self.strict_attacks:
                                            raise RuntimeError(
                                                f"Adversarial generation failed during training for {att}: {e}"
                                            ) from e
                                        LOGGER.warning(f"Adversarial generation failed for {att}: {e}.")
                        finally:
                            if is_current_model:
                                for p, state in zip(self.model.parameters(), param_grad_states):
                                    p.requires_grad = state

                    # 2. Recombine sub-batches into full batch
                    has_spatial_attack = any(
                        getattr(attacker, "transforms_targets", False)
                        for attacker in getattr(self, "attackers", {}).values()
                    )

                    if not has_spatial_attack:
                        for att, sub_batch in sub_batches.items():
                            if att != "raw" and "img" in sub_batch:
                                indices = [idx for idx, st in enumerate(batch["sample_type"]) if st == att]
                                if indices:
                                    batch["img"][indices] = sub_batch["img"]
                    else:
                        new_batch_idx = []
                        new_cls = []
                        new_bboxes = []
                        orig_bi_dim = (
                            batch["batch_idx"].dim() if "batch_idx" in batch and batch["batch_idx"] is not None else 1
                        )
                        orig_cls_dim = batch["cls"].dim() if "cls" in batch and batch["cls"] is not None else 1

                        for t, sub_batch in sub_batches.items():
                            indices = [idx for idx, st in enumerate(batch["sample_type"]) if st == t]
                            if not indices:
                                continue
                            batch["img"][indices] = sub_batch["img"]

                            sub_bi = sub_batch.get("batch_idx")
                            if sub_bi is not None and sub_bi.numel() > 0:
                                sub_bi = sub_bi.view(-1).long()
                                indices_tensor = torch.tensor(indices, device=sub_bi.device, dtype=torch.long)
                                global_bi = indices_tensor[sub_bi]
                                new_batch_idx.append(global_bi)
                                new_cls.append(
                                    sub_batch["cls"].view(-1, 1) if sub_batch["cls"].ndim == 1 else sub_batch["cls"]
                                )
                                new_bboxes.append(
                                    sub_batch["bboxes"].view(-1, 4)
                                    if sub_batch["bboxes"].ndim != 2
                                    else sub_batch["bboxes"]
                                )

                        if new_batch_idx:
                            cat_bi = torch.cat(new_batch_idx, dim=0)
                            cat_cls = torch.cat(new_cls, dim=0)
                            cat_box = torch.cat(new_bboxes, dim=0)

                            bi_flat = cat_bi.view(-1)
                            sort_idx = torch.argsort(bi_flat)
                            cat_bi = cat_bi[sort_idx]
                            cat_cls = cat_cls[sort_idx]
                            cat_box = cat_box[sort_idx]

                            if orig_bi_dim > 1 and cat_bi.dim() == 1:
                                cat_bi = cat_bi.unsqueeze(-1)
                            batch["batch_idx"] = cat_bi

                            if orig_cls_dim == 1 and cat_cls.dim() > 1:
                                cat_cls = cat_cls.squeeze(-1)
                            elif orig_cls_dim > 1 and cat_cls.dim() == 1:
                                cat_cls = cat_cls.unsqueeze(-1)
                            batch["cls"] = cat_cls
                            batch["bboxes"] = cat_box
                        else:
                            dev = batch["img"].device
                            batch["batch_idx"] = torch.zeros(0, dtype=torch.long, device=dev)
                            batch["cls"] = torch.zeros((0, 1), dtype=torch.float32, device=dev)
                            batch["bboxes"] = torch.zeros((0, 4), dtype=torch.float32, device=dev)

                    # 3. Apply configured Defense Preprocessing Pipeline for YOLO detector
                    _, _, h, w = batch["img"].shape
                    if self.pipeline is not None:
                        with torch.no_grad():
                            processed_imgs, pred_matrices = self.pipeline.forward(
                                batch["img"], mode=self.preprocess_mode
                            )
                            batch["img"] = processed_imgs

                            if pred_matrices is not None:
                                batch = transform_batch_bboxes_for_stn(batch, pred_matrices, height=h, width=w)

                    # 4. YOLO Forward and Loss calculation
                    if self.args.compile:
                        preds = self.model(batch["img"])
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        loss, self.loss_items = self.model(batch)

                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= self.world_size
                    if not self.loss_names:
                        self.loss_names = tuple(self.loss_items)

                    self.tloss = (
                        self.loss_items
                        if self.tloss is None
                        else {k: (self.tloss[k] * i + v) / (i + 1) for k, v in self.loss_items.items()}
                    )

                self.scaler.scale(self.loss).backward()

                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    if i % 20 == 0:
                        torch.cuda.empty_cache()

                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

                if RANK in {-1, 0}:
                    loss_length = len(self.tloss)
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *self.tloss.values(),
                            batch.get("cls", batch["img"]).shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            total_epoch_samples = sum(self.epoch_counts.values())
            if total_epoch_samples > 0 and RANK in {-1, 0}:
                ratios = {t: self.epoch_counts[t] / total_epoch_samples for t in self.epoch_counts}
                LOGGER.info(
                    f"Epoch {epoch + 1} processed sample ratios: "
                    + ", ".join(f"{t}: {r:.3f}" for t, r in ratios.items())
                )

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)
                self.metrics, self.fitness = self.validate()

            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0

            # Dynamic attack ratio update at epoch end (Stage 2 only)
            new_ratios = None
            is_updated = False
            if self.stage == 2 and self.adaptive_controller is not None:
                if RANK in {-1, 0}:
                    val_failures = getattr(self.validator, "attack_failures", set())
                    if val_failures:
                        raise RuntimeError(
                            f"Adversarial generation failed during validation for attacks: {val_failures}. "
                            "Cannot silently use Clean mAP or update Controller!"
                        )

                    attack_maps = getattr(self.validator, "last_attack_maps", {})
                    new_ratios, is_updated = self.adaptive_controller.step(
                        epoch=epoch + 1,
                        attack_maps=attack_maps,
                        fitness=self.fitness,
                        save_dir=self.save_dir,
                    )

                if RANK != -1:
                    broadcast_list = [new_ratios if RANK == 0 else None, is_updated if RANK == 0 else None]
                    dist.broadcast_object_list(broadcast_list, src=0)
                    new_ratios, is_updated = broadcast_list[0], broadcast_list[1]
                    if RANK != 0 and new_ratios is not None:
                        self.adaptive_controller.next_train_ratios = dict(new_ratios)
                        self.adaptive_controller.current_train_ratios = dict(new_ratios)

                if is_updated and new_ratios is not None:
                    self.adaptive_controller.current_train_ratios = dict(new_ratios)
                    self.attack_ratios = [new_ratios.get(a, 0.0) for a in self.attack_names]
                    self.args.attack_ratio = self.attack_ratios
                    LOGGER.info(
                        f"Epoch {epoch + 1} controller update applied. New training ratios for Epoch {epoch + 2}+: "
                        + ", ".join(f"{k}: {v:.4f}" for k, v in new_ratios.items())
                    )

            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)

            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")


DetectionTrainer = DetectionTrainerAdvFGDSTNStaged

__all__ = [
    "DetectionTrainer",
    "DetectionTrainerAdvFGDSTNStaged",
    "find_detection_head",
]
