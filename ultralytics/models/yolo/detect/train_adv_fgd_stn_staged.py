# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Staged Adaptive Adversarial Training with Frozen FGD and STN for YOLO."""

from __future__ import annotations

import csv
import io
import json
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
from ultralytics.data.augment import Compose, Format, LetterBox, v8_transforms
from ultralytics.data.dataset import YOLODataset
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
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import (
    autocast,
    convert_optimizer_state_dict_to_fp16,
    qat_state,
    strip_qat,
    unset_deterministic,
    unwrap_model,
)

HEAD_CLASSES = (Detect, Segment, Pose, OBB)


class YOLODatasetStage1Clean(YOLODataset):
    """Stage 1 Dataset: strictly 100% clean raw samples with no adversarial attacks."""

    def __init__(self, cfg, img_path, batch, data, mode="train", rect=False, stride=32, **kwargs):
        self.cfg = cfg
        super().__init__(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=cfg,
            rect=cfg.rect or rect,
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=data,
            fraction=cfg.fraction if mode == "train" else 1.0,
            **kwargs,
        )
        self.attack_names = []
        self.attack_ratios = []
        self.p_clean = 1.0
        self.update_virtual_samples()
        self.transforms_raw = self.build_transforms(self.cfg)
        self.transforms = self.transforms_raw

    def update_virtual_samples(self, attack_ratios=None, seed=None):
        """Construct virtual samples allocating exactly 100% raw samples."""
        self.attack_names = []
        self.attack_ratios = []
        self.p_clean = 1.0
        n_raw = len(self.im_files)
        self.virtual_samples = [(i, "raw") for i in range(n_raw)]
        LOGGER.info(f"Stage 1 sample allocation:\nraw: {n_raw}/{n_raw} (1.000)")

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """Build data augmentation transforms using configured hyp parameters."""
        if hyp is None:
            hyp = self.cfg
        hyp = copy(hyp)
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=getattr(hyp, "mask_ratio", 4),
                mask_overlap=getattr(hyp, "overlap_mask", True),
                bgr=getattr(hyp, "bgr", 0.0) if self.augment else 0.0,
            )
        )
        return transforms

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Fetch image, label, and annotations, marking sample strictly as raw."""
        orig_idx, sample_type = self.virtual_samples[index]
        assert sample_type == "raw", f"Stage 1 sample must be 'raw', got {sample_type}"
        label = self.get_image_and_label(orig_idx)
        label = self.transforms_raw(label)
        label["sample_type"] = "raw"
        label["attack_name"] = None
        return label

    def __len__(self) -> int:
        return len(self.virtual_samples)

    def close_mosaic(self, hyp: dict) -> None:
        """Disable mosaic and other multi-image augmentations."""
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms_raw = self.build_transforms(hyp)
        self.transforms = self.transforms_raw


def _strip_frozen_bn_train(model: nn.Module) -> None:
    """Remove monkeypatched train method from BatchNorm modules to allow clean serialization."""
    if model is None:
        return
    for mod in model.modules():
        if "train" in mod.__dict__:
            del mod.__dict__["train"]


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

        raw_freeze_all_bn = overrides.pop("freeze_all_bn", saved_args.get("freeze_all_bn", None))
        if raw_freeze_all_bn is not None:
            self.freeze_all_bn = (
                raw_freeze_all_bn
                if isinstance(raw_freeze_all_bn, bool)
                else str(raw_freeze_all_bn).lower() in ("true", "1", "yes")
            )
        else:
            self.freeze_all_bn = self.stage == 1

        raw_eval_only = overrides.pop("eval_only", saved_args.get("eval_only", False))
        self.eval_only_mode = (
            raw_eval_only if isinstance(raw_eval_only, bool) else str(raw_eval_only).lower() in ("true", "1", "yes")
        )

        raw_eval_epoch0 = overrides.pop("eval_epoch0", saved_args.get("eval_epoch0", False))
        self.eval_epoch0 = (
            raw_eval_epoch0
            if isinstance(raw_eval_epoch0, bool)
            else str(raw_eval_epoch0).lower() in ("true", "1", "yes")
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
            stage_epochs = 3
            stage_lr0 = 1e-6
            stage_lrf = 1.0
            stage_save_period = 1

            overrides.setdefault("mosaic", saved_args.get("mosaic", 0.0))
            overrides.setdefault("scale", saved_args.get("scale", 0.0))
            overrides.setdefault("translate", saved_args.get("translate", 0.0))
            overrides.setdefault("fliplr", saved_args.get("fliplr", 0.0))
            overrides.setdefault("flipud", saved_args.get("flipud", 0.0))
            overrides.setdefault("hsv_h", saved_args.get("hsv_h", 0.0))
            overrides.setdefault("hsv_s", saved_args.get("hsv_s", 0.0))
            overrides.setdefault("hsv_v", saved_args.get("hsv_v", 0.0))
            overrides.setdefault("mixup", saved_args.get("mixup", 0.0))
            overrides.setdefault("cutmix", saved_args.get("cutmix", 0.0))
            overrides.setdefault("copy_paste", saved_args.get("copy_paste", 0.0))
            overrides.setdefault("close_mosaic", saved_args.get("close_mosaic", 0))
            overrides.setdefault("lrf", saved_args.get("lrf", stage_lrf))
            overrides.setdefault("warmup_epochs", saved_args.get("warmup_epochs", 0.0))
            overrides.setdefault("warmup_bias_lr", saved_args.get("warmup_bias_lr", 0.0))
            overrides.setdefault("save_period", saved_args.get("save_period", stage_save_period))
        else:
            stage_p_clean = 0.65
            stage_att_names = ["pgd", "bim", "mim", "worstk"]
            stage_att_ratios = [0.12, 0.05, 0.10, 0.08]
            stage_adaptive = True
            stage_eval_w = [0.20, 0.20, 0.15, 0.15, 0.30]
            stage_epochs = 20
            stage_lr0 = 1e-5
            stage_lrf = 0.01
            stage_save_period = -1

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
        overrides.setdefault("lr0", saved_args.get("lr0", stage_lr0))
        overrides.setdefault("optimizer", saved_args.get("optimizer", "AdamW"))
        overrides.setdefault("warmup_bias_lr", saved_args.get("warmup_bias_lr", 0.0))
        overrides.setdefault("save_period", saved_args.get("save_period", stage_save_period))

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
        self.args.freeze_all_bn = self.freeze_all_bn
        self.args.eval_only = self.eval_only_mode
        self.args.eval_epoch0 = self.eval_epoch0
        self.args.strict_attacks = self.strict_attacks
        self.save_period = int(overrides.get("save_period", getattr(self.args, "save_period", stage_save_period)))
        self.args.save_period = self.save_period
        self._frozen_param_snapshots: dict[str, torch.Tensor] = {}
        self._bn_stat_snapshots: dict[str, dict[str, torch.Tensor | None]] = {}

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

        unwrapped = unwrap_model(self.model)
        head_module, _ = find_detection_head(unwrapped)
        head_param_ids = {id(p) for p in head_module.parameters()} if head_module is not None else set()

        bn_modules = [
            m
            for m in self.model.modules()
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.SyncBatchNorm))
        ]
        bn_param_ids = {id(p) for m in bn_modules for p in m.parameters()}

        # Freeze all BatchNorm layers if requested
        if getattr(self, "freeze_all_bn", False):
            for m in bn_modules:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                m.train = lambda mode=True, mod=m: mod

        # Re-verify parameter freezing after _setup_train
        if self.train_head_only:
            for k, v in self.model.named_parameters():
                in_head = id(v) in head_param_ids
                in_bn = id(v) in bn_param_ids
                if in_head and ".dfl." not in k and not (getattr(self, "freeze_all_bn", False) and in_bn):
                    v.requires_grad = True
                else:
                    v.requires_grad = False

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            assert len(trainable_params) > 0, "Detection Head has no trainable parameters! Check layer identification."
            assert all(id(p) in head_param_ids for p in trainable_params), (
                "Trainable parameters leak outside Detection Head!"
            )
            if getattr(self, "freeze_all_bn", False):
                assert not any(id(p) in bn_param_ids for p in trainable_params), (
                    "BatchNorm parameters must NOT be trainable when freeze_all_bn is enabled!"
                )
                assert all(not m.training for m in bn_modules), "All BatchNorm modules must be in eval mode!"
                assert all(not p.requires_grad for m in bn_modules for p in m.parameters()), (
                    "All BatchNorm parameters must have requires_grad=False!"
                )

            LOGGER.info(
                f"Model parameter freezing verified: Train Head Only = {self.train_head_only}, "
                f"Freeze All BN = {getattr(self, 'freeze_all_bn', False)}. "
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
            if getattr(self, "freeze_all_bn", False):
                assert not any(id(p) in bn_param_ids for p in group["params"]), (
                    "Optimizer contains BatchNorm parameters while freeze_all_bn is enabled!"
                )

        self._take_frozen_snapshots()

        config_record = {
            "stage": self.stage,
            "stage1_weights": str(self.stage1_weights) if self.stage1_weights else None,
            "train_head_only": self.train_head_only,
            "freeze_bn": self.freeze_bn,
            "freeze_all_bn": getattr(self, "freeze_all_bn", False),
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

    def _take_frozen_snapshots(self):
        """Take snapshots of all frozen parameters and BatchNorm running statistics for runtime verification."""
        self._frozen_param_snapshots = {
            k: v.detach().clone().cpu() for k, v in self.model.named_parameters() if not v.requires_grad
        }
        self._bn_stat_snapshots = {}
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.SyncBatchNorm)) and (
                getattr(self, "freeze_all_bn", False) or not m.training
            ):
                self._bn_stat_snapshots[name] = {
                    "running_mean": m.running_mean.detach().clone().cpu() if m.running_mean is not None else None,
                    "running_var": m.running_var.detach().clone().cpu() if m.running_var is not None else None,
                    "num_batches_tracked": m.num_batches_tracked.detach().clone().cpu()
                    if m.num_batches_tracked is not None
                    else None,
                }

    def check_frozen_invariants(self, step_name: str = "epoch"):
        """Verify that frozen model parameters and BN running statistics remain unchanged at runtime."""
        for name, param in self.model.named_parameters():
            if name in self._frozen_param_snapshots:
                orig = self._frozen_param_snapshots[name]
                curr = param.detach().cpu()
                if not torch.equal(curr, orig):
                    raise RuntimeError(f"Runtime check failed: frozen parameter '{name}' mutated at {step_name}!")

        for name, m in self.model.named_modules():
            if name in self._bn_stat_snapshots:
                snap = self._bn_stat_snapshots[name]
                if getattr(self, "freeze_all_bn", False) and m.training:
                    raise RuntimeError(
                        f"Runtime check failed: BatchNorm '{name}' was accidentally switched to training mode at {step_name}!"
                    )
                if snap["running_mean"] is not None and not torch.equal(
                    m.running_mean.detach().cpu(), snap["running_mean"]
                ):
                    raise RuntimeError(
                        f"Runtime check failed: BatchNorm running_mean for '{name}' changed at {step_name}!"
                    )
                if snap["running_var"] is not None and not torch.equal(
                    m.running_var.detach().cpu(), snap["running_var"]
                ):
                    raise RuntimeError(
                        f"Runtime check failed: BatchNorm running_var for '{name}' changed at {step_name}!"
                    )
                if snap["num_batches_tracked"] is not None and not torch.equal(
                    m.num_batches_tracked.detach().cpu(), snap["num_batches_tracked"]
                ):
                    raise RuntimeError(
                        f"Runtime check failed: BatchNorm num_batches_tracked for '{name}' changed at {step_name}!"
                    )

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation."""
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode == "train":
            if self.stage == 1:
                dataset = YOLODatasetStage1Clean(
                    self.args,
                    img_path,
                    batch,
                    self.data,
                    mode=mode,
                    rect=False,
                    stride=gs,
                )
                assert len(dataset) == len(dataset.im_files), (
                    f"Stage 1 dataset length ({len(dataset)}) must match number of images ({len(dataset.im_files)})!"
                )
                assert all(st == "raw" for _, st in dataset.virtual_samples), (
                    "Stage 1 virtual samples must all be 'raw'!"
                )
                assert len({idx for idx, _ in dataset.virtual_samples}) == len(dataset.im_files), (
                    "Stage 1 must include all images exactly once per epoch!"
                )
                return dataset
            return super().build_dataset(img_path, mode, batch)
        return super().build_dataset(img_path, mode, batch)

    def build_optimizer(self, model, **kwargs):
        """Ensure optimizer groups only contain trainable YOLO parameters, excluding frozen defenses."""
        optimizer = super().build_optimizer(model, **kwargs)
        pipeline_param_ids = {id(p) for p in self.pipeline.parameters()} if self.pipeline is not None else set()

        unwrapped = unwrap_model(model)
        head_module, _ = find_detection_head(unwrapped)
        head_param_ids = {id(p) for p in head_module.parameters()} if head_module is not None else set()

        bn_param_ids = set()
        if getattr(self, "freeze_all_bn", False):
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                    for p in m.parameters():
                        bn_param_ids.add(id(p))

        for group in optimizer.param_groups:
            filtered = []
            for p in group["params"]:
                if id(p) in pipeline_param_ids:
                    continue
                if not p.requires_grad:
                    continue
                if self.train_head_only and id(p) not in head_param_ids:
                    continue
                if getattr(self, "freeze_all_bn", False) and id(p) in bn_param_ids:
                    continue
                filtered.append(p)
            group["params"] = filtered

        return optimizer

    def _model_train(self):
        """Set model in training mode while keeping frozen layers and frozen BN in eval mode."""
        self.model.train()
        if getattr(self, "freeze_all_bn", False):
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                    m.eval()
        elif getattr(self, "freeze_bn", True):
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
            "freeze_all_bn",
            "strict_attacks",
            "eval_only",
            "eval_epoch0",
        ]:
            if hasattr(validator_args, arg):
                try:
                    delattr(validator_args, arg)
                except AttributeError:
                    pass

        validator = DetectionValidatorAdvFGDSTNStaged(
            getattr(self, "test_loader", None),
            save_dir=self.save_dir,
            args=validator_args,
            _callbacks=self.callbacks,
            pipeline=self.pipeline,
            preprocess_mode=self.preprocess_mode,
            fgd_weights=self.fgd_weights,
            stn_weights=self.stn_weights,
            attack_through_preprocess=getattr(self.args, "val_attack_through_preprocess", False),
            stage=self.stage,
            eval_weights=self.eval_weights,
            strict_attacks=self.strict_attacks,
        )
        validator.attacker = getattr(self, "attacker", None)
        validator.attackers = getattr(self, "attackers", {})
        validator.attack_names = getattr(self, "attack_names", [])
        validator.attack_ratios = getattr(self, "attack_ratios", [])
        validator.eval_weights = getattr(self, "eval_weights", None)
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
            if "freeze_all_bn" in ckpt:
                self.freeze_all_bn = bool(ckpt["freeze_all_bn"])
                self.args.freeze_all_bn = self.freeze_all_bn
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
        _strip_frozen_bn_train(ema)

        live_model = deepcopy(unwrap_model(self.model)).half().to(memory_format=torch.contiguous_format)
        if hasattr(live_model, "criterion"):
            live_model.criterion = None
        for v in live_model.state_dict().values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                torch.nan_to_num_(v)
        strip_qat(live_model)
        _strip_frozen_bn_train(live_model)

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
            "freeze_all_bn": getattr(self, "freeze_all_bn", False),
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
        if self.stage == 1:
            if (self.save_period > 0) and ((self.epoch + 1) % self.save_period == 0):
                (self.wdir / f"epoch{self.epoch + 1}.pt").write_bytes(serialized_ckpt)
        else:
            if (self.save_period > 0) and (self.epoch % self.save_period == 0):
                (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)
        return True

    def run_epoch0_eval(self, save_results: bool = True) -> dict[str, Any]:
        """Execute Epoch 0 baseline evaluation using the exact Stage 1 validation pipeline.

        Evaluates the base detector with frozen STN->FGD defense on the validation set without
        initializing the optimizer, scheduler, or adversarial attackers.
        """
        if getattr(self, "stage", 1) != 1:
            msg = (
                f"run_epoch0_eval is only supported for Stage 1 clean baseline evaluation (got stage={self.stage}). "
                "Stage 2 adversarial baseline evaluation is not structurally supported in epoch 0 / eval-only mode."
            )
            LOGGER.error(msg)
            raise NotImplementedError(msg)

        if not isinstance(self.model, nn.Module):
            self.setup_model()
            self.model = self.model.to(self.device)
            gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
            self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
            self.stride = gs
            self.set_model_attributes()

        if getattr(self, "test_loader", None) is None:
            self.test_loader = self.get_dataloader(
                self.data[self.args.split],
                batch_size=self.batch_size if self.args.task in {"obb", "semantic", "depth"} else self.batch_size * 2,
                rank=-1,
                mode="val",
            )

        validator = self.get_validator()
        validator.dataloader = self.test_loader

        if not hasattr(self, "loss_items") or self.loss_items is None:
            self.loss_items = {
                "box_loss": torch.tensor(0.0, device=self.device),
                "cls_loss": torch.tensor(0.0, device=self.device),
                "dfl_loss": torch.tensor(0.0, device=self.device),
            }

        prev_epoch = getattr(self, "epoch", 0)
        self.epoch = 0

        LOGGER.info("\n" + "=" * 60)
        LOGGER.info(f"Starting Epoch 0 Baseline Validation (Stage {self.stage}, Mode: '{self.preprocess_mode}')...")
        LOGGER.info("=" * 60)

        stats = validator(self)

        clean_map50 = float(stats.get("metrics/mAP50(B)", 0.0))
        clean_map50_95 = float(stats.get("metrics/mAP50-95(B)", 0.0))
        fitness = float(stats.get("fitness", clean_map50_95))

        results_record = {
            "epoch": 0,
            "stage": self.stage,
            "model": str(self.args.model),
            "preprocess_mode": self.preprocess_mode,
            "fgd_weights": str(self.fgd_weights) if self.fgd_weights else None,
            "stn_weights": str(self.stn_weights) if self.stn_weights else None,
            "metrics/mAP50(B)": clean_map50,
            "metrics/mAP50-95(B)": clean_map50_95,
            "fitness": fitness,
        }

        if save_results and RANK in {-1, 0}:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            json_file = self.save_dir / "epoch0_results.json"
            csv_file = self.save_dir / "epoch0_results.csv"

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(results_record, f, indent=2)

            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(results_record.keys()))
                writer.writeheader()
                writer.writerow(results_record)

            LOGGER.info(f"Epoch 0 Baseline Results: Clean mAP50={clean_map50:.5f}, Clean mAP50-95={clean_map50_95:.5f}")
            LOGGER.info(f"Saved Epoch 0 results to:\n  JSON: {json_file}\n  CSV:  {csv_file}\n")

        self.epoch = prev_epoch
        return results_record

    def eval_only(self) -> dict[str, Any]:
        """Run standalone evaluation without training."""
        if getattr(self, "stage", 1) != 1:
            msg = (
                f"eval_only mode is only supported for Stage 1 clean baseline evaluation (got stage={self.stage}). "
                "Stage 2 adversarial evaluation requires an active training context and is not structurally supported in eval-only mode."
            )
            LOGGER.error(msg)
            raise NotImplementedError(msg)
        LOGGER.info("Running in --eval_only mode (no optimizer, scheduler, or adversarial attackers).")
        return self.run_epoch0_eval(save_results=True)

    def train(self):
        """Execute training or eval-only depending on configured flags."""
        if getattr(self.args, "eval_only", False) or getattr(self, "eval_only_mode", False):
            return self.eval_only()
        return super().train()

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

        if self.stage == 1 and getattr(self, "eval_epoch0", False):
            LOGGER.info("Evaluating Epoch 0 baseline before commencing Stage 1 training...")
            self.run_epoch0_eval(save_results=True)
            self.best_fitness = None
        elif self.stage != 1 and getattr(self, "eval_epoch0", False):
            LOGGER.warning(
                f"eval_epoch0 is ignored for stage={self.stage}; Epoch 0 baseline evaluation is only supported for Stage 1."
            )

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
                if self.stage == 1:
                    assert all(st == "raw" for st in batch.get("sample_type", [])), (
                        f"Stage 1 must process only 'raw' clean samples, found {batch.get('sample_type')}"
                    )
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

                    if i == 0:
                        self.check_frozen_invariants(f"epoch {epoch + 1} batch 0")

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

            self.check_frozen_invariants(f"epoch {epoch + 1}")
            if RANK in {-1, 0}:
                LOGGER.info(f"Epoch {epoch + 1}: Frozen parameters and BatchNorm statistics verified unchanged.")

            total_epoch_samples = sum(self.epoch_counts.values())
            if self.stage == 1:
                assert self.epoch_counts.get("raw", 0) == total_epoch_samples, (
                    f"Stage 1 must contain exactly 100% raw samples! Counts: {self.epoch_counts}"
                )
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
    "YOLODatasetStage1Clean",
    "find_detection_head",
]
