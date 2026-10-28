# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Combined FGD + STN Defense with Adaptive Adversarial Training."""

from __future__ import annotations

import csv
import io
import math
import random
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import distributed as dist

from ultralytics import __version__
from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch
from ultralytics.attacks.attack_utils import setup_attack_model
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.defenses.fgd_stn_pipeline import (
    MODE_DESCRIPTIONS,
    VALID_MODES,
    FGDSTNPipeline,
    PreprocessDetectorWrapper,
    build_fgd_stn_pipeline,
    canonicalize_mode,
    transform_batch_bboxes_for_stn,
)
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train_adv_adaptive import (
    DetectionTrainer as DetectionTrainerAdvAdaptive,
    DynamicAttackRatioController,
    YOLODatasetAdvAdaptive,
    parse_list_arg,
)
from ultralytics.models.yolo.detect.val_adv_fgd_stn_adaptive import (
    DetectionValidatorAdvFGDSTNAdaptive,
)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, GIT, LOGGER, RANK, YAML, colorstr, TQDM
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import (
    autocast,
    convert_optimizer_state_dict_to_fp16,
    qat_state,
    strip_qat,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
)


class AdaptiveAttackRatioController(DynamicAttackRatioController):
    """Extended DynamicAttackRatioController supporting optional fixed-ratio mode.

    When adaptive_ratio is True, dynamically updates attack ratios every update_interval epochs
    based on per-attack validation mAP50-95 degradations with softmax temperature tau.
    Clean ratio remains strictly fixed at p_clean (default 0.25), and all attack ratios sum to (1 - p_clean).
    When adaptive_ratio is False, initial training ratios remain unchanged throughout training.
    """

    def __init__(
        self,
        attack_names: list[str] | None = None,
        initial_attack_ratios: list[float] | None = None,
        p_clean: float = 0.25,
        tau: float = 5.0,
        update_interval: int = 5,
        eval_weights: list[float] | dict[str, float] | None = None,
        save_dir: Path | str | None = None,
        adaptive_ratio: bool = True,
    ):
        super().__init__(
            attack_names=attack_names,
            initial_attack_ratios=initial_attack_ratios,
            p_clean=p_clean,
            tau=tau,
            update_interval=update_interval,
            eval_weights=eval_weights,
            save_dir=save_dir,
        )
        self.adaptive_ratio = bool(adaptive_ratio)

    def step(
        self,
        epoch: int,
        attack_maps: dict[str, float],
        fitness: float | None = None,
        save_dir: Path | str | None = None,
    ) -> tuple[dict[str, float], bool]:
        """Update ratios if adaptive_ratio is enabled and update_interval reached; log history."""
        clean_map = attack_maps.get("raw", attack_maps.get("clean", None))
        if clean_map is None or not math.isfinite(clean_map):
            raise RuntimeError(f"Clean mAP is missing or non-finite: {clean_map}. Cannot update attack ratios!")

        for att in self.attack_names:
            if att not in attack_maps:
                raise RuntimeError(
                    f"Validation mAP metric missing for attack '{att}'. Cannot silently fall back to Clean mAP!"
                )
            m = attack_maps[att]
            if m is None or not math.isfinite(m):
                raise RuntimeError(
                    f"Validation mAP metric for attack '{att}' is invalid ({m}). Cannot silently fall back to Clean mAP!"
                )

        if fitness is None:
            fitness = self.compute_fitness(attack_maps)

        if not self.adaptive_ratio:
            # Fixed-ratio mode: ratios stay unchanged
            record = {
                "epoch": epoch,
                "mAP_raw": float(clean_map),
                **{f"mAP_{att}": float(attack_maps[att]) for att in self.attack_names},
                "train_ratio_raw": float(self.current_train_ratios.get("raw", self.p_clean)),
                **{f"train_ratio_{att}": float(self.current_train_ratios.get(att, 0.0)) for att in self.attack_names},
                "next_ratio_raw": float(self.next_train_ratios.get("raw", self.p_clean)),
                **{f"next_ratio_{att}": float(self.next_train_ratios.get(att, 0.0)) for att in self.attack_names},
                "fitness": float(fitness),
                "updated": False,
            }
            self.history.append(record)
            target_dir = Path(save_dir) if save_dir else self.save_dir
            if target_dir:
                self._write_history_csv(target_dir / "adaptive_history.csv")
            return copy(self.next_train_ratios), False

        return super().step(epoch=epoch, attack_maps=attack_maps, fitness=fitness, save_dir=save_dir)

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d["adaptive_ratio"] = self.adaptive_ratio
        return d

    def load_state_dict(self, state: dict[str, Any]):
        super().load_state_dict(state)
        if "adaptive_ratio" in state:
            self.adaptive_ratio = bool(state["adaptive_ratio"])


class DetectionTrainerAdvFGDSTNAdaptive(DetectionTrainerAdvAdaptive):
    """Detection Trainer integrating Unified FGD + STN defense with Adaptive Adversarial Training.

    Five Preprocessing Modes:
    - none: YOLO only (Baseline)
    - fgd: FGD residual denoiser -> YOLO
    - stn: STN spatial rectification -> YOLO
    - fgd_stn: FGD denoising -> STN rectification -> YOLO
    - stn_fgd: STN rectification -> FGD denoising -> YOLO

    Features:
    - Frozen defense modules (FGD / STN) in permanent eval mode; only YOLO parameters are updated.
    - Configurable dynamic ratio adjustment (adaptive_ratio=True/False).
    - Preserves fixed Clean ratio (default 25%) and updates attack ratios every 5 epochs via softmax degradation.
    - Ground-truth bounding boxes are correctly transformed into STN rectified coordinate system.
    - Worst-of-K bounding box transformations are preserved and not applied twice.
    - Attack generation through defense pipeline (attack_through_preprocess=True/False).
    - Comprehensive checkpoint restoration (last.pt, best.pt) and logging (adaptive_history.csv).
    """

    pipeline_defaults: ClassVar[dict] = dict(
        preprocess_mode="none",
        fgd_weights=None,
        stn_weights=None,
        adaptive_ratio=True,
        attack_through_preprocess=True,
        p_clean=0.25,
        tau=5.0,
        update_interval=5,
        eval_weights=[0.25, 0.15, 0.15, 0.20],
        attack_name=["pgd", "bim", "mim", "worstk"],
        attack_ratio=[0.10, 0.10, 0.40, 0.15],
    )

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
        attack_weights="",
    ):
        if overrides is None:
            overrides = {}

        # Extract research arguments
        saved_args = {}
        resume = overrides.get("resume", False)
        if resume and isinstance(resume, (str, Path)) and str(resume).endswith(".pt"):
            args_yaml = Path(resume).resolve().parent.parent / "args.yaml"
            if args_yaml.exists():
                try:
                    import yaml

                    with open(args_yaml, "r", encoding="utf-8") as f:
                        saved_args = yaml.safe_load(f) or {}
                except Exception as e:
                    LOGGER.warning(f"Could not load saved configuration from {args_yaml}: {e}")

        # Pop research options
        self._preprocess_mode_arg = canonicalize_mode(
            overrides.pop("preprocess_mode", saved_args.get("preprocess_mode", "none"))
        )
        self._fgd_weights_arg = overrides.pop("fgd_weights", saved_args.get("fgd_weights", None))
        self._stn_weights_arg = overrides.pop("stn_weights", saved_args.get("stn_weights", None))

        raw_adaptive_ratio = overrides.pop("adaptive_ratio", saved_args.get("adaptive_ratio", True))
        if isinstance(raw_adaptive_ratio, bool):
            self._adaptive_ratio_arg = raw_adaptive_ratio
        else:
            self._adaptive_ratio_arg = str(raw_adaptive_ratio).lower() in ("true", "1", "yes")

        raw_att_through = overrides.pop(
            "attack_through_preprocess", saved_args.get("attack_through_preprocess", True)
        )
        if isinstance(raw_att_through, bool):
            self._attack_through_preprocess_arg = raw_att_through
        else:
            self._attack_through_preprocess_arg = str(raw_att_through).lower() in ("true", "1", "yes")

        super().__init__(cfg, overrides, _callbacks, attack_weights=attack_weights)

        # Store attributes on args and self
        self.preprocess_mode = self._preprocess_mode_arg
        self.fgd_weights = self._fgd_weights_arg
        self.stn_weights = self._stn_weights_arg
        self.adaptive_ratio = self._adaptive_ratio_arg
        self.attack_through_preprocess = self._attack_through_preprocess_arg

        self.args.preprocess_mode = self.preprocess_mode
        self.args.fgd_weights = self.fgd_weights
        self.args.stn_weights = self.stn_weights
        self.args.adaptive_ratio = self.adaptive_ratio
        self.args.attack_through_preprocess = self.attack_through_preprocess

        # Validate required checkpoints
        if self.preprocess_mode in {"fgd", "fgd_stn", "stn_fgd"} and not self.fgd_weights:
            LOGGER.warning(f"Preprocessing mode '{self.preprocess_mode}' requested without fgd_weights. Fallback enabled.")
        if self.preprocess_mode in {"stn", "fgd_stn", "stn_fgd"} and not self.stn_weights:
            LOGGER.warning(f"Preprocessing mode '{self.preprocess_mode}' requested without stn_weights. Fallback enabled.")

        # Re-initialize controller with adaptive_ratio toggle
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

        self.pipeline: Optional[FGDSTNPipeline] = None
        self.pipeline_meta: Dict[str, Any] = {}

    def set_model_attributes(self):
        """Build frozen defense pipeline before configuring online adversarial attackers."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

        # 1. Build and permanently freeze Unified Defense Pipeline
        self.pipeline, self.pipeline_meta = build_fgd_stn_pipeline(
            fgd_weights=self.fgd_weights,
            stn_weights=self.stn_weights,
            device=self.device,
            allow_identity_fallback=True,
        )
        self.pipeline.to(self.device)
        self.pipeline.eval().requires_grad_(False)

        LOGGER.info(
            f"Defense Pipeline Initialized: Mode='{self.preprocess_mode}' "
            f"({MODE_DESCRIPTIONS.get(self.preprocess_mode, self.preprocess_mode)}) | "
            f"FGD: {'Loaded' if self.pipeline.has_denoiser else 'None'}, "
            f"STN: {'Loaded' if self.pipeline.has_stn else 'None'}"
        )

        # 2. Setup attack model
        attack_weights = getattr(self.args, "attack_weights", None)
        use_current_model = (attack_weights is None) or (
            str(attack_weights).strip().lower() in {"", "none", "current"}
        )

        try:
            imgsz = getattr(self.args, "imgsz", 640)
            if use_current_model:
                LOGGER.info("Adversarial training: using CURRENT training model as attack generator (white-box AT).")
                self.attack_model = None
                target_detector = unwrap_model(self.model)
            else:
                LOGGER.info(f"Adversarial training: loading fixed attack model from: {attack_weights}")
                self.attack_model = setup_attack_model(
                    attack_weights,
                    device=self.device,
                    nc=self.data["nc"],
                    training=False,
                    imgsz=imgsz,
                )
                self.attack_model.eval()
                target_detector = self.attack_model

            # Wrap detector with defense pipeline if attacks pass through preprocessing
            if self.attack_through_preprocess and self.preprocess_mode != "none":
                LOGGER.info(
                    f"Wrapping attack detector with PreprocessDetectorWrapper (mode='{self.preprocess_mode}')"
                )
                target_attack_model = PreprocessDetectorWrapper(
                    pipeline=self.pipeline,
                    detector=target_detector,
                    mode=self.preprocess_mode,
                )
            else:
                LOGGER.info("Attacks generate directly on detector without defense wrapper.")
                target_attack_model = target_detector

            self.attackers = {}
            for att in self.attack_names:
                if att.lower().strip() not in {"raw", "none", "clean"}:
                    self.attackers[att] = self.build_attacker(att, target_attack_model, imgsz)
                    LOGGER.info(f"Attacker initialized: {att}")

            self.attacker = self.attackers.get(self.attack_names[0], None) if self.attackers else None
        except Exception as e:
            if getattr(self, "strict_attacks", False):
                raise
            LOGGER.warning(f"Failed to load attack model or attacker: {e}")
            import traceback

            LOGGER.warning(traceback.format_exc())
            self.attack_model = None
            self.attacker = None
            self.attackers = {}

    def build_optimizer(self, model, **kwargs):
        """Ensure optimizer groups only contain trainable YOLO parameters, excluding frozen defenses."""
        optimizer = super().build_optimizer(model, **kwargs)
        if self.pipeline is not None:
            pipeline_param_ids = {id(p) for p in self.pipeline.parameters()}
            for group in optimizer.param_groups:
                group["params"] = [p for p in group["params"] if id(p) not in pipeline_param_ids and p.requires_grad]
        return optimizer

    def get_validator(self):
        """Return DetectionValidatorAdvFGDSTNAdaptive configured with active defense pipeline."""
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
        ]:
            if hasattr(validator_args, arg):
                delattr(validator_args, arg)

        validator = DetectionValidatorAdvFGDSTNAdaptive(
            getattr(self, "test_loader", None),
            save_dir=self.save_dir,
            args=validator_args,
            _callbacks=self.callbacks,
            pipeline=self.pipeline,
            preprocess_mode=self.preprocess_mode,
            fgd_weights=self.fgd_weights,
            stn_weights=self.stn_weights,
            attack_through_preprocess=self.attack_through_preprocess,
        )
        validator.attacker = getattr(self, "attacker", None)
        validator.attackers = getattr(self, "attackers", {})
        validator.attack_names = self.attack_names
        validator.attack_ratios = self.attack_ratios
        validator.eval_weights = self.eval_weights
        return validator

    def _setup_train(self):
        """Verify defense parameter freezing and record complete experiment configuration."""
        super()._setup_train()

        if self.pipeline is not None:
            assert not any(
                p.requires_grad for p in self.pipeline.parameters()
            ), "Defense pipeline parameters must not require gradients!"
            assert not self.pipeline.training, "Defense pipeline must be in eval mode!"

            pipeline_ids = {id(p) for p in self.pipeline.parameters()}
            assert not any(
                id(p) in pipeline_ids for group in self.optimizer.param_groups for p in group["params"]
            ), "Defense pipeline parameters must NOT be present in optimizer param groups!"

        config_record = dict(
            preprocess_mode=self.preprocess_mode,
            preprocess_mode_description=MODE_DESCRIPTIONS.get(self.preprocess_mode, self.preprocess_mode),
            fgd_weights=str(self.fgd_weights) if self.fgd_weights else None,
            stn_weights=str(self.stn_weights) if self.stn_weights else None,
            adaptive_ratio=self.adaptive_ratio,
            attack_through_preprocess=self.attack_through_preprocess,
            p_clean=self.p_clean,
            tau=self.tau,
            update_interval=self.update_interval,
            eval_weights=self.eval_weights,
            attack_names=self.attack_names,
            attack_ratios=self.attack_ratios,
            yolo_model=str(self.args.model),
            imgsz=self.args.imgsz,
            epochs=self.args.epochs,
            batch=self.args.batch,
            amp=self.args.amp,
        )

        if RANK in {-1, 0}:
            YAML.save(self.save_dir / "fgd_stn_adaptive_config.yaml", config_record)
            YAML.save(self.save_dir / "args.yaml", vars(self.args))
            LOGGER.info("FGD + STN Adaptive Adversarial Training Setup Summary:")
            for k, v in config_record.items():
                LOGGER.info(f"  {k}: {v}")

    def _load_checkpoint_state(self, ckpt: dict[str, Any]):
        """Load optimizer, scaler, EMA, adaptive_state, and defense pipeline settings from checkpoint."""
        super()._load_checkpoint_state(ckpt)
        if ckpt:
            if "preprocess_mode" in ckpt:
                self.preprocess_mode = canonicalize_mode(ckpt["preprocess_mode"])
                self.args.preprocess_mode = self.preprocess_mode
            if "fgd_weights" in ckpt:
                self.fgd_weights = ckpt["fgd_weights"]
                self.args.fgd_weights = self.fgd_weights
            if "stn_weights" in ckpt:
                self.stn_weights = ckpt["stn_weights"]
                self.args.stn_weights = self.stn_weights
            if "adaptive_ratio" in ckpt:
                self.adaptive_ratio = bool(ckpt["adaptive_ratio"])
                self.args.adaptive_ratio = self.adaptive_ratio
            if "attack_through_preprocess" in ckpt:
                self.attack_through_preprocess = bool(ckpt["attack_through_preprocess"])
                self.args.attack_through_preprocess = self.attack_through_preprocess

            LOGGER.info(
                f"Restored FGD+STN settings from checkpoint: mode='{self.preprocess_mode}', "
                f"adaptive_ratio={self.adaptive_ratio}, attack_through_preprocess={self.attack_through_preprocess}"
            )

    def save_model(self):
        """Save model training checkpoints preserving adaptive_state and defense configuration."""
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

        buffer = io.BytesIO()
        torch.save(
            {
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
                "adaptive_state": self.adaptive_controller.state_dict(),
                "preprocess_mode": self.preprocess_mode,
                "fgd_weights": str(self.fgd_weights) if self.fgd_weights else None,
                "stn_weights": str(self.stn_weights) if self.stn_weights else None,
                "adaptive_ratio": self.adaptive_ratio,
                "attack_through_preprocess": self.attack_through_preprocess,
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
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()

        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)
        return True

    def _do_train(self):
        """Train the model with defense preprocessing and dynamic attack ratio adjustments."""
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
            f"Starting FGD+STN Adaptive Adversarial Training for {self.epochs} epochs "
            f"(mode='{self.preprocess_mode}', adaptive={self.adaptive_ratio})..."
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

            # Dynamic attack ratio allocation update per epoch
            base_seed = getattr(self.args, "seed", 0) or 42
            epoch_seed = base_seed + epoch
            needs_reset = False
            if epoch == self.start_epoch and self.resume:
                if hasattr(self.train_loader.dataset, "update_virtual_samples"):
                    self.train_loader.dataset.update_virtual_samples(
                        attack_ratios=self.adaptive_controller.current_train_ratios,
                        seed=epoch_seed,
                    )
                    needs_reset = True
            elif epoch > self.start_epoch:
                if hasattr(self.train_loader.dataset, "update_virtual_samples"):
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
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                amp_dtype = torch.bfloat16 if getattr(self.args, "amp", None) == "bf16" else self.amp
                with autocast(amp_dtype, device=self.device.type):
                    batch = self.preprocess_batch(batch)

                    for st in batch.get("sample_type", []):
                        if st in self.epoch_counts:
                            self.epoch_counts[st] += 1

                    sub_batches = self.split_batch_by_type(batch)

                    # 1. Generate adversarial images for present attack groups on-the-fly
                    has_generated_adv = False
                    if getattr(self, "attackers", None):
                        is_current_model = getattr(self, "attack_model", None) is None
                        if is_current_model:
                            param_grad_states = [p.requires_grad for p in self.model.parameters()]
                            for p in self.model.parameters():
                                p.requires_grad = False

                        try:
                            for att, attacker in self.attackers.items():
                                if att in sub_batches:
                                    sub_batch = sub_batches[att]
                                    try:
                                        if hasattr(attacker, "proxy_model"):
                                            attacker.proxy_model.current_paths = sub_batch.get("im_file", [])
                                        if hasattr(attacker, "estimator") and hasattr(attacker.estimator, "model"):
                                            attacker.estimator.model.current_paths = sub_batch.get("im_file", [])
                                        if self.attack_model is not None and hasattr(self.attack_model, "current_paths"):
                                            self.attack_model.current_paths = sub_batch.get("im_file", [])

                                        if getattr(self.args, "amp", None) == "bf16":
                                            with autocast(torch.bfloat16, device=self.device.type):
                                                if att == "worstk" and callable(getattr(attacker, "forward_batch", None)):
                                                    attacker.forward_batch(sub_batch)
                                                    imgs_adv = sub_batch["img"]
                                                else:
                                                    imgs_adv = run_attack_on_batch(
                                                        attacker, sub_batch, label_policy="largest_box"
                                                    )
                                        else:
                                            with torch.amp.autocast(device_type="cuda", enabled=False):
                                                if att == "worstk" and callable(getattr(attacker, "forward_batch", None)):
                                                    attacker.forward_batch(sub_batch)
                                                    imgs_adv = sub_batch["img"]
                                                else:
                                                    imgs_adv = run_attack_on_batch(
                                                        attacker, sub_batch, label_policy="largest_box"
                                                    )

                                        if imgs_adv is not None:
                                            sub_batch["img"] = imgs_adv.detach().to(sub_batch["img"].dtype).to(self.device)
                                            has_generated_adv = True
                                            del imgs_adv
                                    except Exception as e:
                                        if getattr(self, "strict_attacks", False):
                                            raise
                                        LOGGER.warning(f"Adversarial generation failed for {att}: {e}. Keeping raw images.")
                                        import traceback

                                        LOGGER.warning(traceback.format_exc())
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
                            batch["batch_idx"].dim()
                            if "batch_idx" in batch and batch["batch_idx"] is not None
                            else 1
                        )
                        orig_cls_dim = (
                            batch["cls"].dim() if "cls" in batch and batch["cls"] is not None else 1
                        )

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

                            # Sort targets by batch_idx to maintain non-decreasing order
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

                            # Transform ground-truth bounding boxes if STN rectification was applied
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

            # Dynamic attack ratio update at epoch end
            new_ratios = None
            is_updated = False
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

            # DDP synchronization: broadcast new ratios from rank 0 to all ranks
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


DetectionTrainer = DetectionTrainerAdvFGDSTNAdaptive

__all__ = [
    "AdaptiveAttackRatioController",
    "DetectionTrainerAdvFGDSTNAdaptive",
    "DetectionTrainer",
]
