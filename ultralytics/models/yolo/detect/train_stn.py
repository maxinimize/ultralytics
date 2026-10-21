# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Independent Spatial Transformer Network (STN) Trainer for Worst-of-k Defense."""

from __future__ import annotations

import math
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ultralytics.attacks.attack_utils import BatchContainer
from ultralytics.attacks.spatial_worst_of_k_stn import SpatialWorstOfKWithMetadata
from ultralytics.cfg import get_cfg
from ultralytics.defenses.spatial_transformer import (
    SpatialTransformer,
    compute_control_points_loss,
    compute_masked_recon_loss,
    compute_validity_mask,
    transform_boxes,
)
from ultralytics.data import build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, YAML, IterableSimpleNamespace, colorstr
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import EarlyStopping, ModelEMA, TORCH_2_4, select_device, unwrap_model


class STNTrainer(DetectionTrainer):
    """Independent trainer for Spatial Transformer Network (STN) defense under Worst-of-k attacks.

    Freezes guidance detection model (e.g. YOLO12L), generates online Worst-of-k geometric attacks,
    optimizes only the STN localization parameters, and saves dedicated STN checkpoints with best.pt
    selected by validation Worst-of-k mAP50-95.
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: Optional[dict] = None,
        _callbacks: Optional[dict] = None,
    ) -> None:
        """Initialize STNTrainer with research overrides and default configuration."""
        overrides = dict(overrides or {})
        defaults = dict(
            degrees=30.0,
            translate=0.10,
            worstk_k=10,
            attack_weights=None,
            loss_geom_weight=1.0,
            loss_recon_weight=1.0,
            loss_det_weight=0.0,
            val_interval=1,
        )
        self.stn_cfg = {key: overrides.pop(key, value) for key, value in defaults.items()}

        baseline = dict(
            task="detect",
            optimizer="Adam",
            lr0=1e-3,
            weight_decay=1e-5,
            warmup_epochs=0.0,
            project="runs/train_stn",
            name="exp",
            val=True,
            plots=False,
        )
        super().__init__(cfg, {**baseline, **overrides}, _callbacks)

        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError(f"STNTrainer requires CPU or CUDA device, got {self.device.type}")

        self.stn_cfg["degrees"] = float(self.stn_cfg["degrees"])
        self.stn_cfg["translate"] = float(self.stn_cfg["translate"])
        self.stn_cfg["worstk_k"] = int(self.stn_cfg["worstk_k"])
        self.stn_cfg["loss_geom_weight"] = float(self.stn_cfg["loss_geom_weight"])
        self.stn_cfg["loss_recon_weight"] = float(self.stn_cfg["loss_recon_weight"])
        self.stn_cfg["loss_det_weight"] = float(self.stn_cfg["loss_det_weight"])
        self.stn_cfg["val_interval"] = int(self.stn_cfg["val_interval"])

        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        YAML.save(self.save_dir / "stn_config.yaml", self.stn_cfg)

    def check_resume(self, overrides: dict) -> None:
        """Load STN checkpoint without passing through YOLO detection checkpoint loader."""
        self.resume = bool(self.args.resume)
        self.stn_checkpoint = None
        if self.resume:
            if not isinstance(self.args.resume, (str, Path)):
                raise ValueError("STN resume requires a valid checkpoint path (e.g. weights/last.pt)")
            resume_path = Path(self.args.resume)
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            self.stn_checkpoint = torch_load(resume_path, map_location="cpu")
            self.stn_cfg = self.stn_checkpoint.get("stn_config", self.stn_cfg)
            saved = self.stn_checkpoint.get("train_args", {}).copy()
            saved.update({k: overrides[k] for k in ("device", "workers", "epochs", "batch") if k in overrides})
            saved.update(resume=str(resume_path), exist_ok=True)
            self.args = get_cfg(DEFAULT_CFG, saved)

    def _setup_train(self) -> None:
        """Prepare frozen guidance model, online Worst-of-k attacker, and STN defense model."""
        # 1. Load guidance detection model and freeze all parameters
        self.guidance_model, _ = load_checkpoint(self.args.model, device=self.device)
        self.args.model = str(Path(self.guidance_model.pt_path).resolve())
        self.guidance_model.eval().requires_grad_(False)
        if hasattr(self.guidance_model, "args") and isinstance(self.guidance_model.args, dict):
            self.guidance_model.args = IterableSimpleNamespace(**self.guidance_model.args)
        self.guidance_loss = unwrap_model(self.guidance_model).loss

        if self.guidance_model.names != self.data["names"]:
            raise ValueError("Guidance model class names do not match dataset class names.")

        self.stride = max(int(self.guidance_model.stride.max()), 32)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=self.stride, floor=self.stride, max_dim=1)

        # 2. Instantiate STN defense model
        self.stn = SpatialTransformer(
            max_degrees=self.stn_cfg["degrees"] * 1.5,
            max_translate=self.stn_cfg["translate"] * 1.5,
            fill=114.0 / 255.0,
        ).to(self.device)
        self.stn.stride = self.guidance_model.stride
        self.stn.names = self.guidance_model.names
        # Disable detector EMA because STN trains directly with Adam
        self.ema = ModelEMA(self.stn)
        self.ema.enabled = False
        self.stn.loss = self._loss
        self.model = self.stn  # BaseTrainer uses self.model

        # 3. Setup attack model for online Worst-of-k attack
        attack_weights = self.stn_cfg.get("attack_weights")
        if attack_weights is None or str(attack_weights).strip().lower() in {"", "none", "current"}:
            self.attack_model = self.guidance_model
            self.stn_cfg["attack_weights"] = self.args.model
        else:
            self.attack_model, _ = load_checkpoint(attack_weights, device=self.device)
            self.stn_cfg["attack_weights"] = str(Path(self.attack_model.pt_path).resolve())
            self.attack_model.eval().requires_grad_(False)
            if hasattr(self.attack_model, "args") and isinstance(self.attack_model.args, dict):
                self.attack_model.args = IterableSimpleNamespace(**self.attack_model.args)

        # 4. Instantiate online SpatialWorstOfKWithMetadata attacker
        self.attacker = SpatialWorstOfKWithMetadata(
            model=self.attack_model,
            k=self.stn_cfg["worstk_k"],
            degrees=self.stn_cfg["degrees"],
            translate=self.stn_cfg["translate"],
            img_size=self.args.imgsz,
            fill=114.0 / 255.0,
            seed=int(self.args.seed) if self.args.seed is not None else None,
        )

        # 5. Build standard YOLO dataloaders
        self.args.nbs = self.batch_size
        self._build_train_pipeline()

        # Training runtime utilities required by BaseTrainer._do_train
        self.freeze_layer_names = []
        self.amp = self.args.amp not in {False, "fp32"}
        self.amp = bool(self.amp)
        if self.device.type == "npu":
            import torch_npu

            self.scaler = torch_npu.npu.amp.GradScaler(enabled=self.amp and self.args.amp != "bf16")
        elif self.device.type == "cpu":
            self.scaler = torch.amp.GradScaler("cpu", enabled=False)
        else:
            self.scaler = (
                torch.amp.GradScaler(
                    self.device.type if self.device.type == "xpu" else "cuda",
                    enabled=self.amp and self.args.amp != "bf16",
                )
                if TORCH_2_4
                else torch.cuda.amp.GradScaler(enabled=self.amp and self.args.amp != "bf16")
            )
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        if hasattr(self, "scheduler") and (not hasattr(self.scheduler, "last_epoch") or self.scheduler.last_epoch < 0):
            self.scheduler.last_epoch = self.start_epoch - 1

        # 6. Setup validator
        self.validator = self.get_validator()
        self.loss_names = ("total_loss", "geom_loss", "recon_loss", "det_loss")
        self.metrics = {}
        self.best_fitness = 0.0

        # 7. Restore from checkpoint if resuming
        if self.stn_checkpoint is not None:
            ckpt = self.stn_checkpoint
            self.stn.load_state_dict(ckpt["stn"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt and hasattr(self, "scheduler"):
                self.scheduler.load_state_dict(ckpt["scheduler"])
            self.start_epoch = ckpt["epoch"] + 1
            self.best_fitness = float(ckpt.get("best_fitness", 0.0))
            if self.start_epoch >= self.epochs:
                raise ValueError(f"--epochs ({self.epochs}) must exceed checkpoint epoch ({self.start_epoch}).")
            self.stn_checkpoint = None

        YAML.save(self.save_dir / "args.yaml", vars(self.args))
        YAML.save(self.save_dir / "stn_config.yaml", self.stn_cfg)
        LOGGER.info(
            f"STN Trainer initialized: guidance={self.args.model}, k={self.stn_cfg['worstk_k']}, "
            f"deg={self.stn_cfg['degrees']}, trans={self.stn_cfg['translate']}"
        )
        self.run_callbacks("on_pretrain_routine_end")

    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        """Build standard YOLO dataset using guidance model stride."""
        gs = max(int(unwrap_model(self.guidance_model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def _build_train_pipeline(self):
        """Build dataloaders and configure guidance model detection attributes."""
        self.args.nbs = self.batch_size
        BaseTrainer._build_train_pipeline(self)
        if hasattr(unwrap_model(self.guidance_model), "set_head_attr"):
            datasets = {"train": self.train_loader.dataset, "val": self.test_loader.dataset}
            yolo.detect.DetectionValidator._check_max_det(self.args, datasets)
            unwrap_model(self.guidance_model).set_head_attr(max_det=self.args.max_det)

    def set_class_weights(self):
        """No class weights needed for STN localization training."""
        pass

    def build_optimizer(self, model: nn.Module, **kwargs):
        """Build optimizer updating only STN localization network parameters."""
        lr = float(self.args.lr0) if hasattr(self.args, "lr0") else 1e-3
        weight_decay = float(self.args.weight_decay) if hasattr(self.args, "weight_decay") else 1e-5
        optimizer = torch.optim.Adam(
            self.stn.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        return optimizer

    def preprocess_batch(self, batch: dict) -> dict:
        """Generate online Worst-of-k attack while preserving clean images and GT metadata."""
        batch = super().preprocess_batch(batch)
        # SpatialWorstOfKWithMetadata populates batch['clean_img'], batch['gt_matrix'], batch['gt_params']
        self.attacker.forward_batch(batch)
        return batch

    def _loss(self, batch: dict, preds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """Compute composite STN loss: geometry + masked reconstruction + optional frozen-YOLO detection."""
        images = batch["img"]  # Attacked images [B, C, H, W]
        clean_images = batch["clean_img"]
        gt_matrices = batch["gt_matrix"]
        _, _, h, w = images.shape

        if preds is None:
            rect_images, pred_matrices = self.stn(images)
        else:
            rect_images, pred_matrices = preds

        w_geom = self.stn_cfg["loss_geom_weight"]
        w_recon = self.stn_cfg["loss_recon_weight"]
        w_det = self.stn_cfg["loss_det_weight"]

        # 1. Normalized control-point geometric loss
        if w_geom > 0.0:
            loss_geom = compute_control_points_loss(pred_matrices, gt_matrices, height=h, width=w)
        else:
            loss_geom = torch.tensor(0.0, device=images.device)

        # 2. Masked image reconstruction loss (excluding unrecoverable padding/crops)
        if w_recon > 0.0:
            mask = compute_validity_mask(gt_matrices, pred_matrices, height=h, width=w)
            loss_recon = compute_masked_recon_loss(rect_images, clean_images, mask)
        else:
            loss_recon = torch.tensor(0.0, device=images.device)

        # 3. Optional frozen-YOLO detection loss on rectified image with clean ground truth
        if w_det > 0.0:
            clean_batch = BatchContainer(
                img=rect_images,
                batch_idx=batch["clean_batch_idx"],
                cls=batch["clean_cls"],
                bboxes=batch["clean_bboxes"],
            )
            # Evaluate guidance model detection loss
            guidance_training = self.guidance_model.training
            try:
                self.guidance_model.train()
                for m in self.guidance_model.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)):
                        m.eval()
                det_preds = self.guidance_model(rect_images, augment=False)
                det_loss_items, _ = self.guidance_loss(clean_batch, det_preds)
                loss_det = det_loss_items.sum()
            finally:
                self.guidance_model.train(guidance_training)
        else:
            loss_det = torch.tensor(0.0, device=images.device)

        total_loss = w_geom * loss_geom + w_recon * loss_recon + w_det * loss_det

        loss_items = {
            "total_loss": total_loss.detach(),
            "geom_loss": loss_geom.detach(),
            "recon_loss": loss_recon.detach(),
            "det_loss": loss_det.detach(),
        }
        return total_loss, loss_items

    def get_validator(self):
        """Construct independent STN validation pipeline."""
        from ultralytics.models.yolo.detect.val_stn import STNDetectionValidator

        validator = STNDetectionValidator(
            dataloader=self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator.stn = self.stn
        validator.guidance_model = self.guidance_model
        validator.attacker = self.attacker
        return validator

    def validate(self):
        """Evaluate STN defense on validation dataset and return robust metrics dictionary."""
        if (self.epoch + 1) % self.stn_cfg["val_interval"] != 0 and (self.epoch + 1) < self.epochs:
            return {}, 0.0

        self.stn.eval()
        self.validator.stn = self.stn
        self.validator.guidance_model = self.guidance_model
        self.validator.attacker = self.attacker

        results = self.validator(trainer=self)
        worstk_map = float(results.get("worstk/mAP50-95", 0.0))
        self.stn.train().requires_grad_(True)
        return results, worstk_map

    def save_model(self):
        """Save dedicated STN checkpoints; best.pt is selected by validation Worst-of-k mAP50-95."""
        checkpoint = {
            "epoch": self.epoch,
            "stn": self.stn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_fitness": self.best_fitness,
            "train_args": vars(self.args),
            "stn_config": self.stn_cfg,
            "guidance_model": str(self.args.model),
            "metrics": getattr(self, "metrics", {}),
        }

        # Save last.pt
        last_path = self.wdir / "last.pt"
        torch.save(checkpoint, last_path)

        # Update best.pt if Worst-of-k mAP50-95 improved
        current_fitness = float(self.fitness) if self.fitness is not None else 0.0
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            checkpoint["best_fitness"] = self.best_fitness
            best_path = self.wdir / "best.pt"
            torch.save(checkpoint, best_path)
            LOGGER.info(f"STN new best checkpoint saved: Worst-of-k mAP50-95={current_fitness:.5f}")

        if self.save_period > 0 and (self.epoch + 1) % self.save_period == 0:
            torch.save(checkpoint, self.wdir / f"epoch{self.epoch + 1}.pt")
        return True

    def final_eval(self):
        """Report final evaluation results for the trained STN model."""
        best_path = self.wdir / "best.pt"
        LOGGER.info(f"STN training completed. Best checkpoint: {best_path} (mAP50-95={self.best_fitness:.5f})")
