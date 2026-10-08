"""Isolated, single-device FP32 FGD training using the existing engine and online attacks."""

from __future__ import annotations

import math
from copy import copy
from pathlib import Path

import torch

from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch
from ultralytics.cfg import get_cfg
from ultralytics.defenses.hgd_fgd import FGDModel
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.train_adv_test import DetectionTrainer as AdversarialTrainer
from ultralytics.models.yolo.detect.train_adv_test import YOLODatasetAdvTest, parse_list_arg
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, YAML, IterableSimpleNamespace, callbacks
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import TORCH_2_4, EarlyStopping, ModelEMA


class FGDTrainer(DetectionTrainer):
    """Reuse BaseTrainer's loop with paired online attacks and denoiser-only optimizer/checkpoint.

    Checkpoints retain the last epoch, not a purported best robustness score. Detection mAP evaluation and DDP are
    outside this initial baseline. Resume restores the optimizer, schedule, and original attack partition.
    """

    split_batch_by_type = AdversarialTrainer.split_batch_by_type

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """Separate research arguments from the standard Ultralytics configuration."""
        overrides = dict(overrides or {})
        defaults = dict(
            attack_name=["pgd", "bim", "mim"],
            attack_ratio=None,
            attack_weights=None,
        )
        self.fgd = {key: overrides.pop(key, value) for key, value in defaults.items()}
        baseline = dict(
            task="detect",
            amp=False,
            plots=False,
            val=False,
            optimizer="Adam",
            lr0=1e-4,
            weight_decay=0.0,
            warmup_epochs=0.0,
            patience=0,
            project="runs/fgd",
        )
        super().__init__(cfg, {**baseline, **overrides}, _callbacks)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if self.world_size > 1 or RANK != -1 or self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Initial FGD training supports one CPU or CUDA device.")
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        if self.args.amp or self.args.compile or self.args.distill_model or self.args.quantize == 8:
            raise ValueError("Initial FGD training requires FP32 without compile, QAT, or distillation.")
        if self.args.batch < 1 or self.args.rect or self.args.single_cls:
            raise ValueError("FGD requires an explicit positive batch size, rect=False, and single_cls=False.")
        if self.args.plots or self.args.val or self.args.warmup_epochs or self.args.time:
            raise ValueError("Initial FGD uses epoch-based training without detection validation, plots, or warmup.")
        self.attack_names = parse_list_arg(self.fgd["attack_name"])
        ratios = self.fgd["attack_ratio"]
        self.attack_ratios = (
            parse_list_arg(ratios, float)
            if ratios is not None
            else [1 / len(self.attack_names)] * len(self.attack_names)
            if self.attack_names
            else []
        )
        if (
            not self.attack_names
            or len(set(self.attack_names)) != len(self.attack_names)
            or set(self.attack_names) - {"pgd", "bim", "mim"}
        ):
            raise ValueError("Choose unique attack names from pgd, bim, mim.")
        if (
            len(self.attack_ratios) != len(self.attack_names)
            or any(not math.isfinite(r) or r < 0 for r in self.attack_ratios)
            or not 0 < sum(self.attack_ratios) <= 1
        ):
            raise ValueError("Provide one non-negative ratio per attack, with sum in (0, 1].")
        self.fgd.update(attack_name=self.attack_names, attack_ratio=self.attack_ratios)
        YAML.save(self.save_dir / "fgd.yaml", self.fgd)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Pass attack partition settings only to the existing adversarial training dataset."""
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        cfg = copy(self.args)
        cfg.attack_name, cfg.attack_ratio = self.attack_names, self.attack_ratios
        return YOLODatasetAdvTest(cfg, img_path, batch, self.data, stride=self.stride)

    def check_resume(self, overrides: dict):
        """Load FGD state dictionaries without routing them through YOLO checkpoint loading."""
        self.resume = bool(self.args.resume)
        self.fgd_checkpoint = None
        if self.resume:
            if not isinstance(self.args.resume, (str, Path)):
                raise ValueError("FGD resume requires an explicit path to weights/last.pt.")
            self.fgd_checkpoint = torch_load(self.args.resume, map_location="cpu")
            self.fgd = self.fgd_checkpoint["fgd_config"]
            saved = self.fgd_checkpoint["train_args"].copy()
            saved.update({k: overrides[k] for k in ("device", "workers", "epochs") if k in overrides})
            saved.update(resume=str(self.args.resume), exist_ok=True)
            self.args = get_cfg(DEFAULT_CFG, saved)

    def _setup_train(self):
        """Prepare frozen detector roles without detector training setup that re-enables frozen parameters."""
        self.teacher_model, _ = load_checkpoint(self.args.model, device=self.device)
        self.args.model = str(Path(self.teacher_model.pt_path).resolve())
        head = self.teacher_model.model[-1]
        if type(head) is not Detect or head.nl != 3 or head.export:
            raise ValueError("FGD requires an unfused three-scale Detect model with raw feature outputs.")
        if self.teacher_model.names != self.data["names"]:
            raise ValueError("Teacher class names/order must match the dataset; use a matching trained checkpoint.")
        self.stride = max(int(self.teacher_model.stride.max()), 32)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=self.stride, floor=self.stride, max_dim=1)
        self.model = FGDModel(self.teacher_model).to(self.device)
        attack_weights = self.fgd["attack_weights"]
        if attack_weights is None or str(attack_weights).strip().lower() in {"", "none", "current"}:
            attack_weights = self.args.model
        self.attack_model, _ = load_checkpoint(attack_weights, device=self.device)
        self.fgd["attack_weights"] = str(Path(self.attack_model.pt_path).resolve())
        if self.attack_model.names != self.data["names"]:
            raise ValueError("Attack model class names/order must match the dataset.")
        self.attack_model.requires_grad_(False).eval()
        self.attack_model.args = IterableSimpleNamespace(**self.attack_model.args)
        self.attack_model.criterion = None
        self.attackers = {
            name: build_attacker(
                name,
                model=self.attack_model,
                img_size=self.args.imgsz,
            )
            for name in self.attack_names
        }
        self.freeze_layer_names = ["teacher_model"]
        self.amp = False
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=False) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=False)
        )
        self._build_train_pipeline()
        self.validator = self.get_validator()
        self.loss_names = ("fgd_loss", "P3", "P4", "P5")
        self.metrics = {}
        self.ema = ModelEMA(self.model.denoiser)
        self.ema.enabled = False
        self.stopper, self.stop = EarlyStopping(patience=0), False
        if self.fgd_checkpoint is not None:
            ckpt = self.fgd_checkpoint
            self.model.denoiser.load_state_dict(ckpt["denoiser"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.start_epoch = ckpt["epoch"] + 1
            if self.start_epoch >= self.epochs:
                raise ValueError("--epochs must exceed the completed checkpoint epoch count.")
            self.train_loader.dataset.virtual_samples = ckpt["attack_partition"]
            self.train_loader.reset()
            self.fgd_checkpoint = None
        self.scheduler.last_epoch = self.start_epoch - 1
        YAML.save(self.save_dir / "args.yaml", vars(self.args))
        YAML.save(self.save_dir / "fgd.yaml", self.fgd)
        LOGGER.info(
            "FGD: equal-weight P3/P4/P5 L1; last/periodic checkpoints only, no robustness-based best selection."
        )
        self.run_callbacks("on_pretrain_routine_end")

    def _build_train_pipeline(self):
        """Reuse data loading and optimizer scheduling, with one optimizer update per batch."""
        self.args.nbs = self.batch_size
        BaseTrainer._build_train_pipeline(self)

    def build_optimizer(self, model: FGDModel, **kwargs):
        """Expose only denoiser parameters to the existing optimizer builder."""
        return super().build_optimizer(model.denoiser, **kwargs)

    def preprocess_batch(self, batch: dict) -> dict:
        """Save augmented clean inputs before writing online attacks back at the same sample indices."""
        batch = super().preprocess_batch(batch)
        batch["clean_img"] = batch["img"].detach().clone()
        batch["fgd_type"] = list(batch["sample_type"])
        sub_batches = self.split_batch_by_type(batch)
        for name, attacker in self.attackers.items():
            if name not in sub_batches:
                continue
            indices = [i for i, kind in enumerate(batch["sample_type"]) if kind == name]
            adv = run_attack_on_batch(attacker, sub_batches[name])
            if adv is None:
                for i in indices:
                    batch["fgd_type"][i] = "raw"
            else:
                batch["img"][indices] = adv.detach()
        return batch

    def label_loss_items(self, loss_items: dict | None = None, prefix: str = "train"):
        """Report sample-weighted epoch means and counts, including actual generated attack groups."""
        if loss_items is None or prefix != "train":
            return super().label_loss_items(loss_items, prefix)
        result, totals, count = {}, 0, 0
        for group in ["raw", *self.attack_names]:
            n = self.model.group_counts.get(group, 0)
            result[f"{prefix}/count_{group}"] = n
            values = self.model.group_sums[group] / n if n else torch.full((3,), float("nan"))
            result[f"{prefix}/fgd_{group}"] = float(values.mean())
            result.update({f"{prefix}/{scale}_{group}": float(v) for scale, v in zip(("P3", "P4", "P5"), values)})
            if n:
                totals = totals + self.model.group_sums[group]
                count += n
        means = totals / count
        result.update({f"{prefix}/{scale}": float(v) for scale, v in zip(("P3", "P4", "P5"), means)})
        result[f"{prefix}/fgd_loss"] = float(means.mean())
        LOGGER.info("FGD sample counts: " + str(self.model.group_counts))
        return result

    def validate(self):
        """Do not use training feature loss as a proxy for detection robustness or checkpoint selection."""
        return {}, None

    def optimizer_step(self):
        """Reject non-finite denoiser gradients before the optimizer step can corrupt a checkpoint."""
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.denoiser.parameters()):
            raise FloatingPointError("Non-finite FGD denoiser gradient.")
        super().optimizer_step()

    def save_model(self):
        """Save only denoiser training state and detector references, retaining full optimizer precision."""
        checkpoint = dict(
            epoch=self.epoch,
            denoiser=self.model.denoiser.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            train_args=vars(self.args),
            fgd_config=self.fgd,
            guidance_model=str(self.args.model),
            attack_partition=self.train_loader.dataset.virtual_samples,
        )
        temporary = self.last.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(self.last)
        if self.save_period > 0 and (self.epoch + 1) % self.save_period == 0:
            torch.save(checkpoint, self.wdir / f"epoch{self.epoch + 1}.pt")
        return True

    def final_eval(self):
        """Leave FGD checkpoints resumable; detector final_eval expects an incompatible YOLO checkpoint."""
        LOGGER.info(f"FGD checkpoint: {self.last}. Robust detection mAP must be evaluated separately.")
