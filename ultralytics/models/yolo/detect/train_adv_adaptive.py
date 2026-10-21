# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import csv
import io
import math
import random
import time
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import distributed as dist

from ultralytics import __version__
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, GIT, LOGGER, RANK, colorstr, TQDM
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
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose, Format, LetterBox, v8_transforms

from ultralytics.models.yolo.detect.val_adv_adaptive import DetectionValidator as DetectionValidatorAdvAdaptive
from ultralytics.attacks.attack_utils import setup_attack_model
from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch


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
    val = str(val).strip()
    if not val:
        return []
    if ',' in val:
        parts = val.split(',')
    else:
        parts = val.split()
    return [type_conv(p.strip()) for p in parts if p.strip()]


class DynamicAttackRatioController:
    """Controller for dynamic attack ratio adversarial training based on validation mAP."""

    def __init__(
        self,
        attack_names: list[str] | None = None,
        initial_attack_ratios: list[float] | None = None,
        p_clean: float = 0.25,
        tau: float = 5.0,
        update_interval: int = 5,
        eval_weights: list[float] | dict[str, float] | None = None,
        save_dir: Path | str | None = None,
    ):
        self.attack_names = list(attack_names or ["pgd", "bim", "mim", "worstk"])
        self.p_clean = float(p_clean)
        self.tau = float(tau)
        self.update_interval = int(update_interval)
        self.save_dir = Path(save_dir) if save_dir else None

        default_eval = {"pgd": 0.25, "bim": 0.15, "mim": 0.15, "worstk": 0.20}
        if isinstance(eval_weights, dict):
            self.eval_weights = {str(k).lower().strip(): float(v) for k, v in eval_weights.items()}
        elif eval_weights is not None:
            parsed_ew = [float(x) for x in parse_list_arg(eval_weights, float)]
            self.eval_weights = {
                att: (parsed_ew[i] if i < len(parsed_ew) else default_eval.get(att, 0.25))
                for i, att in enumerate(self.attack_names)
            }
        else:
            self.eval_weights = {att: default_eval.get(att, 0.25) for att in self.attack_names}

        default_ratios = {"pgd": 0.10, "bim": 0.10, "mim": 0.40, "worstk": 0.15}
        if initial_attack_ratios is not None:
            parsed_r = [float(x) for x in parse_list_arg(initial_attack_ratios, float)]
            init_r = {
                att: (parsed_r[i] if i < len(parsed_r) else default_ratios.get(att, 0.10))
                for i, att in enumerate(self.attack_names)
            }
        else:
            init_r = {att: default_ratios.get(att, 0.10) for att in self.attack_names}

        self.initial_train_ratios = {"raw": self.p_clean, **init_r}
        self.current_train_ratios = copy(self.initial_train_ratios)
        self.next_train_ratios = copy(self.initial_train_ratios)
        self.last_updated_epoch = 0
        self.history = []

    def compute_fitness(self, attack_maps: dict[str, float]) -> float:
        """Compute fitness using normalized fixed evaluation weights. Clean weight is strictly 0."""
        w_sum = sum(self.eval_weights.get(a, 0.0) for a in self.attack_names)
        if w_sum <= 0:
            norm_w = {a: 1.0 / len(self.attack_names) for a in self.attack_names}
        else:
            norm_w = {a: self.eval_weights.get(a, 0.0) / w_sum for a in self.attack_names}

        fitness = sum(norm_w.get(a, 0.0) * float(attack_maps.get(a, 0.0)) for a in self.attack_names)
        return float(fitness)

    def step(
        self,
        epoch: int,
        attack_maps: dict[str, float],
        fitness: float | None = None,
        save_dir: Path | str | None = None,
    ) -> tuple[dict[str, float], bool]:
        """Process epoch validation results, update attack ratios if update interval reached, and log history.
        
        Args:
            epoch (int): 1-based epoch number (e.g. 1, 2, ..., 5, 6, ..., 10, 11).
            attack_maps (dict[str, float]): Map of attack names (and 'raw') to mAP50-95.
            fitness (float, optional): Fitness score. If None, computed via compute_fitness.
            save_dir (Path, optional): Directory to save adaptive_history.csv.
        """
        clean_map = attack_maps.get("raw", attack_maps.get("clean", None))
        if clean_map is None or not math.isfinite(clean_map):
            raise RuntimeError(f"Clean mAP is missing or non-finite: {clean_map}. Cannot update attack ratios!")

        for att in self.attack_names:
            if att not in attack_maps:
                raise RuntimeError(f"Validation mAP metric missing for attack '{att}'. Cannot silently fall back to Clean mAP!")
            m = attack_maps[att]
            if m is None or not math.isfinite(m):
                raise RuntimeError(f"Validation mAP metric for attack '{att}' is invalid ({m}). Cannot silently fall back to Clean mAP!")

        if fitness is None:
            fitness = self.compute_fitness(attack_maps)

        is_update_epoch = (epoch % self.update_interval == 0)
        used_train_ratios = copy(self.current_train_ratios)

        if is_update_epoch:
            diffs = [clean_map - float(attack_maps[att]) for att in self.attack_names]
            scaled = [self.tau * d for d in diffs]
            max_s = max(scaled)
            exps = [math.exp(s - max_s) for s in scaled]
            sum_exps = sum(exps)
            softmax_probs = [e / sum_exps for e in exps]

            budget = max(0.0, 1.0 - self.p_clean)
            new_ratios = {"raw": self.p_clean}
            for att, prob in zip(self.attack_names, softmax_probs):
                new_ratios[att] = budget * prob

            self.next_train_ratios = copy(new_ratios)
            self.last_updated_epoch = epoch
            LOGGER.info(
                f"\n[DynamicAttackRatioController] Epoch {epoch} reached update interval ({self.update_interval}).\n"
                f"  mClean: {clean_map:.4f}, Degradations: " + ", ".join(f"{att}: {d:.4f}" for att, d in zip(self.attack_names, diffs)) + "\n"
                f"  Computed new ratios for next epochs: " + ", ".join(f"{k}: {v:.4f}" for k, v in self.next_train_ratios.items()) + "\n"
            )
        else:
            self.next_train_ratios = copy(self.current_train_ratios)

        record = {
            "epoch": epoch,
            "mAP_raw": float(clean_map),
            **{f"mAP_{att}": float(attack_maps[att]) for att in self.attack_names},
            "train_ratio_raw": float(used_train_ratios.get("raw", self.p_clean)),
            **{f"train_ratio_{att}": float(used_train_ratios.get(att, 0.0)) for att in self.attack_names},
            "next_ratio_raw": float(self.next_train_ratios.get("raw", self.p_clean)),
            **{f"next_ratio_{att}": float(self.next_train_ratios.get(att, 0.0)) for att in self.attack_names},
            "fitness": float(fitness),
            "updated": bool(is_update_epoch),
        }
        self.history.append(record)

        target_dir = Path(save_dir) if save_dir else self.save_dir
        if target_dir:
            self._write_history_csv(target_dir / "adaptive_history.csv")

        return copy(self.next_train_ratios), is_update_epoch

    def _write_history_csv(self, csv_file: Path):
        """Write or update adaptive_history.csv."""
        if not self.history:
            return
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.history[0].keys())
        try:
            with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.history:
                    formatted_row = {}
                    for k, v in row.items():
                        if isinstance(v, float):
                            formatted_row[k] = f"{v:.5f}"
                        else:
                            formatted_row[k] = v
                    writer.writerow(formatted_row)
        except Exception as e:
            LOGGER.warning(f"Failed to write adaptive_history.csv: {e}")

    def state_dict(self) -> dict[str, Any]:
        """Serialize controller state for checkpointing."""
        return {
            "attack_names": list(self.attack_names),
            "p_clean": float(self.p_clean),
            "tau": float(self.tau),
            "update_interval": int(self.update_interval),
            "eval_weights": dict(self.eval_weights),
            "initial_train_ratios": dict(self.initial_train_ratios),
            "current_train_ratios": dict(self.current_train_ratios),
            "next_train_ratios": dict(self.next_train_ratios),
            "last_updated_epoch": int(self.last_updated_epoch),
            "history": list(self.history),
        }

    def load_state_dict(self, state: dict[str, Any]):
        """Load controller state from checkpoint."""
        if "attack_names" in state:
            self.attack_names = list(state["attack_names"])
        if "p_clean" in state:
            self.p_clean = float(state["p_clean"])
        if "tau" in state:
            self.tau = float(state["tau"])
        if "update_interval" in state:
            self.update_interval = int(state["update_interval"])
        if "eval_weights" in state:
            self.eval_weights = dict(state["eval_weights"])
        if "initial_train_ratios" in state:
            self.initial_train_ratios = dict(state["initial_train_ratios"])
        if "current_train_ratios" in state:
            self.current_train_ratios = dict(state["current_train_ratios"])
        if "next_train_ratios" in state:
            self.next_train_ratios = dict(state["next_train_ratios"])
        if "last_updated_epoch" in state:
            self.last_updated_epoch = int(state["last_updated_epoch"])
        if "history" in state:
            self.history = list(state["history"])


class YOLODatasetAdvAdaptive(YOLODataset):
    """Dataset class for loading clean images and mapping to online adversarial attack types with dynamic reallocation."""

    def __init__(self, cfg, img_path, batch, data, mode="train", rect=False, stride=32, attack_names=None, attack_ratios=None, p_clean=0.25, **kwargs):
        self.cfg = cfg
        self.attack_names = list(attack_names) if attack_names is not None else parse_list_arg(getattr(cfg, "attack_name", "pgd"), str)
        if not self.attack_names:
            self.attack_names = ["pgd", "bim", "mim", "worstk"]
        self.p_clean = float(p_clean if p_clean is not None else getattr(cfg, "p_clean", 0.25))

        if attack_ratios is not None:
            if isinstance(attack_ratios, dict):
                self.attack_ratios = [float(attack_ratios.get(att, 0.1)) for att in self.attack_names]
            else:
                self.attack_ratios = [float(x) for x in parse_list_arg(attack_ratios, float)]
        else:
            self.attack_ratios = parse_list_arg(getattr(cfg, "attack_ratio", 0.5), float)

        if len(self.attack_ratios) < len(self.attack_names):
            last = self.attack_ratios[-1] if self.attack_ratios else 0.1
            self.attack_ratios.extend([last] * (len(self.attack_names) - len(self.attack_ratios)))
        elif len(self.attack_ratios) > len(self.attack_names):
            self.attack_ratios = self.attack_ratios[:len(self.attack_names)]

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
            **kwargs
        )

        self.update_virtual_samples()
        self.transforms_raw = self.build_transforms(self.cfg)

    def build_transforms(self, hyp: dict | None = None) -> Compose:
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
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    def update_virtual_samples(self, attack_ratios: list[float] | dict[str, float] | None = None, seed: int | None = None):
        """Update attack ratios and rebuild virtual sample allocation.
        
        Guarantees that every image in the dataset is used exactly once per epoch,
        and images are randomly assigned to clean or attack categories.
        """
        if attack_ratios is not None:
            if isinstance(attack_ratios, dict):
                if "raw" in attack_ratios or "clean" in attack_ratios:
                    self.p_clean = float(attack_ratios.get("raw", attack_ratios.get("clean", self.p_clean)))
                self.attack_ratios = [float(attack_ratios.get(att, 0.0)) for att in self.attack_names]
            elif isinstance(attack_ratios, (list, tuple)):
                self.attack_ratios = [float(x) for x in attack_ratios]

        n_raw = len(self.im_files)
        self.adv_indices_by_attack = {att: list(range(n_raw)) for att in self.attack_names}

        cats = ["raw"] + list(self.attack_names)
        raw_ratios = [self.p_clean] + list(self.attack_ratios)
        total_r = sum(raw_ratios)
        if total_r > 0:
            norm_ratios = [r / total_r for r in raw_ratios]
        else:
            norm_ratios = [1.0] + [0.0] * len(self.attack_names)

        splits = []
        cum_ratio = 0.0
        for r in norm_ratios:
            cum_ratio += r
            splits.append(round(n_raw * cum_ratio))
        if splits:
            splits[-1] = n_raw

        rng = random.Random(seed) if seed is not None else random
        all_indices = list(range(n_raw))
        rng.shuffle(all_indices)

        self.virtual_samples = []
        start_idx = 0
        for cat, end_idx in zip(cats, splits):
            for idx in all_indices[start_idx:end_idx]:
                self.virtual_samples.append((idx, cat))
            start_idx = end_idx

        rng.shuffle(self.virtual_samples)
        LOGGER.info(
            f"YOLODatasetAdvAdaptive: Rebuilt {len(self.virtual_samples)} virtual samples. "
            f"Ratios: " + ", ".join(f"{c}: {splits[i] - (splits[i-1] if i > 0 else 0)}/{n_raw} ({norm_ratios[i]:.3f})" for i, c in enumerate(cats))
        )

    def build_virtual_samples(self):
        """Backward-compatible alias for update_virtual_samples."""
        self.update_virtual_samples()

    def __getitem__(self, index: int) -> dict[str, Any]:
        orig_idx, sample_type = self.virtual_samples[index]
        label = self.get_image_and_label(orig_idx)
        label = self.transforms_raw(label)
        label["sample_type"] = sample_type
        label["attack_name"] = sample_type if sample_type != "raw" else None
        return label

    def __len__(self) -> int:
        return len(self.virtual_samples)

    def close_mosaic(self, hyp: dict) -> None:
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms_raw = self.build_transforms(hyp)

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        union_keys = sorted(list(set().union(*(b.keys() for b in batch))))
        for b in batch:
            for k in union_keys:
                if k not in b:
                    b[k] = None
        batch = [dict(sorted(b.items())) for b in batch]

        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))

        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats", "sem_masks"}:
                valid_tensors = [v for v in value if isinstance(v, torch.Tensor)]
                if len(valid_tensors) == len(value):
                    new_batch[k] = torch.stack(valid_tensors, 0)
                else:
                    new_batch[k] = value
            elif k == "visuals":
                valid_tensors = [v for v in value if isinstance(v, torch.Tensor)]
                if len(valid_tensors) == len(value):
                    new_batch[k] = torch.nn.utils.rnn.pad_sequence(valid_tensors, batch_first=True)
                else:
                    new_batch[k] = value
            elif k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                valid_tensors = [v for v in value if isinstance(v, torch.Tensor)]
                if len(valid_tensors) == len(value):
                    new_batch[k] = torch.cat(valid_tensors, 0)
                else:
                    if valid_tensors:
                        new_batch[k] = torch.cat(valid_tensors, 0)
                    else:
                        new_batch[k] = None
            else:
                new_batch[k] = value

        if "batch_idx" in new_batch and new_batch["batch_idx"] is not None:
            new_batch_idx = []
            for i, b in enumerate(batch):
                b_idx = b.get("batch_idx")
                if b_idx is not None:
                    new_batch_idx.append(b_idx + i)
            if new_batch_idx:
                new_batch["batch_idx"] = torch.cat(new_batch_idx, 0)

        return new_batch


# Backward-compatible alias
YOLODatasetAdvTest = YOLODatasetAdvAdaptive


class DetectionTrainer(BaseTrainer):
    """A class extending BaseTrainer for dynamic attack ratio adversarial training."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None, attack_weights=""):
        if overrides is None:
            overrides = {}

        overrides.setdefault("amp", "bf16")

        saved_args = {}
        resume = overrides.get("resume", False)
        if resume and isinstance(resume, (str, Path)) and str(resume).endswith(".pt"):
            args_yaml = Path(resume).resolve().parent.parent / "args.yaml"
            if args_yaml.exists():
                try:
                    import yaml
                    with open(args_yaml, "r") as f:
                        saved_args = yaml.safe_load(f) or {}
                except Exception as e:
                    LOGGER.warning(f"Could not load attack configuration from {args_yaml}: {e}")

        popped_attack_weights = overrides.pop(
            "attack_weights", saved_args.get("attack_weights", "")
        )
        self._attack_weights_arg = attack_weights or popped_attack_weights
        self._attack_name_arg = overrides.pop(
            "attack_name", saved_args.get("attack_name", ["pgd", "bim", "mim", "worstk"])
        )
        self._attack_ratio_arg = overrides.pop(
            "attack_ratio", saved_args.get("attack_ratio", [0.10, 0.10, 0.40, 0.15])
        )
        self._p_clean_arg = overrides.pop(
            "p_clean", saved_args.get("p_clean", 0.25)
        )
        self._tau_arg = overrides.pop(
            "tau", saved_args.get("tau", 5.0)
        )
        self._update_interval_arg = overrides.pop(
            "update_interval", saved_args.get("update_interval", 5)
        )
        self._eval_weights_arg = overrides.pop(
            "eval_weights", saved_args.get("eval_weights", [0.25, 0.15, 0.15, 0.20])
        )
        overrides.pop("attack_num", None)

        super().__init__(cfg, overrides, _callbacks)
        if hasattr(self.args, "attack_num"):
            delattr(self.args, "attack_num")

        self.args.attack_weights = self._attack_weights_arg
        self.args.attack_name = self._attack_name_arg
        self.args.attack_ratio = self._attack_ratio_arg
        self.args.p_clean = float(self._p_clean_arg)
        self.args.tau = float(self._tau_arg)
        self.args.update_interval = int(self._update_interval_arg)
        self.args.eval_weights = self._eval_weights_arg

        self.attack_names = parse_list_arg(self.args.attack_name, str)
        if not self.attack_names:
            self.attack_names = ["pgd", "bim", "mim", "worstk"]
        self.attack_ratios = parse_list_arg(self.args.attack_ratio, float)
        if not self.attack_ratios:
            self.attack_ratios = [0.10, 0.10, 0.40, 0.15]

        if len(self.attack_ratios) < len(self.attack_names):
            last = self.attack_ratios[-1] if self.attack_ratios else 0.10
            self.attack_ratios.extend([last] * (len(self.attack_names) - len(self.attack_ratios)))
        elif len(self.attack_ratios) > len(self.attack_names):
            self.attack_ratios = self.attack_ratios[:len(self.attack_names)]

        self.attack_num = len(self.attack_names)
        self.p_clean = self.args.p_clean
        self.tau = self.args.tau
        self.update_interval = self.args.update_interval
        self.eval_weights = self.args.eval_weights

        self.attack_model = None
        self.attacker = None
        self.attackers = {}

        self.adaptive_controller = DynamicAttackRatioController(
            attack_names=self.attack_names,
            initial_attack_ratios=self.attack_ratios,
            p_clean=self.p_clean,
            tau=self.tau,
            update_interval=self.update_interval,
            eval_weights=self.eval_weights,
            save_dir=self.save_dir,
        )

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation."""
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode == "train":
            current_ratios = getattr(self, "adaptive_controller", None)
            ratios = current_ratios.current_train_ratios if current_ratios else self.attack_ratios
            return YOLODatasetAdvAdaptive(
                self.args,
                img_path,
                batch,
                self.data,
                mode=mode,
                rect=mode == "val",
                stride=gs,
                attack_names=self.attack_names,
                attack_ratios=ratios,
                p_clean=self.p_clean,
            )
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess a batch of images."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255

        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

        attack_weights = getattr(self.args, "attack_weights", None)
        LOGGER.info(f"Debug: attack_weights from args = {attack_weights}")

        use_current_model = (attack_weights is None) or (str(attack_weights).strip().lower() in {"", "none", "current"})

        try:
            imgsz = getattr(self.args, "imgsz", 640)
            if use_current_model:
                LOGGER.info("Adversarial training: using CURRENT training model as attack generator (white-box AT).")
                self.attack_model = None
                target_attack_model = unwrap_model(self.model)
            else:
                LOGGER.info(f"Adversarial training: loading fixed attack model from: {attack_weights}")
                self.attack_model = setup_attack_model(
                    attack_weights,
                    device=self.device,
                    nc=self.data["nc"],
                    training=False,
                    imgsz=imgsz,
                )
                LOGGER.info(f"Attack model loaded successfully from {attack_weights}")
                self.attack_model.eval()
                target_attack_model = self.attack_model

            self.attackers = {}
            for att in self.attack_names:
                if att.lower().strip() not in {"raw", "none", "clean"}:
                    self.attackers[att] = self.build_attacker(att, target_attack_model, imgsz)
                    LOGGER.info(f"attacker initialized: {att}")

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

    def build_attacker(self, name, model, imgsz):
        """Build the online attack."""
        return build_attacker(name, model=model, img_size=imgsz)

    def preprocess_for_detector(self, images):
        return images

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def setup_model(self):
        ckpt = super().setup_model()
        if self.resume and ckpt and ckpt.get("model") is not None:
            live_model = ckpt["model"]
            try:
                if isinstance(live_model, torch.nn.Module):
                    unwrap_model(self.model).load_state_dict(live_model.float().state_dict())
                elif isinstance(live_model, dict):
                    unwrap_model(self.model).load_state_dict(live_model)
                LOGGER.info("Resumed live training model weights from checkpoint 'model' field.")
            except Exception as e:
                LOGGER.warning(f"Failed to restore live model from checkpoint in setup_model: {e}")
        return ckpt

    def _load_checkpoint_state(self, ckpt):
        """Load optimizer, scaler, EMA, best_fitness, adaptive_state, and live training model from checkpoint."""
        super()._load_checkpoint_state(ckpt)
        if ckpt and "adaptive_state" in ckpt:
            self.adaptive_controller.load_state_dict(ckpt["adaptive_state"])
            self.attack_ratios = [self.adaptive_controller.current_train_ratios.get(a, 0.0) for a in self.attack_names]
            self.args.attack_ratio = self.attack_ratios
            LOGGER.info(
                f"Restored adaptive_state from checkpoint: current ratios = {self.adaptive_controller.current_train_ratios}, "
                f"last_updated_epoch = {self.adaptive_controller.last_updated_epoch}"
            )
        if self.resume and ckpt and ckpt.get("model") is not None:
            live_model = ckpt["model"]
            try:
                if isinstance(live_model, torch.nn.Module):
                    unwrap_model(self.model).load_state_dict(live_model.float().state_dict())
                elif isinstance(live_model, dict):
                    unwrap_model(self.model).load_state_dict(live_model)
                LOGGER.info("Verified/Restored live training model weights in _load_checkpoint_state.")
            except Exception as e:
                LOGGER.warning(f"Failed to restore live model in _load_checkpoint_state: {e}")

    def save_model(self):
        """Save model training checkpoints with live model weights and adaptive_state preserved."""
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

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"

        validator_args = copy(self.args)
        for arg in ['attack_weights', 'attack_name', 'attack_ratio', 'attack_num', 'p_clean', 'tau', 'update_interval', 'eval_weights']:
            if hasattr(validator_args, arg):
                delattr(validator_args, arg)

        validator = DetectionValidatorAdvAdaptive(
            getattr(self, "test_loader", None), save_dir=self.save_dir, args=validator_args, _callbacks=self.callbacks
        )
        validator.attacker = getattr(self, "attacker", None)
        validator.attackers = getattr(self, "attackers", {})
        validator.attack_names = self.attack_names
        validator.attack_ratios = self.attack_ratios
        validator.eval_weights = self.eval_weights
        return validator

    def label_loss_items(self, loss_items: dict[str, Any] | None = None, prefix: str = "train"):
        if loss_items is None:
            all_keys = [f"{prefix}/{x}" for x in self.loss_names]
            if prefix == "train" and hasattr(self, "attack_names"):
                for t in ["raw"] + self.attack_names:
                    all_keys.extend([f"{prefix}/{x}_{t}" for x in self.loss_names])
            return all_keys

        loss_dict = {f"{prefix}/{k}": round(float(v), 5) for k, v in loss_items.items()}
        if prefix == "train" and hasattr(self, "group_tloss"):
            for t, g_tloss in self.group_tloss.items():
                if g_tloss is not None:
                    for k, v in g_tloss.items():
                        loss_dict[f"{prefix}/{k}_{t}"] = round(float(v), 5)
        return loss_dict

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        del train_dataset
        return super().auto_batch(max_num_obj)

    def split_batch_by_type(self, batch: dict) -> dict[str, dict]:
        sample_types = batch["sample_type"]
        unique_types = set(sample_types)
        sub_batches = {}

        for t in unique_types:
            indices = [i for i, st in enumerate(sample_types) if st == t]
            sub_batch = {}
            sub_batch["img"] = batch["img"][indices]

            for k in ["im_file", "sample_type", "attack_name", "ori_shape", "ratio_pad"]:
                if k in batch:
                    sub_batch[k] = [batch[k][i] for i in indices]

            if "batch_idx" in batch:
                batch_idx = batch["batch_idx"]
                mask = torch.zeros_like(batch_idx, dtype=torch.bool)
                for idx in indices:
                    mask |= (batch_idx == idx)

                sub_batch["cls"] = batch["cls"][mask]
                sub_batch["bboxes"] = batch["bboxes"][mask]

                idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
                old_batch_idx = batch_idx[mask]
                new_batch_idx = torch.tensor([idx_map[int(x)] for x in old_batch_idx], dtype=old_batch_idx.dtype, device=old_batch_idx.device)
                sub_batch["batch_idx"] = new_batch_idx

            for k in ["masks", "keypoints", "segments", "obb", "visuals", "sem_masks"]:
                if k in batch and batch[k] is not None:
                    if k in {"visuals", "sem_masks"}:
                        sub_batch[k] = batch[k][indices]
                    elif k in {"masks", "keypoints", "segments", "obb"}:
                        sub_batch[k] = batch[k][mask]

            sub_batches[t] = sub_batch

        return sub_batches

    def _do_train(self):
        """Train the model with dynamic attack ratio adversarial training."""
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
            f"Starting dynamic attack ratio adversarial training for {self.epochs} epochs..."
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
            # Re-randomize image assignment and apply current training ratios
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

                    # Generate adversarial images for present attack groups on-the-fly
                    has_generated_adv = False
                    if getattr(self, "attackers", None):
                        is_current_model = (getattr(self, "attack_model", None) is None)
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
                                                imgs_adv = run_attack_on_batch(attacker, sub_batch, label_policy="largest_box")
                                        else:
                                            with torch.amp.autocast(device_type="cuda", enabled=False):
                                                imgs_adv = run_attack_on_batch(attacker, sub_batch, label_policy="largest_box")

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
                        orig_bi_dim = batch["batch_idx"].dim() if "batch_idx" in batch and batch["batch_idx"] is not None else 1
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
                                new_cls.append(sub_batch["cls"].view(-1, 1) if sub_batch["cls"].ndim == 1 else sub_batch["cls"])
                                new_bboxes.append(sub_batch["bboxes"].view(-1, 4) if sub_batch["bboxes"].ndim != 2 else sub_batch["bboxes"])

                        if new_batch_idx:
                            cat_bi = torch.cat(new_batch_idx, dim=0)
                            cat_cls = torch.cat(new_cls, dim=0)
                            cat_box = torch.cat(new_bboxes, dim=0)

                            # Sort targets by batch_idx to maintain strictly non-decreasing order required by Ultralytics loss
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

                    batch["img"] = self.preprocess_for_detector(batch["img"])
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
                LOGGER.info(f"Epoch {epoch + 1} processed sample ratios: " + ", ".join(f"{t}: {r:.3f}" for t, r in ratios.items()))

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
