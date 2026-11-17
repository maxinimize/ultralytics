# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import contextlib
import csv
import io
import math
import random
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import distributed as dist
from torch import nn

from ultralytics import __version__
from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch
from ultralytics.attacks.attack_utils import rollback_batch_state, setup_attack_model, snapshot_batch_state
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.augment import Compose, Format, LetterBox, v8_transforms
from ultralytics.data.dataset import YOLODataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect.val_adv_adaptive import DetectionValidator as DetectionValidatorAdvAdaptive
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, GIT, LOGGER, RANK, TQDM, colorstr
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
    if "," in val:
        parts = val.split(",")
    else:
        parts = val.split()
    return [type_conv(p.strip()) for p in parts if p.strip()]


def _load_saved_args(resume: Any) -> dict[str, Any]:
    """Extract saved configuration from checkpoint and sidecar files on resume.

    Priority:
    1. Checkpoint train_args (highest authority for custom research arguments).
    2. Sidecar phase2_fgd.yaml (for Phase 2 FGD runs).
    3. Sidecar args.yaml (standard Ultralytics run configuration).
    """
    if not resume:
        return {}

    ckpt_path = None
    if isinstance(resume, (str, Path)):
        p = Path(resume)
        if p.exists():
            ckpt_path = p.resolve()
        else:
            try:
                from ultralytics.utils.checks import check_file

                ckpt_path = Path(check_file(str(resume))).resolve()
            except Exception:
                ckpt_path = None
    elif resume is True:
        try:
            from ultralytics.utils.files import get_latest_run

            ckpt_path = Path(get_latest_run()).resolve()
        except Exception:
            ckpt_path = None

    if not ckpt_path or not ckpt_path.is_file():
        return {}

    saved_args: dict[str, Any] = {}

    # 1. Read sidecar args.yaml if present
    for parent in (ckpt_path.parent.parent, ckpt_path.parent):
        args_yaml = parent / "args.yaml"
        if args_yaml.is_file():
            try:
                import yaml

                with open(args_yaml, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if isinstance(content, dict):
                        saved_args.update(content)
                break
            except Exception as e:
                LOGGER.warning(f"Could not load saved configuration from {args_yaml}: {e}")

    # 2. Read sidecar phase2_fgd.yaml if present
    for parent in (ckpt_path.parent.parent, ckpt_path.parent):
        phase2_yaml = parent / "phase2_fgd.yaml"
        if phase2_yaml.is_file():
            try:
                import yaml

                with open(phase2_yaml, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if isinstance(content, dict):
                        saved_args.update(content)
                break
            except Exception as e:
                LOGGER.warning(f"Could not load saved configuration from {phase2_yaml}: {e}")

    # 3. Read checkpoint train_args directly (highest priority)
    try:
        from ultralytics.nn.tasks import torch_safe_load

        ckpt, _ = torch_safe_load(ckpt_path)
        if isinstance(ckpt, dict):
            train_args = ckpt.get("train_args")
            if isinstance(train_args, dict):
                saved_args.update(train_args)
            if "stage" in ckpt and "stage" not in saved_args:
                saved_args["stage"] = ckpt["stage"]
    except Exception as e:
        LOGGER.warning(f"Could not load train_args from checkpoint {ckpt_path}: {e}")

    return saved_args


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

        is_update_epoch = epoch % self.update_interval == 0
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
                f"  mClean: {clean_map:.4f}, Degradations: "
                + ", ".join(f"{att}: {d:.4f}" for att, d in zip(self.attack_names, diffs))
                + "\n"
                "  Computed new ratios for next epochs: "
                + ", ".join(f"{k}: {v:.4f}" for k, v in self.next_train_ratios.items())
                + "\n"
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

    def __init__(
        self,
        cfg,
        img_path,
        batch,
        data,
        mode="train",
        rect=False,
        stride=32,
        attack_names=None,
        attack_ratios=None,
        p_clean=0.25,
        **kwargs,
    ):
        self.cfg = cfg
        self.attack_names = (
            list(attack_names) if attack_names is not None else parse_list_arg(getattr(cfg, "attack_name", "pgd"), str)
        )
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
            self.attack_ratios = self.attack_ratios[: len(self.attack_names)]

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

        seed_val = getattr(self.cfg, "seed", None)
        init_seed = int(DEFAULT_CFG.seed if seed_val is None else seed_val)
        self.update_virtual_samples(seed=init_seed)
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

    def update_virtual_samples(
        self, attack_ratios: list[float] | dict[str, float] | None = None, seed: int | None = None
    ):
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

        if seed is not None:
            rng = random.Random(int(seed))
        else:
            seed_val = getattr(self.cfg, "seed", None)
            init_seed = int(DEFAULT_CFG.seed if seed_val is None else seed_val)
            rng = random.Random(init_seed)
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
            f"Ratios: "
            + ", ".join(
                f"{c}: {splits[i] - (splits[i - 1] if i > 0 else 0)}/{n_raw} ({norm_ratios[i]:.3f})"
                for i, c in enumerate(cats)
            )
        )

    def build_virtual_samples(self):
        """Backward-compatible alias for update_virtual_samples."""
        seed_val = getattr(self.cfg, "seed", None)
        init_seed = int(DEFAULT_CFG.seed if seed_val is None else seed_val)
        self.update_virtual_samples(seed=init_seed)

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

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
        attack_weights="",
        strict_attacks: bool = True,
    ):
        if overrides is None:
            overrides = {}

        overrides.setdefault("amp", "bf16")

        saved_args = _load_saved_args(overrides.get("resume", False))

        raw_strict = overrides.pop("strict_attacks", saved_args.get("strict_attacks", strict_attacks))
        self.strict_attacks = (
            raw_strict if isinstance(raw_strict, bool) else str(raw_strict).lower() in ("true", "1", "yes")
        )

        popped_attack_weights = overrides.pop("attack_weights", None)
        if popped_attack_weights is not None:
            self._attack_weights_arg = popped_attack_weights
        elif attack_weights:
            self._attack_weights_arg = attack_weights
        elif "attack_weights" in saved_args:
            self._attack_weights_arg = saved_args["attack_weights"]
        else:
            self._attack_weights_arg = ""

        if "attack_name" in overrides:
            self._attack_name_arg = overrides.pop("attack_name")
        elif "attack_name" in saved_args:
            self._attack_name_arg = saved_args["attack_name"]
        else:
            self._attack_name_arg = ["pgd", "bim", "mim", "worstk"]

        if "attack_ratio" in overrides:
            self._attack_ratio_arg = overrides.pop("attack_ratio")
        elif "attack_ratio" in saved_args:
            self._attack_ratio_arg = saved_args["attack_ratio"]
        else:
            self._attack_ratio_arg = [0.10, 0.10, 0.40, 0.15]

        if "p_clean" in overrides:
            self._p_clean_arg = overrides.pop("p_clean")
        elif "p_clean" in saved_args:
            self._p_clean_arg = saved_args["p_clean"]
        else:
            self._p_clean_arg = 0.25

        if "tau" in overrides:
            self._tau_arg = overrides.pop("tau")
        elif "tau" in saved_args:
            self._tau_arg = saved_args["tau"]
        else:
            self._tau_arg = 5.0

        if "update_interval" in overrides:
            self._update_interval_arg = overrides.pop("update_interval")
        elif "update_interval" in saved_args:
            self._update_interval_arg = saved_args["update_interval"]
        else:
            self._update_interval_arg = 5

        if "eval_weights" in overrides:
            self._eval_weights_arg = overrides.pop("eval_weights")
        elif "eval_weights" in saved_args:
            self._eval_weights_arg = saved_args["eval_weights"]
        else:
            self._eval_weights_arg = [0.25, 0.15, 0.15, 0.20]
        overrides.pop("attack_num", None)

        super().__init__(cfg, overrides, _callbacks)
        if hasattr(self.args, "attack_num"):
            delattr(self.args, "attack_num")

        self.args.strict_attacks = self.strict_attacks
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
            self.attack_ratios = self.attack_ratios[: len(self.attack_names)]

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
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
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
            for att in self.attack_names:
                if att.lower().strip() not in {"raw", "none", "clean"}:
                    if att not in self.attackers or self.attackers[att] is None:
                        if self.strict_attacks:
                            raise RuntimeError(f"Attacker '{att}' failed to initialize.")
        except Exception as e:
            if getattr(self, "strict_attacks", True):
                raise
            LOGGER.warning(f"Failed to load attack model or attacker: {e}")
            import traceback

            LOGGER.warning(traceback.format_exc())
            self.attack_model = None
            self.attacker = None
            self.attackers = {}

    def build_attacker(self, name, model, imgsz, seed=None):
        """Build the online attack."""
        seed_val = getattr(self.args, "seed", None) if seed is None else seed
        return build_attacker(name, model=model, img_size=imgsz, seed=seed_val)

    def preprocess_for_detector(self, images):
        return images

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def setup_model(self):
        ckpt = super().setup_model()
        if self.resume and ckpt and "random_state" in ckpt:
            self._resume_random_state = ckpt["random_state"]
        if self.resume and ckpt and ("attacker_state" in ckpt or "attackers_state" in ckpt):
            self._resume_attacker_state = ckpt.get("attacker_state", ckpt.get("attackers_state"))
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
        if ckpt and "strict_attacks" in ckpt:
            self.strict_attacks = bool(ckpt["strict_attacks"])
            self.args.strict_attacks = self.strict_attacks
        if ckpt and "adaptive_state" in ckpt:
            self.adaptive_controller.load_state_dict(ckpt["adaptive_state"])
            self.attack_ratios = [self.adaptive_controller.current_train_ratios.get(a, 0.0) for a in self.attack_names]
            self.args.attack_ratio = self.attack_ratios
            LOGGER.info(
                f"Restored adaptive_state from checkpoint: current ratios = {self.adaptive_controller.current_train_ratios}, "
                f"last_updated_epoch = {self.adaptive_controller.last_updated_epoch}"
            )
        if self.resume and ckpt and "random_state" in ckpt:
            self._resume_random_state = ckpt["random_state"]
        if self.resume and ckpt and ("attacker_state" in ckpt or "attackers_state" in ckpt):
            self._resume_attacker_state = ckpt.get("attacker_state", ckpt.get("attackers_state"))
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

    def _restore_resume_rng_and_attackers(self):
        """Restore training RNG states and attacker generator states on resume after setup_train completes."""
        if not getattr(self, "resume", False):
            self._resume_attacker_state = None
            self._resume_random_state = None
            return

        if getattr(self, "_resume_attacker_state", None):
            atk_st = self._resume_attacker_state
            if hasattr(self, "attackers") and isinstance(self.attackers, dict):
                for att_name, st in atk_st.items():
                    if att_name in self.attackers and hasattr(self.attackers[att_name], "load_state_dict"):
                        self.attackers[att_name].load_state_dict(st)
            if hasattr(self, "attacker") and self.attacker is not None and "attacker" in atk_st:
                if hasattr(self.attacker, "load_state_dict"):
                    self.attacker.load_state_dict(atk_st["attacker"])
            self._resume_attacker_state = None

        if getattr(self, "_resume_random_state", None):
            rnd = self._resume_random_state
            if "torch" in rnd and rnd["torch"] is not None:
                torch.set_rng_state(rnd["torch"])
            if torch.cuda.is_available() and rnd.get("cuda") is not None:
                with contextlib.suppress(RuntimeError, TypeError, ValueError):
                    torch.cuda.set_rng_state_all(rnd["cuda"])
            if "numpy" in rnd and rnd["numpy"] is not None:
                np.random.set_state(rnd["numpy"])
            if "python" in rnd and rnd["python"] is not None:
                random.setstate(rnd["python"])
            if "dataloader" in rnd and rnd["dataloader"] is not None:
                if hasattr(self, "train_loader") and getattr(self.train_loader, "generator", None) is not None:
                    self.train_loader.generator.set_state(rnd["dataloader"])
            self._resume_random_state = None
            LOGGER.info("Restored training random state and attacker states from checkpoint.")

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

        attacker_states = {}
        if hasattr(self, "attackers") and isinstance(self.attackers, dict):
            for att_name, att_obj in self.attackers.items():
                if hasattr(att_obj, "state_dict"):
                    attacker_states[att_name] = att_obj.state_dict()
        elif hasattr(self, "attacker") and self.attacker is not None and hasattr(self.attacker, "state_dict"):
            attacker_states["attacker"] = self.attacker.state_dict()

        random_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "dataloader": (
                self.train_loader.generator.get_state().clone()
                if hasattr(self, "train_loader") and getattr(self.train_loader, "generator", None) is not None
                else None
            ),
        }

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
                "attacker_state": attacker_states,
                "random_state": random_state,
                "strict_attacks": getattr(self, "strict_attacks", True),
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
        for arg in [
            "attack_weights",
            "attack_name",
            "attack_ratio",
            "attack_num",
            "p_clean",
            "tau",
            "update_interval",
            "eval_weights",
            "strict_attacks",
        ]:
            if hasattr(validator_args, arg):
                delattr(validator_args, arg)

        raw_att_w = (
            getattr(self, "attack_weights", None)
            or getattr(self, "_attack_weights_arg", None)
            or getattr(self.args, "attack_weights", None)
        )
        validator = DetectionValidatorAdvAdaptive(
            getattr(self, "test_loader", None),
            save_dir=self.save_dir,
            args=validator_args,
            _callbacks=self.callbacks,
            attack_weights=raw_att_w,
            strict_attacks=self.strict_attacks,
        )
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
                    mask |= batch_idx == idx

                sub_batch["cls"] = batch["cls"][mask]
                sub_batch["bboxes"] = batch["bboxes"][mask]

                idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
                old_batch_idx = batch_idx[mask]
                new_batch_idx = torch.tensor(
                    [idx_map[int(x)] for x in old_batch_idx], dtype=old_batch_idx.dtype, device=old_batch_idx.device
                )
                sub_batch["batch_idx"] = new_batch_idx

                for pre_k in ["clean_cls", "clean_bboxes", "pre_attack_cls", "pre_attack_bboxes"]:
                    if pre_k in batch and batch[pre_k] is not None:
                        sub_batch[pre_k] = batch[pre_k][mask]
                for pre_bi in ["clean_batch_idx", "pre_attack_batch_idx"]:
                    if pre_bi in batch and batch[pre_bi] is not None:
                        old_pbi = batch[pre_bi][mask]
                        sub_batch[pre_bi] = torch.tensor(
                            [idx_map[int(x)] for x in old_pbi],
                            dtype=old_pbi.dtype,
                            device=old_pbi.device,
                        )

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

        if self.resume:
            self._restore_resume_rng_and_attackers()

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        seed_val = getattr(self.args, "seed", None)
        base_seed = int(DEFAULT_CFG.seed if seed_val is None else seed_val)
        seed_source = "checkpoint" if self.resume else ("cli/override" if seed_val is not None else "default_cfg")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Experimental Seed: {base_seed} (source: {seed_source}), "
            f"Validation Seed: {base_seed}, Allocation Rule: epoch_seed = seed + epoch\n"
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
            seed_val = getattr(self.args, "seed", None)
            base_seed = int(DEFAULT_CFG.seed if seed_val is None else seed_val)
            epoch_seed = base_seed + epoch
            needs_reset = False
            if epoch == self.start_epoch and self.resume or epoch > self.start_epoch:
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

            self.epoch_attack_stats = {
                "attempted": {att: 0 for att in self.attack_names},
                "generated": {att: 0 for att in self.attack_names},
                "skipped": {att: 0 for att in self.attack_names},
                "failed": {att: 0 for att in self.attack_names},
                "actual_raw": 0,
            }

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

                    sample_types = batch.get("sample_type", [])
                    if not sample_types:
                        sample_types = ["raw"] * batch["img"].shape[0]

                    for st in sample_types:
                        if st in self.epoch_counts:
                            self.epoch_counts[st] += 1

                    sub_batches = self.split_batch_by_type(batch)
                    if "raw" in sub_batches:
                        self.epoch_attack_stats["actual_raw"] += len(sub_batches["raw"]["img"])

                    # Generate adversarial images for present attack groups on-the-fly
                    has_generated_adv = False
                    for att in sub_batches:
                        if att != "raw" and (
                            not getattr(self, "attackers", None)
                            or att not in self.attackers
                            or self.attackers[att] is None
                        ):
                            n_sub = len(sub_batches[att]["img"])
                            self.epoch_attack_stats["attempted"][att] += n_sub
                            if self.strict_attacks:
                                raise RuntimeError(
                                    f"Attacker '{att}' was not initialized for adversarial training at batch {i}."
                                )
                            self.epoch_attack_stats["failed"][att] += n_sub
                            self.epoch_attack_stats["actual_raw"] += n_sub

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
                                    n_sub = len(sub_batch["img"])
                                    self.epoch_attack_stats["attempted"][att] += n_sub

                                    has_targets = (
                                        "bboxes" in sub_batch
                                        and sub_batch["bboxes"] is not None
                                        and sub_batch["bboxes"].numel() > 0
                                    )
                                    if not has_targets and att != "worstk":
                                        # Legitimate skip due to no targets
                                        self.epoch_attack_stats["skipped"][att] += n_sub
                                        self.epoch_attack_stats["actual_raw"] += n_sub
                                        continue

                                    # Backup sub_batch before running attack for potential rollback
                                    input_img_shape = tuple(sub_batch["img"].shape)
                                    sub_backup = snapshot_batch_state(sub_batch)

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

                                        if imgs_adv is None:
                                            raise RuntimeError(
                                                f"Adversarial generation returned None for attack '{att}' at batch {i} on batch with targets."
                                            )
                                        if not torch.isfinite(imgs_adv).all():
                                            raise RuntimeError(
                                                f"Adversarial generation produced NaN or Inf for attack '{att}' at batch {i}."
                                            )
                                        if tuple(imgs_adv.shape) != input_img_shape:
                                            raise RuntimeError(
                                                f"Adversarial generation output shape {imgs_adv.shape} mismatch with input {input_img_shape} for attack '{att}' at batch {i}."
                                            )

                                        sub_batch["img"] = imgs_adv.detach().to(sub_batch["img"].dtype).to(self.device)
                                        self.epoch_attack_stats["generated"][att] += n_sub
                                        has_generated_adv = True
                                        del imgs_adv
                                    except Exception as e:
                                        if self.strict_attacks:
                                            raise RuntimeError(
                                                f"Adversarial generation failed for attack '{att}' at batch {i}: {e}"
                                            ) from e
                                        LOGGER.warning(
                                            f"Adversarial generation failed for attack '{att}' at batch {i}: {e}. Rolling back to raw."
                                        )
                                        # Rollback sub_batch to guarantee paired image and labels
                                        rollback_batch_state(sub_batch, sub_backup, attacker)
                                        self.epoch_attack_stats["failed"][att] += n_sub
                                        self.epoch_attack_stats["actual_raw"] += n_sub
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
            total_gen = sum(self.epoch_attack_stats["generated"].values())
            assert total_gen + self.epoch_attack_stats["actual_raw"] == total_epoch_samples, (
                f"Sample count conservation mismatch: generated ({total_gen}) + actual_raw ({self.epoch_attack_stats['actual_raw']}) != total planned ({total_epoch_samples})"
            )
            for att in self.attack_names:
                att_cnt = self.epoch_attack_stats["attempted"][att]
                gen_cnt = self.epoch_attack_stats["generated"][att]
                skip_cnt = self.epoch_attack_stats["skipped"][att]
                fail_cnt = self.epoch_attack_stats["failed"][att]
                assert att_cnt == gen_cnt + skip_cnt + fail_cnt, (
                    f"Attack count conservation violated for {att}: attempted ({att_cnt}) != gen ({gen_cnt}) + skip ({skip_cnt}) + fail ({fail_cnt})"
                )

            if total_epoch_samples > 0 and RANK in {-1, 0}:
                planned_ratios = {t: self.epoch_counts[t] / total_epoch_samples for t in self.epoch_counts}
                actual_counts = {"raw": self.epoch_attack_stats["actual_raw"]}
                for att in self.attack_names:
                    actual_counts[att] = self.epoch_attack_stats["generated"][att]
                total_actual = sum(actual_counts.values())
                actual_ratios = {t: actual_counts[t] / total_actual for t in actual_counts} if total_actual > 0 else {}
                LOGGER.info(
                    f"Epoch {epoch + 1} sample statistics (Planned vs Actual Executed):\n"
                    f"  Planned:  "
                    + ", ".join(f"{t}: {r:.3f} ({self.epoch_counts[t]})" for t, r in planned_ratios.items())
                    + "\n"
                    "  Actual:   "
                    + ", ".join(f"{t}: {r:.3f} ({actual_counts[t]})" for t, r in actual_ratios.items())
                    + "\n"
                    f"  Details:  attempted={self.epoch_attack_stats['attempted']}, generated={self.epoch_attack_stats['generated']}, "
                    f"skipped={self.epoch_attack_stats['skipped']}, failed={self.epoch_attack_stats['failed']}, actual_raw={self.epoch_attack_stats['actual_raw']}"
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
