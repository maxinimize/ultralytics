# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
import time
from copy import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr, TQDM
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model, unset_deterministic, autocast
import warnings
from torch import distributed as dist
from pathlib import Path
import cv2
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose, Format, LetterBox, v8_transforms

from ultralytics.models.yolo.detect.val_adv_pregen_multi import DetectionValidator as DetectionValidatorAdv
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
    # assume string
    val = str(val).strip()
    if not val:
        return []
    # If it contains commas, split by comma; otherwise split by whitespace
    if ',' in val:
        parts = val.split(',')
    else:
        parts = val.split()
    return [type_conv(p.strip()) for p in parts if p.strip()]


class YOLODatasetAdvTest(YOLODataset):
    """Dataset class for loading clean images and their corresponding pre-generated adversarial images with multiple attacks."""

    def __init__(self, cfg, img_path, batch, data, mode="train", rect=False, stride=32, **kwargs):
        """Initialize YOLODatasetAdvTest with configuration and extract attack parameters."""
        self.cfg = cfg
        self.attack_num = int(getattr(cfg, "attack_num", 1))
        
        # Parse attack name and ratio
        self.attack_names = parse_list_arg(getattr(cfg, "attack_name", "pgd"), str)
        self.attack_ratios = parse_list_arg(getattr(cfg, "attack_ratio", 0.5), float)
        
        # Align lengths with attack_num
        if len(self.attack_names) < self.attack_num:
            last = self.attack_names[-1] if self.attack_names else "pgd"
            self.attack_names.extend([last] * (self.attack_num - len(self.attack_names)))
        else:
            self.attack_names = self.attack_names[:self.attack_num]
            
        if len(self.attack_ratios) < self.attack_num:
            last = self.attack_ratios[-1] if self.attack_ratios else 0.5
            self.attack_ratios.extend([last] * (self.attack_num - len(self.attack_ratios)))
        else:
            self.attack_ratios = self.attack_ratios[:self.attack_num]

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
        
        self.use_pregenerated_adv = getattr(self.cfg, "use_pregenerated_adv", False)
        
        # Build virtual samples
        self.build_virtual_samples()
        
        # Setup separate transform pipelines
        self.transforms_raw = self.build_transforms(self.cfg)
        self.transforms_adv = self.build_adv_transforms(self.cfg)

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """Build transforms for clean images."""
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

    def build_adv_transforms(self, hyp: dict | None = None) -> Compose:
        """Build minimal transforms for adversarial images."""
        transforms = [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
        transforms = Compose(transforms)
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

    def build_virtual_samples(self):
        """Build virtual samples mapped to raw or specific adversarial counterparts based on ratios."""
        self.adv_indices_by_attack = {}
        n_raw = len(self.im_files)
        
        for att in self.attack_names:
            self.adv_indices_by_attack[att] = []
            if self.use_pregenerated_adv:
                for i, f in enumerate(self.im_files):
                    p = Path(f)
                    adv_path = p.parent / f"{p.stem}_{att}{p.suffix}"
                    if adv_path.exists():
                        self.adv_indices_by_attack[att].append(i)
                LOGGER.info(f"YOLODatasetAdvTest: Found {len(self.adv_indices_by_attack[att])}/{n_raw} pre-generated images for attack: {att}")
            else:
                self.adv_indices_by_attack[att] = list(range(n_raw))

        ratio_sum = sum(self.attack_ratios)

        # Disjoint partitioning: every image in the dataset is used exactly once per epoch
        # either as clean (raw) or as one of the specified adversarial attack types.
        if ratio_sum <= 0.0:
            cats = ["raw"]
            ratios = [1.0]
        else:
            cats = ["raw"] + self.attack_names
            ratios = [1.0 - ratio_sum] + self.attack_ratios

        # Compute cumulative split points
        splits = []
        cum_ratio = 0.0
        for r in ratios:
            cum_ratio += r
            splits.append(round(n_raw * cum_ratio))
        if splits:
            splits[-1] = n_raw

        # Shuffle indices to randomly distribute the images across categories
        all_indices = list(range(n_raw))
        random.shuffle(all_indices)

        self.virtual_samples = []
        start_idx = 0
        for cat, end_idx in zip(cats, splits):
            for idx in all_indices[start_idx:end_idx]:
                self.virtual_samples.append((idx, cat))
            start_idx = end_idx

        # Shuffle virtual samples so that batches contain a mix of different categories
        random.shuffle(self.virtual_samples)
        LOGGER.info(f"YOLODatasetAdvTest: Built {len(self.virtual_samples)} virtual samples.")

    def load_adv_image(self, i: int, attack_name: str, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load pre-generated adversarial image from disk."""
        orig_path = self.im_files[i]
        p = Path(orig_path)
        adv_path = p.parent / f"{p.stem}_{attack_name}{p.suffix}"
        
        if not adv_path.exists():
            LOGGER.warning(f"Adversarial image not found: {adv_path}. Falling back to clean image {orig_path}.")
            adv_path = orig_path

        im = cv2.imread(str(adv_path), flags=self.cv2_flag)
        if im is None:
            raise FileNotFoundError(f"Failed to load image: {adv_path}")
            
        h0, w0 = im.shape[:2]
        if rect_mode:
            r = self.imgsz / max(h0, w0)
            if r != 1:
                w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        if im.ndim == 2:
            im = im[..., None]
        return im, (h0, w0), im.shape[:2]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return transformed clean or adversarial label information for given virtual index."""
        orig_idx, sample_type = self.virtual_samples[index]
        
        if sample_type == "raw":
            label = self.get_image_and_label(orig_idx)
            label = self.transforms_raw(label)
            label["sample_type"] = "raw"
            label["attack_name"] = None
        else:
            if self.use_pregenerated_adv:
                original_load_image = self.load_image
                try:
                    self.load_image = lambda idx, rect_mode=True: self.load_adv_image(idx, sample_type, rect_mode=rect_mode)
                    label = self.get_image_and_label(orig_idx)
                finally:
                    self.load_image = original_load_image
                label = self.transforms_adv(label)
            else:
                # If on-the-fly, load raw image and use raw transforms (adversarial is generated in training loop)
                label = self.get_image_and_label(orig_idx)
                label = self.transforms_raw(label)
            
            label["sample_type"] = sample_type
            label["attack_name"] = sample_type
            
        return label

    def __len__(self) -> int:
        """Return length of virtual samples list."""
        return len(self.virtual_samples)

    def close_mosaic(self, hyp: dict) -> None:
        """Disable mosaic, copy_paste, mixup and cutmix augmentations for raw images."""
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms_raw = self.build_transforms(hyp)

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate data samples into batches, handling mismatched keys due to different augmentations."""
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


class DetectionTrainer(BaseTrainer):
    """A class extending the BaseTrainer class for training based on a detection model."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None, attack_weights = ""):
        """Initialize a DetectionTrainer object for training YOLO object detection model."""
        if overrides is None:
            overrides = {}
        self._attack_weights_arg = attack_weights or overrides.pop("attack_weights", "")
        self._attack_name_arg = overrides.pop("attack_name", "cw")
        self._use_pregenerated_adv_arg = overrides.pop("use_pregenerated_adv", False)
        self._attack_num_arg = overrides.pop("attack_num", 1)
        self._attack_ratio_arg = overrides.pop("attack_ratio", 0.5)

        super().__init__(cfg, overrides, _callbacks)

        if not hasattr(self.args, "attack_weights"):
            self.args.attack_weights = self._attack_weights_arg
        if not hasattr(self.args, "attack_name"):
            self.args.attack_name = self._attack_name_arg
        if not hasattr(self.args, "use_pregenerated_adv"):
            self.args.use_pregenerated_adv = self._use_pregenerated_adv_arg
        if not hasattr(self.args, "attack_num"):
            self.args.attack_num = self._attack_num_arg
        if not hasattr(self.args, "attack_ratio"):
            self.args.attack_ratio = self._attack_ratio_arg

        self.attack_names = parse_list_arg(self.args.attack_name, str)
        self.attack_ratios = parse_list_arg(self.args.attack_ratio, float)
        self.attack_num = int(self.args.attack_num)
        
        # Align lengths
        if len(self.attack_names) < self.attack_num:
            last = self.attack_names[-1] if self.attack_names else "pgd"
            self.attack_names.extend([last] * (self.attack_num - len(self.attack_names)))
        else:
            self.attack_names = self.attack_names[:self.attack_num]
            
        if len(self.attack_ratios) < self.attack_num:
            last = self.attack_ratios[-1] if self.attack_ratios else 0.5
            self.attack_ratios.extend([last] * (self.attack_num - len(self.attack_ratios)))
        else:
            self.attack_ratios = self.attack_ratios[:self.attack_num]

        self.attack_model = None
        self.attacker = None
        self.attackers = {}

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation."""
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode == "train":
            return YOLODatasetAdvTest(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
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

        if attack_weights:
            try:
                LOGGER.info(f"Attempting to load attack model from: {attack_weights}")
                imgsz = getattr(self.args, "imgsz", 640)
                self.attack_model = setup_attack_model(
                    attack_weights,
                    device=self.device,
                    nc=self.data["nc"],
                    training=False,
                    imgsz=imgsz,
                )
                LOGGER.info(f"Attack model loaded successfully from {attack_weights}")

                self.attack_model.eval()
                for p in self.attack_model.parameters():
                    p.requires_grad = True

                self.attackers = {}
                for att in self.attack_names:
                    self.attackers[att] = build_attacker(att, model=self.attack_model, img_size=imgsz)
                    LOGGER.info(f"attacker initialized: {att}")
                
                # Keep self.attacker for validation (which uses the first attack)
                self.attacker = self.attackers[self.attack_names[0]]
            except Exception as e:
                LOGGER.warning(f"Failed to load attack model or attacker: {e}")
                import traceback
                LOGGER.warning(traceback.format_exc())
                self.attack_model = None
                self.attacker = None
                self.attackers = {}
        else:
            LOGGER.info("No attack weights provided, adversarial training disabled")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"

        validator_args = copy(self.args)
        if hasattr(validator_args, 'attack_weights'):
            delattr(validator_args, 'attack_weights')
        if hasattr(validator_args, 'attack_name'):
            delattr(validator_args, 'attack_name')
        if hasattr(validator_args, 'use_pregenerated_adv'):
            delattr(validator_args, 'use_pregenerated_adv')
        for arg in ['attack_num', 'attack_ratio']:
            if hasattr(validator_args, arg):
                delattr(validator_args, arg)

        validator = DetectionValidatorAdv(
            self.test_loader, save_dir=self.save_dir, args=validator_args, _callbacks=self.callbacks
        )
        validator.attacker = getattr(self, "attacker", None)
        validator.attackers = getattr(self, "attackers", {})
        validator.attack_names = self.attack_names
        validator.attack_ratios = self.attack_ratios
        return validator

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items tensor."""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            loss_dict = dict(zip(keys, loss_items))
            if prefix == "train" and hasattr(self, "group_tloss"):
                for t, g_tloss in self.group_tloss.items():
                    if g_tloss is not None:
                        g_keys = [f"{prefix}/{x}_{t}" for x in self.loss_names]
                        g_vals = [round(float(x), 5) for x in g_tloss]
                        loss_dict.update(dict(zip(g_keys, g_vals)))
            return loss_dict
        else:
            all_keys = list(keys)
            if prefix == "train" and hasattr(self, "attack_names"):
                for t in ["raw"] + self.attack_names:
                    all_keys.extend([f"{prefix}/{x}_{t}" for x in self.loss_names])
            return all_keys

    def progress_string(self):
        """Return formatted progress bar headers."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples."""
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """Plot training labels."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get optimal batch size."""
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        del train_dataset
        return super().auto_batch(max_num_obj)

    def split_batch_by_type(self, batch: dict) -> dict[str, dict]:
        """Split a batch dictionary into sub-batches based on the sample_type field."""
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
        """Train the model with multi-attack capability."""
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
            f"Starting multi-attack adversarial training for {self.epochs} epochs..."
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
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None

            # Reset group-specific epoch metrics
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
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)

                    for st in batch.get("sample_type", []):
                        if st in self.epoch_counts:
                            self.epoch_counts[st] += 1

                    sub_batches = self.split_batch_by_type(batch)

                    # If on-the-fly, generate adversarial images for present attack groups
                    if not getattr(self.args, "use_pregenerated_adv", False) and getattr(self, "attackers", None):
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

                                    with torch.amp.autocast(device_type="cuda", enabled=False):
                                        imgs_adv = run_attack_on_batch(attacker, sub_batch, label_policy="largest_box")

                                    if imgs_adv is not None:
                                        sub_batch["img"] = imgs_adv.to(sub_batch["img"].dtype).to(self.device)
                                except Exception as e:
                                    LOGGER.warning(f"Adversarial generation failed for {att}: {e}. Keeping raw images.")
                                    import traceback
                                    LOGGER.warning(traceback.format_exc())

                    raw_ratio = max(0.0, 1.0 - sum(self.attack_ratios))
                    group_weights = {"raw": raw_ratio}
                    for att, r in zip(self.attack_names, self.attack_ratios):
                        group_weights[att] = r

                    present_groups = list(sub_batches.keys())
                    present_weights = {g: group_weights.get(g, 0.0) for g in present_groups}
                    total_weight = sum(present_weights.values())
                    if total_weight > 0:
                        normalized_weights = {g: w / total_weight for g, w in present_weights.items()}
                    else:
                        normalized_weights = {g: 1.0 / len(present_groups) for g in present_groups}

                    total_loss = 0.0
                    total_loss_items = torch.zeros(3, device=self.device)

                    for g in present_groups:
                        sub_batch = sub_batches[g]
                        if sub_batch["img"].shape[0] == 0:
                            continue

                        if self.args.compile:
                            preds = self.model(sub_batch["img"])
                            loss, loss_items = unwrap_model(self.model).loss(sub_batch, preds)
                        else:
                            loss, loss_items = self.model(sub_batch)

                        w = normalized_weights[g]
                        total_loss = total_loss + w * loss.sum()
                        total_loss_items += w * loss_items

                        if self.group_tloss.get(g) is None:
                            self.group_tloss[g] = loss_items.clone().detach()
                            self.group_batch_count[g] = 1
                        else:
                            self.group_tloss[g] = (self.group_tloss[g] * self.group_batch_count[g] + loss_items.detach()) / (self.group_batch_count[g] + 1)
                            self.group_batch_count[g] += 1

                    self.loss = total_loss
                    if RANK != -1:
                        self.loss *= self.world_size
                    self.loss_items = total_loss_items
                    self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)

                self.scaler.scale(self.loss).backward()

                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
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

