# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
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
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr, TQDM, LOCAL_RANK
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model, unset_deterministic, autocast
import warnings
from torch import distributed as dist
from pathlib import Path
import cv2
from ultralytics.data.dataset import YOLODataset

from ultralytics.models.yolo.detect.val_adv_single import DetectionValidator as DetectionValidatorAdv
from ultralytics.attacks.attack_utils import setup_attack_model
from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch


class RepeatDataset(torch.utils.data.Dataset):
    """Dataset wrapper to repeat each item in a dataset repeats times."""

    def __init__(self, dataset, repeats):
        self.dataset = dataset
        self.repeats = repeats

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, index):
        orig_idx = index // self.repeats
        aug_idx = index % self.repeats
        sample = self.dataset[orig_idx]
        return {
            "img": sample["img"],
            "cls": sample["cls"],
            "bboxes": sample["bboxes"],
            "orig_idx": orig_idx,
            "aug_idx": aug_idx
        }


def collate_fn_list(batch):
    """Custom collate function to return batch samples as a list of dicts."""
    return batch


def worker_init_fn(worker_id):
    """Initialize worker seed to ensure randomness across workers."""
    import random
    import numpy as np
    import time
    seed = (worker_id + 1) * int(time.time() * 1000) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class YOLODynamicPoolDataset(YOLODataset):
    """Dataset class for loading clean and pre-generated attacked augmented images from disk with a custom mix ratio."""

    def __init__(self, cfg, clean_dir, attacked_dir, batch, data, mode="train", rect=False, stride=32, **kwargs):
        """Initialize YOLODynamicPoolDataset."""
        self.cfg = cfg
        self.clean_dir = Path(clean_dir)
        self.attacked_dir = Path(attacked_dir)
        self.attack_mix_ratio = getattr(cfg, "attack_mix_ratio", 0.5)

        super().__init__(
            img_path=str(self.clean_dir / "images"),
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=False,  # Disable on-the-fly random transformations
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

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load the pre-generated clean or attacked image from disk depending on attack_mix_ratio."""
        orig_path = self.im_files[i]
        p = Path(orig_path)
        
        # Decide whether to load the attacked image or clean image based on attack_mix_ratio
        use_attacked = random.random() < self.attack_mix_ratio
        
        if use_attacked:
            img_path = self.attacked_dir / "images" / p.name
            if not img_path.exists():
                img_path = self.clean_dir / "images" / p.name
        else:
            img_path = self.clean_dir / "images" / p.name

        im = cv2.imread(str(img_path), flags=self.cv2_flag)
        if im is None:
            raise FileNotFoundError(f"Image Not Found: {img_path}")
            
        h0, w0 = im.shape[:2]  # orig hw
        if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        if im.ndim == 2:
            im = im[..., None]
        return im, (h0, w0), im.shape[:2]


class DetectionTrainer(BaseTrainer):
    """A class extending the BaseTrainer class for dynamic pool-based adversarial training."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize a DetectionTrainer object."""
        if overrides is None:
            overrides = {}
        self._attack_name_arg = overrides.pop("attack_name", "cw")
        self._use_pregenerated_adv_arg = overrides.pop("use_pregenerated_adv", False)
        
        # Get dynamic pool overrides
        self._attack_mix_ratio_arg = overrides.pop("attack_mix_ratio", 0.5)
        self._num_aug_arg = overrides.pop("num_aug", 5)
        self._pool_update_period_arg = overrides.pop("pool_update_period", 5)

        super().__init__(cfg, overrides, _callbacks)

        if not hasattr(self.args, "attack_name"):
            self.args.attack_name = self._attack_name_arg
        if not hasattr(self.args, "use_pregenerated_adv"):
            self.args.use_pregenerated_adv = self._use_pregenerated_adv_arg
        if not hasattr(self.args, "attack_mix_ratio"):
            self.args.attack_mix_ratio = self._attack_mix_ratio_arg
        if not hasattr(self.args, "num_aug"):
            self.args.num_aug = self._num_aug_arg
        if not hasattr(self.args, "pool_update_period"):
            self.args.pool_update_period = self._pool_update_period_arg
        
        self.attacker = None

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation."""
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode == "train":
            # For training, load from the dynamic pool in /scratch
            scratch_base = os.environ.get("SCRATCH", "/scratch/czhan295")
            temp_dir = Path(scratch_base) / "train_adv_temp" / self.save_dir.name
            clean_dir = temp_dir / "clean"
            attacked_dir = temp_dir / "attacked"
            return YOLODynamicPoolDataset(
                self.args,
                clean_dir=clean_dir,
                attacked_dir=attacked_dir,
                batch=batch,
                data=self.data,
                mode=mode,
                rect=mode == "val",
                stride=gs,
            )
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return dataloader for the specified mode."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
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
        """Preprocess a batch of images by scaling and converting to float."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if "img_adv" in batch:
            batch["img_adv"] = batch["img_adv"].float() / 255

        if self.args.multi_scale:
            imgs = batch["img"]
            imgs_adv = batch.get("img_adv")
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
                if imgs_adv is not None:
                    imgs_adv = nn.functional.interpolate(imgs_adv, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
            if imgs_adv is not None:
                batch["img_adv"] = imgs_adv
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

        attack_name = getattr(self.args, "attack_name", "cw")
        try:
            imgsz = getattr(self.args, "imgsz", 640)
            self.attacker = build_attacker(attack_name, model=unwrap_model(self.model), img_size=imgsz)
            LOGGER.info(f"Attacker initialized using training model weights: {attack_name}")
        except Exception as e:
            LOGGER.warning(f"Failed to initialize attacker: {e}")
            import traceback
            LOGGER.warning(traceback.format_exc())
            self.attacker = None

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def setup_model(self):
        """Override to cache ckpt for resume support when called early."""
        if hasattr(self, "_cached_ckpt"):
            ckpt = self._cached_ckpt
            delattr(self, "_cached_ckpt")
            return ckpt
        if isinstance(self.model, torch.nn.Module):
            return getattr(self, "_resumed_ckpt", None)
            
        ckpt = super().setup_model()
        self._resumed_ckpt = ckpt
        return ckpt

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"

        validator_args = copy(self.args)
        for key in ['attack_weights', 'attack_name', 'use_pregenerated_adv', 'attack_mix_ratio', 'num_aug', 'pool_update_period']:
            if hasattr(validator_args, key):
                delattr(validator_args, key)

        validator = DetectionValidatorAdv(
            self.test_loader, save_dir=self.save_dir, args=validator_args, _callbacks=self.callbacks
        )
        validator.attacker = getattr(self, "attacker", None)
        validator.attack_name = getattr(self.args, "attack_name", None)
        return validator

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items tensor."""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations."""
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model."""
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj)

    def _generate_adversarial_batch(self, batch: dict) -> torch.Tensor | None:
        """Helper to generate adversarial images for a single batch."""
        if self.attacker is None:
            return None

        try:
            if hasattr(self.attacker, "proxy_model"):
                self.attacker.proxy_model.current_paths = batch.get("im_file", [])
            if hasattr(self.attacker, "estimator") and hasattr(self.attacker.estimator, "model"):
                self.attacker.estimator.model.current_paths = batch.get("im_file", [])
            model_to_attack = unwrap_model(self.model)
            if hasattr(model_to_attack, "current_paths"):
                model_to_attack.current_paths = batch.get("im_file", [])

            with torch.amp.autocast(device_type="cuda", enabled=False):
                imgs_adv = run_attack_on_batch(self.attacker, batch, label_policy="largest_box")

            return None if imgs_adv is None else imgs_adv.to(batch["img"].dtype)
        except Exception as e:
            LOGGER.warning(f"Adversarial generation failed: {e}. Skipping attack for this batch.")
            return None

    def generate_adversarial(self, batch: dict) -> torch.Tensor | None:
        """Override to disable on-the-fly generation during the training step."""
        return None

    def generate_pools(self, gen_epoch=None):
        """
        Generate num_aug random augmented images per input image,
        apply the attack using updated model weights, and save them.
        """
        import shutil
        import json
        from PIL import Image

        if gen_epoch is None:
            gen_epoch = getattr(self, "epoch", 0)

        # Ensure we have our temp directories in /scratch
        scratch_base = os.environ.get("SCRATCH", "/scratch/czhan295")
        temp_dir = Path(scratch_base) / "train_adv_temp" / self.save_dir.name
        clean_dir = temp_dir / "clean"
        attacked_dir = temp_dir / "attacked"
        info_file = temp_dir / "pool_info.json"

        if RANK in {-1, 0}:
            LOGGER.info(f"Dynamic Pool Generation: Starting pool generation under {temp_dir} for epoch {gen_epoch}...")
            
            # Remove info file to mark pool as incomplete during generation
            if info_file.exists():
                info_file.unlink()

            # Recreate directories to be clean
            if clean_dir.exists():
                shutil.rmtree(clean_dir)
            if attacked_dir.exists():
                shutil.rmtree(attacked_dir)

            (clean_dir / "images").mkdir(parents=True, exist_ok=True)
            (clean_dir / "labels").mkdir(parents=True, exist_ok=True)
            (attacked_dir / "images").mkdir(parents=True, exist_ok=True)
            
            # Put the model in eval mode for attack generation
            self.model.eval()

            # Create a generator dataset representing the original dataset with augmentations
            gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
            gen_dataset = build_yolo_dataset(
                self.args,
                self.data["train"],
                self.args.batch,
                self.data,
                mode="train",
                rect=False,
                stride=gs,
            )

            N = len(gen_dataset)
            num_aug = getattr(self.args, "num_aug", 5)
            LOGGER.info(f"Dynamic Pool Generation: Original dataset has {N} images. Generating {num_aug * N} augmented images...")

            # 1. Generate clean augmented images and labels in parallel using DataLoader
            repeat_dataset = RepeatDataset(gen_dataset, num_aug)
            clean_generator_loader = torch.utils.data.DataLoader(
                repeat_dataset,
                batch_size=self.args.batch if self.args.batch else 16,
                num_workers=self.args.workers if self.args.workers is not None else 3,
                collate_fn=collate_fn_list,
                worker_init_fn=worker_init_fn,
                shuffle=False,
                pin_memory=False,
            )

            for batch in clean_generator_loader:
                for sample in batch:
                    try:
                        img_tensor = sample["img"]  # (3, H, W)
                        cls = sample["cls"]  # (num_objs, 1)
                        bboxes = sample["bboxes"]  # (num_objs, 4)
                        i = sample["orig_idx"]
                        j = sample["aug_idx"]

                        # Convert from CHW PyTorch tensor to HWC NumPy uint8
                        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                        img_pil = Image.fromarray(img_np)

                        # Save clean augmented image
                        img_name = f"img_{i}_aug_{j}.jpg"
                        img_pil.save(clean_dir / "images" / img_name)

                        # Save label (YOLO text format)
                        lbl_name = f"img_{i}_aug_{j}.txt"
                        with open(clean_dir / "labels" / lbl_name, "w") as f:
                            for c_val, box in zip(cls, bboxes):
                                f.write(f"{int(c_val)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                    except Exception as e:
                        LOGGER.warning(f"Failed to generate augmented image sample: {e}")

            LOGGER.info("Dynamic Pool Generation: Clean augmented dataset generated. Running attacks...")

            # 2. If attacker is active, generate attacked images
            if self.attacker is not None:
                # We build a YOLODataset over the generated clean augmented images
                clean_aug_dataset = YOLODataset(
                    img_path=str(clean_dir / "images"),
                    imgsz=self.args.imgsz,
                    batch_size=self.args.batch,
                    augment=False,
                    hyp=self.args,
                    rect=False,
                    stride=gs,
                    single_cls=self.args.single_cls or False,
                    classes=self.args.classes,
                    data=self.data,
                    task=self.args.task,
                )

                clean_aug_loader = build_dataloader(
                    clean_aug_dataset,
                    batch=self.args.batch,
                    workers=self.args.workers,
                    shuffle=False,
                    rank=-1,  # Serial processing for pool generation is safer and simpler
                )

                # Process batches to generate attacks
                for batch in clean_aug_loader:
                    batch = self.preprocess_batch(batch)
                    
                    # Generate adversarial images for this batch
                    imgs_adv = self._generate_adversarial_batch(batch)
                    
                    if imgs_adv is None:
                        imgs_adv = batch["img"]
                        
                    im_files = batch["im_file"]
                    for idx_in_batch in range(len(im_files)):
                        img_name = Path(im_files[idx_in_batch]).name
                        attacked_path = attacked_dir / "images" / img_name
                        
                        img_adv = imgs_adv[idx_in_batch]  # (3, H, W)
                        img_adv_np = (img_adv.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                        
                        Image.fromarray(img_adv_np).save(attacked_path)

            # Write pool_info.json to mark completion
            pool_info = {
                "epoch": gen_epoch,
                "num_aug": num_aug,
                "completed": True
            }
            with open(info_file, "w") as f:
                json.dump(pool_info, f, indent=4)
            LOGGER.info("Dynamic Pool Generation: Successfully generated clean and attacked image pools.")

        if self.world_size > 1:
            dist.barrier()

    def check_pool_valid(self, target_epoch: int) -> bool:
        """Check if the pre-generated pool matches the target epoch, expected num_aug, and was fully generated."""
        import json
        scratch_base = os.environ.get("SCRATCH", "/scratch/czhan295")
        pool_dir = Path(scratch_base) / "train_adv_temp" / self.save_dir.name
        info_file = pool_dir / "pool_info.json"
        clean_img_dir = pool_dir / "clean" / "images"

        if not info_file.exists() or not clean_img_dir.exists():
            return False

        try:
            with open(info_file, "r") as f:
                info = json.load(f)
            
            completed = info.get("completed", False)
            gen_epoch = info.get("epoch", -1)
            num_aug = info.get("num_aug", -1)
            
            expected_num_aug = getattr(self.args, "num_aug", 5)

            if completed and gen_epoch == target_epoch and num_aug == expected_num_aug:
                if len(list(clean_img_dir.glob("*.jpg"))) > 0:
                    return True
        except Exception:
            pass
        return False

    def _do_train(self):
        """Train the model with the specified world size."""
        if self.world_size > 1:
            self._setup_ddp()

        # Load/instantiate the model early so we can unwrap it and generate pools
        if not isinstance(self.model, torch.nn.Module):
            ckpt = self.setup_model()
            self.model = self.model.to(self.device)
            self.set_model_attributes()
            # Cache the checkpoint for standard trainer initialization
            self._cached_ckpt = ckpt

        # Initial pool generation check
        pool_update_period = getattr(self.args, "pool_update_period", 5)
        expected_gen_epoch = (self.start_epoch // pool_update_period) * pool_update_period

        if not self.check_pool_valid(expected_gen_epoch):
            LOGGER.info(f"Dynamic Pool Check: No valid complete pool found for epoch {expected_gen_epoch}. Regenerating...")
            self.generate_pools(gen_epoch=expected_gen_epoch)
        else:
            LOGGER.info(f"Dynamic Pool Check: Valid complete pool found for epoch {expected_gen_epoch}. Reusing existing pool.")

        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            # Pool update check
            pool_update_period = getattr(self.args, "pool_update_period", 5)
            if epoch > self.start_epoch and epoch % pool_update_period == 0:
                LOGGER.info(f"Dynamic Pool Update: Regenerating clean and attacked image pools at epoch {epoch}...")
                self.generate_pools()
                # Re-initialize training dataloader to read new pool files
                batch_size = self.batch_size // max(self.world_size, 1)
                self.train_loader = self.get_dataloader(
                    self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
                )

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
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward (Single forward/backward pass using the mixed pool batch)
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    if self.args.compile:
                        preds = self.model(batch["img"])
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        loss, self.loss_items = self.model(batch)
                
                # Backward
                self.scaler.scale(loss.sum()).backward()

                # Logging updates
                self.loss = loss.sum()
                if RANK != -1:
                    self.loss *= self.world_size
                self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)  # prevent VRAM spike
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # Do final val with best.pt
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")
