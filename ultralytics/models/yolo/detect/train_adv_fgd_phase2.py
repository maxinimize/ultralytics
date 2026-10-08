"""Frozen Phase 1 FGD preprocessing with the existing online adversarial training loop."""

from copy import copy
from pathlib import Path
from typing import ClassVar

from ultralytics.attacks.attack_bridge import build_attacker
from ultralytics.defenses.hgd_fgd import FGDDetectorWrapper, apply_fgd, load_fgd_denoiser
from ultralytics.models.yolo.detect.train_adv_test import DetectionTrainer
from ultralytics.models.yolo.detect.val_adv_fgd_phase2 import DetectionValidator
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, YAML


class FGDAdversarialTrainer(DetectionTrainer):
    """Keep FGD external to YOLO, EMA and optimizer while attacks differentiate through both models."""

    defaults: ClassVar[dict] = dict(
        use_fgd=True,
        fgd_weights=None,
        attack_through_fgd=True,
        attack_epsilon=0.031372549,
        attack_steps=5,
        attack_step_size=0.00784313725,
    )

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, attack_weights=""):
        """Restore Phase 2 settings on resume and preserve them in args for DDP reconstruction."""
        overrides = dict(overrides or {})
        explicit = {key: overrides.pop(key) for key in self.defaults if key in overrides}
        self.phase2 = {**self.defaults, **explicit}
        self.denoiser = None
        super().__init__(cfg, overrides, _callbacks, attack_weights)
        if self.resume:
            # Ultralytics strips research arguments from completed .pt files; keep the external config authoritative.
            metadata = Path(self.args.resume).resolve().parent.parent / "phase2_fgd.yaml"
            if metadata.is_file():
                saved = YAML.load(metadata)
                self.phase2.update({key: saved[key] for key in self.defaults if key in saved})
            self.phase2.update({key: getattr(self.args, key) for key in self.defaults if hasattr(self.args, key)})
            self.phase2.update(explicit)
        if self.phase2["use_fgd"]:
            weights = self.phase2["fgd_weights"]
            if not weights or not Path(weights).is_file():
                raise FileNotFoundError(f"Phase 2 requires an existing FGD checkpoint: {weights!r}")
            self.phase2["fgd_weights"] = str(Path(weights).resolve())
        self.strict_attacks = self.phase2["use_fgd"]
        for key, value in self.phase2.items():
            setattr(self.args, key, value)

    def set_model_attributes(self):
        """Load frozen FGD before the inherited trainer creates online attacks."""
        if self.phase2["use_fgd"]:
            self.denoiser, self.fgd_metadata = load_fgd_denoiser(self.phase2["fgd_weights"], self.device)
        super().set_model_attributes()

    def build_attacker(self, name, model, imgsz):
        """Use the current detector or configured transfer detector behind the same frozen FGD."""
        if not self.phase2["use_fgd"]:
            return super().build_attacker(name, model, imgsz)
        if self.phase2["attack_through_fgd"]:
            model = FGDDetectorWrapper(self.denoiser, model)
        kwargs = {}
        if name.lower().strip() in {"pgd", "bim", "mim"}:
            kwargs = dict(
                epsilon=self.phase2["attack_epsilon"],
                lr=self.phase2["attack_step_size"],
                epoch=self.phase2["attack_steps"],
            )
        return build_attacker(name, model=model, img_size=imgsz, **kwargs)

    def preprocess_for_detector(self, images):
        """Preprocess every clean/attacked input after the inherited attack and label recombination."""
        return apply_fgd(self.denoiser, images) if self.denoiser is not None else images

    def build_optimizer(self, model, **kwargs):
        """Keep inherited optimizer groups/hyperparameters while excluding frozen detector weights."""
        optimizer = super().build_optimizer(model, **kwargs)
        if self.phase2["use_fgd"]:
            for group in optimizer.param_groups:
                group["params"] = [p for p in group["params"] if p.requires_grad]
        return optimizer

    def get_validator(self):
        """Reuse existing attack metrics and FGD preprocessing, including final best.pt evaluation."""
        args = copy(self.args)
        for key in (*self.defaults, "attack_weights", "attack_name", "attack_ratio", "attack_num"):
            if hasattr(args, key):
                delattr(args, key)
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        validator = DetectionValidator(
            getattr(self, "test_loader", None), save_dir=self.save_dir, args=args, _callbacks=self.callbacks
        )
        validator.attacker, validator.attackers = self.attacker, self.attackers
        validator.attack_names, validator.attack_ratios = self.attack_names, self.attack_ratios
        validator.denoiser = self.denoiser
        validator.strict_attacks = self.strict_attacks
        if self.phase2["use_fgd"]:
            validator.attack_builder = self.build_attacker
            validator.whitebox = self.attack_model is None
        return validator

    def _setup_train(self):
        """Check optimizer isolation and record the complete experiment after inherited setup."""
        super()._setup_train()
        if self.denoiser is None:
            return
        fgd_ids = {id(p) for p in self.denoiser.parameters()}
        assert not any(p.requires_grad for p in self.denoiser.parameters())
        assert not self.denoiser.training
        assert not any(id(p) in fgd_ids for group in self.optimizer.param_groups for p in group["params"])
        assert all(p.requires_grad for group in self.optimizer.param_groups for p in group["params"])
        metadata = dict(
            **self.phase2,
            phase="Phase 2 FGD adversarial training",
            fgd_frozen=True,
            fgd_eval_mode=True,
            fgd_trainable_params=0,
            fgd_total_params=sum(p.numel() for p in self.denoiser.parameters()),
            yolo_trainable_params=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            yolo_initial_weights=str(self.args.model),
            attack_weights=self.args.attack_weights,
            attack_name=self.attack_names,
            attack_ratio=self.attack_ratios,
            imgsz=self.args.imgsz,
            epochs=self.args.epochs,
            batch=self.args.batch,
            amp=self.args.amp,
            fgd_checkpoint_metadata=self.fgd_metadata,
            checkpoint_selection="Existing attack-ratio-weighted validation mAP50-95",
        )
        if RANK in {-1, 0}:
            YAML.save(self.save_dir / "phase2_fgd.yaml", metadata)
            YAML.save(self.save_dir / "args.yaml", vars(self.args))
            LOGGER.info("Phase 2 FGD adversarial training: enabled")
            for key, value in metadata.items():
                LOGGER.info(f"{key}: {value}")
