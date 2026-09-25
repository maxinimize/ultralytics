"""Ultralytics-native trainer for HYDRA score search and masked fine-tuning.

Milestone 2:
    Implement clean detection HYDRA training while inheriting Ultralytics'
    existing dataloader, DetectionModel, loss, validator, logging, checkpointing,
    and optimizer infrastructure.
"""

from __future__ import annotations

import io
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ultralytics.cfg import DEFAULT_CFG_DICT, cfg2dict
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.pruning.hydra_convert import convert_to_hydra, iter_hydra_layers
from ultralytics.nn.pruning.hydra_layer import HydraConv2d
from ultralytics.nn.pruning.hydra_mask import (
    HydraSparsityReport,
    configure_masked_finetune,
    configure_score_search,
    enforce_pruned_weights_zero,
    freeze_hydra_masks,
    set_hydra_keep_ratio,
)
from ultralytics.nn.pruning.hydra_score import initialize_hydra_scores
from ultralytics.utils import __version__, GIT, LOGGER
from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16, unwrap_model


HYDRA_DEFAULTS = {
    # Enable HYDRA-specific behavior.
    "hydra": True,

    # Training stage:
    #   "score"    -> freeze dense weights and optimize popup_scores only
    #   "finetune" -> freeze masks/scores and fine-tune retained weights
    "hydra_stage": "score",

    # Fraction of weights retained in every selected HYDRA layer.
    # keep_ratio=0.5 means 50% retained and 50% pruned.
    "hydra_keep_ratio": 1.0,

    # HYDRA importance-score initialization method.
    "hydra_score_init": "scaled",

    # Milestones 1-3 use layer-wise pruning only.
    "hydra_global_pruning": False,

    # Score search should use the live model instead of EMA-smoothed scores.
    "hydra_use_ema": False,
}

HYDRA_CONFIG_KEYS = frozenset(HYDRA_DEFAULTS.keys())


def prepare_hydra_cfg(cfg=None) -> dict[str, Any]:
    """Normalize and inject HYDRA default arguments into a configuration dict."""
    if cfg is None:
        hydra_cfg = deepcopy(DEFAULT_CFG_DICT)
    elif isinstance(cfg, dict):
        hydra_cfg = deepcopy(cfg)
    else:
        hydra_cfg = deepcopy(cfg2dict(cfg))

    for key, value in HYDRA_DEFAULTS.items():
        hydra_cfg.setdefault(key, value)

    return hydra_cfg


class HydraTrainerMixin:
    """Mixin class providing HYDRA stage lifecycle and parameter management.

    Can be combined with DetectionTrainer or AdversarialDetectionTrainer to add:
      - Automatic HYDRA leaf conversion and score initialization;
      - Dynamic score mask search with Straight-Through Estimators;
      - Fixed-mask fine-tuning with strict zero-pruned-weight enforcement;
      - Checkpoint serialization preserving HYDRA state and metadata.
    """

    def init_hydra_attributes(self) -> None:
        """Initialize and validate trainer-level HYDRA state attributes."""
        self.hydra_stage = str(self.args.hydra_stage)
        self.hydra_keep_ratio = float(self.args.hydra_keep_ratio)
        self.hydra_score_init = str(self.args.hydra_score_init)
        self.hydra_global_pruning = bool(self.args.hydra_global_pruning)
        self.hydra_use_ema = bool(self.args.hydra_use_ema)
        self._validate_hydra_args()

    def _validate_hydra_args(self) -> None:
        """Validate Milestone-2 HYDRA configuration values early."""
        if self.hydra_stage not in {"score", "finetune"}:
            raise ValueError(
                f"hydra_stage must be 'score' or 'finetune', got {self.hydra_stage!r}"
            )

        if not 0.0 <= self.hydra_keep_ratio <= 1.0:
            raise ValueError(
                f"hydra_keep_ratio must be in [0, 1], got {self.hydra_keep_ratio}"
            )

        if self.hydra_score_init not in {"scaled"}:
            raise ValueError(
                f"Unsupported hydra_score_init: {self.hydra_score_init!r}"
            )

        if self.hydra_global_pruning:
            raise ValueError(
                "Global HYDRA pruning is not implemented in Milestones 1-3. "
                "Use hydra_global_pruning=False."
            )

    def setup_model(self):
        """Construct model, load weights, and attach HYDRA components."""
        ckpt = super().setup_model()
        self._attach_hydra_after_weight_load(ckpt)
        return ckpt

    def _attach_hydra_after_weight_load(self, ckpt=None) -> None:
        """Convert loaded model and configure requested HYDRA stage parameters."""
        hydra_layers = list(iter_hydra_layers(self.model))
        is_already_hydra = len(hydra_layers) > 0

        if not is_already_hydra:
            convert_to_hydra(self.model, keep_ratio=self.hydra_keep_ratio)

        # Restore popup_scores and fixed_mask from checkpoint if present
        scores_restored = False
        masks_restored = False

        src_state = None
        if ckpt is not None:
            if isinstance(ckpt, dict):
                src_model = ckpt.get("ema") or ckpt.get("model")
                if isinstance(src_model, torch.nn.Module):
                    src_state = src_model.state_dict()
                elif isinstance(src_model, dict):
                    src_state = src_model
                elif "state_dict" in ckpt:
                    src_state = ckpt["state_dict"]
            elif isinstance(ckpt, torch.nn.Module):
                src_state = ckpt.state_dict()

        if src_state is None and hasattr(self, "args") and getattr(self.args, "model", None):
            model_path = str(self.args.model)
            if model_path.endswith(".pt") and Path(model_path).exists():
                try:
                    loaded_ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
                    if isinstance(loaded_ckpt, dict):
                        src_model = loaded_ckpt.get("ema") or loaded_ckpt.get("model")
                        if isinstance(src_model, torch.nn.Module):
                            src_state = src_model.state_dict()
                        elif isinstance(src_model, dict):
                            src_state = src_model
                    elif isinstance(loaded_ckpt, torch.nn.Module):
                        src_state = loaded_ckpt.state_dict()
                except Exception as e:
                    LOGGER.warning(f"HYDRA: Could not load fallback checkpoint from {model_path}: {e}")

        if src_state is not None:
            # Build prefix-normalized lookup dict
            norm_src_state = {}
            for k, v in src_state.items():
                norm_k = k
                for pfx in ("module.", "model."):
                    if norm_k.startswith(pfx):
                        norm_k = norm_k[len(pfx):]
                norm_src_state[norm_k] = v

            num_scores = 0
            num_masks = 0
            for name, layer in iter_hydra_layers(self.model):
                norm_name = name
                for pfx in ("module.", "model."):
                    if norm_name.startswith(pfx):
                        norm_name = norm_name[len(pfx):]

                score_k = f"{norm_name}.popup_scores"
                if score_k in norm_src_state:
                    layer.popup_scores.data.copy_(norm_src_state[score_k].to(layer.popup_scores.device))
                    num_scores += 1

                mask_k = f"{norm_name}.fixed_mask"
                if mask_k in norm_src_state:
                    loaded_mask = norm_src_state[mask_k].to(layer.fixed_mask.device)
                    # Check if the mask has actual non-trivial sparsity (i.e. not all 1s)
                    if (loaded_mask < 0.5).any():
                        layer.fixed_mask.data.copy_(loaded_mask)
                        num_masks += 1

            if num_scores > 0:
                scores_restored = True
                LOGGER.info(f"HYDRA: Restored popup_scores for {num_scores} layer(s) from checkpoint.")
            if num_masks > 0:
                masks_restored = True
                LOGGER.info(f"HYDRA: Restored fixed_mask for {num_masks} layer(s) from checkpoint.")

        if self.hydra_stage == "score":
            if not scores_restored:
                initialize_hydra_scores(self.model, method=self.hydra_score_init)
            set_hydra_keep_ratio(self.model, self.hydra_keep_ratio)
            configure_score_search(self.model)
        elif self.hydra_stage == "finetune":
            set_hydra_keep_ratio(self.model, self.hydra_keep_ratio)
            if masks_restored:
                total_m = sum(l.fixed_mask.numel() for _, l in iter_hydra_layers(self.model))
                kept_m = sum(int((l.fixed_mask > 0.5).sum().item()) for _, l in iter_hydra_layers(self.model))
                sparsity = 1.0 - (kept_m / total_m) if total_m > 0 else 0.0
                LOGGER.info(f"HYDRA: Using restored fixed_mask from checkpoint: sparsity={sparsity:.4f} ({kept_m}/{total_m} retained)")
            else:
                if not scores_restored:
                    LOGGER.warning("HYDRA ⚠️ No popup_scores found in checkpoint to freeze masks from!")
                report = freeze_hydra_masks(self.model)
                LOGGER.info(f"HYDRA: Froze masks from popup_scores: sparsity={report.sparsity:.4f}")
            configure_masked_finetune(self.model)
            enforce_pruned_weights_zero(self.model)
        else:
            raise ValueError(f"Unknown HYDRA stage: {self.hydra_stage}")

    def _setup_train(self):
        """Set up model, loader, optimizer, and enforce HYDRA stage invariants."""
        super()._setup_train()
        if self.hydra_stage == "score":
            configure_score_search(self.model)
        elif self.hydra_stage == "finetune":
            configure_masked_finetune(self.model)
            enforce_pruned_weights_zero(self.model)

    def _model_train(self):
        """Set model in training mode and enforce BN eval mode during score search."""
        if not hasattr(self, "freeze_layer_names"):
            self.freeze_layer_names = [".dfl"]
        super()._model_train()
        if self.hydra_stage == "score":
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()

    def build_optimizer(
        self,
        model: nn.Module,
        name: str = "auto",
        lr: float = 0.001,
        momentum: float = 0.9,
        decay: float = 1e-5,
        iterations: int = 1e5,
    ):
        """Build stage-specific optimizer parameter groups."""
        if self.hydra_stage == "score":
            configure_score_search(model)

            score_params = [
                p for name_p, p in unwrap_model(model).named_parameters() if "popup_scores" in name_p
            ]
            if not score_params:
                raise RuntimeError("No popup_scores parameters found for score search optimizer!")

            # Verify no normal model weights appear in trainable parameter list.
            for name_p, p in unwrap_model(model).named_parameters():
                if "popup_scores" not in name_p and p.requires_grad:
                    raise RuntimeError(f"Dense parameter '{name_p}' is trainable during score search!")

            if name in ("auto", "MuSGD", "SGD", "sgd"):
                optimizer = torch.optim.SGD(score_params, lr=lr, momentum=momentum, nesterov=True)
            elif name in ("Adam", "AdamW", "adam", "adamw"):
                optimizer = torch.optim.AdamW(score_params, lr=lr, betas=(momentum, 0.999))
            else:
                optimizer = torch.optim.SGD(score_params, lr=lr, momentum=momentum)

            # Programmatically verify optimizer param groups contain ONLY popup_scores.
            score_param_set = set(score_params)
            for group in optimizer.param_groups:
                for p in group["params"]:
                    assert p in score_param_set, "Non-score parameter found in score optimizer param_groups!"

            return optimizer

        elif self.hydra_stage == "finetune":
            configure_masked_finetune(model)

            optimizer = super().build_optimizer(
                model=model,
                name=name,
                lr=lr,
                momentum=momentum,
                decay=decay,
                iterations=iterations,
            )
            score_param_set = {
                p for name_p, p in unwrap_model(model).named_parameters() if "popup_scores" in name_p
            }
            for group in optimizer.param_groups:
                group["params"] = [p for p in group["params"] if p not in score_param_set]

            # Verify no popup_scores remain in fine-tune optimizer.
            for group in optimizer.param_groups:
                for p in group["params"]:
                    assert p not in score_param_set, "popup_scores found in fine-tune optimizer param_groups!"

            return optimizer

        else:
            raise ValueError(f"Unknown HYDRA stage: {self.hydra_stage}")

    def optimizer_step(self) -> None:
        """Run optimizer step and enforce HYDRA stage invariants."""
        # DEBUG: inspect popup_scores gradients before the FIRST step.
        if self.hydra_stage == "score" and not hasattr(self, "_hydra_grad_checked"):
            model = unwrap_model(self.model)

            score_params = [
                (name, p)
                for name, p in model.named_parameters()
                if "popup_scores" in name
            ]

            total = len(score_params)
            nonzero_grad = [
                name
                for name, p in score_params
                if p.grad is not None
                and torch.count_nonzero(p.grad).item() > 0
            ]

            assert total > 0, (
                "[FAIL] No popup_scores parameters found before optimizer step."
            )
            assert len(nonzero_grad) > 0, (
                "[FAIL] No popup_scores received a non-zero gradient "
                "from the detection loss."
            )
            self._hydra_grad_checked = True

        # Perform normal optimizer step.
        super().optimizer_step()

        # FINETUNE invariant: pruned weights must remain physically zero.
        if self.hydra_stage == "finetune":
            enforce_pruned_weights_zero(self.model)

    def save_model(self):
        """Save model checkpoint preserving HYDRA state and metadata."""
        if self.hydra_stage == "score":
            buffer = io.BytesIO()
            torch.save(
                {
                    "epoch": self.epoch,
                    "best_fitness": self.best_fitness,
                    "model": unwrap_model(self.model),
                    "ema": None,
                    "updates": 0,
                    "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                    "scaler": self.scaler.state_dict(),
                    "train_args": vars(self.args),
                    "train_metrics": {**self.metrics, **{"fitness": self.fitness}} if self.metrics else {},
                    "train_results": self.read_results_csv(),
                    "date": datetime.now().isoformat(),
                    "version": __version__,
                    "license": "AGPL-3.0 (https://ultralytics.com/license)",
                    "docs": "https://docs.ultralytics.com",
                    "hydra_stage": self.hydra_stage,
                    "hydra_keep_ratio": self.hydra_keep_ratio,
                    "hydra_score_init": self.hydra_score_init,
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
        else:
            super().save_model()

    def finalize_score_search(self) -> HydraSparsityReport:
        """Freeze score rankings into permanent masks and return sparsity report."""
        return freeze_hydra_masks(self.model)


class HydraDetectionTrainer(HydraTrainerMixin, DetectionTrainer):
    """Clean detection trainer that adds HYDRA stage management."""

    def __init__(
        self,
        cfg=None,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
    ):
        """Initialize a DetectionTrainer with a private Ultralytics + HYDRA config."""
        hydra_cfg = prepare_hydra_cfg(cfg)
        overrides = deepcopy(overrides) if overrides is not None else {}

        super().__init__(
            cfg=hydra_cfg,
            overrides=overrides,
            _callbacks=_callbacks,
        )
        self.init_hydra_attributes()

    def get_validator(self):
        """Return the standard validator with trainer-only HYDRA keys removed."""
        validator_args = {
            key: value
            for key, value in vars(self.args).items()
            if key not in HYDRA_CONFIG_KEYS
        }

        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=validator_args,
            _callbacks=self.callbacks,
        )
