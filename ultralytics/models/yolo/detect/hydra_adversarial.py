"""Milestone-3 adapter contract and implementation for robust HYDRA training.

Reuses the project's existing adversarial-training pipeline from
`train_adv_test.py` (canonical reference for Milestone 3) without duplicate
reimplementation of PGD/BIM/MIM/Worst-of-k attack algorithms or detection loss.

Class hierarchy:
    BaseTrainer
        -> ExistingAdversarialDetectionTrainer (train_adv_test.py)
            -> HydraAdversarialDetectionTrainer (hydra_adversarial.py)
                (with HydraTrainerMixin from hydra_train.py)

Key properties:
    - Default attack source: current dynamically masked HYDRA model (SCORE) or
      fixed-mask HYDRA model (FINETUNE) when no explicit attack weight is configured.
    - Fixed-source compatibility: separate fixed attack model when explicit attack
      weights are configured.
    - SCORE stage: updates popup_scores only; dense YOLO weights remain frozen.
    - FINETUNE stage: updates retained weights only; popup_scores and masks are fixed;
      pruned weights stay physically zero.
    - Gradient hygiene: inner attack gradients are isolated from outer training updates;
      adversarial batch images are detached before the outer forward pass.
"""

from __future__ import annotations

import io
from copy import deepcopy
from datetime import datetime
from typing import Any, Protocol

import torch

from ultralytics.models.yolo.detect.hydra_train import (
    HYDRA_CONFIG_KEYS,
    HYDRA_DEFAULTS,
    HydraTrainerMixin,
    prepare_hydra_cfg,
)
from ultralytics.models.yolo.detect.train_adv_test import (
    DetectionTrainer as AdversarialDetectionTrainer,
)
from ultralytics.utils import __version__
from ultralytics.utils.torch_utils import (
    convert_optimizer_state_dict_to_fp16,
    unwrap_model,
)


class RobustBatchAdapter(Protocol):
    """Interface the existing adversarial pipeline satisfies conceptually."""

    def build_robust_objective(self, batch: dict, model) -> tuple[torch.Tensor, dict]:
        """Return robust detection loss and logging items."""
        ...


def integration_checklist() -> tuple[str, ...]:
    """Return Milestone-3 integration requirements for agents and reviewers."""
    return (
        "Reuse existing PGD/BIM/MIM/Worst-of-k attack implementations from train_adv_test.py.",
        "Generate attacks against the current masked model during score search by default.",
        "Generate attacks against the current fixed-mask model during fine-tuning by default.",
        "Preserve fixed-source attack mode when explicit attack weights are configured.",
        "Do not replace Ultralytics detection loss with YOLOv3 loss code.",
        "Verify only popup_scores change during robust score search.",
        "Verify masks remain fixed and pruned weights stay zero during robust fine-tuning.",
        "Ensure inner attack gradients are detached and do not corrupt outer optimization.",
    )


class HydraAdversarialDetectionTrainer(HydraTrainerMixin, AdversarialDetectionTrainer):
    """Adversarial detection trainer that adds HYDRA stage management.

    Inherits:
      - AdversarialDetectionTrainer (train_adv_test.py): multi-attack dataloader,
        disjoint virtual sample ratios, online sample generation, sub-batch routing,
        weighted loss combination, and gradient isolation.
      - HydraTrainerMixin (hydra_train.py): HYDRA model conversion, score search,
        popup_scores optimization, fixed-mask fine-tuning, and zero-weight enforcement.
    """

    def __init__(
        self,
        cfg=None,
        overrides: dict[str, Any] | None = None,
        _callbacks=None,
        attack_weights="",
    ):
        """Initialize an adversarial detection trainer with HYDRA stage management."""
        hydra_cfg = prepare_hydra_cfg(cfg)
        overrides = deepcopy(overrides) if overrides is not None else {}

        super().__init__(
            cfg=hydra_cfg,
            overrides=overrides,
            _callbacks=_callbacks,
            attack_weights=attack_weights,
        )

        self.init_hydra_attributes()

    def get_validator(self):
        """Return the project's multi-attack validator with trainer-only HYDRA keys removed."""
        # Temporarily remove trainer-only HYDRA keys from self.args so validator args check passes
        removed_keys = {}
        for key in HYDRA_CONFIG_KEYS:
            if hasattr(self.args, key):
                removed_keys[key] = getattr(self.args, key)
                delattr(self.args, key)

        try:
            validator = super().get_validator()
        finally:
            # Restore HYDRA keys on trainer.args
            for key, val in removed_keys.items():
                setattr(self.args, key, val)

        return validator

    def save_model(self):
        """Save model checkpoint preserving HYDRA state, attack metadata, and results."""
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
                    "attack_name": getattr(self.args, "attack_name", None),
                    "attack_ratio": getattr(self.args, "attack_ratio", None),
                    "attack_weights": getattr(self.args, "attack_weights", None),
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
