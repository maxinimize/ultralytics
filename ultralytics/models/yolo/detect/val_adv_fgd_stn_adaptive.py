# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Adaptive Adversarial Validation Pipeline with configurable FGD + STN defense preprocessing."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any, Sequence

import torch

from ultralytics.attacks.attack_bridge import run_attack_on_batch
from ultralytics.defenses.fgd_stn_pipeline import (
    MODE_DESCRIPTIONS,
    FGDSTNPipeline,
    build_fgd_stn_pipeline,
    canonicalize_mode,
    transform_batch_bboxes_for_stn,
)
from ultralytics.defenses.spatial_transformer import transform_boxes
from ultralytics.models.yolo.detect.val_adv_adaptive import (
    DetectionValidator as DetectionValidatorAdvAdaptive,
)
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import autocast


class DetectionValidatorAdvFGDSTNAdaptive(DetectionValidatorAdvAdaptive):
    """Detection validator supporting Unified FGD + STN preprocessing with Adaptive Adversarial evaluation.

    Evaluates Clean (raw) validation images followed by each specified attack (PGD, BIM, MIM, Worst-of-K).
    Images are preprocessed through the configured defense mode (none, fgd, stn, fgd_stn, stn_fgd).
    STN bounding boxes are unwarped back to input coordinates before evaluation metrics are calculated.
    Fitness is calculated exclusively using normalized fixed evaluation weights on attack mAP50-95
    (Clean weight is strictly 0.0), matching the adaptive trainer's checkpoint selection.
    """

    def __init__(
        self,
        dataloader=None,
        save_dir: Path | None = None,
        args: dict | Any | None = None,
        _callbacks: dict | None = None,
        pipeline: FGDSTNPipeline | None = None,
        preprocess_mode: str = "none",
        fgd_weights: str | Path | None = None,
        stn_weights: str | Path | None = None,
        attack_through_preprocess: bool = False,
        eval_weights: Sequence[float] | None = None,
        attack_weights: str | Path | None = None,
        strict_attacks: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize validator with defense pipeline and preprocessing options."""
        raw_mode = preprocess_mode
        raw_fgd_w = fgd_weights
        raw_stn_w = stn_weights
        raw_att_through = attack_through_preprocess
        raw_attack_weights = attack_weights
        raw_strict = strict_attacks

        if isinstance(args, dict):
            raw_mode = args.pop("preprocess_mode", raw_mode)
            raw_fgd_w = args.pop("fgd_weights", raw_fgd_w)
            raw_stn_w = args.pop("stn_weights", raw_stn_w)
            raw_att_through = args.pop("attack_through_preprocess", raw_att_through)
            raw_attack_weights = args.pop("attack_weights", raw_attack_weights)
            raw_strict = args.pop("strict_attacks", raw_strict)
        elif args is not None:
            for k in (
                "preprocess_mode",
                "fgd_weights",
                "stn_weights",
                "attack_through_preprocess",
                "attack_weights",
                "strict_attacks",
            ):
                if hasattr(args, k):
                    val = getattr(args, k)
                    if k == "preprocess_mode":
                        raw_mode = val
                    elif k == "fgd_weights":
                        raw_fgd_w = val
                    elif k == "stn_weights":
                        raw_stn_w = val
                    elif k == "attack_through_preprocess":
                        raw_att_through = val
                    elif k == "attack_weights":
                        raw_attack_weights = val
                    elif k == "strict_attacks":
                        raw_strict = val
                    try:
                        delattr(args, k)
                    except AttributeError:
                        pass

        super().__init__(
            dataloader=dataloader,
            save_dir=save_dir,
            args=args,
            _callbacks=_callbacks,
            attack_weights=raw_attack_weights,
            strict_attacks=raw_strict,
            **kwargs,
        )

        self.preprocess_mode = canonicalize_mode(raw_mode or "none")
        self.fgd_weights = raw_fgd_w
        self.stn_weights = raw_stn_w
        self.attack_through_preprocess = (
            raw_att_through
            if isinstance(raw_att_through, bool)
            else str(raw_att_through).lower() in ("true", "1", "yes")
        )
        self.strict_attacks = (
            raw_strict if isinstance(raw_strict, bool) else str(raw_strict).lower() in ("true", "1", "yes")
        )
        self.args.strict_attacks = self.strict_attacks

        self.pipeline: FGDSTNPipeline | None = pipeline
        self.current_pred_matrix: torch.Tensor | None = None
        self._current_hw: tuple[int, int] = (640, 640)
        self.attack_failures = set()
        self.eval_weights = eval_weights
        self.end2end = False

    def _setup_pipeline(self, device: torch.device):
        """Ensure pipeline is built and frozen on the active device."""
        if self.pipeline is None:
            if self.preprocess_mode in {"fgd", "fgd_stn", "stn_fgd"} and (
                not self.fgd_weights or not Path(self.fgd_weights).exists()
            ):
                raise FileNotFoundError(
                    f"FGD checkpoint is required for preprocess_mode '{self.preprocess_mode}' but was not found at: "
                    f"'{self.fgd_weights}'. Silent fallback to identity is prohibited."
                )
            if self.preprocess_mode in {"stn", "fgd_stn", "stn_fgd"} and (
                not self.stn_weights or not Path(self.stn_weights).exists()
            ):
                raise FileNotFoundError(
                    f"STN checkpoint is required for preprocess_mode '{self.preprocess_mode}' but was not found at: "
                    f"'{self.stn_weights}'. Silent fallback to identity is prohibited."
                )

            self.pipeline, _ = build_fgd_stn_pipeline(
                fgd_weights=self.fgd_weights,
                stn_weights=self.stn_weights,
                device=device,
                allow_identity_fallback=False,
            )

            if self.preprocess_mode in {"fgd", "fgd_stn", "stn_fgd"} and not self.pipeline.has_denoiser:
                raise RuntimeError("FGD denoiser module failed to load from checkpoint!")
            if self.preprocess_mode in {"stn", "fgd_stn", "stn_fgd"} and not self.pipeline.has_stn:
                raise RuntimeError("STN module failed to load from checkpoint!")

        self.pipeline.to(device)
        self.pipeline.eval().requires_grad_(False)

    def __call__(self, trainer=None, model=None):
        """Execute multi-attack validation with configured defense preprocessing pipeline."""
        if trainer is not None:
            if hasattr(trainer, "strict_attacks"):
                self.strict_attacks = bool(trainer.strict_attacks)
                self.args.strict_attacks = self.strict_attacks
            if hasattr(trainer, "preprocess_mode"):
                self.preprocess_mode = canonicalize_mode(trainer.preprocess_mode)
            if hasattr(trainer, "fgd_weights"):
                self.fgd_weights = trainer.fgd_weights
            if hasattr(trainer, "stn_weights"):
                self.stn_weights = trainer.stn_weights
            if hasattr(trainer, "pipeline") and trainer.pipeline is not None:
                self.pipeline = trainer.pipeline

        target_device = trainer.device if trainer is not None else getattr(self, "device", None)
        if target_device is None:
            target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._setup_pipeline(target_device)

        if self.attack_through_preprocess and self.preprocess_mode != "none":
            att_desc = "Validation attacks configured to pass through defense preprocessing"
        else:
            att_desc = "Validation attacks target YOLO detector directly (attack_through_preprocess=False)"
        LOGGER.info(
            f"Unified FGD+STN Adaptive Validator running with mode: '{self.preprocess_mode}' "
            f"({MODE_DESCRIPTIONS.get(self.preprocess_mode, self.preprocess_mode)}) | {att_desc}"
        )

        try:
            return super().__call__(trainer=trainer, model=model)
        finally:
            self.current_pred_matrix = None
            self._current_hw = None

    def compute_validation_loss(
        self,
        batch: dict[str, Any],
        preds: Any,
        model: torch.nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute validation loss, transforming ground truth bboxes to rectified coordinates if STN was applied.

        Args:
            batch (dict[str, Any]): Batch dictionary containing annotations and input images.
            preds (Any): Model predictions for the batch.
            model (nn.Module, optional): Model instance to compute loss with. Defaults to self.model.

        Returns:
            (tuple[torch.Tensor, dict[str, torch.Tensor]]): Tuple of (loss, loss_items).
        """
        target = model if model is not None else getattr(self, "model", None)
        if target is None:
            raise ValueError("No model provided or available on validator to compute validation loss.")
        if self.current_pred_matrix is not None and getattr(self, "pipeline", None) is not None:
            h, w = self._current_hw if self._current_hw is not None else (batch["img"].shape[2], batch["img"].shape[3])
            batch_loss = {k: v.clone() if torch.is_tensor(v) else copy(v) for k, v in batch.items()}
            batch_loss = transform_batch_bboxes_for_stn(batch_loss, self.current_pred_matrix, height=h, width=w)
            return target.loss(batch_loss, preds)
        return target.loss(batch, preds)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch: move to device, generate online attack, and apply defense pipeline."""
        self.current_pred_matrix = None
        self._current_hw = None
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = (batch["img"].half() if self.args.quantize == 16 else batch["img"].float()) / 255.0

        attack_name = getattr(self, "current_attack", None) or getattr(self, "attack_name", None)
        if attack_name and attack_name.lower().strip() not in {"raw", "none", "clean"}:
            attacker = None
            if hasattr(self, "attackers") and isinstance(self.attackers, dict):
                attacker = self.attackers.get(attack_name, None)
            if attacker is None:
                attacker = getattr(self, "attacker", None)

            batch_i = getattr(self, "batch_i", 0)
            if attacker is None:
                self.attack_failures.add(attack_name)
                msg = f"Attacker for '{attack_name}' was not initialized at batch {batch_i}. Cannot substitute clean data."
                LOGGER.error(msg)
                if self.strict_attacks:
                    raise RuntimeError(msg)
            else:
                has_targets = "bboxes" in batch and batch["bboxes"] is not None and batch["bboxes"].numel() > 0
                if not has_targets and attack_name != "worstk":
                    # Legitimate skip due to no targets
                    pass
                else:
                    try:
                        im_files = batch.get("im_file", [])
                        if hasattr(attacker, "proxy_model"):
                            attacker.proxy_model.current_paths = im_files
                        if hasattr(attacker, "estimator") and hasattr(attacker.estimator, "model"):
                            attacker.estimator.model.current_paths = im_files

                        if self.device.type == "cuda" and getattr(self.args, "amp", None) == "bf16":
                            amp_ctx = autocast(torch.bfloat16, device=self.device.type)
                        elif self.device.type == "cuda" and (
                            getattr(self.args, "amp", False)
                            or getattr(self.args, "half", False)
                            or getattr(self.args, "quantize", None) == 16
                        ):
                            amp_ctx = autocast(True, device=self.device.type)
                        else:
                            amp_ctx = autocast(False, device=self.device.type)

                        adv_batch_size = int(getattr(self.args, "adv_val_batch", 16))
                        with amp_ctx:
                            if attack_name == "worstk" and callable(getattr(attacker, "forward_batch", None)):
                                attacker.forward_batch(batch)
                                imgs_adv = batch["img"]
                            else:
                                imgs_adv = run_attack_on_batch(
                                    attacker, batch, label_policy="largest_box", max_batch_size=adv_batch_size
                                )

                        if imgs_adv is None:
                            raise RuntimeError(
                                f"Adversarial generation returned None for attack '{attack_name}' at batch {batch_i} on batch with targets."
                            )
                        if not torch.isfinite(imgs_adv).all():
                            raise RuntimeError(
                                f"Adversarial generation produced NaN or Inf for attack '{attack_name}' at batch {batch_i}."
                            )
                        if imgs_adv.shape != batch["img"].shape:
                            raise RuntimeError(
                                f"Adversarial generation output shape {imgs_adv.shape} mismatch with input {batch['img'].shape} for attack '{attack_name}' at batch {batch_i}."
                            )

                        batch["img"] = imgs_adv.to(batch["img"].dtype).to(self.device)
                    except Exception as e:
                        self.attack_failures.add(attack_name)
                        msg = f"Adversarial generation failed for attack '{attack_name}' at batch {batch_i}: {e}"
                        LOGGER.error(msg)
                        if self.strict_attacks:
                            raise RuntimeError(msg) from e

        _, _, h, w = batch["img"].shape
        self._current_hw = (h, w)

        # Apply defense preprocessing pipeline according to configured mode
        if self.pipeline is not None:
            with torch.no_grad():
                processed_img, pred_matrix = self.pipeline.forward(batch["img"], mode=self.preprocess_mode)
                batch["img"] = processed_img
                self.current_pred_matrix = pred_matrix
        else:
            self.current_pred_matrix = None

        return batch

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Perform NMS and unwarp bounding boxes back to canvas coordinates if STN was applied."""
        if isinstance(preds, (list, tuple)) and len(preds) > 0 and isinstance(preds[0], dict):
            processed_preds = preds
        else:
            processed_preds = super().postprocess(preds)

        if self.current_pred_matrix is not None and self.pipeline is not None:
            h, w = self._current_hw
            unwarped_preds = []
            for si, pred_dict in enumerate(processed_preds):
                boxes = pred_dict.get("bboxes")
                if boxes is None or boxes.numel() == 0:
                    unwarped_preds.append(pred_dict)
                    continue

                m_si = self.current_pred_matrix[si]
                # transform_boxes maps rectified coords -> canvas coords using forward matrix m_si
                mapped_boxes = transform_boxes(boxes, m_si, height=h, width=w, clip=True)

                # Filter degenerate boxes
                bw = mapped_boxes[:, 2] - mapped_boxes[:, 0]
                bh = mapped_boxes[:, 3] - mapped_boxes[:, 1]
                valid = (bw > 0.0) & (bh > 0.0)

                new_dict = {
                    "bboxes": mapped_boxes[valid],
                    "conf": pred_dict["conf"][valid],
                    "cls": pred_dict["cls"][valid],
                }
                if "extra" in pred_dict and pred_dict["extra"] is not None:
                    new_dict["extra"] = pred_dict["extra"][valid]
                unwarped_preds.append(new_dict)

            return unwarped_preds

        return processed_preds


DetectionValidator = DetectionValidatorAdvFGDSTNAdaptive

__all__ = ["DetectionValidator", "DetectionValidatorAdvFGDSTNAdaptive"]
