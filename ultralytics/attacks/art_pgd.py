from __future__ import annotations

import contextlib

import numpy as np
import torch
from torch import nn

from ultralytics.attacks.attacker import Attacker
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import unwrap_model

# ART
try:
    # newer versions of ART using the generic object detector
    from art.estimators.object_detection import PyTorchObjectDetector as ARTDetector
except (ImportError, AttributeError):
    # older versions of ART
    from art.estimators.object_detection import PyTorchYolo as ARTDetector

from art.attacks.evasion import ProjectedGradientDescent

from ultralytics.attacks.attack_utils import (
    BatchContainer,
    capture_module_training_states,
    restore_module_training_states,
)


class YoloV5ForART(nn.Module):
    """
    Wrapper for YOLOv8 to fit ART's (images, targets) -> losses interface.
    """

    def __init__(self, yolo_model: nn.Module, img_size: int):
        super().__init__()
        self.model = yolo_model
        self.img_size = img_size

        # get YOLOv8 loss function
        unwrapped = unwrap_model(self.model)
        self._loss_fn = unwrapped.loss

        # ensure model is in eval mode but allow gradients
        self.model.eval()
        self.current_paths = []

    def zero_grad(self, set_to_none: bool = False):
        """Prevent ART from clearing outer training model accumulated parameter gradients."""

    @torch.enable_grad()
    def forward(self, images: torch.Tensor, targets=None):
        """
        images: (N,C,H,W) float in [0,1]
        targets: list[dict] with keys 'boxes'(xyxy, pixel), 'labels' (int64)
        """
        from ultralytics.utils import LOGGER

        # inference mode
        if targets is None:
            saved_inference_states = capture_module_training_states(self.model)
            try:
                self.model.eval()
                with torch.no_grad():
                    preds = self.model(images, augment=False)
            finally:
                restore_module_training_states(saved_inference_states)
            dets = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.6, max_det=300)
            out = []
            for det in dets:
                if det is None or len(det) == 0:
                    out.append(
                        {
                            "boxes": torch.zeros((0, 4), device=images.device),
                            "labels": torch.zeros((0,), dtype=torch.int64, device=images.device),
                            "scores": torch.zeros((0,), device=images.device),
                        }
                    )
                else:
                    boxes = det[:, :4]
                    scores = det[:, 4]
                    labels = det[:, 5].long()
                    out.append({"boxes": boxes, "labels": labels, "scores": scores})
            return out

        # training mode: convert targets
        _n, _, H, W = images.shape
        yolo_tgts = []

        for i, tgt in enumerate(targets):
            boxes = tgt.get("boxes")
            labels = tgt.get("labels")

            if boxes is None or labels is None:
                continue

            b = torch.as_tensor(boxes, device=images.device, dtype=torch.float32)
            c = torch.as_tensor(labels, device=images.device, dtype=torch.int64)

            if b.numel() == 0 or c.numel() == 0 or b.size(0) == 0:
                continue

            if b.dim() == 1:
                b = b.unsqueeze(0)
            if c.dim() == 0:
                c = c.unsqueeze(0)

            if b.size(-1) != 4:
                continue

            # xyxy (pixel) -> xywh (normalized)
            xy = (b[:, 0:2] + b[:, 2:4]) * 0.5
            wh = (b[:, 2:4] - b[:, 0:2]).clamp(min=1e-6)

            xy[:, 0] /= W
            xy[:, 1] /= H
            wh[:, 0] /= W
            wh[:, 1] /= H

            if torch.any(torch.isnan(xy)) or torch.any(torch.isnan(wh)):
                continue

            img_idx = torch.full((b.size(0), 1), i, device=images.device, dtype=torch.float32)
            cls = c.to(dtype=torch.float32).unsqueeze(1)

            yolo_tgts.append(torch.cat([img_idx, cls, xy, wh], dim=1))

        if len(yolo_tgts) == 0:
            return {"loss": torch.tensor(0.0, device=images.device, requires_grad=True)}

        yolo_targets = torch.cat(yolo_tgts, dim=0)

        if yolo_targets.size(0) == 0 or yolo_targets.size(1) != 6:
            return {"loss": torch.tensor(0.0, device=images.device, requires_grad=True)}

        # use YOLOv8 loss
        saved_states = capture_module_training_states(self.model)
        try:
            # Note: We do NOT freeze model parameters here.
            # self.attack_model is separate from the trained model, so its gradients are harmless (just unused).
            # Freezing them caused the computation graph to break in some environments.

            # Force enable grad for the entire block to ensure operations are recorded
            with torch.enable_grad():
                # ensure input images requires gradients
                if not images.requires_grad:
                    images.requires_grad_(True)

                # Double check status
                if not torch.is_grad_enabled():
                    LOGGER.warning("⚠️ torch.is_grad_enabled() IS FALSE inside wrapper! Forcing it.")
                    torch.set_grad_enabled(True)

                # set train mode to get feature maps (for loss computation)
                # but force BN layers and Dropout to eval mode (use pretrained statistics / deterministic)
                self.model.train()
                for m in self.model.modules():
                    if isinstance(
                        m,
                        (
                            torch.nn.BatchNorm1d,
                            torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d,
                            torch.nn.SyncBatchNorm,
                            torch.nn.Dropout,
                        ),
                    ):
                        m.eval()
                if hasattr(self.model, "pipeline") and self.model.pipeline is not None:
                    self.model.pipeline.eval()

                # forward propagation
                preds = self.model(images, augment=False)

                # LOGGER.info(f"Preds type: {type(preds)}")
                if isinstance(preds, (list, tuple)):
                    LOGGER.info(f"Preds len: {len(preds)}")
                    if len(preds) > 0 and isinstance(preds[0], torch.Tensor):
                        LOGGER.info(f"Preds[0] req_grad: {preds[0].requires_grad}")
                        LOGGER.info(f"Preds[0] grad_fn: {preds[0].grad_fn}")

                # construct batch
                batch = BatchContainer(
                    img=images,
                    batch_idx=yolo_targets[:, 0].long(),
                    cls=yolo_targets[:, 1:2],
                    bboxes=yolo_targets[:, 2:6],
                )

                # call loss function
                loss, _loss_items = self._loss_fn(batch, preds)

                # Ensure loss is scalar for backward()
                final_loss = loss.sum()

                # Verify graph integrity
                if not final_loss.requires_grad:
                    raise RuntimeError(f"Adversarial loss does not require grad! Input req_grad={images.requires_grad}")

            if images.requires_grad and not images.is_leaf:
                images.retain_grad()

            return {"loss": final_loss}
        finally:
            restore_module_training_states(saved_states)


class ARTPGD(Attacker):
    attack_mode = "detector"
    requires_targets = True
    target_format = "detector_targets"
    """
    fit attacker.forward(x, targets)
      epsilon -> PGD.eps
      lr      -> PGD.eps_step
      epoch   -> PGD.max_iter
    """

    def __init__(
        self,
        model,
        config=None,
        target=None,
        epsilon: float = 0.031372549,
        lr: float = 0.00784313725,
        epoch: int = 5,
        img_size: int = 640,
        num_random_init: int = 1,
    ):
        super().__init__(model, config, epsilon)
        self.device = next(model.parameters()).device
        self.epsilon = float(epsilon)
        self.lr = float(lr)
        self.epoch = int(epoch)
        self.img_size = int(img_size)
        self.num_random_init = int(num_random_init)
        if self.num_random_init > 1:
            raise ValueError(
                f"num_random_init={self.num_random_init} > 1 is not supported for detection PGD. Set num_random_init=1."
            )

        saved_states = capture_module_training_states(self.model)
        try:
            # Set model to eval mode (batchnorm/dropout will also be handled in ART wrapper)
            self.model.eval()
            # Note: Do not force p.requires_grad_(True) to preserve frozen layers and avoid parameter gradient calculation

            # wrap yolo model for ART
            wrapped = YoloV5ForART(self.model, img_size)
            self.wrapped_model = wrapped

            # ART Detector with dynamic device resolution to avoid moving CPU models to GPU
            resolved_device_type = "gpu" if self.device.type == "cuda" else "cpu"
            self.estimator = ARTDetector(
                model=wrapped,
                input_shape=(3, img_size, img_size),
                clip_values=(0.0, 1.0),
                channels_first=True,
                device_type=resolved_device_type,
                attack_losses=("loss",),
            )
            # disable batchnorm/dropout
            if hasattr(self.estimator, "set_batchnorm"):
                self.estimator.set_batchnorm(False)
            if hasattr(self.estimator, "set_dropout"):
                self.estimator.set_dropout(False)

            # PGD
            self.attack = ProjectedGradientDescent(
                estimator=self.estimator,
                norm=np.inf,
                eps=epsilon,
                eps_step=lr,
                max_iter=epoch,
                targeted=False,
                num_random_init=self.num_random_init,
                verbose=False,
            )
        finally:
            restore_module_training_states(saved_states)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        x: (N,C,H,W) in [0,1] float
        targets: (M,6) [img_idx, cls, x, y, w, h] normalized xywh
        """
        is_mim = getattr(self, "decay", 0.0) > 0.0
        num_random_init = int(getattr(self.attack, "num_random_init", getattr(self, "num_random_init", 1)))

        # Route MIM and zero-random-init attacks to ART's generate path
        if is_mim or num_random_init == 0:
            return self._forward_art(x, targets)

        if num_random_init > 1:
            raise ValueError(
                f"num_random_init={num_random_init} > 1 is not supported for detection PGD. Set num_random_init=1."
            )

        # Native PyTorch L_inf PGD with random initialization
        return self._forward_native_pgd(x, targets)

    def _forward_art(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """ART generate execution path for MIM and zero random initialization."""
        saved_states = capture_module_training_states(self.model)
        try:
            n, _, h, w = x.shape
            y_art = self._to_art_labels(targets, h, w, n)

            x_np = x.detach().cpu().numpy()
            if hasattr(self.attack, "set_params"):
                with contextlib.suppress(Exception):
                    self.attack.set_params(batch_size=n)
            if hasattr(self.attack, "_attack") and hasattr(self.attack._attack, "batch_size"):
                self.attack._attack.batch_size = n
            if hasattr(self.attack, "batch_size"):
                self.attack.batch_size = n
            x_adv_np = self.attack.generate(x=x_np, y=y_art)
            x_adv = torch.from_numpy(x_adv_np).to(self.device).type_as(x)
            return x_adv
        finally:
            restore_module_training_states(saved_states)

    def _forward_native_pgd(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Native PyTorch L_inf PGD execution path with random initialization for object detection."""
        orig_dtype = x.dtype
        orig_device = x.device

        eps = float(getattr(self.attack, "eps", self.epsilon))
        eps_step = float(getattr(self.attack, "eps_step", self.lr))
        max_iter = int(getattr(self.attack, "max_iter", self.epoch))

        # Save original center in FP32 on model device without modifying x
        x0 = x.detach().clone().float().to(self.device)

        # Uniform random perturbation in [-epsilon, +epsilon], clipped to [0, 1]
        noise = torch.empty_like(x0).uniform_(-eps, eps)
        x_adv = torch.clamp(x0 + noise, 0.0, 1.0)

        n, _, h, w = x.shape
        y_art = self._to_art_labels(targets, h, w, n)
        wrapped = getattr(self, "wrapped_model", getattr(self.estimator, "model", None))

        saved_states = capture_module_training_states(self.model)
        try:
            for _ in range(max_iter):
                x_adv = x_adv.detach().requires_grad_(True)
                out = wrapped(x_adv, y_art)
                loss = out.get("loss")
                if loss is None or not loss.requires_grad:
                    break
                grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
                if not torch.isfinite(grad).all():
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

                # Step along gradient sign
                x_adv = x_adv.detach() + eps_step * grad.sign()

                # Project onto L_inf ball of radius eps centered at original x0
                x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)

                # Clip to valid image range [0, 1]
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

            return x_adv.detach().to(dtype=orig_dtype, device=orig_device)
        finally:
            restore_module_training_states(saved_states)

    @staticmethod
    def _to_art_labels(targets: torch.Tensor, H: int, W: int, batch_size: int) -> list[dict[str, np.ndarray]]:
        """Convert YOLO targets to ART label format.

        Args:
            targets: (M, 6) [img_idx, cls, x, y, w, h] (xywh normalized)
            H: Image height
            W: Image width
            batch_size: Number of images in batch

        Returns:
            List of dicts with 'boxes' (N, 4) xyxy and 'labels' (N,)
        """
        out: list[dict[str, np.ndarray]] = []

        # input validation
        if targets is None or targets.numel() == 0:
            for _ in range(batch_size):
                out.append({"boxes": np.zeros((0, 4), dtype=np.float32), "labels": np.zeros((0,), dtype=np.int64)})
            return out

        t = targets.detach().cpu()

        # ensure targets is 2D tensor
        if t.dim() == 1:
            t = t.unsqueeze(0)
        elif t.dim() > 2:
            t = t.squeeze()
            if t.dim() == 1:
                t = t.unsqueeze(0)

        # check number of columns
        if t.size(1) < 6:
            raise ValueError(f"targets must have 6 columns, got {t.size(1)}")

        for i in range(batch_size):
            ti = t[t[:, 0] == i]

            # use size(0) to check number of rows
            if ti.size(0) == 0:
                out.append({"boxes": np.zeros((0, 4), dtype=np.float32), "labels": np.zeros((0,), dtype=np.int64)})
                continue

            # ensure ti is 2D
            if ti.dim() == 1:
                ti = ti.unsqueeze(0)

            # xywh (normalized) -> xyxy (pixel)
            xywh = ti[:, 2:6].clone()
            xywh[:, 0] *= W
            xywh[:, 1] *= H
            xywh[:, 2] *= W
            xywh[:, 3] *= H

            xyxy = torch.zeros_like(xywh)
            xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
            xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
            xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
            xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

            # clip to image boundary
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, W)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, H)

            labels = ti[:, 1].to(torch.int64)

            out.append(
                {
                    "boxes": xyxy.numpy().astype(np.float32),
                    "labels": labels.numpy().astype(np.int64),
                }
            )

        return out
