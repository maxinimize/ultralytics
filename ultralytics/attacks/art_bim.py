import numpy as np
import torch
from typing import List, Dict, Optional, Sequence, Tuple

from ultralytics.attacks.attacker import Attacker
from .art_pgd import YoloV5ForART  # reuse the wrapper

# ART
try:
    from art.estimators.object_detection import PyTorchObjectDetector as ARTDetector
except Exception:
    from art.estimators.object_detection import PyTorchYolo as ARTDetector

from art.attacks.evasion import BasicIterativeMethod


class ARTBIM(Attacker):
    """
    BIM through ART's object-detector estimator.

    This version intentionally keeps the old PGD-style object-detector route:
      YOLO model -> YoloV5ForART wrapper -> ART object detector -> BasicIterativeMethod

    Parameter mapping:
      epsilon / eps     -> BIM.eps
      lr / eps_step     -> BIM.eps_step
      epoch / max_iter  -> BIM.max_iter
    """

    # Metadata used by attack_bridge.infer_attack_mode() and run_attack_on_batch().
    # Keeping this as detector mode means attack_bridge will pass YOLO detector targets:
    # (M, 6) [img_idx, cls, x, y, w, h].
    attack_mode = "detector"
    requires_targets = True
    target_format = "yolo_detection"

    def __init__(
        self,
        model,
        config=None,
        target=None,
        epsilon: float = 0.05,
        lr: float = 0.005,
        epoch: int = 20,
        img_size: int = 640,
        *,
        eps: Optional[float] = None,
        eps_step: Optional[float] = None,
        max_iter: Optional[int] = None,
        targeted: bool = False,
        batch_size: int = 32,
        verbose: bool = False,
        clip_values: Tuple[float, float] = (0.0, 1.0),
        channels_first: bool = True,
        device_type: Optional[str] = None,
        attack_losses: Sequence[str] = ("loss",),
        **kwargs,
    ):
        super().__init__(model, config, epsilon if eps is None else eps)
        self.device = next(model.parameters()).device

        # Allow either the local project names or ART-style names.
        self.epsilon = float(epsilon if eps is None else eps)
        self.lr = float(lr if eps_step is None else eps_step)
        self.epoch = int(epoch if max_iter is None else max_iter)
        self.img_size = int(img_size)
        self.targeted = bool(targeted)
        self.batch_size = int(batch_size)
        self.verbose = bool(verbose)
        self.clip_values = clip_values
        self.channels_first = channels_first
        self.attack_losses = tuple(attack_losses)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        self.wrapped_model = YoloV5ForART(self.model, self.img_size)
        resolved_device_type = device_type or ("gpu" if self.device.type == "cuda" else "cpu")

        detector_kwargs = dict(
            model=self.wrapped_model,
            input_shape=(3, self.img_size, self.img_size),
            clip_values=self.clip_values,
            channels_first=self.channels_first,
            device_type=resolved_device_type,
            attack_losses=self.attack_losses,
        )

        # Different ART versions expose slightly different object-detector constructors.
        # Try the full constructor first, then fall back by removing optional arguments.
        try:
            self.estimator = ARTDetector(**detector_kwargs)
        except TypeError:
            detector_kwargs.pop("attack_losses", None)
            self.estimator = ARTDetector(**detector_kwargs)

        if hasattr(self.estimator, "set_batchnorm"):
            self.estimator.set_batchnorm(False)
        if hasattr(self.estimator, "set_dropout"):
            self.estimator.set_dropout(False)

        attack_kwargs = dict(
            eps=self.epsilon,
            eps_step=self.lr,
            max_iter=self.epoch,
            targeted=self.targeted,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        # Newer ART uses estimator=..., older ART variants may still accept classifier=...
        try:
            self.attack = BasicIterativeMethod(estimator=self.estimator, **attack_kwargs)
        except TypeError:
            self.attack = BasicIterativeMethod(classifier=self.estimator, **attack_kwargs)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        x: (N,C,H,W) in [0,1]
        targets: (M,6) [img_idx, cls, x, y, w, h] (xywh normalized)
        """
        n, _, h, w = x.shape
        y_art = self._to_art_labels(targets, h, w, n)

        x_np = x.detach().cpu().numpy()
        x_adv_np = self.attack.generate(x=x_np, y=y_art)
        x_adv = torch.from_numpy(x_adv_np).to(self.device).type_as(x)
        return x_adv

    @staticmethod
    def _to_art_labels(targets: torch.Tensor, H: int, W: int, batch_size: int) -> List[Dict[str, np.ndarray]]:
        """Convert YOLO targets to ART label format.
        
        Args:
            targets: (M, 6) [img_idx, cls, x, y, w, h] (xywh normalized)
            H: Image height
            W: Image width
            batch_size: Number of images in batch
            
        Returns:
            List of dicts with 'boxes' (N, 4) xyxy and 'labels' (N,)
        """
        out: List[Dict[str, np.ndarray]] = []
        
        # add target validation
        if targets is None or targets.numel() == 0:
            # return empty labels
            for _ in range(batch_size):
                out.append({
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "labels": np.zeros((0,), dtype=np.int64)
                })
            return out
        
        t = targets.detach().cpu()
        
        # ensure targets is 2D tensor
        if t.dim() == 1:
            t = t.unsqueeze(0)  # (6,) -> (1, 6)
        elif t.dim() > 2:
            t = t.squeeze()
            if t.dim() == 1:
                t = t.unsqueeze(0)
        
        # ensure targets has 6 columns
        if t.size(1) < 6:
            raise ValueError(f"targets must have 6 columns, got {t.size(1)}")
        
        for i in range(batch_size):
            ti = t[t[:, 0] == i]  # extract targets for image i
            
            # check if there are any targets
            if ti.size(0) == 0:  # no targets
                out.append({
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "labels": np.zeros((0,), dtype=np.int64)
                })
                continue
            
            # ensure ti is 2D
            if ti.dim() == 1:
                ti = ti.unsqueeze(0)
            
            # convert xywh (normalized) to xyxy (absolute coordinates)
            xywh = ti[:, 2:6].clone()
            xywh[:, 0] *= W  # x_center
            xywh[:, 1] *= H  # y_center
            xywh[:, 2] *= W  # width
            xywh[:, 3] *= H  # height
            
            # xywh -> xyxy
            xyxy = torch.zeros_like(xywh)
            xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
            xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
            xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
            xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2
            
            # clamp to image boundary
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, W)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, H)
            
            labels = ti[:, 1].to(torch.int64)
            
            out.append({
                "boxes": xyxy.numpy().astype(np.float32),
                "labels": labels.numpy().astype(np.int64),
            })
        
        return out
