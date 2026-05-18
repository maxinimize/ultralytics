import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

from ultralytics.attacks.attacker import Attacker
from .art_pgd import YoloV5ForART  # reuse the wrapper

# ART
try:
    from art.estimators.object_detection import PyTorchObjectDetector as ARTDetector
except Exception:
    from art.estimators.object_detection import PyTorchYolo as ARTDetector

from art.attacks.evasion import FastGradientMethod


class ARTFGSM(Attacker):
    attack_mode = "detector"
    requires_targets = True
    target_format = "detector_targets"
    """
    FGSM epsilon -> FGSM.eps
    """
    def __init__(self, model, config=None, target=None,
                 epsilon=0.05, img_size=640):
        super().__init__(model, config, epsilon)
        self.device = next(model.parameters()).device

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        wrapped = YoloV5ForART(self.model, img_size)

        self.estimator = ARTDetector(
            model=wrapped,
            input_shape=(3, img_size, img_size),
            clip_values=(0.0, 1.0),
            channels_first=True,
            device_type="gpu",
            attack_losses=("loss",),
        )
        if hasattr(self.estimator, "set_batchnorm"):
            self.estimator.set_batchnorm(False)
        if hasattr(self.estimator, "set_dropout"):
            self.estimator.set_dropout(False)

        self.attack = FastGradientMethod(
            estimator=self.estimator,
            eps=epsilon,
            targeted=False,
            num_random_init=0,
        )

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        x: (N,C,H,W) in [0,1]
        targets: (M,6) [img_idx, cls, x, y, w, h] (xywh 归一化)
        """
        n, _, h, w = x.shape
        y_art = self._to_art_labels(targets, h, w, n)

        x_np = x.detach().cpu().numpy()
        x_adv_np = self.attack.generate(x=x_np, y=y_art)
        x_adv = torch.from_numpy(x_adv_np).to(self.device).type_as(x)
        return x_adv

    # @staticmethod
    # def _to_art_labels(targets: torch.Tensor, H: int, W: int, batch_size: int) -> List[Dict[str, np.ndarray]]:
    #     out: List[Dict[str, np.ndarray]] = []
    #     t = targets.detach().cpu()
    #     for i in range(batch_size):
    #         ti = t[t[:, 0] == i]
    #         if ti.numel() == 0:
    #             out.append({"boxes": np.zeros((0, 4), dtype=np.float32),
    #                         "labels": np.zeros((0,), dtype=np.int64)})
    #             continue
    #         xywh = ti[:, 2:6].clone()
    #         xywh[:, 0] *= W; xywh[:, 1] *= H
    #         xywh[:, 2] *= W; xywh[:, 3] *= H
    #         xyxy = torch.zeros_like(xywh)
    #         xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    #         xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    #         xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    #         xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    #         labels = ti[:, 1].to(torch.int64)
    #         out.append({
    #             "boxes": xyxy.numpy().astype(np.float32),
    #             "labels": labels.numpy().astype(np.int64),
    #         })
    #     return out

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
        
        # ✅ 增加输入验证
        if targets is None or targets.numel() == 0:
            # 返回空标签
            for _ in range(batch_size):
                out.append({
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "labels": np.zeros((0,), dtype=np.int64)
                })
            return out
        
        t = targets.detach().cpu()
        
        # ✅ 确保 targets 是 2D 张量
        if t.dim() == 1:
            t = t.unsqueeze(0)  # (6,) -> (1, 6)
        elif t.dim() > 2:
            t = t.squeeze()
            if t.dim() == 1:
                t = t.unsqueeze(0)
        
        # ✅ 验证 targets 至少有 6 列
        if t.size(1) < 6:
            raise ValueError(f"targets must have 6 columns, got {t.size(1)}")
        
        for i in range(batch_size):
            ti = t[t[:, 0] == i]  # 提取属于图像 i 的目标
            
            # ✅ 改进空检查: 检查行数而不是 numel
            if ti.size(0) == 0:  # 没有目标
                out.append({
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "labels": np.zeros((0,), dtype=np.int64)
                })
                continue
            
            # ✅ 确保 ti 是 2D
            if ti.dim() == 1:
                ti = ti.unsqueeze(0)
            
            # 转换 xywh (归一化) -> xyxy (绝对坐标)
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
            
            # ✅ 裁剪到图像边界
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, W)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, H)
            
            labels = ti[:, 1].to(torch.int64)
            
            out.append({
                "boxes": xyxy.numpy().astype(np.float32),
                "labels": labels.numpy().astype(np.int64),
            })
        
        return out