import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

from ultralytics.attacks.attacker import Attacker
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils.nms import non_max_suppression

# ART
try:
    # newer versions of ART using the generic object detector
    from art.estimators.object_detection import PyTorchObjectDetector as ARTDetector
except Exception:
    # older versions of ART
    from art.estimators.object_detection import PyTorchYolo as ARTDetector

from art.attacks.evasion import ProjectedGradientDescent


class BatchContainer:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._data = kwargs
    
    def __getitem__(self, key):
        return self._data[key]
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def __contains__(self, key):
        return key in self._data

# class YoloV5ForART(nn.Module):
#     """
#     wrap YOLOv5 to fit ART's (images, targets) -> losses interface.
#     """
#     def __init__(self, yolo_model: nn.Module, img_size: int):
#         super().__init__()
#         self.model = yolo_model
#         self.img_size = img_size
#         self._loss_fn = unwrap_model(self.model).loss

#     # @torch.enable_grad()
#     # def forward(self, images: torch.Tensor, targets=None):
#     #     """
#     #     images: (N,C,H,W) float in [0,1]
#     #     targets: list[dict] with keys 'boxes'(xyxy, pixel), 'labels' (int64)
#     #     return: if targets is None, return torchvision style outputs;
#     #     """
#     #     # YOLOv5 forward
#     #     preds = self.model(images, augment=False)

#     #     if targets is None:
#     #         # if targets is None, return torchvision style outputs;
#     #         dets = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.6, max_det=300)
#     #         out = []
#     #         for det in dets:
#     #             if det is None or len(det) == 0:
#     #                 out.append({"boxes": torch.zeros((0, 4), device=images.device),
#     #                             "labels": torch.zeros((0,), dtype=torch.int64, device=images.device),
#     #                             "scores": torch.zeros((0,), device=images.device)})
#     #             else:
#     #                 boxes = det[:, :4]          # xyxy
#     #                 scores = det[:, 4]
#     #                 labels = det[:, 5].long()
#     #                 out.append({"boxes": boxes, "labels": labels, "scores": scores})
#     #         return out

#     #     # ART → YOLOv5：list[dict](xyxy)  ->  Tensor(N,6)  xywh
#     #     n, _, H, W = images.shape
#     #     yolo_tgts = []
#     #     for i, tgt in enumerate(targets):
#     #         if tgt["boxes"] is None or len(tgt["boxes"]) == 0:
#     #             continue
#     #         b = torch.as_tensor(tgt["boxes"], device=images.device, dtype=torch.float32)  # (K,4) xyxy
#     #         c = torch.as_tensor(tgt["labels"], device=images.device, dtype=torch.int64)   # (K,)
#     #         # xyxy -> xywh (pixel)
#     #         xy = (b[:, 0:2] + b[:, 2:4]) * 0.5
#     #         wh = (b[:, 2:4] - b[:, 0:2]).clamp(min=1e-6)
#     #         # -> [0,1]
#     #         xy[:, 0] /= W; xy[:, 1] /= H
#     #         wh[:, 0] /= W; wh[:, 1] /= H
#     #         # -> [img_idx, cls, x, y, w, h]
#     #         img_idx = torch.full((b.size(0), 1), i, device=images.device, dtype=torch.float32)
#     #         cls = c.to(dtype=torch.float32).unsqueeze(1)
#     #         yolo_tgts.append(torch.cat([img_idx, cls, xy, wh], dim=1))

#     #     if len(yolo_tgts) == 0:
#     #         yolo_targets = images.new_zeros((0, 6))
#     #     else:
#     #         yolo_targets = torch.cat(yolo_tgts, dim=0)

#     #     # compute loss
#     #     total_loss, _ = self._loss_fn(preds, yolo_targets)
#     #     # only return total_loss with gradient (loss_items are detached, cannot be used for backprop)
#     #     return {"loss": total_loss}

#     @torch.enable_grad()
#     def forward(self, images: torch.Tensor, targets=None):
#         """
#         images: (N,C,H,W) float in [0,1]
#         targets: list[dict] with keys 'boxes'(xyxy, pixel), 'labels' (int64)
#         """
#         from ultralytics.utils import LOGGER
        
#         preds = self.model(images, augment=False)

#         if targets is None:
#             # inference mode
#             dets = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.6, max_det=300)
#             out = []
#             for det in dets:
#                 if det is None or len(det) == 0:
#                     out.append({
#                         "boxes": torch.zeros((0, 4), device=images.device),
#                         "labels": torch.zeros((0,), dtype=torch.int64, device=images.device),
#                         "scores": torch.zeros((0,), device=images.device)
#                     })
#                 else:
#                     boxes = det[:, :4]
#                     scores = det[:, 4]
#                     labels = det[:, 5].long()
#                     out.append({"boxes": boxes, "labels": labels, "scores": scores})
#             return out

#         # training mode: ART format -> YOLO format
#         n, _, H, W = images.shape
#         yolo_tgts = []
        
#         for i, tgt in enumerate(targets):
#             boxes = tgt.get("boxes")
#             labels = tgt.get("labels")
            
#             if boxes is None or labels is None:
#                 continue
            
#             b = torch.as_tensor(boxes, device=images.device, dtype=torch.float32)
#             c = torch.as_tensor(labels, device=images.device, dtype=torch.int64)
            
#             if b.numel() == 0 or c.numel() == 0 or b.size(0) == 0:
#                 continue
            
#             if b.dim() == 1:
#                 b = b.unsqueeze(0)
#             if c.dim() == 0:
#                 c = c.unsqueeze(0)
            
#             # check boxes columns
#             if b.size(-1) != 4:
#                 LOGGER.warning(f"Invalid boxes shape: {b.shape}, expected (N, 4)")
#                 continue
            
#             xy = (b[:, 0:2] + b[:, 2:4]) * 0.5
#             wh = (b[:, 2:4] - b[:, 0:2]).clamp(min=1e-6)
            
#             xy[:, 0] /= W; xy[:, 1] /= H
#             wh[:, 0] /= W; wh[:, 1] /= H
            
#             # check normalized values
#             if torch.any(torch.isnan(xy)) or torch.any(torch.isnan(wh)):
#                 LOGGER.warning(f"NaN detected in normalized targets, skipping")
#                 continue
            
#             img_idx = torch.full((b.size(0), 1), i, device=images.device, dtype=torch.float32)
#             cls = c.to(dtype=torch.float32).unsqueeze(1)
            
#             yolo_tgts.append(torch.cat([img_idx, cls, xy, wh], dim=1))

#         # handle empty targets
#         if len(yolo_tgts) == 0:
#             # critical: return a small dummy loss to avoid calling _loss_fn
#             LOGGER.debug(f"No valid targets in batch, returning dummy loss")
#             return {"loss": torch.tensor(0.0, device=images.device, requires_grad=True)}
        
#         yolo_targets = torch.cat(yolo_tgts, dim=0)
        
#         # check yolo_targets shape
#         if yolo_targets.size(0) == 0 or yolo_targets.size(1) != 6:
#             LOGGER.warning(f"Invalid yolo_targets shape: {yolo_targets.shape}")
#             return {"loss": torch.tensor(0.0, device=images.device, requires_grad=True)}
        
#         # compute loss (only when there are valid targets)
#         try:
#             total_loss, _ = self._loss_fn(preds, yolo_targets)
#             return {"loss": total_loss}
#         except Exception as e:
#             LOGGER.warning(f"Loss computation failed: {e}, returning dummy loss")
#             return {"loss": torch.tensor(0.0, device=images.device, requires_grad=True)}

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
        # for p in self.model.parameters():
        #     p.requires_grad = True
            
        self.current_paths = []

    @torch.enable_grad()
    def forward(self, images: torch.Tensor, targets=None):
        """
        images: (N,C,H,W) float in [0,1]
        targets: list[dict] with keys 'boxes'(xyxy, pixel), 'labels' (int64)
        """
        from ultralytics.utils import LOGGER
        
        # inference mode
        if targets is None:
            with torch.no_grad():
                preds = self.model(images, augment=False)
            dets = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.6, max_det=300)
            out = []
            for det in dets:
                if det is None or len(det) == 0:
                    out.append({
                        "boxes": torch.zeros((0, 4), device=images.device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=images.device),
                        "scores": torch.zeros((0,), device=images.device)
                    })
                else:
                    boxes = det[:, :4]
                    scores = det[:, 4]
                    labels = det[:, 5].long()
                    out.append({"boxes": boxes, "labels": labels, "scores": scores})
            return out

        # training mode: convert targets
        n, _, H, W = images.shape
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
            
            xy[:, 0] /= W; xy[:, 1] /= H
            wh[:, 0] /= W; wh[:, 1] /= H
            
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
        try:
            # save original model state
            original_training = self.model.training
            
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

                # # DEBUG
                # LOGGER.info(f"DEBUG DIAGNOSTICS:")
                # LOGGER.info(f"  Model training mode: {self.model.training}")
                # try:
                #     p = next(self.model.parameters())
                #     LOGGER.info(f"  Model param requires_grad: {p.requires_grad}")
                #     LOGGER.info(f"  Model param device: {p.device}")
                # except Exception:
                #     LOGGER.info("  Could not check model params")
                
                # LOGGER.info(f"  Input images: shape={images.shape}, req_grad={images.requires_grad}, is_leaf={images.is_leaf}")
                # LOGGER.info(f"  Grad enabled: {torch.is_grad_enabled()}")
                
                # set train mode to get feature maps (for loss computation)
                # but force BN layers to eval mode (use pretrained statistics)
                self.model.train()
                for m in self.model.modules():
                    if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                        m.eval()
                
                # LOGGER.info(f"Model training mode after set: {self.model.training}")

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
                loss, loss_items = self._loss_fn(batch, preds)
                
                # Ensure loss is scalar for backward()
                final_loss = loss.sum()

                # Verify graph integrity
                if not final_loss.requires_grad:
                    LOGGER.warning(f"Critical: Loss does not require grad! Input req_grad={images.requires_grad}")
                    # Debug preds
                    if isinstance(preds, (list, tuple)) and len(preds) > 0:
                        LOGGER.warning(f"DEBUG: preds[0] req_grad={preds[0].requires_grad}")
                    # Construct a valid dummy loss connected to input
                    final_loss = (images * 0).sum() 

            self.model.train(original_training)
            
            if images.requires_grad and not images.is_leaf:
                 images.retain_grad()

            return {"loss": final_loss}
            
        except Exception as e:
            LOGGER.warning(f"Loss computation failed: {e}")
            
            self.model.train(original_training)
            
            import traceback
            LOGGER.debug(traceback.format_exc())
            
            # Return a safe dummy loss
            dummy = torch.tensor(0.0, device=images.device, requires_grad=True)
            if images.requires_grad:
                 dummy = dummy + (images.mean() * 0.0)
            
            return {"loss": dummy}

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
    def __init__(self, model, config=None, target=None,
                 epsilon=0.05, lr=0.005, epoch=20, img_size=640):
        super().__init__(model, config, epsilon)
        self.device = next(model.parameters()).device

        # might not be necessary as batchnorm/dropout will be disabled in ART wrapper later
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        # wrap yolo model for ART
        wrapped = YoloV5ForART(self.model, img_size)

        # ART Detector
        self.estimator = ARTDetector(
            model=wrapped,
            input_shape=(3, img_size, img_size),
            clip_values=(0.0, 1.0),
            channels_first=True,
            device_type="gpu",
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
            num_random_init=0,
            verbose=False,
        )

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        x: (N,C,H,W) in [0,1] float
        targets: (M,6) [img_idx, cls, x, y, w, h] 归一化 xywh
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
    #         # xywh -> xyxy
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
        
        # input validation
        if targets is None or targets.numel() == 0:
            for _ in range(batch_size):
                out.append({
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "labels": np.zeros((0,), dtype=np.int64)
                })
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
                out.append({
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "labels": np.zeros((0,), dtype=np.int64)
                })
                continue
            
            # ensure ti is 2D
            if ti.dim() == 1:
                ti = ti.unsqueeze(0)
            
            # xywh (normalized) -> xyxy (pixel)
            xywh = ti[:, 2:6].clone()
            xywh[:, 0] *= W; xywh[:, 1] *= H
            xywh[:, 2] *= W; xywh[:, 3] *= H
            
            xyxy = torch.zeros_like(xywh)
            xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
            xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
            xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
            xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
            
            # clip to image boundary
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, W)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, H)
            
            labels = ti[:, 1].to(torch.int64)
            
            out.append({
                "boxes": xyxy.numpy().astype(np.float32),
                "labels": labels.numpy().astype(np.int64),
            })
        
        return out