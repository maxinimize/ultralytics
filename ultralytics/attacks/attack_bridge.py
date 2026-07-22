from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from art.estimators.classification import PyTorchClassifier
except Exception:
    from art.estimators.classification.pytorch import PyTorchClassifier


def infer_num_classes(model: nn.Module, fallback: int = 80) -> int:
    """Infer number of classes from a YOLO model."""
    nc = getattr(model, "nc", None)
    if isinstance(nc, int) and nc > 0:
        return nc

    names = getattr(model, "names", None)
    if isinstance(names, (list, tuple)) and len(names):
        return len(names)
    if isinstance(names, dict) and len(names):
        return len(names)

    unwrapped = model
    while hasattr(unwrapped, "model"):
        candidate = getattr(unwrapped, "model", None)
        if candidate is None or candidate is unwrapped:
            break
        unwrapped = candidate
        nc = getattr(unwrapped, "nc", None)
        if isinstance(nc, int) and nc > 0:
            return nc
        names = getattr(unwrapped, "names", None)
        if isinstance(names, (list, tuple)) and len(names):
            return len(names)
        if isinstance(names, dict) and len(names):
            return len(names)

    return fallback


def _find_prediction_tensor(outputs):
    """Recursively find a tensor prediction output."""
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, dict):
        for value in outputs.values():
            found = _find_prediction_tensor(value)
            if found is not None:
                return found
    if isinstance(outputs, (list, tuple)):
        for value in outputs:
            found = _find_prediction_tensor(value)
            if found is not None:
                return found
    return None


class SingleLabelProxyLoss(nn.Module):
    """Cross-entropy over image-level class scores."""

    def forward(self, outputs: torch.Tensor, targets) -> torch.Tensor:
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).to(outputs.device)
        elif not torch.is_tensor(targets):
            targets = torch.as_tensor(targets, device=outputs.device)

        if targets.ndim > 1:
            targets = targets.argmax(dim=1)

        return F.cross_entropy(outputs.float(), targets.long())


class YoloSingleLabelClassifierProxy(nn.Module):
    """
    Wrap a YOLO detector as an image-level single-label classifier proxy.

    The proxy aggregates dense detector class scores into one class-score vector per image
    by taking the maximum class score over all candidate boxes.
    """

    def __init__(self, yolo_model: nn.Module, img_size: int, num_classes: Optional[int] = None):
        super().__init__()
        self.model = yolo_model
        self.img_size = img_size
        self.nc = int(num_classes or infer_num_classes(yolo_model))
        self.current_paths = []

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

    def _normalize_prediction_shape(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim == 2:
            return raw
        if raw.ndim != 3:
            raise ValueError(f"Expected a 2D/3D tensor from YOLO proxy, got shape {tuple(raw.shape)}")

        # Prefer (B, num_boxes, channels)
        if raw.shape[-1] in {self.nc, self.nc + 4, self.nc + 5}:
            return raw

        # Common YOLO layout: (B, channels, num_boxes)
        if raw.shape[1] in {self.nc, self.nc + 4, self.nc + 5}:
            return raw.transpose(1, 2)

        # Fallback heuristic: smaller dim is usually channels
        if raw.shape[1] < raw.shape[2]:
            return raw.transpose(1, 2)

        return raw

    def _aggregate_class_scores(self, raw: torch.Tensor) -> torch.Tensor:
        raw = raw.float()

        if raw.ndim == 2:
            if raw.shape[1] < self.nc:
                raise ValueError(f"2D proxy output has too few channels for nc={self.nc}: {tuple(raw.shape)}")
            return raw[:, : self.nc]

        raw = self._normalize_prediction_shape(raw)
        channels = raw.shape[-1]

        if channels >= self.nc + 5:
            obj = raw[..., 4:5]
            cls_scores = raw[..., 5 : 5 + self.nc]
            cls_scores = obj * cls_scores
        elif channels >= self.nc + 4:
            cls_scores = raw[..., 4 : 4 + self.nc]
        elif channels >= self.nc:
            cls_scores = raw[..., : self.nc]
        else:
            raise ValueError(f"Cannot infer class channels from proxy output shape {tuple(raw.shape)}")

        return cls_scores.amax(dim=1)

    @torch.enable_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        preds = self.model(images, augment=False)
        raw = _find_prediction_tensor(preds)
        if raw is None:
            raise TypeError(f"Could not find a tensor prediction inside model output of type {type(preds)}")
        return self._aggregate_class_scores(raw)


def build_classifier_estimator(model: nn.Module, img_size: int, num_classes: Optional[int] = None):
    """Create a PyTorchClassifier estimator around the YOLO single-label proxy."""
    proxy = YoloSingleLabelClassifierProxy(model, img_size=img_size, num_classes=num_classes)
    device_type = "gpu" if next(model.parameters()).device.type != "cpu" else "cpu"

    estimator = PyTorchClassifier(
        model=proxy,
        loss=SingleLabelProxyLoss(),
        input_shape=(3, img_size, img_size),
        nb_classes=proxy.nc,
        clip_values=(0.0, 1.0),
        channels_first=True,
        device_type=device_type,
    )
    return proxy, estimator


def normalize_classifier_targets(targets, *, as_one_hot: bool = False, nb_classes: Optional[int] = None) -> np.ndarray:
    """Convert labels to numpy class indices or one-hot encodings."""
    if targets is None:
        raise ValueError("Classifier targets must not be None.")

    if torch.is_tensor(targets):
        arr = targets.detach().cpu().numpy()
    else:
        arr = np.asarray(targets)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    elif arr.ndim > 2:
        raise ValueError(f"Unsupported classifier target shape: {arr.shape}")

    if arr.ndim == 2:
        labels = arr.argmax(axis=1).astype(np.int64)
    else:
        labels = arr.astype(np.int64).reshape(-1)

    if not as_one_hot:
        return labels

    if nb_classes is None:
        raise ValueError("nb_classes is required when as_one_hot=True")

    one_hot = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def infer_attack_mode(attacker) -> str:
    """Return 'detector' or 'classifier'."""
    mode = getattr(attacker, "attack_mode", None)
    if mode in {"detector", "classifier"}:
        return mode

    estimator = getattr(attacker, "estimator", None)
    if estimator is not None:
        name = estimator.__class__.__name__.lower()
        if "objectdetector" in name or "yolo" in name:
            return "detector"
        if "classifier" in name:
            return "classifier"

    return "detector"


def build_detector_targets(batch: dict) -> Optional[torch.Tensor]:
    """Build YOLO detector targets of shape (N, 6) from a training/validation batch."""
    batch_idx = batch.get("batch_idx")
    cls = batch.get("cls")
    bboxes = batch.get("bboxes")

    if batch_idx is None or cls is None or bboxes is None:
        return None

    if batch_idx.dim() == 1:
        batch_idx = batch_idx.unsqueeze(1)
    elif batch_idx.dim() > 2:
        batch_idx = batch_idx.squeeze()
        if batch_idx.dim() == 1:
            batch_idx = batch_idx.unsqueeze(1)

    if cls.dim() == 1:
        cls = cls.unsqueeze(1)
    elif cls.dim() > 2:
        cls = cls.squeeze()
        if cls.dim() == 1:
            cls = cls.unsqueeze(1)

    if bboxes.dim() == 3:
        bboxes = bboxes.squeeze(1)

    if batch_idx.size(0) != cls.size(0) or batch_idx.size(0) != bboxes.size(0):
        return None

    return torch.cat([batch_idx, cls, bboxes], dim=1)


def build_single_labels_from_batch(batch: dict, policy: str = "largest_box") -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Convert a detection batch into one single-label class target per image.

    Returns:
        labels: Tensor[K] with one class id per attacked image.
        attacked_idx: Tensor[K] with the image indices that received labels.
    """
    imgs = batch.get("img")
    batch_idx = batch.get("batch_idx")
    cls = batch.get("cls")
    bboxes = batch.get("bboxes")

    if imgs is None or batch_idx is None or cls is None or bboxes is None:
        return None, None

    device = imgs.device
    bs = imgs.shape[0]

    if batch_idx.dim() > 1:
        batch_idx = batch_idx.squeeze(-1)
    if cls.dim() > 1:
        cls = cls.squeeze(-1)
    if bboxes.dim() == 3:
        bboxes = bboxes.squeeze(1)

    labels = []
    attacked_idx = []

    for i in range(bs):
        mask = batch_idx == i
        if mask.sum() == 0:
            continue

        cls_i = cls[mask].long()
        boxes_i = bboxes[mask]

        if policy == "largest_box":
            areas = boxes_i[:, 2] * boxes_i[:, 3]
            chosen = torch.argmax(areas)
        elif policy == "first":
            chosen = 0
        else:
            raise ValueError(f"Unknown single-label policy: {policy}")

        labels.append(cls_i[chosen])
        attacked_idx.append(torch.tensor(i, device=device))

    if not labels:
        return None, None

    return torch.stack(labels).to(device), torch.stack(attacked_idx).long().to(device)


def run_attack_on_batch(attacker, batch: dict, *, label_policy: str = "largest_box") -> Optional[torch.Tensor]:
    """
    Apply either a detector-style attacker or a classifier-style attacker to a batch.

    Detector attackers receive YOLO detector targets.
    Classifier attackers receive one class label per image, chosen via the single-label bridge.
    """
    imgs = batch["img"]
    mode = infer_attack_mode(attacker)

    if mode == "detector":
        targets = build_detector_targets(batch)
        if targets is None or targets.numel() == 0:
            return None
        adv = attacker.forward(imgs.float(), targets)
        return adv.to(imgs.dtype)

    labels, attacked_idx = build_single_labels_from_batch(batch, policy=label_policy)
    if labels is None or attacked_idx is None or attacked_idx.numel() == 0:
        return None

    adv_imgs = imgs.clone()
    adv_subset = attacker.forward(imgs[attacked_idx].float(), labels)
    adv_imgs[attacked_idx] = adv_subset.to(imgs.dtype)
    return adv_imgs


def build_attacker(name: str, model: nn.Module, **kwargs):
    """
    Central factory for attack creation.

    Using one factory avoids training and validation silently using different attacks.
    """
    normalized = name.lower().strip()

    if normalized in {"fgsm"}:
        from ultralytics.attacks.art_fgsm import ARTFGSM
        return ARTFGSM(model=model, **kwargs)
    if normalized in {"pgd"}:
        from ultralytics.attacks.art_pgd import ARTPGD
        return ARTPGD(model=model, **kwargs)
    if normalized in {"dpatch"}:
        from ultralytics.attacks.art_dpatch import ARTDPatch
        return ARTDPatch(model=model, **kwargs)
    if normalized in {"mim"}:
        from ultralytics.attacks.art_mim import ARTMIM
        return ARTMIM(model=model, **kwargs)
    if normalized in {"cw"}:
        from ultralytics.attacks.art_cw import ARTCW
        return ARTCW(model=model, **kwargs)
    if normalized in {"bim"}:
        from ultralytics.attacks.art_bim import ARTBIM
        return ARTBIM(model=model, **kwargs)
    if normalized in {"deepfool"}:
        from ultralytics.attacks.art_deepfool import ARTDeepFool
        return ARTDeepFool(model=model, **kwargs)
    if normalized in {"jsma"}:
        from ultralytics.attacks.art_jsma import ARTJSMA
        return ARTJSMA(model=model, **kwargs)
    if normalized in {"uap"}:
        from ultralytics.attacks.art_uap import ARTUAP
        return ARTUAP(model=model, **kwargs)
    if normalized in {"autoattack"}:
        from ultralytics.attacks.art_autoattack import ARTAutoAttack
        return ARTAutoAttack(model=model, **kwargs)

    raise ValueError(f"Unsupported attack name: {name}")
