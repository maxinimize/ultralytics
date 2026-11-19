from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.attacks.attack_utils import (
    capture_module_training_states,
    restore_module_training_states,
)

try:
    from art.estimators.classification import PyTorchClassifier
except (ImportError, AttributeError):
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


def build_single_labels_from_batch(
    batch: dict, policy: str = "largest_box"
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
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


def slice_batch_indices(batch: dict, indices: list[int]) -> dict:
    """Slice a batch dictionary for the specified sample indices."""
    sub_batch = {}
    n_imgs = batch["img"].shape[0] if "img" in batch and torch.is_tensor(batch["img"]) else None

    # Target group definitions: (batch_idx_key, associated_keys)
    target_groups = [
        ("batch_idx", ["cls", "bboxes", "masks", "keypoints", "segments", "obb"]),
        ("pre_attack_batch_idx", ["pre_attack_cls", "pre_attack_bboxes"]),
        ("clean_batch_idx", ["clean_cls", "clean_bboxes"]),
    ]
    all_target_keys = set()
    for bi_k, f_keys in target_groups:
        all_target_keys.add(bi_k)
        all_target_keys.update(f_keys)

    for k, v in batch.items():
        if k in all_target_keys:
            continue
        elif k == "img":
            sub_batch["img"] = v[indices]
        elif k in {"im_file", "ori_shape", "ratio_pad", "sample_type", "attack_name"}:
            if isinstance(v, (list, tuple)):
                sub_batch[k] = [v[i] for i in indices]
            elif isinstance(v, torch.Tensor):
                sub_batch[k] = v[indices]
            else:
                sub_batch[k] = v
        elif k in {
            "visuals",
            "sem_masks",
            "text_feats",
            "depth",
            "semantic_mask",
            "gt_matrix",
            "attack_matrices",
            "gt_params",
        }:
            if isinstance(v, torch.Tensor):
                sub_batch[k] = v[indices]
            elif isinstance(v, (list, tuple)):
                sub_batch[k] = [v[i] for i in indices]
            else:
                sub_batch[k] = v
        elif isinstance(v, torch.Tensor) and n_imgs is not None and v.shape[0] == n_imgs:
            sub_batch[k] = v[indices]
        elif isinstance(v, (list, tuple)) and n_imgs is not None and len(v) == n_imgs:
            sub_batch[k] = [v[i] for i in indices]
        else:
            sub_batch[k] = v

    # Slice each target group independently using its own batch_idx tensor
    for bi_k, f_keys in target_groups:
        if bi_k in batch and batch[bi_k] is not None:
            batch_idx = batch[bi_k]
            b_idx_1d = batch_idx.squeeze(-1) if batch_idx.dim() > 1 else batch_idx
            start_idx = indices[0]
            end_idx = indices[-1] + 1
            if indices == list(range(start_idx, end_idx)):
                mask = (b_idx_1d >= start_idx) & (b_idx_1d < end_idx)
                new_b_idx = (b_idx_1d[mask] - start_idx).to(batch_idx.dtype)
            else:
                mask = torch.zeros(b_idx_1d.shape, dtype=torch.bool, device=batch_idx.device)
                for idx in indices:
                    mask |= b_idx_1d == idx
                idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
                new_b_idx = torch.tensor(
                    [idx_map[int(x)] for x in b_idx_1d[mask].tolist()],
                    dtype=batch_idx.dtype,
                    device=batch_idx.device,
                )
            if batch_idx.dim() > 1:
                new_b_idx = new_b_idx.unsqueeze(-1)
            sub_batch[bi_k] = new_b_idx

            for box_k in f_keys:
                if box_k in batch and batch[box_k] is not None:
                    sub_batch[box_k] = batch[box_k][mask]

    return sub_batch


def validate_batch_annotations(batch: dict, batch_size: int, classes: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Validate batch annotations and return boolean mask of shape [batch_size] indicating which images have targets."""
    batch_idx = batch.get("batch_idx")
    cls = batch.get("cls")
    bboxes = batch.get("bboxes")

    if batch_idx is None or cls is None or bboxes is None:
        raise ValueError(
            f"Batch annotations corrupted: missing batch_idx, cls, or bboxes in batch keys {list(batch.keys())}"
        )
    if not (torch.is_tensor(batch_idx) and torch.is_tensor(cls) and torch.is_tensor(bboxes)):
        raise ValueError("Batch annotations corrupted: batch_idx, cls, and bboxes must be torch.Tensor")

    bi = batch_idx.squeeze(-1) if batch_idx.ndim > 1 else batch_idx
    c = cls.squeeze(-1) if cls.ndim > 1 else cls
    b = bboxes

    if not (bi.ndim == 1 and c.ndim == 1 and b.ndim == 2 and b.shape[-1] == 4):
        raise ValueError(
            f"Batch annotations corrupted: shape mismatch bi={tuple(bi.shape)}, c={tuple(c.shape)}, b={tuple(b.shape)}"
        )
    if not (bi.shape[0] == c.shape[0] == b.shape[0]):
        raise ValueError(
            f"Batch annotations corrupted: length mismatch bi={bi.shape[0]}, c={c.shape[0]}, b={b.shape[0]}"
        )
    if not (torch.isfinite(bi).all() and torch.isfinite(c).all() and torch.isfinite(b).all()):
        raise ValueError("Batch annotations corrupted: non-finite values found in annotations")
    if bi.numel() > 0:
        min_bi = int(bi.min().item())
        max_bi = int(bi.max().item())
        if min_bi < 0 or max_bi >= batch_size:
            raise ValueError(
                f"Batch annotations corrupted: batch_idx out of bounds [0, {batch_size - 1}], got [{min_bi}, {max_bi}]"
            )

    has_targets = torch.zeros((batch_size,), dtype=torch.bool, device=bi.device)
    if bi.numel() > 0:
        if classes is not None:
            cls_set = torch.tensor(list(classes), device=c.device)
            valid_cls_mask = (c.unsqueeze(-1) == cls_set).any(dim=-1)
        else:
            valid_cls_mask = torch.ones_like(c, dtype=torch.bool)

        valid_bi = bi[valid_cls_mask]
        if valid_bi.numel() > 0:
            has_targets.scatter_(0, valid_bi.long(), True)

    return has_targets


def _check_attack_output(adv: Any, batch: dict, attack_name: str, batch_i: int = -1, sub_idx: int = 0) -> None:
    """Strict validation of attack generation output on sub-batches with valid targets."""
    im_files = batch.get("im_file", [])
    has_targets = False
    if (
        "cls" in batch
        and batch["cls"] is not None
        and batch["cls"].numel() > 0
        or "bboxes" in batch
        and batch["bboxes"] is not None
        and batch["bboxes"].numel() > 0
    ):
        has_targets = True

    if not has_targets:
        return

    if adv is None:
        raise RuntimeError(
            f"Attack '{attack_name}' failed at batch {batch_i}, sub-batch {sub_idx}: "
            f"Attacker returned None for batch with targets. Relevant image paths: {im_files}"
        )
    if not torch.is_tensor(adv):
        raise RuntimeError(
            f"Attack '{attack_name}' failed at batch {batch_i}, sub-batch {sub_idx}: "
            f"Attacker returned non-tensor output of type {type(adv)}. Relevant image paths: {im_files}"
        )
    if not adv.is_floating_point():
        raise RuntimeError(
            f"Attack '{attack_name}' failed at batch {batch_i}, sub-batch {sub_idx}: "
            f"Attacker returned non-floating point tensor with dtype {adv.dtype}. Relevant image paths: {im_files}"
        )
    expected_shape = batch["img"].shape
    if adv.shape != expected_shape:
        raise RuntimeError(
            f"Attack '{attack_name}' failed at batch {batch_i}, sub-batch {sub_idx}: "
            f"Attacker output shape {tuple(adv.shape)} does not match expected shape {tuple(expected_shape)}. "
            f"Relevant image paths: {im_files}"
        )
    if not torch.isfinite(adv).all():
        raise RuntimeError(
            f"Attack '{attack_name}' failed at batch {batch_i}, sub-batch {sub_idx}: "
            f"Attacker output contains NaN or Inf values. Relevant image paths: {im_files}"
        )
    min_val = adv.amin().item()
    max_val = adv.amax().item()
    if min_val < -1e-6 or max_val > 1.0 + 1e-6:
        raise RuntimeError(
            f"Attack '{attack_name}' failed at batch {batch_i}, sub-batch {sub_idx}: "
            f"Attacker output pixels out of [0, 1] bounds (min: {min_val}, max: {max_val}, tolerance 1e-6). "
            f"Relevant image paths: {im_files}"
        )


def run_attack_on_batch(
    attacker,
    batch: dict,
    *,
    label_policy: str = "largest_box",
    max_batch_size: Optional[int] = None,
    strict: bool = False,
    attack_name: str = "attack",
    batch_i: int = -1,
    sub_idx: int = 0,
    **kwargs: Any,
) -> Optional[torch.Tensor]:
    """
    Apply an attacker to a training/validation batch.

    Batch-aware spatial attacks are handled first because they must update both
    images and detection targets. Existing detector/classifier attacks keep the
    original image-only return path unchanged.
    """
    target_model = getattr(attacker, "model", None)
    saved_states = capture_module_training_states(target_model)
    try:
        imgs = batch["img"]
        bs = imgs.shape[0]

        limit = max_batch_size if max_batch_size is not None else getattr(attacker, "max_batch_size", None)
        if limit is not None and limit > 0 and bs > limit:
            adv_imgs = imgs.clone()
            is_batch_aware = callable(getattr(attacker, "forward_batch", None)) and bool(
                getattr(attacker, "batch_aware", False)
            )
            possible_target_keys = [
                "batch_idx",
                "cls",
                "bboxes",
                "masks",
                "keypoints",
                "segments",
                "obb",
                "clean_batch_idx",
                "clean_cls",
                "clean_bboxes",
                "pre_attack_batch_idx",
                "pre_attack_cls",
                "pre_attack_bboxes",
            ]
            collected_targets = {k: [] for k in possible_target_keys}
            seen_target_keys = {k for k in possible_target_keys if k in batch and batch[k] is not None}
            collected_gt_matrices = []
            collected_gt_params = []
            collected_attack_matrices = []

            for start_idx in range(0, bs, limit):
                end_idx = min(start_idx + limit, bs)
                indices = list(range(start_idx, end_idx))
                sub_batch = slice_batch_indices(batch, indices)
                im_files = sub_batch.get("im_file", [])
                if hasattr(attacker, "proxy_model"):
                    attacker.proxy_model.current_paths = im_files
                if hasattr(attacker, "estimator") and hasattr(attacker.estimator, "model"):
                    attacker.estimator.model.current_paths = im_files
                try:
                    adv_sub = run_attack_on_batch(
                        attacker,
                        sub_batch,
                        label_policy=label_policy,
                        max_batch_size=None,
                        strict=strict,
                        attack_name=attack_name,
                        batch_i=batch_i,
                        sub_idx=start_idx,
                    )
                except Exception:
                    if strict:
                        raise
                    adv_sub = None

                if strict:
                    _check_attack_output(
                        adv_sub, sub_batch, attack_name=attack_name, batch_i=batch_i, sub_idx=start_idx
                    )

                if adv_sub is not None:
                    adv_imgs[start_idx:end_idx] = adv_sub.to(adv_imgs.dtype)

                if "gt_matrix" in sub_batch and sub_batch["gt_matrix"] is not None:
                    collected_gt_matrices.append(sub_batch["gt_matrix"])
                elif hasattr(attacker, "last_selected_matrices") and attacker.last_selected_matrices is not None:
                    collected_gt_matrices.append(attacker.last_selected_matrices)
                if "gt_params" in sub_batch and sub_batch["gt_params"] is not None:
                    collected_gt_params.append(sub_batch["gt_params"])
                elif hasattr(attacker, "last_selected_params") and attacker.last_selected_params is not None:
                    collected_gt_params.append(attacker.last_selected_params)
                if "attack_matrices" in sub_batch and sub_batch["attack_matrices"] is not None:
                    collected_attack_matrices.append(sub_batch["attack_matrices"])

                if is_batch_aware:
                    for bi_k in ["batch_idx", "clean_batch_idx", "pre_attack_batch_idx"]:
                        if bi_k in sub_batch and sub_batch[bi_k] is not None:
                            seen_target_keys.add(bi_k)
                            sub_bi = sub_batch[bi_k]
                            if sub_bi.numel() > 0:
                                collected_targets[bi_k].append(sub_bi + start_idx)
                    for k in possible_target_keys:
                        if (
                            k not in {"batch_idx", "clean_batch_idx", "pre_attack_batch_idx"}
                            and k in sub_batch
                            and sub_batch[k] is not None
                        ):
                            seen_target_keys.add(k)
                            collected_targets[k].append(sub_batch[k])

                del sub_batch, adv_sub
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if is_batch_aware:
                for k in seen_target_keys:
                    if collected_targets[k]:
                        batch[k] = torch.cat(collected_targets[k], dim=0)
                    elif k in batch:
                        batch[k] = batch[k][:0]
                if collected_gt_matrices:
                    all_matrices = torch.cat(collected_gt_matrices, dim=0)
                    batch["gt_matrix"] = all_matrices
                    if hasattr(attacker, "last_selected_matrices"):
                        attacker.last_selected_matrices = all_matrices
                if collected_gt_params:
                    all_params = torch.cat(collected_gt_params, dim=0)
                    batch["gt_params"] = all_params
                    if hasattr(attacker, "last_selected_params"):
                        attacker.last_selected_params = all_params
                if collected_attack_matrices:
                    batch["attack_matrices"] = torch.cat(collected_attack_matrices, dim=0)

            return adv_imgs

        # Spatial attacks such as Worst-of-k must transform GT boxes together with
        # the image. forward_batch() updates batch_idx/cls/bboxes in-place and still
        # returns an image tensor, preserving the existing caller contract.
        forward_batch = getattr(attacker, "forward_batch", None)
        if callable(forward_batch) and bool(getattr(attacker, "batch_aware", False)):
            try:
                adv = forward_batch(batch)
            except Exception:
                if strict:
                    raise
                return None
            if strict:
                _check_attack_output(adv, batch, attack_name=attack_name, batch_i=batch_i, sub_idx=sub_idx)
            return adv.to(imgs.dtype)

        mode = infer_attack_mode(attacker)

        if mode == "detector":
            targets = build_detector_targets(batch)
            if targets is None or targets.numel() == 0:
                return None
            try:
                adv = attacker.forward(imgs.float(), targets)
            except Exception:
                if strict:
                    raise
                return None
            del targets
            if strict:
                _check_attack_output(adv, batch, attack_name=attack_name, batch_i=batch_i, sub_idx=sub_idx)
            res_adv = adv.to(imgs.dtype)
            del adv
            return res_adv

        labels, attacked_idx = build_single_labels_from_batch(batch, policy=label_policy)
        if labels is None or attacked_idx is None or attacked_idx.numel() == 0:
            return None

        try:
            adv_subset = attacker.forward(imgs[attacked_idx].float(), labels)
        except Exception:
            if strict:
                raise
            return None

        if strict:
            sub_b = {"img": imgs[attacked_idx], "cls": labels}
            if batch.get("im_file"):
                sub_b["im_file"] = [batch["im_file"][i] for i in attacked_idx.tolist()]
            _check_attack_output(adv_subset, sub_b, attack_name=attack_name, batch_i=batch_i, sub_idx=sub_idx)

        adv_imgs = imgs.clone()
        adv_imgs[attacked_idx] = adv_subset.to(imgs.dtype)
        del adv_subset, labels, attacked_idx
        return adv_imgs
        del adv_subset, labels, attacked_idx
        return adv_imgs
    finally:
        restore_module_training_states(saved_states)


def build_attacker(name: str, model: nn.Module, **kwargs):
    """
    Central factory for attack creation.

    Using one factory avoids training and validation silently using different attacks.
    """
    saved_states = capture_module_training_states(model)
    try:
        normalized = name.lower().strip()
        seed = kwargs.pop("seed", None)

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
        if normalized in {"worstk", "worst_of_k", "spatial_worst_of_k"}:
            from ultralytics.attacks.spatial_worst_of_k import SpatialWorstOfK

            return SpatialWorstOfK(model=model, seed=seed, **kwargs)

        raise ValueError(f"Unsupported attack name: {name}")
    finally:
        restore_module_training_states(saved_states)
