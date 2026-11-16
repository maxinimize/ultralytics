from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, IterableSimpleNamespace


def setup_attack_model(attack_weights, device, nc, training=False, imgsz=640):
    """
    Set up the attack model based on the provided parameters.

    Args:
        attack_weights (str | Path | torch.nn.Module): Path to the attack model weights or existing module.
        device (torch.device): Device.
        nc (int): Number of classes.
        training (bool): Whether in training mode.
        imgsz (int): Input image size.

    Returns:
        attack_model: Configured attack model.
    """
    if not attack_weights:
        raise ValueError("attack_weights must be provided for setup_attack_model()")

    if isinstance(attack_weights, torch.nn.Module):
        model = attack_weights.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    if isinstance(attack_weights, (str, Path)) and not Path(attack_weights).exists():
        raise FileNotFoundError(f"Attack weights file '{attack_weights}' not found.")

    # load checkpoint
    ckpt = torch.load(str(attack_weights), map_location="cpu", weights_only=False)

    # model configuration
    cfg = ckpt.get("model").yaml if hasattr(ckpt.get("model", {}), "yaml") else None

    # create model
    if cfg:
        attack_model = DetectionModel(cfg, ch=3, nc=nc).to(device)
    else:
        # reconstruct model from checkpoint
        attack_model = ckpt["model"].to(device)
        if hasattr(attack_model, "nc"):
            attack_model.nc = nc

    # load weights
    if "model" in ckpt:
        state_dict = ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
        attack_model.load_state_dict(state_dict, strict=False)
        LOGGER.info(f"Loaded attack model from {attack_weights}")

    # set hyperparameters
    attack_model.args = deepcopy(DEFAULT_CFG_DICT)
    attack_model.args["imgsz"] = imgsz
    if "hyp" in ckpt:
        for k, v in ckpt["hyp"].items():
            if k in attack_model.args:
                attack_model.args[k] = v

    # convert to IterableSimpleNamespace for compatibility with loss functions
    attack_model.args = IterableSimpleNamespace(**attack_model.args)

    # set model to eval mode if not training
    if not training:
        attack_model.eval()
    for p in attack_model.parameters():
        p.requires_grad_(False)
    if hasattr(attack_model, "init_criterion") and getattr(attack_model, "criterion", None) is None:
        try:
            attack_model.criterion = attack_model.init_criterion()
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            LOGGER.warning(f"Could not pre-initialize criterion on surrogate: {e}")

    return attack_model


class BatchContainer:
    """Small dict/object hybrid matching the batch interface used by Ultralytics detection losses."""

    def __init__(self, **kwargs):
        self._data = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value
        setattr(self, key, value)

    def __delitem__(self, key):
        del self._data[key]
        if hasattr(self, key):
            delattr(self, key)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def copy(self):
        return BatchContainer(**self._data.copy())

    def __copy__(self):
        return BatchContainer(**self._data.copy())


def extract_detection_model(model: Any) -> torch.nn.Module:
    """Extract full DetectionModel from AutoBackend or training model without stripping to nn.Sequential.

    Args:
        model (Any): An AutoBackend instance, DetectionModel, or nn.Module.

    Returns:
        (torch.nn.Module): The unwrapped complete detection model.

    Raises:
        ValueError: If model is an AutoBackend with an unsupported non-PyTorch backend.
        TypeError: If the extracted model is nn.Sequential or missing loss/forward capabilities.
    """
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.utils.torch_utils import unwrap_model

    if isinstance(model, AutoBackend):
        fmt = getattr(model, "format", None)
        if fmt != "pt":
            raise ValueError(
                f"Adversarial surrogate extraction only supports native PyTorch models ('pt'), "
                f"but received AutoBackend format='{fmt}'."
            )
        backend = getattr(model, "backend", None)
        if backend is not None and hasattr(backend, "model"):
            detector = unwrap_model(backend.model)
        else:
            detector = unwrap_model(getattr(model, "model", model))
    else:
        detector = unwrap_model(model)

    if isinstance(detector, torch.nn.Sequential):
        raise TypeError("Extracted detector is an nn.Sequential instance, not a complete DetectionModel.")

    if not hasattr(detector, "loss") or not callable(getattr(detector, "loss", None)):
        raise TypeError(
            f"Extracted detector of type '{type(detector).__name__}' does not have a callable 'loss' method."
        )

    return detector
