from __future__ import annotations

import contextlib
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
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


def capture_module_training_states(model: torch.nn.Module | None) -> dict[torch.nn.Module, bool]:
    """Capture training mode of every submodule in model without recursive side effects.

    Args:
        model (Optional[torch.nn.Module]): Target module whose submodule training modes should be recorded.

    Returns:
        dict[torch.nn.Module, bool]: Mapping of each submodule to its boolean training state.
    """
    if model is None or not isinstance(model, torch.nn.Module):
        return {}
    states = {m: m.training for m in model.modules()}
    for attr in ("detector", "pipeline", "model"):
        sub = getattr(model, attr, None)
        if isinstance(sub, torch.nn.Module):
            for m in sub.modules():
                if m not in states:
                    states[m] = m.training
    return states


def restore_module_training_states(states: dict[torch.nn.Module, bool]) -> None:
    """Restore training mode of every submodule from snapshot without recursive train() side effects.

    Args:
        states (dict[torch.nn.Module, bool]): Mapping from submodules to their saved training states.
    """
    if not states:
        return
    for m, mode in states.items():
        m.training = mode


@contextlib.contextmanager
def preserve_module_training_states(*models: torch.nn.Module | None):
    """Context manager to preserve and restore submodule training states across operations or exceptions.

    Args:
        *models: One or more PyTorch modules to snapshot and restore.
    """
    snapshots = [capture_module_training_states(m) for m in models]
    try:
        yield
    finally:
        for snap in snapshots:
            restore_module_training_states(snap)


@contextlib.contextmanager
def isolate_rng(
    seed: int | None = None,
    *,
    attackers: Any = None,
    dataloader: Any = None,
):
    """Context manager to isolate RNG states during validation, evaluation, or attacks.

    Captures:
      - Python random state
      - NumPy random state
      - PyTorch CPU RNG state
      - PyTorch CUDA RNG states (all devices) if CUDA is available
      - Private generator state of attackers (e.g. SpatialWorstOfK)
      - DataLoader generator state (if present)

    Sets `seed` if provided (for Python, NumPy, PyTorch CPU, PyTorch CUDA, and attackers via reset_seed).
    Guarantees restoration of all captured states in `finally`.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # Capture attacker states
    atk_list = []
    if attackers is not None:
        if isinstance(attackers, dict):
            atk_list = list(attackers.values())
        elif isinstance(attackers, (list, tuple, set)):
            atk_list = list(attackers)
        else:
            atk_list = [attackers]
    saved_atk_states = []
    for atk in atk_list:
        if hasattr(atk, "state_dict"):
            state = None
            with contextlib.suppress(RuntimeError, TypeError, ValueError, AttributeError):
                state = atk.state_dict()
            saved_atk_states.append((atk, state))
        elif hasattr(atk, "_generator") and isinstance(atk._generator, torch.Generator):
            saved_atk_states.append((atk, {"generator_state": atk._generator.get_state().clone()}))
        else:
            saved_atk_states.append((atk, None))

    # Capture DataLoader generator state
    saved_dl_gen_state = None
    if dataloader is not None and hasattr(dataloader, "generator") and dataloader.generator is not None:
        with contextlib.suppress(RuntimeError, TypeError, ValueError, AttributeError):
            saved_dl_gen_state = dataloader.generator.get_state().clone()

    # Apply isolated seed if specified
    if seed is not None:
        int_seed = int(seed)
        random.seed(int_seed)
        np.random.seed(int_seed)
        torch.manual_seed(int_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int_seed)
        for atk, _ in saved_atk_states:
            if hasattr(atk, "reset_seed"):
                atk.reset_seed(int_seed)
            elif hasattr(atk, "_generator") and isinstance(atk._generator, torch.Generator):
                atk._generator.manual_seed(int_seed)

    try:
        yield
    finally:
        # Restore all states unconditionally
        for atk, state in saved_atk_states:
            if state is not None:
                if hasattr(atk, "load_state_dict"):
                    with contextlib.suppress(RuntimeError, TypeError, ValueError, AttributeError):
                        atk.load_state_dict(state)
                elif hasattr(atk, "_generator") and "generator_state" in state:
                    with contextlib.suppress(RuntimeError, TypeError, ValueError, AttributeError):
                        atk._generator.set_state(state["generator_state"])

        if (
            saved_dl_gen_state is not None
            and dataloader is not None
            and getattr(dataloader, "generator", None) is not None
        ):
            with contextlib.suppress(RuntimeError, TypeError, ValueError, AttributeError):
                dataloader.generator.set_state(saved_dl_gen_state)

        if cuda_states is not None and torch.cuda.is_available():
            with contextlib.suppress(RuntimeError, TypeError, ValueError, AttributeError):
                torch.cuda.set_rng_state_all(cuda_states)

        torch.set_rng_state(torch_cpu_state)
        np.random.set_state(np_state)
        random.setstate(py_state)
