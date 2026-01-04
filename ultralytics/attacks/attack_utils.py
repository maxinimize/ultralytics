# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Attack utility functions for adversarial training in Ultralytics YOLO.

This module provides utilities for setting up attack models and related functionality
for adversarial training and evaluation.
"""

import torch
from pathlib import Path
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import LOGGER, DEFAULT_CFG
from ultralytics.utils.checks import check_suffix
from ultralytics.utils.torch_utils import unwrap_model
from copy import deepcopy


def setup_attack_model(attack_weights, device, nc, training=False, imgsz=640):
    """
    Set up the attack model for adversarial training/evaluation.
    
    This function loads a YOLO detection model specifically for generating adversarial
    examples. The model is loaded from weights and configured for the attack pipeline.
    
    Args:
        attack_weights (str): Path to the attack model weights file (.pt).
        device (torch.device): Device to load the model on (cpu, cuda, etc.).
        nc (int): Number of classes in the dataset.
        training (bool): Whether the model is being used during training.
        imgsz (int): Image size for the model.
    
    Returns:
        (DetectionModel): Configured attack model ready for adversarial example generation.
    
    Raises:
        ValueError: If attack_weights is empty or not provided.
        FileNotFoundError: If the weights file doesn't exist.
    
    Examples:
        >>> from ultralytics.attacks.attack_utils import setup_attack_model
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> attack_model = setup_attack_model(
        ...     attack_weights='yolo11n.pt',
        ...     device=device,
        ...     nc=80,
        ...     training=True,
        ...     imgsz=640
        ... )
    """
    if not attack_weights or attack_weights == "":
        raise ValueError(
            "attack_weights must be provided for setup_attack_model(). "
            "Please provide a valid path to the model weights file."
        )
    
    # Check file extension
    check_suffix(attack_weights, ".pt")
    
    # Check if file exists
    attack_weights_path = Path(attack_weights)
    if not attack_weights_path.exists():
        raise FileNotFoundError(f"Attack weights file not found: {attack_weights}")
    
    try:
        # Load checkpoint using Ultralytics method
        LOGGER.info(f"Loading attack model from {attack_weights}...")
        attack_model, ckpt = load_checkpoint(attack_weights, device=device, inplace=True, fuse=False)
        
        # Model is already loaded and on the correct device
        
        # Set model attributes
        attack_model.nc = nc
        attack_model.names = {i: f"class{i}" for i in range(nc)}  # dummy names
        attack_model.args = DEFAULT_CFG  # attach default config
        
        # Set model to eval mode and freeze parameters
        attack_model.eval()
        for param in attack_model.parameters():
            param.requires_grad = False
        
        LOGGER.info(
            f"Attack model loaded successfully:\n"
            f"  - Classes: {nc}\n"
            f"  - Device: {device}\n"
            f"  - Image size: {imgsz}\n"
            f"  - Training mode: {training}"
        )
        
        return attack_model
        
    except Exception as e:
        LOGGER.error(f"Failed to load attack model: {e}")
        raise


def prepare_attack_batch(batch, device):
    """
    Prepare a batch for adversarial attack.
    
    Args:
        batch (dict): Batch dictionary from dataloader.
        device (torch.device): Target device.
    
    Returns:
        (dict): Prepared batch ready for attack.
    """
    attack_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            attack_batch[k] = v.to(device, non_blocking=True)
        else:
            attack_batch[k] = v
    return attack_batch


def copy_model_anchors(source_model, target_model):
    """
    Copy anchors from source model to target model.
    
    Args:
        source_model (nn.Module): Source model to copy anchors from.
        target_model (nn.Module): Target model to copy anchors to.
    
    Returns:
        (bool): True if anchors were successfully copied, False otherwise.
    """
    try:
        source_head = unwrap_model(source_model).model[-1]
        target_head = unwrap_model(target_model).model[-1]
        
        if hasattr(source_head, 'anchors') and hasattr(target_head, 'anchors'):
            if source_head.anchors.shape == target_head.anchors.shape:
                target_head.anchors.data.copy_(
                    source_head.anchors.detach().to(
                        dtype=target_head.anchors.dtype,
                        device=target_head.anchors.device
                    )
                )
                LOGGER.info("Successfully copied anchors between models")
                return True
            else:
                LOGGER.warning(
                    f"Anchor shape mismatch: "
                    f"source {source_head.anchors.shape} vs "
                    f"target {target_head.anchors.shape}"
                )
        else:
            LOGGER.warning("One or both models do not have anchor attributes")
        return False
    except Exception as e:
        LOGGER.warning(f"Could not copy anchors: {e}")
        return False