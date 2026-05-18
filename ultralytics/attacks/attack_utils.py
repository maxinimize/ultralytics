import torch
from ultralytics.utils import LOGGER, DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.nn.tasks import DetectionModel
from copy import deepcopy

def setup_attack_model(attack_weights, device, nc, training=False, imgsz=640):
    """
    Set up the attack model based on the provided parameters.
    
    Args:
        attack_weights (str): Path to the attack model weights.
        device (torch.device): Device.
        nc (int): Number of classes.
        training (bool): Whether in training mode.
        imgsz (int): Input image size.

    Returns:
        attack_model: Configured attack model.
    """
    if not attack_weights:
        raise ValueError("attack_weights must be provided for setup_attack_model()")
    
    # load checkpoint
    ckpt = torch.load(attack_weights, map_location='cpu', weights_only=False)
    
    # model configuration
    cfg = ckpt.get('model').yaml if hasattr(ckpt.get('model', {}), 'yaml') else None
    
    # create model
    if cfg:
        attack_model = DetectionModel(cfg, ch=3, nc=nc).to(device)
    else:
        # reconstruct model from checkpoint
        attack_model = ckpt['model'].to(device)
        if hasattr(attack_model, 'nc'):
            attack_model.nc = nc
    
    # load weights
    if 'model' in ckpt:
        state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
        attack_model.load_state_dict(state_dict, strict=False)
        LOGGER.info(f"Loaded attack model from {attack_weights}")
    
    # set hyperparameters
    attack_model.args = deepcopy(DEFAULT_CFG_DICT)
    attack_model.args['imgsz'] = imgsz
    if 'hyp' in ckpt:
        for k, v in ckpt['hyp'].items():
            if k in attack_model.args:
                attack_model.args[k] = v
    
    # convert to IterableSimpleNamespace for compatibility with loss functions
    attack_model.args = IterableSimpleNamespace(**attack_model.args)
    
    # set model to eval mode if not training
    if not training:
        attack_model.eval()
    
    return attack_model