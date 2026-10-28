# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Spatial Transformer Network (STN) defense for Worst-of-k geometric attacks."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils import LOGGER
from ultralytics.utils.patches import torch_load


def build_forward_matrix(
    params: torch.Tensor,
    height: int,
    width: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build pixel-space forward affine transformation matrices input -> output.

    Consistent with Ultralytics SpatialWorstOfK:
    Rotation is performed around the image center ((width - 1)/2, (height - 1)/2),
    followed by horizontal translation tx and vertical translation ty.

    Args:
        params (torch.Tensor): Shape [..., 3] containing [angle_deg, tx_px, ty_px].
        height (int): Image height in pixels.
        width (int): Image width in pixels.
        dtype (torch.dtype, optional): Matrix data type. Defaults to params.dtype.

    Returns:
        torch.Tensor: Shape [..., 3, 3] forward transformation matrices.
    """
    if dtype is None:
        dtype = params.dtype

    angles = params[..., 0] * (math.pi / 180.0)
    tx = params[..., 1]
    ty = params[..., 2]

    c = torch.cos(angles)
    s = torch.sin(angles)
    cx = float(width - 1.0) / 2.0
    cy = float(height - 1.0) / 2.0

    shape = params.shape[:-1] + (3, 3)
    m = torch.zeros(shape, device=params.device, dtype=dtype)
    m[..., 0, 0] = c
    m[..., 0, 1] = -s
    m[..., 1, 0] = s
    m[..., 1, 1] = c
    m[..., 0, 2] = cx + tx - c * cx + s * cy
    m[..., 1, 2] = cy + ty - s * cx - c * cy
    m[..., 2, 2] = 1.0
    return m


def matrix_to_grid(
    forward_matrices: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert pixel-space forward matrices to PyTorch affine_grid normalized sampling coordinates.

    In STN rectification:
    The input to grid_sample is the attacked image I_adv.
    The output is the rectified image I_rect.
    A pixel coordinate in I_rect corresponds to p_adv = M_fwd @ p_rect in I_adv.
    For PyTorch affine_grid (align_corners=False), normalized coordinates u in [-1, 1]
    are mapped to pixel coordinates x via:
        x = (W / 2) * u + (W - 1) / 2  <=>  [x, y, 1]^T = P @ [u, v, 1]^T.
    Therefore, the normalized sampling grid theta mapping output to input is:
        theta_h = P_inv @ M_fwd @ P
        theta = theta_h[:, :2, :].

    Args:
        forward_matrices (torch.Tensor): Shape [B, 3, 3] pixel-space forward matrices.
        height (int): Image height.
        width (int): Image width.

    Returns:
        torch.Tensor: Sampling grid with shape [B, H, W, 2] for F.grid_sample.
    """
    batch_size = forward_matrices.shape[0]
    device = forward_matrices.device
    dtype = forward_matrices.dtype

    p = torch.tensor(
        [
            [width / 2.0, 0.0, (width - 1.0) / 2.0],
            [0.0, height / 2.0, (height - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    p_inv = torch.linalg.inv(p)

    theta_h = p_inv.unsqueeze(0) @ forward_matrices.float() @ p.unsqueeze(0)
    theta = theta_h[:, :2, :].to(dtype=dtype)

    grid = F.affine_grid(theta, size=(batch_size, 3, height, width), align_corners=False)
    return grid


def rectify_images(
    images: torch.Tensor,
    forward_matrices: torch.Tensor,
    fill: float = 114.0 / 255.0,
) -> torch.Tensor:
    """Warp attacked NCHW images to rectified canonical frame using forward matrices.

    Args:
        images (torch.Tensor): Shape [B, C, H, W] in [0, 1].
        forward_matrices (torch.Tensor): Shape [B, 3, 3] forward transform matrices.
        fill (float): Border fill value for out-of-boundary samples.

    Returns:
        torch.Tensor: Rectified image tensor of shape [B, C, H, W] in [0, 1].
    """
    _, _, h, w = images.shape
    grid = matrix_to_grid(forward_matrices, h, w)
    if grid.dtype != images.dtype:
        grid = grid.to(dtype=images.dtype)

    centered = images - fill
    rectified = (
        F.grid_sample(
            centered,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        + fill
    )
    return rectified.clamp(0.0, 1.0)


def transform_boxes(
    boxes: torch.Tensor,
    matrices: torch.Tensor,
    height: int,
    width: int,
    clip: bool = True,
) -> torch.Tensor:
    """Transform bounding boxes between coordinate frames using affine matrices.

    Args:
        boxes (torch.Tensor): Shape [N, 4] or [N, >=4] in xyxy format [x1, y1, x2, y2, ...].
        matrices (torch.Tensor): Shape [3, 3] or [N, 3, 3] affine transformation matrix.
        height (int): Image canvas height in pixels.
        width (int): Image canvas width in pixels.
        clip (bool): Whether to clip transformed boxes to [0, width] x [0, height].

    Returns:
        torch.Tensor: Transformed boxes with the same leading dimensions.
    """
    if boxes.numel() == 0:
        return boxes.clone()

    if matrices.ndim == 2:
        matrices = matrices.unsqueeze(0).expand(boxes.shape[0], 3, 3)
    elif matrices.ndim == 3 and matrices.shape[0] == 1 and boxes.shape[0] > 1:
        matrices = matrices.expand(boxes.shape[0], 3, 3)

    matrices = matrices.to(device=boxes.device, dtype=boxes.dtype)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    ones = torch.ones_like(x1)

    # 4 corners in homogeneous coordinates: [N, 4, 3]
    corners = torch.stack(
        (
            torch.stack((x1, y1, ones), dim=-1),
            torch.stack((x2, y1, ones), dim=-1),
            torch.stack((x2, y2, ones), dim=-1),
            torch.stack((x1, y2, ones), dim=-1),
        ),
        dim=1,
    )

    # Transform corners: [N, 4, 3] @ [N, 3, 3]^T -> [N, 4, 3]
    transformed = torch.einsum("nij,npj->npi", matrices, corners)
    xy = transformed[..., :2]

    new_x1 = xy[..., 0].amin(dim=1)
    new_y1 = xy[..., 1].amin(dim=1)
    new_x2 = xy[..., 0].amax(dim=1)
    new_y2 = xy[..., 1].amax(dim=1)

    if clip:
        new_x1 = new_x1.clamp(0.0, float(width))
        new_y1 = new_y1.clamp(0.0, float(height))
        new_x2 = new_x2.clamp(0.0, float(width))
        new_y2 = new_y2.clamp(0.0, float(height))

    out = boxes.clone()
    out[:, 0] = new_x1
    out[:, 1] = new_y1
    out[:, 2] = new_x2
    out[:, 3] = new_y2
    return out


def compute_control_points_loss(
    pred_matrices: torch.Tensor,
    gt_matrices: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Compute normalized control-point geometric loss between predicted and ground-truth matrices.

    Control points are placed at canonical positions (4 corners and center).
    Differences are normalized by width and height to maintain scale and dimension invariance.

    Args:
        pred_matrices (torch.Tensor): Shape [B, 3, 3] predicted forward matrices.
        gt_matrices (torch.Tensor): Shape [B, 3, 3] ground-truth forward matrices.
        height (int): Image height.
        width (int): Image width.

    Returns:
        torch.Tensor: Scalar geometric distance loss.
    """
    device = pred_matrices.device
    dtype = pred_matrices.dtype

    w_f = float(width - 1.0)
    h_f = float(height - 1.0)

    # 5 canonical control points: corners + center
    pts = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [w_f, 0.0, 1.0],
            [w_f, h_f, 1.0],
            [0.0, h_f, 1.0],
            [w_f / 2.0, h_f / 2.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )  # [5, 3]

    # Map control points: [B, 5, 2]
    pred_pts = torch.einsum("bij,pj->bpi", pred_matrices, pts)[..., :2]
    gt_pts = torch.einsum("bij,pj->bpi", gt_matrices, pts)[..., :2]

    # Normalize coordinate errors
    scale = torch.tensor([float(width), float(height)], device=device, dtype=dtype)
    diff = (pred_pts - gt_pts) / scale  # [B, 5, 2]
    point_dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-12)  # [B, 5]
    return point_dist.mean()


def compute_validity_mask(
    gt_matrices: torch.Tensor,
    pred_matrices: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Compute binary validity mask of uncropped, unpadded pixels in the rectified frame.

    A pixel p_rect in I_rect is valid if and only if:
    1. Its corresponding point p_adv = pred_M @ p_rect lies within the attacked canvas.
    2. Under ground-truth attack gt_M, p_clean = gt_M_inv @ p_adv lies within the clean canvas.

    Args:
        gt_matrices (torch.Tensor): Shape [B, 3, 3] ground-truth attack matrices.
        pred_matrices (torch.Tensor): Shape [B, 3, 3] predicted forward matrices.
        height (int): Image height.
        width (int): Image width.

    Returns:
        torch.Tensor: Binary float mask of shape [B, 1, H, W] with 1 for valid pixels and 0 for invalid.
    """
    batch_size = gt_matrices.shape[0]
    device = gt_matrices.device
    dtype = gt_matrices.dtype

    # Create coordinate grid of rectified canvas: [H, W, 3]
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(grid_x)
    p_rect = torch.stack((grid_x, grid_y, ones), dim=-1).reshape(-1, 3)  # [HW, 3]

    # p_adv = pred_M @ p_rect: [B, HW, 3]
    p_adv = torch.einsum("bij,pj->bpi", pred_matrices, p_rect)
    adv_x = p_adv[..., 0]
    adv_y = p_adv[..., 1]
    adv_valid = (adv_x >= 0.0) & (adv_x <= float(width - 1.0)) & (adv_y >= 0.0) & (adv_y <= float(height - 1.0))

    # p_clean = gt_M_inv @ p_adv: [B, HW, 3]
    gt_inv = torch.linalg.inv(gt_matrices)
    p_clean = torch.einsum("bij,bpj->bpi", gt_inv, p_adv)
    clean_x = p_clean[..., 0]
    clean_y = p_clean[..., 1]
    clean_valid = (
        (clean_x >= 0.0) & (clean_x <= float(width - 1.0)) & (clean_y >= 0.0) & (clean_y <= float(height - 1.0))
    )

    mask = (adv_valid & clean_valid).float().reshape(batch_size, 1, height, width)
    return mask


def compute_masked_recon_loss(
    rect_img: torch.Tensor,
    clean_img: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute L1 reconstruction loss on valid unpadded/uncropped pixels.

    Args:
        rect_img (torch.Tensor): Rectified image [B, C, H, W].
        clean_img (torch.Tensor): Target clean image [B, C, H, W].
        mask (torch.Tensor): Validity mask [B, 1, H, W].

    Returns:
        torch.Tensor: Scalar reconstruction loss.
    """
    diff = torch.abs(rect_img - clean_img) * mask
    total_valid = torch.sum(mask) * float(clean_img.shape[1])
    if total_valid < 1.0:
        return torch.tensor(0.0, device=rect_img.device, dtype=rect_img.dtype)
    return torch.sum(diff) / total_valid


class SpatialLocalizationNet(nn.Module):
    """Lightweight Convolutional Localization Network for estimating spatial transformations.

    Predicts rotation angle θ (degrees) and translations tx, ty (pixels).
    Weights and biases of the final linear layer are initialized to zeros,
    guaranteeing an exact identity transformation at initialization.
    """

    def __init__(
        self,
        max_degrees: float = 45.0,
        max_translate: float = 0.25,
        in_channels: int = 3,
    ) -> None:
        """Initialize localization CNN.

        Args:
            max_degrees (float): Maximum absolute rotation angle in degrees.
            max_translate (float): Maximum translation as a fraction of image dimensions.
            in_channels (int): Input image channels (default: 3).
        """
        super().__init__()
        self.max_degrees = float(max_degrees)
        self.max_translate = float(max_translate)

        # 4-stage lightweight strided conv feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),  # /2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # /4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # /8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # /16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # 128 * 2 * 2 = 512
        self.fc1 = nn.Linear(512, 64)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 3)

        # Exact identity initialization: zero weights and zero biases
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict transformation parameters [angle_deg, tx_px, ty_px].

        Args:
            x (torch.Tensor): Input images of shape [B, 3, H, W] in [0, 1].

        Returns:
            torch.Tensor: Estimated parameters of shape [B, 3].
        """
        _, _, h, w = x.shape
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        hidden = self.relu(self.fc1(pooled))
        raw = self.fc2(hidden)  # [B, 3]

        angle = torch.tanh(raw[:, 0]) * self.max_degrees
        tx = torch.tanh(raw[:, 1]) * (self.max_translate * float(w))
        ty = torch.tanh(raw[:, 2]) * (self.max_translate * float(h))

        return torch.stack((angle, tx, ty), dim=-1)


class SpatialTransformer(nn.Module):
    """Spatial Transformer Network (STN) for rectifying Worst-of-k spatial attacks.

    Autonomous defense module that takes attacked images, estimates rotation and translation
    without access to ground-truth attack parameters, and outputs rectified images and
    forward transformation matrices.
    """

    def __init__(
        self,
        max_degrees: float = 45.0,
        max_translate: float = 0.25,
        fill: float = 114.0 / 255.0,
    ) -> None:
        """Initialize Spatial Transformer.

        Args:
            max_degrees (float): Max rotation angle in degrees for localization head.
            max_translate (float): Max translation ratio for localization head.
            fill (float): Border padding fill value in [0, 1].
        """
        super().__init__()
        self.max_degrees = float(max_degrees)
        self.max_translate = float(max_translate)
        self.fill = float(fill)
        self.localization = SpatialLocalizationNet(
            max_degrees=self.max_degrees,
            max_translate=self.max_translate,
        )
        self.stride = torch.tensor([32.0])
        self.criterion = None

    @property
    def stn_architecture(self) -> dict[str, Any]:
        """Return architectural configuration dictionary of STN."""
        return {
            "max_degrees": self.max_degrees,
            "max_translate": self.max_translate,
            "fill": self.fill,
        }

    def __deepcopy__(self, memo):
        """Deepcopy STN module while ignoring transient attached loss callbacks."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "loss":
                continue
            result.__dict__[k] = deepcopy(v, memo)
        return result

    def forward(self, images: torch.Tensor | dict, *args, **kwargs) -> Any:
        """Perform autonomous rectification on input images or evaluate batch loss.

        Args:
            images (Union[torch.Tensor, dict]): Shape [B, 3, H, W] in [0, 1] or batch dictionary.
            *args (Any): Variable arguments passed to attached loss.
            **kwargs (Any): Keyword arguments passed to attached loss.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] or Loss output:
                - If input is Tensor: (rectified_images, forward_matrices)
                - If input is dict: loss, loss_items
        """
        if isinstance(images, dict):
            if hasattr(self, "loss") and callable(self.loss):
                return self.loss(images, *args, **kwargs)
            raise ValueError("SpatialTransformer received a batch dictionary, but no loss method is attached.")

        _, _, h, w = images.shape
        loc_dtype = next(self.localization.parameters()).dtype
        params = self.localization(images.to(dtype=loc_dtype))
        matrices = build_forward_matrix(params, height=h, width=w, dtype=images.dtype)
        rectified = rectify_images(images, matrices, fill=self.fill)
        return rectified, matrices

    def predict_matrix(self, images: torch.Tensor) -> torch.Tensor:
        """Estimate forward transformation matrix without warping images."""
        _, _, h, w = images.shape
        loc_dtype = next(self.localization.parameters()).dtype
        params = self.localization(images.to(dtype=loc_dtype))
        return build_forward_matrix(params, height=h, width=w, dtype=images.dtype)

    def predict_params(self, images: torch.Tensor) -> torch.Tensor:
        """Estimate transformation parameters [angle_deg, tx_px, ty_px]."""
        loc_dtype = next(self.localization.parameters()).dtype
        return self.localization(images.to(dtype=loc_dtype))


class STNDetectorWrapper(nn.Module):
    """Integrated defense wrapper combining STN with a frozen YOLO detector.

    During inference:
    1. STN estimates the spatial distortion and rectifies the image.
    2. Frozen YOLO detector predicts bounding boxes on the rectified image.
    3. Predicted bounding boxes are mapped back to attacked-image coordinates.
    """

    def __init__(
        self,
        stn: SpatialTransformer,
        detector: nn.Module,
    ) -> None:
        """Initialize wrapper.

        Args:
            stn (SpatialTransformer): Trained or initialized STN module.
            detector (nn.Module): Detection model (e.g. YOLO12L), frozen.
        """
        super().__init__()
        self.stn = stn
        self.detector = detector
        self.detector.eval().requires_grad_(False)

        # Delegate essential detector attributes for validator compatibility
        for attr in ("names", "nc", "stride", "pt_path", "yaml", "args"):
            if hasattr(detector, attr):
                setattr(self, attr, getattr(detector, attr))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:
        """Forward pass rectifying image and returning detections.

        Args:
            x (torch.Tensor): Input images [B, 3, H, W].

        Returns:
            Detection outputs from YOLO detector.
        """
        rect_img, pred_m = self.stn(x)
        self.last_pred_matrices = pred_m
        return self.detector(rect_img, *args, **kwargs)


def load_stn_checkpoint(
    weights_path: str | Path,
    device: torch.device,
) -> tuple[SpatialTransformer, dict[str, Any]]:
    """Load trained STN model from checkpoint.

    Args:
        weights_path (Union[str, Path]): Path to STN checkpoint (.pt).
        device (torch.device): Device to place the loaded model.

    Returns:
        Tuple[SpatialTransformer, Dict[str, Any]]: Loaded STN and checkpoint metadata.
    """
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"STN checkpoint not found: {path}")

    ckpt = torch_load(path, map_location=device)
    stn_cfg = ckpt.get("stn_config", {})

    # Determine STN architectural parameters in priority order
    if "stn_architecture" in ckpt:
        arch = ckpt["stn_architecture"]
        max_deg = float(arch["max_degrees"])
        max_trans = float(arch["max_translate"])
        fill = float(arch.get("fill", 114.0 / 255.0))
    elif "max_degrees" in stn_cfg and "max_translate" in stn_cfg:
        max_deg = float(stn_cfg["max_degrees"])
        max_trans = float(stn_cfg["max_translate"])
        fill = float(stn_cfg.get("fill", 114.0 / 255.0))
    elif "stn_max_degrees" in stn_cfg and "stn_max_translate" in stn_cfg:
        max_deg = float(stn_cfg["stn_max_degrees"])
        max_trans = float(stn_cfg["stn_max_translate"])
        fill = float(stn_cfg.get("stn_fill", stn_cfg.get("fill", 114.0 / 255.0)))
    elif "degrees" in stn_cfg and "translate" in stn_cfg:
        # Legacy STNTrainer checkpoint: STN was instantiated with degrees * 1.5, translate * 1.5
        deg = float(stn_cfg["degrees"])
        trans = float(stn_cfg["translate"])
        max_deg = deg * 1.5
        max_trans = trans * 1.5
        fill = float(stn_cfg.get("fill", 114.0 / 255.0))
        LOGGER.warning(
            f"STN checkpoint at {path} lacks explicit 'stn_architecture'. Applying legacy STNTrainer formula: "
            f"max_degrees = degrees ({deg}) * 1.5 = {max_deg}, max_translate = translate ({trans}) * 1.5 = {max_trans}."
        )
    else:
        max_deg = 45.0
        max_trans = 0.15
        fill = 114.0 / 255.0
        LOGGER.warning(
            f"STN checkpoint at {path} lacks architectural metadata and STN config. "
            f"Defaulting to max_degrees={max_deg}, max_translate={max_trans}, fill={fill}."
        )

    stn = SpatialTransformer(max_degrees=max_deg, max_translate=max_trans, fill=fill).to(device)
    if "stn" in ckpt:
        stn.load_state_dict(ckpt["stn"])
    elif "model" in ckpt:
        stn.load_state_dict(ckpt["model"])
    else:
        stn.load_state_dict(ckpt)

    stn.eval()
    LOGGER.info(
        f"Loaded STN defense from {path} onto {device} (max_degrees={max_deg}, max_translate={max_trans}, fill={fill})"
    )
    return stn, ckpt
