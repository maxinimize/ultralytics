"""
PyTorch implementation of Feature Distillation (FD) for Ultralytics/YOLO pipelines.

Based on the official implementation of:
Liu et al., "Feature Distillation: DNN-Oriented JPEG Compression Against
Adversarial Examples", CVPR 2019.

This module intentionally preserves the core algorithm used by the official
repository:
    [0, 1] image tensor
      -> scale to [0, 255]
      -> split each channel into non-overlapping 8x8 blocks
      -> orthonormal 2-D DCT
      -> quantize using the official fixed q_table
      -> de-quantize
      -> inverse DCT
      -> scale back to [0, 1] and clip

Engineering changes for modern YOLO/PyTorch:
    - NCHW [B, C, H, W] instead of NHWC
    - native torch implementation instead of NumPy/SciPy
    - works on CPU or CUDA
    - vectorized batch/channel/block processing
    - processes batch index 0 (the official repository loop starts from 1)
    - optionally pads non-multiple-of-8 image sizes
    - supports arbitrary channel count, although YOLO normally uses C=3

Important:
    - This is a deterministic preprocessing defense, not a trainable network.
    - The default hard torch.round() is faithful to the preprocessing defense,
      but it blocks useful input gradients. Adaptive attacks should use a
      separately defined BPDA/STE evaluation rather than silently changing
      this implementation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_orthonormal_dct_matrix(
    n: int = 8,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create the orthonormal DCT-II matrix used by scipy.fftpack.dct(..., norm='ortho')."""
    sample_idx = torch.arange(n, dtype=dtype)
    freq_idx = torch.arange(n, dtype=dtype).unsqueeze(1)

    matrix = torch.cos(
        (math.pi / n) * (sample_idx + 0.5) * freq_idx
    )
    matrix[0] *= math.sqrt(1.0 / n)
    matrix[1:] *= math.sqrt(2.0 / n)
    return matrix


def official_fd_quantization_table(
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return the fixed 8x8 quantization table used by the public official code.

    Official repository:
        q_table = np.ones((8, 8)) * 30
        q_table[0:4, 0:4] = 25
    """
    q_table = torch.full((8, 8), 30.0, dtype=dtype)
    q_table[:4, :4] = 25.0
    return q_table


class FeatureDistillation(nn.Module):
    """
    Fixed Feature Distillation preprocessing for image tensors.

    Expected input
    --------------
    Tensor of shape [B, C, H, W], floating point, normally in [0, 1].

    Parameters
    ----------
    q_table:
        Optional custom 8x8 quantization table. If None, the fixed table from
        the official public implementation is used.

    allow_padding:
        The original implementation effectively requires H and W to be
        divisible by 8. The default False preserves that assumption.
        Set True only if you intentionally want this PyTorch implementation to
        pad to the next multiple of 8 and crop back afterward.

    padding_mode:
        torch.nn.functional.pad mode used when allow_padding=True.
        "reflect" is the default engineering choice; padding is NOT part of the
        original Feature Distillation algorithm.

    preserve_dtype:
        Compute internally in float32 for stable DCT/quantization. If True,
        cast the result back to the input dtype before returning.
    """

    block_size: int = 8

    def __init__(
        self,
        q_table: Optional[torch.Tensor] = None,
        *,
        allow_padding: bool = False,
        padding_mode: str = "reflect",
        preserve_dtype: bool = True,
    ) -> None:
        super().__init__()

        if q_table is None:
            q_table = official_fd_quantization_table()
        else:
            q_table = torch.as_tensor(q_table, dtype=torch.float32)

        if tuple(q_table.shape) != (8, 8):
            raise ValueError(
                f"Feature Distillation expects an 8x8 q_table, got {tuple(q_table.shape)}."
            )
        if torch.any(q_table <= 0):
            raise ValueError("All quantization-table entries must be > 0.")

        self.allow_padding = allow_padding
        self.padding_mode = padding_mode
        self.preserve_dtype = preserve_dtype

        # Buffers move automatically with .to(device), but are not trainable parameters.
        self.register_buffer("q_table", q_table.clone().detach())
        self.register_buffer(
            "dct_matrix",
            _make_orthonormal_dct_matrix(self.block_size),
        )

    @property
    def is_trainable(self) -> bool:
        """FD has no trainable parameters."""
        return False

    def _pad_if_required(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        h, w = x.shape[-2:]
        pad_h = (-h) % self.block_size
        pad_w = (-w) % self.block_size

        if pad_h == 0 and pad_w == 0:
            return x, 0, 0

        if not self.allow_padding:
            raise ValueError(
                "Feature Distillation uses non-overlapping 8x8 blocks, so H and W "
                f"must be divisible by 8. Received H={h}, W={w}. "
                "For normal Ultralytics model-input sizes such as 640x640 this is "
                "already satisfied. Set allow_padding=True only if padding is intended."
            )

        x = F.pad(
            x,
            (0, pad_w, 0, pad_h),
            mode=self.padding_mode,
        )
        return x, pad_h, pad_w

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape [B, C, H, W], "
                f"got {tuple(images.shape)}."
            )

        if not images.is_floating_point():
            raise TypeError(
                "FeatureDistillation expects a floating-point tensor in [0, 1]. "
                "Convert uint8 images with images.float() / 255.0 first."
            )

        if images.numel() == 0:
            return images

        original_dtype = images.dtype
        original_h, original_w = images.shape[-2:]

        # The official NumPy code ultimately returns float32. Using float32
        # internally also avoids doing DCT/rounding in fp16 under AMP.
        x = images.to(dtype=torch.float32)
        x, pad_h, pad_w = self._pad_if_required(x)

        b, c, h, w = x.shape
        bs = self.block_size

        # Official implementation multiplies normalized input by 255 before DCT.
        x = x * 255.0

        # [B, C, H, W]
        # -> [B, C, H/8, 8, W/8, 8]
        # -> [B, C, H/8, W/8, 8, 8]
        blocks = (
            x.reshape(b, c, h // bs, bs, w // bs, bs)
            .permute(0, 1, 2, 4, 3, 5)
            .contiguous()
        )

        dct = self.dct_matrix.to(dtype=torch.float32)
        q_table = self.q_table.to(dtype=torch.float32)

        # 2-D orthonormal DCT:
        #     C = D @ X @ D^T
        coeff = torch.matmul(torch.matmul(dct, blocks), dct.transpose(-1, -2))

        # Faithful to the official algorithm:
        # quantization -> rounding -> de-quantization.
        coeff_quantized = torch.round(coeff / q_table)
        coeff_dequantized = coeff_quantized * q_table

        # 2-D inverse DCT:
        #     X_hat = D^T @ C_hat @ D
        reconstructed = torch.matmul(
            torch.matmul(dct.transpose(-1, -2), coeff_dequantized),
            dct,
        )

        # [B, C, H/8, W/8, 8, 8]
        # -> [B, C, H, W]
        reconstructed = (
            reconstructed.permute(0, 1, 2, 4, 3, 5)
            .contiguous()
            .reshape(b, c, h, w)
        )

        # Official implementation divides by 255 and clips to [0, 1].
        reconstructed = (reconstructed / 255.0).clamp_(0.0, 1.0)

        # Padding is an engineering extension; crop it away if used.
        if pad_h != 0 or pad_w != 0:
            reconstructed = reconstructed[..., :original_h, :original_w]

        if self.preserve_dtype:
            reconstructed = reconstructed.to(dtype=original_dtype)

        return reconstructed


def build_feature_distillation(
    device: Optional[torch.device | str] = None,
    *,
    allow_padding: bool = False,
) -> FeatureDistillation:
    """Convenience factory. Instantiate once and reuse it for every batch."""
    defense = FeatureDistillation(allow_padding=allow_padding)
    if device is not None:
        defense = defense.to(device)
    defense.eval()
    return defense


if __name__ == "__main__":
    # Minimal smoke test.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fd = build_feature_distillation(device=device)

    x = torch.rand(2, 3, 640, 640, device=device)
    y = fd(x)

    print("device:", y.device)
    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print("output range:", float(y.min()), float(y.max()))
    print("trainable parameters:", sum(p.numel() for p in fd.parameters()))
