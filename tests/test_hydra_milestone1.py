"""Milestone-1 correctness tests.

Run these tests before any score-search experiment.
"""

import copy
import math

import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.pruning.hydra_convert import convert_to_hydra, iter_hydra_layers
from ultralytics.nn.pruning.hydra_layer import HydraConv2d, HydraMode
from ultralytics.nn.pruning.hydra_score import initialize_hydra_scores


def test_conv2d_attribute_preservation():
    """`HydraConv2d.from_conv` must preserve allConv2d geometry and parameters."""
    conv = nn.Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(2, 2),
        groups=4,
        bias=True,
    )
    torch.nn.init.normal_(conv.weight)
    torch.nn.init.normal_(conv.bias)

    hydra_conv = HydraConv2d.from_conv(conv, keep_ratio=0.8)

    assert hydra_conv.in_channels == conv.in_channels
    assert hydra_conv.out_channels == conv.out_channels
    assert hydra_conv.kernel_size == conv.kernel_size
    assert hydra_conv.stride == conv.stride
    assert hydra_conv.padding == conv.padding
    assert hydra_conv.dilation == conv.dilation
    assert hydra_conv.groups == conv.groups
    assert hydra_conv.padding_mode == conv.padding_mode
    assert hydra_conv.keep_ratio == 0.8
    assert hydra_conv.weight.device == conv.weight.device
    assert hydra_conv.weight.dtype == conv.weight.dtype
    assert torch.equal(hydra_conv.weight, conv.weight)
    assert torch.equal(hydra_conv.bias, conv.bias)
    assert hydra_conv.popup_scores.shape == conv.weight.shape


def test_ste_gradient_flow():
    """Gradients must pass through STE to popup_scores while weight remains detached in SCORE mode."""
    conv = nn.Conv2d(8, 16, 3, padding=1)
    hydra_conv = HydraConv2d.from_conv(conv, keep_ratio=0.5)
    hydra_conv.set_mode(HydraMode.SCORE)

    x = torch.randn(2, 8, 16, 16, requires_grad=True)
    out = hydra_conv(x)
    loss = out.sum()
    loss.backward()

    assert hydra_conv.popup_scores.grad is not None
    assert torch.abs(hydra_conv.popup_scores.grad).sum() > 0
    assert hydra_conv.weight.grad is None


def test_layerwise_topk_mask_sparsity():
    """Dynamic masks must keep exact top-k fraction for various keep_ratio values."""
    conv = nn.Conv2d(10, 10, 3, padding=1, bias=False)  # 900 params
    hydra_conv = HydraConv2d.from_conv(conv)
    torch.nn.init.normal_(hydra_conv.popup_scores)

    numel = hydra_conv.weight.numel()

    for ratio in [1.0, 0.75, 0.5, 0.25, 0.0]:
        hydra_conv.set_keep_ratio(ratio)
        mask = hydra_conv.dynamic_mask()
        expected_ones = int(round(ratio * numel))
        assert mask.sum().item() == expected_ones


def test_scaled_initialization_formula():
    """Score initialization must match the HYDRA scaled formula exactly."""
    model = nn.Sequential(
        Conv(3, 16, 3),
        Conv(16, 32, 3),
    )
    convert_to_hydra(model)
    initialize_hydra_scores(model)

    for _, layer in iter_hydra_layers(model):
        w = layer.weight
        fan_in = w.shape[1:].numel()
        scale = math.sqrt(6.0 / fan_in)
        max_abs = w.abs().max()
        expected_scores = scale * (w / max_abs)
        assert torch.allclose(layer.popup_scores, expected_scores, atol=1e-6)


def test_keep_ratio_one_equivalence():
    """Converted SCORE-mode model with k=1 must match the dense model."""
    # Test 1: Sequential model with Conv wrapper
    dense_seq = nn.Sequential(Conv(3, 16, 3), Conv(16, 32, 3)).eval()
    hydra_seq = copy.deepcopy(dense_seq).eval()

    x_dummy = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        y_dense = dense_seq(x_dummy)

    convert_to_hydra(hydra_seq, keep_ratio=1.0)
    initialize_hydra_scores(hydra_seq)

    for mode in [HydraMode.DENSE, HydraMode.SCORE, HydraMode.FINETUNE]:
        for _, layer in iter_hydra_layers(hydra_seq):
            layer.set_mode(mode)

        with torch.no_grad():
            y_hydra = hydra_seq(x_dummy)

        diff = (y_dense - y_hydra).abs().max().item()
        assert diff < 1e-5, f"Mode {mode} failed equivalence check: max diff = {diff}"

    def compute_max_diff(out1, out2):
        if isinstance(out1, torch.Tensor):
            return (out1 - out2).abs().max().item()
        elif isinstance(out1, (list, tuple)):
            return max([compute_max_diff(i1, i2) for i1, i2 in zip(out1, out2)])
        elif isinstance(out1, dict):
            return max([compute_max_diff(out1[k], out2[k]) for k in out1])
        return 0.0

    # Test 2: Ultralytics DetectionModel
    yolo_model = YOLO("yolov8n.yaml").model.eval()
    dense_yolo = copy.deepcopy(yolo_model).eval()
    hydra_yolo = copy.deepcopy(yolo_model).eval()

    x_img = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        out_dense = dense_yolo(x_img)

    convert_to_hydra(hydra_yolo, keep_ratio=1.0)
    initialize_hydra_scores(hydra_yolo)

    for mode in [HydraMode.DENSE, HydraMode.SCORE, HydraMode.FINETUNE]:
        for _, layer in iter_hydra_layers(hydra_yolo):
            layer.set_mode(mode)

        with torch.no_grad():
            out_hydra = hydra_yolo(x_img)

        diff = compute_max_diff(out_dense, out_hydra)
        assert diff < 1e-4, f"YOLO Mode {mode} failed equivalence check: max diff = {diff}"


if __name__ == "__main__":
    print("Running test_conv2d_attribute_preservation...")
    test_conv2d_attribute_preservation()
    print("Running test_ste_gradient_flow...")
    test_ste_gradient_flow()
    print("Running test_layerwise_topk_mask_sparsity...")
    test_layerwise_topk_mask_sparsity()
    print("Running test_scaled_initialization_formula...")
    test_scaled_initialization_formula()
    print("Running test_keep_ratio_one_equivalence...")
    test_keep_ratio_one_equivalence()
    print("All Milestone-1 tests PASSED successfully!")
