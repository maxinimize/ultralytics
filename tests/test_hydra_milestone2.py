"""Milestone-2 correctness tests for score search and fixed-mask fine-tuning."""

import copy
import tempfile
from pathlib import Path

import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.pruning.hydra_convert import convert_to_hydra, iter_hydra_layers
from ultralytics.nn.pruning.hydra_export import materialize_hydra_model
from ultralytics.nn.pruning.hydra_layer import HydraConv2d, HydraMode
from ultralytics.nn.pruning.hydra_mask import (
    configure_masked_finetune,
    configure_score_search,
    enforce_pruned_weights_zero,
    freeze_hydra_masks,
    set_hydra_keep_ratio,
)
from ultralytics.nn.pruning.hydra_score import initialize_hydra_scores


def test_score_search_parameter_update_and_membership():
    """Score search must update only popup_scores, leaving dense weights untouched."""
    model = nn.Sequential(
        Conv(3, 16, 3),
        Conv(16, 32, 3),
    )
    convert_to_hydra(model, keep_ratio=0.5)
    initialize_hydra_scores(model)
    configure_score_search(model)

    # Snapshot initial state
    dense_weights_before = [layer.weight.clone() for _, layer in iter_hydra_layers(model)]
    popup_scores_before = [layer.popup_scores.clone() for _, layer in iter_hydra_layers(model)]

    # Check optimizer parameter membership
    score_params = [p for name, p in model.named_parameters() if "popup_scores" in name]
    optimizer = torch.optim.SGD(score_params, lr=0.1)

    for group in optimizer.param_groups:
        for p in group["params"]:
            assert any(p is sp for sp in score_params), "Non-score parameter found in score optimizer!"
            for _, layer in iter_hydra_layers(model):
                assert p is not layer.weight, "Dense Conv2d weight found in score optimizer!"
                if layer.bias is not None:
                    assert p is not layer.bias, "Dense Conv2d bias found in score optimizer!"

    # Run one forward/backward/step
    x = torch.randn(2, 3, 16, 16)
    out = model(x)
    loss = out.sum()
    loss.backward()
    optimizer.step()

    # Verify dense weights are identical and popup_scores changed
    for (_, layer), w_before, s_before in zip(iter_hydra_layers(model), dense_weights_before, popup_scores_before):
        assert torch.equal(layer.weight, w_before), "Dense weight changed during score search!"
        assert not torch.equal(layer.popup_scores, s_before), "popup_scores failed to update!"


def test_requested_sparsity():
    """Layer-wise masks must achieve requested keep_ratio values."""
    yolo_model = YOLO("yolov8n.yaml").model
    convert_to_hydra(yolo_model)
    initialize_hydra_scores(yolo_model)

    keep_ratios = [1.0, 0.75, 0.50, 0.10]
    for ratio in keep_ratios:
        set_hydra_keep_ratio(yolo_model, ratio)
        report = freeze_hydra_masks(yolo_model)
        expected_kept = 0
        total_elems = 0
        for _, layer in iter_hydra_layers(yolo_model):
            expected_kept += int(round(ratio * layer.weight.numel()))
            total_elems += layer.weight.numel()

        assert report.total_mask_elements == total_elems
        assert report.kept_mask_elements == expected_kept
        expected_sparsity = 1.0 - (expected_kept / total_elems)
        assert abs(report.sparsity - expected_sparsity) < 1e-5


def test_fixed_mask_immutability_and_finetune_behavior():
    """Fixed masks must remain immutable and pruned weights stay zero during fine-tuning."""
    model = nn.Sequential(
        Conv(3, 16, 3),
        Conv(16, 32, 3),
    )
    convert_to_hydra(model, keep_ratio=0.5)
    initialize_hydra_scores(model)
    freeze_hydra_masks(model)
    configure_masked_finetune(model)

    masks_before = [layer.fixed_mask.clone() for _, layer in iter_hydra_layers(model)]
    scores_before = [layer.popup_scores.clone() for _, layer in iter_hydra_layers(model)]

    # Optimizer for fine-tuning
    params = [p for name, p in model.named_parameters() if "popup_scores" not in name and p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.1)

    # Initial zeroing of pruned weights
    enforce_pruned_weights_zero(model)
    weights_before = [layer.weight.clone() for _, layer in iter_hydra_layers(model)]

    x = torch.randn(2, 3, 16, 16)
    out = model(x)
    loss = out.sum()
    loss.backward()
    optimizer.step()
    enforce_pruned_weights_zero(model)

    for (_, layer), m_before, s_before, w_before in zip(iter_hydra_layers(model), masks_before, scores_before, weights_before):
        assert torch.equal(layer.fixed_mask, m_before), "fixed_mask changed during optimizer step!"
        assert torch.equal(layer.popup_scores, s_before), "popup_scores changed during fine-tuning!"
        assert not torch.equal(layer.weight, w_before), "Retained weights did not update during fine-tuning!"
        pruned_weights = layer.weight * (1.0 - layer.fixed_mask)
        assert torch.abs(pruned_weights).max().item() == 0.0, "Pruned weights are not physically zero!"


def test_materialization_equivalence():
    """Materialized ordinary Conv2d model output must match masked HYDRA output."""
    yolo_model = YOLO("yolov8n.yaml").model.eval()
    convert_to_hydra(yolo_model, keep_ratio=0.5)
    initialize_hydra_scores(yolo_model)
    freeze_hydra_masks(yolo_model)
    configure_masked_finetune(yolo_model)

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        hydra_out = yolo_model(x)

    materialized_model = materialize_hydra_model(yolo_model, inplace=False).eval()

    for m in materialized_model.modules():
        assert not isinstance(m, HydraConv2d), "HydraConv2d module remaining after materialization!"

    with torch.no_grad():
        mat_out = materialized_model(x)

    def compute_max_diff(o1, o2):
        if isinstance(o1, torch.Tensor):
            return (o1 - o2).abs().max().item()
        elif isinstance(o1, (list, tuple)):
            return max(compute_max_diff(a, b) for a, b in zip(o1, o2))
        elif isinstance(o1, dict):
            return max(compute_max_diff(o1[k], o2[k]) for k in o1)
        return 0.0

    max_diff = compute_max_diff(hydra_out, mat_out)
    assert max_diff < 1e-4, f"Materialized model output differs from HYDRA output: max_diff={max_diff}"


def test_checkpoint_serialization_and_deserialization():
    """Saved and reloaded HYDRA checkpoint must match scores, masks, and outputs."""
    model = YOLO("yolov8n.yaml").model.eval()
    convert_to_hydra(model, keep_ratio=0.5)
    initialize_hydra_scores(model)
    freeze_hydra_masks(model)
    configure_masked_finetune(model)

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        orig_out = model(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "hydra_test.pt"
        torch.save({"model": model, "hydra_stage": "finetune", "hydra_keep_ratio": 0.5}, ckpt_path)

        ckpt = torch.load(ckpt_path, weights_only=False)
        reloaded_model = ckpt["model"].eval()

        with torch.no_grad():
            reloaded_out = reloaded_model(x)

        def compute_max_diff(o1, o2):
            if isinstance(o1, torch.Tensor):
                return (o1 - o2).abs().max().item()
            elif isinstance(o1, (list, tuple)):
                return max(compute_max_diff(a, b) for a, b in zip(o1, o2))
            elif isinstance(o1, dict):
                return max(compute_max_diff(o1[k], o2[k]) for k in o1)
            return 0.0

        max_diff = compute_max_diff(orig_out, reloaded_out)
        assert max_diff < 1e-5, f"Reloaded model output differs: max_diff={max_diff}"

        for (_, orig_layer), (_, reload_layer) in zip(iter_hydra_layers(model), iter_hydra_layers(reloaded_model)):
            assert torch.equal(orig_layer.popup_scores, reload_layer.popup_scores), "Scores differ after reload!"
            assert torch.equal(orig_layer.fixed_mask, reload_layer.fixed_mask), "Fixed masks differ after reload!"


if __name__ == "__main__":
    print("Running test_score_search_parameter_update_and_membership...")
    test_score_search_parameter_update_and_membership()
    print("Running test_requested_sparsity...")
    test_requested_sparsity()
    print("Running test_fixed_mask_immutability_and_finetune_behavior...")
    test_fixed_mask_immutability_and_finetune_behavior()
    print("Running test_materialization_equivalence...")
    test_materialization_equivalence()
    print("Running test_checkpoint_serialization_and_deserialization...")
    test_checkpoint_serialization_and_deserialization()
    print("All Milestone-2 tests PASSED successfully!")
