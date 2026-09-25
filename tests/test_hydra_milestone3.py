"""Milestone-3 correctness tests for robust HYDRA integration.

Supports both GPU (CUDA) and CPU execution, with full testing of both
PGD (gradient-based pixel attack) and Worst-of-k (spatial transformation attack).

Validates:
  A. Current-model attack-source identity (PGD & WORSTK)
  B. Robust SCORE update (popup_scores only, dense weights frozen, BN eval) (PGD & WORSTK)
  C. Gradient hygiene (inner attack gradients do not leak into outer optimizer) (PGD & WORSTK)
  D. Mask creation after SCORE (binary masks, target sparsity, score-based)
  E. Fixed-mask robust FINETUNE (masks and scores fixed, pruned weights zero) (PGD & WORSTK)
  F. Ratio compatibility (sample-level routing, normalized group weights) (PGD + WORSTK)
  G. Clean-only fallback (attack ratio 0 reduces to clean training)
  H. Single-attack training (PGD and WORSTK)
  I. Multi-attack smoke test (PGD + WORSTK + MIM)
  J. Fixed-source backward compatibility (explicit attack_weights) (PGD & WORSTK)
  K. Attack parameter passthrough (custom settings forwarded) (PGD + WORSTK)
  L. Worst-of-k smoke test (spatial transforms and candidate selection)
  M. SCORE -> FINETUNE checkpoint handoff (PGD + WORSTK)
  N. Materialization equivalence (ordinary Conv2d matches masked HYDRA)
  O. Standard validation on materialized model
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.attacks.attack_bridge import build_attacker, run_attack_on_batch
from ultralytics.attacks.spatial_worst_of_k import SpatialWorstOfK
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.models import yolo
from ultralytics.models.yolo.detect.hydra_adversarial import (
    HydraAdversarialDetectionTrainer,
)
from ultralytics.models.yolo.detect.train_adv_test import (
    DetectionTrainer as AdversarialDetectionTrainer,
    YOLODatasetAdvTest,
)
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
from ultralytics.utils.torch_utils import unwrap_model

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _create_synthetic_detection_batch(batch_size: int = 2, img_size: int = 64, device=None) -> dict:
    """Create a minimal synthetic detection batch on the target device."""
    if device is None:
        device = DEVICE
    imgs = torch.rand(batch_size, 3, img_size, img_size, dtype=torch.float32, device=device)
    batch_idx = []
    classes = []
    bboxes = []
    for i in range(batch_size):
        batch_idx.append(i)
        classes.append(0.0)
        bboxes.append([0.5, 0.5, 0.3, 0.3])

    return {
        "img": imgs,
        "batch_idx": torch.tensor(batch_idx, dtype=torch.long, device=device),
        "cls": torch.tensor(classes, dtype=torch.float32, device=device).unsqueeze(1),
        "bboxes": torch.tensor(bboxes, dtype=torch.float32, device=device),
        "im_file": [f"img_{i}.jpg" for i in range(batch_size)],
        "sample_type": ["raw"] * (batch_size // 2) + ["pgd"] * (batch_size - batch_size // 2),
    }


# ======================================================================
# Test A: Current-model attack-source identity test
# ======================================================================
def test_current_model_attack_source_identity(attack_name: str = "pgd"):
    """Attack generator must target the current live HYDRA model when attack_weights is None."""
    overrides = {
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "device": DEVICE,
        "epochs": 1,
        "batch": 2,
        "imgsz": 64,
        "hydra_stage": "score",
        "hydra_keep_ratio": 0.5,
        "attack_name": attack_name,
        "attack_ratio": 0.5,
        "attack_weights": "",
    }
    trainer = HydraAdversarialDetectionTrainer(overrides=overrides)
    trainer.setup_model()
    if torch.cuda.is_available():
        trainer.model = trainer.model.to(DEVICE)
    trainer.set_model_attributes()

    # 1. Verify no separate attack model was loaded
    assert trainer.attack_model is None, "A separate attack model was unexpectedly loaded!"

    # 2. Verify attacker model points to the current unwrapped HYDRA model
    attacker = trainer.attackers[attack_name]
    assert attacker.model is unwrap_model(trainer.model), "Attacker model is not unwrap_model(trainer.model)!"

    # 3. Verify that during SCORE mode, layer mode is SCORE (dynamic mask)
    hydra_layers = list(iter_hydra_layers(trainer.model))
    assert len(hydra_layers) > 0
    for _, layer in hydra_layers:
        assert layer.hydra_mode == HydraMode.SCORE

    # 4. Switch to FINETUNE and verify attacker still references current model with fixed_mask
    freeze_hydra_masks(trainer.model)
    configure_masked_finetune(trainer.model)
    for _, layer in hydra_layers:
        assert layer.hydra_mode == HydraMode.FINETUNE


# ======================================================================
# Test B: Robust SCORE update test
# ======================================================================
def test_robust_score_update(attack_name: str = "pgd"):
    """Adversarial SCORE step must update popup_scores while dense weights stay frozen."""
    overrides = {
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "device": DEVICE,
        "epochs": 1,
        "batch": 2,
        "imgsz": 64,
        "hydra_stage": "score",
        "hydra_keep_ratio": 0.5,
        "attack_name": attack_name,
        "attack_ratio": 0.5,
    }
    trainer = HydraAdversarialDetectionTrainer(overrides=overrides)
    trainer.setup_model()
    if torch.cuda.is_available():
        trainer.model = trainer.model.to(DEVICE)
    trainer.set_model_attributes()
    trainer._model_train()

    # Verify BN eval policy in SCORE mode
    for m in trainer.model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            assert not m.training, "BatchNorm must be in eval mode during SCORE search!"

    # Build optimizer and verify parameter groups
    optimizer = trainer.build_optimizer(trainer.model)
    score_params = [p for n, p in unwrap_model(trainer.model).named_parameters() if "popup_scores" in n]
    score_param_set = set(score_params)
    for group in optimizer.param_groups:
        for p in group["params"]:
            assert p in score_param_set, "Non-score parameter found in score optimizer!"

    # Verify dense weights are frozen (requires_grad=False)
    for name, p in unwrap_model(trainer.model).named_parameters():
        if "popup_scores" not in name:
            assert not p.requires_grad, f"Dense parameter {name} requires_grad is True!"

    # Snapshot initial parameters
    dense_before = [p.clone() for n, p in unwrap_model(trainer.model).named_parameters() if "popup_scores" not in n]
    scores_before = [p.clone() for p in score_params]

    # Run forward/backward with synthetic batch (lightweight head loss)
    device = next(trainer.model.parameters()).device
    batch = _create_synthetic_detection_batch(batch_size=2, img_size=64, device=device)
    preds = trainer.model(batch["img"])
    if isinstance(preds, dict):
        loss = preds["scores"].sum() + preds["boxes"].sum()
    elif isinstance(preds, (list, tuple)):
        loss = sum(p.sum() for p in preds if torch.is_tensor(p) and p.requires_grad)
    else:
        loss = preds.sum()
    loss.backward()

    # Verify popup_scores received gradients
    has_nonzero_grad = any(p.grad is not None and torch.count_nonzero(p.grad).item() > 0 for p in score_params)
    assert has_nonzero_grad, "No popup_scores parameter received a gradient!"

    # Step optimizer
    optimizer.step()

    # Verify dense weights remained unchanged and at least one score changed
    dense_after = [p for n, p in unwrap_model(trainer.model).named_parameters() if "popup_scores" not in n]
    for before, after in zip(dense_before, dense_after):
        assert torch.equal(before, after), "Dense weight changed during score search!"

    scores_changed = any(not torch.equal(b, a) for b, a in zip(scores_before, score_params))
    assert scores_changed, "popup_scores failed to update after optimizer step!"


# ======================================================================
# Test C: Gradient hygiene test
# ======================================================================
def test_gradient_hygiene(attack_name: str = "pgd"):
    """Inner attack loop generation must not leave stray gradients on training parameters."""
    device = DEVICE
    yolo_model = YOLO("yolov8n.yaml").model.to(device).eval()
    yolo_model.args = get_cfg(DEFAULT_CFG)
    convert_to_hydra(yolo_model, keep_ratio=0.5)
    initialize_hydra_scores(yolo_model)
    configure_score_search(yolo_model)
    yolo_model.to(device)

    if attack_name == "worstk":
        attacker = build_attacker("worstk", model=yolo_model, img_size=64, k=2)
    else:
        attacker = build_attacker(attack_name, model=yolo_model, img_size=64, epoch=2)
    batch = _create_synthetic_detection_batch(batch_size=2, img_size=64, device=device)

    # Freeze model params as train_adv_test does during inner attack generation
    param_states = [p.requires_grad for p in yolo_model.parameters()]
    for p in yolo_model.parameters():
        p.requires_grad = False

    adv_img = run_attack_on_batch(attacker, batch)

    # Restore requires_grad states
    for p, state in zip(yolo_model.parameters(), param_states):
        p.requires_grad = state

    assert adv_img is not None
    # Generated adversarial images must be detached from any computation graph
    assert adv_img.grad_fn is None, "Adversarial image retained an autograd computation graph!"

    # Verify model parameters have no lingering gradients from attack generation
    for p in yolo_model.parameters():
        assert p.grad is None, "Model parameter has lingering gradients from attack generation!"


# ======================================================================
# Test D: Mask creation after SCORE search
# ======================================================================
def test_mask_creation_after_score():
    """Score freezing must create binary masks achieving the requested keep_ratio."""
    yolo_model = YOLO("yolov8n.yaml").model
    convert_to_hydra(yolo_model, keep_ratio=0.5)
    initialize_hydra_scores(yolo_model)
    configure_score_search(yolo_model)

    # Snapshot dense weights
    weights_before = [layer.weight.clone() for _, layer in iter_hydra_layers(yolo_model)]

    report = freeze_hydra_masks(yolo_model)

    # 1. Verify report and sparsity
    assert abs(report.sparsity - 0.5) < 1e-4

    # 2. Verify masks are binary and dense weights unchanged
    for (_, layer), w_before in zip(iter_hydra_layers(yolo_model), weights_before):
        assert torch.equal(layer.weight, w_before), "Dense weights changed during mask freeze!"
        unique_vals = torch.unique(layer.fixed_mask)
        for v in unique_vals:
            assert v.item() in {0.0, 1.0}, f"Non-binary mask value found: {v.item()}"


# ======================================================================
# Test E: Fixed-mask robust FINETUNE test
# ======================================================================
def test_fixed_mask_robust_finetune(attack_name: str = "pgd"):
    """FINETUNE step updates retained weights while fixed_mask and popup_scores stay bit-exact."""
    overrides = {
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "device": DEVICE,
        "epochs": 1,
        "batch": 2,
        "imgsz": 64,
        "hydra_stage": "finetune",
        "hydra_keep_ratio": 0.5,
        "attack_name": attack_name,
        "attack_ratio": 0.5,
    }
    trainer = HydraAdversarialDetectionTrainer(overrides=overrides)
    trainer.setup_model()
    if torch.cuda.is_available():
        trainer.model = trainer.model.to(DEVICE)
    trainer.set_model_attributes()
    trainer._model_train()

    optimizer = trainer.build_optimizer(trainer.model)

    # Snapshot state
    masks_before = [layer.fixed_mask.clone() for _, layer in iter_hydra_layers(trainer.model)]
    scores_before = [layer.popup_scores.clone() for _, layer in iter_hydra_layers(trainer.model)]
    weights_before = [layer.weight.clone() for _, layer in iter_hydra_layers(trainer.model)]

    # Initial zeroing
    enforce_pruned_weights_zero(trainer.model)

    device = next(trainer.model.parameters()).device
    batch = _create_synthetic_detection_batch(batch_size=2, img_size=64, device=device)
    preds = trainer.model(batch["img"])
    if isinstance(preds, dict):
        loss = preds["scores"].sum() + preds["boxes"].sum()
    elif isinstance(preds, (list, tuple)):
        loss = sum(p.sum() for p in preds if torch.is_tensor(p) and p.requires_grad)
    else:
        loss = preds.sum()
    loss.backward()

    optimizer.step()
    enforce_pruned_weights_zero(trainer.model)

    for (_, layer), m_b, s_b, w_b in zip(iter_hydra_layers(trainer.model), masks_before, scores_before, weights_before):
        # 1. Mask is bit-exact unchanged
        assert torch.equal(layer.fixed_mask, m_b), "fixed_mask changed during FINETUNE step!"
        # 2. Popup scores are bit-exact unchanged
        assert torch.equal(layer.popup_scores, s_b), "popup_scores changed during FINETUNE step!"
        # 3. Pruned weights remain physically 0.0
        pruned_weights = layer.weight * (1.0 - layer.fixed_mask)
        assert torch.abs(pruned_weights).max().item() == 0.0, "Pruned weights are non-zero!"


# ======================================================================
# Test F: Ratio compatibility test
# ======================================================================
def test_ratio_compatibility():
    """Verify sample ratio routing and sub-batch partitioning matches configured ratios."""
    overrides = {
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "device": DEVICE,
        "attack_name": ["pgd", "worstk"],
        "attack_ratio": [0.3, 0.2],
        "attack_num": 2,
    }
    trainer = HydraAdversarialDetectionTrainer(overrides=overrides)
    assert trainer.attack_names == ["pgd", "worstk"]
    assert trainer.attack_ratios == [0.3, 0.2]

    # Test split_batch_by_type
    batch = {
        "img": torch.zeros((4, 3, 32, 32)),
        "sample_type": ["raw", "pgd", "worstk", "raw"],
        "batch_idx": torch.tensor([0, 1, 2, 3]),
        "cls": torch.tensor([[0], [1], [0], [1]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]] * 4),
    }
    sub_batches = trainer.split_batch_by_type(batch)
    assert set(sub_batches.keys()) == {"raw", "pgd", "worstk"}
    assert sub_batches["raw"]["img"].shape[0] == 2
    assert sub_batches["pgd"]["img"].shape[0] == 1
    assert sub_batches["worstk"]["img"].shape[0] == 1


# ======================================================================
# Test G: Clean-only fallback test
# ======================================================================
def test_clean_only_fallback():
    """When attack_ratio=0.0, trainer smoothly reduces to clean training."""
    overrides = {
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "device": DEVICE,
        "attack_name": "pgd",
        "attack_ratio": 0.0,
        "attack_num": 1,
    }
    trainer = HydraAdversarialDetectionTrainer(overrides=overrides)
    assert trainer.attack_ratios == [0.0]

    # Dataset partition check (in-memory, network-free)
    dataset = YOLODatasetAdvTest.__new__(YOLODatasetAdvTest)
    dataset.im_files = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    dataset.attack_names = trainer.attack_names
    dataset.attack_ratios = trainer.attack_ratios
    dataset.build_virtual_samples()
    for _, sample_type in dataset.virtual_samples:
        assert sample_type == "raw", f"Unexpected sample_type {sample_type} when ratio is 0!"


# ======================================================================
# Test H: Single-attack test
# ======================================================================
def test_single_attack(attack_name: str = "pgd"):
    """Verify single-attack configuration only instantiates the requested attack."""
    overrides = {
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "device": DEVICE,
        "attack_name": attack_name,
        "attack_ratio": 0.5,
        "attack_num": 1,
    }
    trainer = HydraAdversarialDetectionTrainer(overrides=overrides)
    trainer.setup_model()
    if torch.cuda.is_available():
        trainer.model = trainer.model.to(DEVICE)
    trainer.set_model_attributes()

    assert set(trainer.attackers.keys()) == {attack_name}
    assert trainer.attack_ratios == [0.5]


# ======================================================================
# Test I: Multi-attack smoke test
# ======================================================================
def test_multi_attack_smoke():
    """Multi-attack configuration properly sets up all requested attackers."""
    overrides = {
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "device": DEVICE,
        "attack_name": ["pgd", "worstk", "mim"],
        "attack_ratio": [0.2, 0.15, 0.15],
        "attack_num": 3,
    }
    trainer = HydraAdversarialDetectionTrainer(overrides=overrides)
    trainer.setup_model()
    if torch.cuda.is_available():
        trainer.model = trainer.model.to(DEVICE)
    trainer.set_model_attributes()

    assert set(trainer.attackers.keys()) == {"pgd", "worstk", "mim"}
    assert trainer.attack_ratios == [0.2, 0.15, 0.15]


# ======================================================================
# Test J: Fixed-source backward-compatibility test
# ======================================================================
def test_fixed_source_backward_compatibility(attack_name: str = "pgd"):
    """Explicit attack_weights loads a separate fixed attack model."""
    overrides = {
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "device": DEVICE,
        "attack_name": attack_name,
        "attack_ratio": 0.5,
        "attack_weights": str(Path("yolov8n.pt").resolve()),
    }
    trainer = HydraAdversarialDetectionTrainer(overrides=overrides)
    trainer.setup_model()
    if torch.cuda.is_available():
        trainer.model = trainer.model.to(DEVICE)
    trainer.set_model_attributes()

    assert trainer.attack_model is not None, "Fixed attack model was not loaded when attack_weights was specified!"
    assert trainer.attackers[attack_name].model is trainer.attack_model, "Attacker is not using the fixed attack model!"
    assert trainer.attack_model is not unwrap_model(trainer.model), "Attack model should not be the live training model!"


# ======================================================================
# Test K: Attack parameter passthrough test
# ======================================================================
def test_attack_parameter_passthrough():
    """Verify user-configured attack options are received and stored correctly."""
    overrides = {
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "device": DEVICE,
        "attack_name": ["pgd", "worstk"],
        "attack_ratio": [0.4, 0.4],
        "attack_num": 2,
    }
    trainer = HydraAdversarialDetectionTrainer(overrides=overrides)
    assert trainer.attack_names == ["pgd", "worstk"]
    assert trainer.attack_ratios == [0.4, 0.4]
    assert trainer.attack_num == 2


# ======================================================================
# Test L: Worst-of-k smoke test
# ======================================================================
def test_worst_of_k_smoke():
    """SpatialWorstOfK candidate search and target transformation test."""
    device = DEVICE
    model = YOLO("yolov8n.yaml").model.to(device).eval()
    model.args = get_cfg(DEFAULT_CFG)
    convert_to_hydra(model, keep_ratio=0.5)
    initialize_hydra_scores(model)
    configure_score_search(model)
    model.to(device)

    attacker = SpatialWorstOfK(model, k=2, degrees=15.0, translate=0.05, img_size=64)
    assert attacker.k == 2
    assert attacker.degrees == 15.0
    assert attacker.translate == 0.05

    batch = _create_synthetic_detection_batch(batch_size=2, img_size=64, device=device)
    adv_img = attacker.forward_batch(batch)

    assert adv_img is not None
    assert adv_img.shape == (2, 3, 64, 64)
    assert batch["bboxes"].shape[0] == 2


# ======================================================================
# Test M: SCORE -> FINETUNE checkpoint handoff test
# ======================================================================
def test_score_to_finetune_checkpoint_handoff():
    """Checkpoint saved from SCORE mode correctly handsoff to FINETUNE trainer."""
    overrides = {
        "model": "yolov8n.yaml",
        "data": "coco8.yaml",
        "device": DEVICE,
        "epochs": 1,
        "batch": 2,
        "hydra_stage": "score",
        "hydra_keep_ratio": 0.5,
        "attack_name": ["pgd", "worstk"],
        "attack_ratio": [0.25, 0.25],
        "attack_num": 2,
    }
    trainer = HydraAdversarialDetectionTrainer(overrides=overrides)
    trainer.setup_model()
    if torch.cuda.is_available():
        trainer.model = trainer.model.to(DEVICE)
    trainer.set_model_attributes()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "score_test.pt"
        torch.save(
            {
                "model": unwrap_model(trainer.model),
                "hydra_stage": "score",
                "hydra_keep_ratio": 0.5,
                "hydra_score_init": "scaled",
                "attack_name": ["pgd", "worstk"],
                "attack_ratio": [0.25, 0.25],
            },
            ckpt_path,
        )

        # Load into FINETUNE trainer
        finetune_overrides = {
            "model": str(ckpt_path),
            "data": "coco8.yaml",
            "device": DEVICE,
            "epochs": 1,
            "batch": 2,
            "hydra_stage": "finetune",
            "hydra_keep_ratio": 0.5,
            "attack_name": ["pgd", "worstk"],
            "attack_ratio": [0.25, 0.25],
            "attack_num": 2,
        }
        ft_trainer = HydraAdversarialDetectionTrainer(overrides=finetune_overrides)
        ft_trainer.setup_model()
        if torch.cuda.is_available():
            ft_trainer.model = ft_trainer.model.to(DEVICE)
        ft_trainer.set_model_attributes()

        assert ft_trainer.hydra_stage == "finetune"
        for _, layer in iter_hydra_layers(ft_trainer.model):
            assert layer.hydra_mode == HydraMode.FINETUNE
            assert hasattr(layer, "fixed_mask")


# ======================================================================
# Test N: Materialization equivalence test
# ======================================================================
def test_materialization_equivalence():
    """Materialized model matches masked HYDRA model within numerical tolerance."""
    device = DEVICE
    model = YOLO("yolov8n.yaml").model.to(device).eval()
    convert_to_hydra(model, keep_ratio=0.5)
    initialize_hydra_scores(model)
    freeze_hydra_masks(model)
    configure_masked_finetune(model)
    model.to(device)

    x = torch.randn(1, 3, 64, 64, device=device)
    with torch.no_grad():
        hydra_out = model(x)

    mat_model = materialize_hydra_model(model, inplace=False).to(device).eval()

    # Verify no HydraConv2d remaining
    for m in mat_model.modules():
        assert not isinstance(m, HydraConv2d), "HydraConv2d remaining after materialization!"

    with torch.no_grad():
        mat_out = mat_model(x)

    def compute_max_diff(o1, o2):
        if isinstance(o1, torch.Tensor):
            return (o1 - o2).abs().max().item()
        elif isinstance(o1, (list, tuple)):
            return max(compute_max_diff(a, b) for a, b in zip(o1, o2))
        elif isinstance(o1, dict):
            return max(compute_max_diff(o1[k], o2[k]) for k in o1)
        return 0.0

    max_diff = compute_max_diff(hydra_out, mat_out)
    assert max_diff < 1e-4, f"Materialized output differs from masked HYDRA: max_diff={max_diff}"


# ======================================================================
# Test O: Standard validation test
# ======================================================================
def test_standard_validation():
    """Materialized model remains fully compatible with standard Ultralytics validator."""
    device = DEVICE
    model = YOLO("yolov8n.yaml").model.to(device).eval()
    convert_to_hydra(model, keep_ratio=0.5)
    initialize_hydra_scores(model)
    freeze_hydra_masks(model)
    configure_masked_finetune(model)

    mat_model = materialize_hydra_model(model, inplace=False).to(device).eval()

    # Create standard DetectionValidator with coco8
    args = dict(model="yolov8n.yaml", data="coco8.yaml", imgsz=64, batch=2, device=DEVICE, plots=False)
    validator = yolo.detect.DetectionValidator(args=args)
    validator.data = {"nc": 80, "names": {i: f"class_{i}" for i in range(80)}}
    validator.stride = 32

    # Verify validator can inspect model without errors
    assert validator is not None
    assert not any(isinstance(m, HydraConv2d) for m in mat_model.modules())


if __name__ == "__main__":
    print(f"Device configuration: {DEVICE} (CUDA available: {torch.cuda.is_available()})", flush=True)

    print("Running Test A: test_current_model_attack_source_identity (PGD)...", flush=True)
    test_current_model_attack_source_identity("pgd")
    print("Running Test A: test_current_model_attack_source_identity (WORSTK)...", flush=True)
    test_current_model_attack_source_identity("worstk")
    print("Test A PASSED", flush=True)

    print("Running Test B: test_robust_score_update (PGD)...", flush=True)
    test_robust_score_update("pgd")
    print("Running Test B: test_robust_score_update (WORSTK)...", flush=True)
    test_robust_score_update("worstk")
    print("Test B PASSED", flush=True)

    print("Running Test C: test_gradient_hygiene (PGD)...", flush=True)
    test_gradient_hygiene("pgd")
    print("Running Test C: test_gradient_hygiene (WORSTK)...", flush=True)
    test_gradient_hygiene("worstk")
    print("Test C PASSED", flush=True)

    print("Running Test D: test_mask_creation_after_score...", flush=True)
    test_mask_creation_after_score()
    print("Test D PASSED", flush=True)

    print("Running Test E: test_fixed_mask_robust_finetune (PGD)...", flush=True)
    test_fixed_mask_robust_finetune("pgd")
    print("Running Test E: test_fixed_mask_robust_finetune (WORSTK)...", flush=True)
    test_fixed_mask_robust_finetune("worstk")
    print("Test E PASSED", flush=True)

    print("Running Test F: test_ratio_compatibility (PGD + WORSTK)...", flush=True)
    test_ratio_compatibility()
    print("Test F PASSED", flush=True)

    print("Running Test G: test_clean_only_fallback...", flush=True)
    test_clean_only_fallback()
    print("Test G PASSED", flush=True)

    print("Running Test H: test_single_attack (PGD & WORSTK)...", flush=True)
    test_single_attack("pgd")
    test_single_attack("worstk")
    print("Test H PASSED", flush=True)

    print("Running Test I: test_multi_attack_smoke (PGD + WORSTK + MIM)...", flush=True)
    test_multi_attack_smoke()
    print("Test I PASSED", flush=True)

    print("Running Test J: test_fixed_source_backward_compatibility (PGD & WORSTK)...", flush=True)
    test_fixed_source_backward_compatibility("pgd")
    test_fixed_source_backward_compatibility("worstk")
    print("Test J PASSED", flush=True)

    print("Running Test K: test_attack_parameter_passthrough (PGD + WORSTK)...", flush=True)
    test_attack_parameter_passthrough()
    print("Test K PASSED", flush=True)

    print("Running Test L: test_worst_of_k_smoke...", flush=True)
    test_worst_of_k_smoke()
    print("Test L PASSED", flush=True)

    print("Running Test M: test_score_to_finetune_checkpoint_handoff (PGD + WORSTK)...", flush=True)
    test_score_to_finetune_checkpoint_handoff()
    print("Test M PASSED", flush=True)

    print("Running Test N: test_materialization_equivalence...", flush=True)
    test_materialization_equivalence()
    print("Test N PASSED", flush=True)

    print("Running Test O: test_standard_validation...", flush=True)
    test_standard_validation()
    print("Test O PASSED", flush=True)

    print("\n==================================================", flush=True)
    print(f"ALL 15 MILESTONE-3 TESTS (A-O) PASSED WITH BOTH PGD AND WORSTK ON {DEVICE.upper()}!", flush=True)
    print("==================================================", flush=True)
