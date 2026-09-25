"""Trainer-level smoke test for HYDRA Milestone 2.

This test complements m1test.py and m2test.py by exercising the real
Ultralytics DetectionTrainer lifecycle, real YOLO detection loss, checkpoint
handoff, masked fine-tuning, materialization, and normal validation.

Expected Milestone-2 API:
    ultralytics.models.yolo.detect.hydra_train.HydraDetectionTrainer
    ultralytics.nn.pruning.iter_hydra_layers
    ultralytics.nn.pruning.HydraMode
    ultralytics.nn.pruning.hydra_mask.freeze_hydra_masks
    ultralytics.nn.pruning.hydra_mask.enforce_pruned_weights_zero
    ultralytics.nn.pruning.hydra_mask.set_hydra_keep_ratio
    ultralytics.nn.pruning.hydra_export.materialize_hydra_model

If your implementation used different names, change only the import/adapter
section and config-key names. Do not weaken the assertions.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.models.yolo.detect.hydra_train import HydraDetectionTrainer
from ultralytics.nn.pruning import HydraMode, iter_hydra_layers
from ultralytics.nn.pruning.hydra_export import materialize_hydra_model
from ultralytics.nn.pruning.hydra_mask import (
    enforce_pruned_weights_zero,
    freeze_hydra_masks,
    set_hydra_keep_ratio,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = "yolo12s.pt"
DATA = "coco8.yaml"
IMGSZ = 320
BATCH = 2
WORKERS = 2
SCORE_EPOCHS = 2
FINETUNE_EPOCHS = 2
KEEP_RATIO = 0.50
DEVICE = "0" if torch.cuda.is_available() else "cpu"
PROJECT = "runs/hydra_m2_trainer_test"
SCORE_NAME = "score_smoke"
FINETUNE_NAME = "finetune_smoke"
OUTPUT_TOLERANCE = 1e-3
CLEAN_PREVIOUS_RUNS = True


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model if DDP/DataParallel wrapping is present."""
    return model.module if hasattr(model, "module") else model


def compute_max_diff(a: Any, b: Any) -> float:
    """Recursively compute maximum absolute difference between model outputs."""
    if isinstance(a, torch.Tensor):
        return (a - b).abs().max().item()
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max((compute_max_diff(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((compute_max_diff(a[k], b[k]) for k in a), default=0.0)
    return 0.0


def snapshot_scores(model):
    model = unwrap_model(model)
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if "popup_scores" in n
    }


def snapshot_non_scores(model):
    model = unwrap_model(model)
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if "popup_scores" not in n
    }


def snapshot_masks(model):
    model = unwrap_model(model)
    return {
        n: layer.fixed_mask.detach().cpu().clone()
        for n, layer in iter_hydra_layers(model)
    }


def snapshot_hydra_weights(model):
    model = unwrap_model(model)
    return {
        n: layer.weight.detach().cpu().clone()
        for n, layer in iter_hydra_layers(model)
    }


def changed_names(before, after):
    assert before.keys() == after.keys(), "Snapshot keys changed unexpectedly."
    return [n for n in before if not torch.equal(before[n], after[n])]


def optimizer_parameter_names(trainer):
    model = unwrap_model(trainer.model)
    ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    return [n for n, p in model.named_parameters() if id(p) in ids]


def assert_binary_masks(model):
    for name, layer in iter_hydra_layers(unwrap_model(model)):
        values = set(torch.unique(layer.fixed_mask.detach()).cpu().tolist())
        assert values.issubset({0.0, 1.0}), f"Non-binary mask in {name}: {values}"


def assert_layerwise_density(model, expected):
    errors = []
    for name, layer in iter_hydra_layers(unwrap_model(model)):
        mask = layer.fixed_mask.detach()
        total = mask.numel()
        density = float(mask.sum().item()) / total
        allowed = 1.0 / total + 1e-12
        if abs(density - expected) > allowed:
            errors.append((name, total, density))
    assert not errors, "Layer-wise density mismatch:\n" + "\n".join(map(str, errors[:20]))


def assert_pruned_weights_zero(model):
    bad = []
    for name, layer in iter_hydra_layers(unwrap_model(model)):
        pruned = layer.weight.detach()[layer.fixed_mask == 0]
        if pruned.numel() and torch.count_nonzero(pruned).item() != 0:
            bad.append(name)
    assert not bad, "Pruned weights became non-zero:\n" + "\n".join(bad[:20])


def assert_no_hydra_state(model):
    assert len(list(iter_hydra_layers(model))) == 0, "HydraConv2d remains after materialization."
    assert not [n for n, _ in model.named_parameters() if "popup_scores" in n]
    assert not [n for n, _ in model.named_buffers() if "fixed_mask" in n]


@dataclass
class ScoreAudit:
    """Audit the real trainer while it runs the SCORE stage."""

    started: bool = False
    checked_batch: bool = False
    scores_before: dict[str, torch.Tensor] = field(default_factory=dict)
    weights_before: dict[str, torch.Tensor] = field(default_factory=dict)

    def on_train_start(self, trainer):
        model = unwrap_model(trainer.model)
        layers = list(iter_hydra_layers(model))
        assert layers, "No HYDRA layers at on_train_start."
        assert all(layer.hydra_mode == HydraMode.SCORE for _, layer in layers)

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trainable
        assert all("popup_scores" in n for n in trainable), (
            "Non-score trainable parameters:\n" +
            "\n".join(n for n in trainable if "popup_scores" not in n)
        )

        opt_names = optimizer_parameter_names(trainer)
        assert opt_names, "Score optimizer is empty."
        bad = [n for n in opt_names if "popup_scores" not in n]
        assert not bad, "Score optimizer contains non-score parameters:\n" + "\n".join(bad[:50])

        self.scores_before = snapshot_scores(model)
        self.weights_before = snapshot_non_scores(model)
        self.started = True
        print(f"[PASS] SCORE trainer: {len(layers)} HYDRA layers, optimizer contains scores only.")

    def on_train_batch_start(self, trainer):
        # BaseTrainer may call model.train() every epoch. Check the BN policy at
        # the point where a real training batch is about to execute.
        if self.checked_batch:
            return
        bad = []
        for name, m in unwrap_model(trainer.model).named_modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                affine = ((m.weight is not None and m.weight.requires_grad) or
                          (m.bias is not None and m.bias.requires_grad))
                if affine or m.training:
                    bad.append((name, affine, m.training))
        assert not bad, "BatchNorm was re-enabled by trainer lifecycle:\n" + "\n".join(map(str, bad[:20]))
        self.checked_batch = True
        print("[PASS] BatchNorm remains frozen/eval inside the real training loop.")

    def verify(self, trainer):
        assert self.started
        changed_scores = changed_names(self.scores_before, snapshot_scores(trainer.model))
        changed_weights = changed_names(self.weights_before, snapshot_non_scores(trainer.model))
        assert changed_scores, "No popup_scores changed during real score training."
        assert not changed_weights, "Ordinary weights changed during SCORE:\n" + "\n".join(changed_weights[:50])
        print(f"[PASS] SCORE stage changed {len(changed_scores)} score tensors and 0 ordinary parameters.")


@dataclass
class FinetuneAudit:
    """Audit the real trainer while it runs fixed-mask fine-tuning."""

    started: bool = False
    scores_before: dict[str, torch.Tensor] = field(default_factory=dict)
    masks_before: dict[str, torch.Tensor] = field(default_factory=dict)
    weights_before: dict[str, torch.Tensor] = field(default_factory=dict)

    def on_train_start(self, trainer):
        model = unwrap_model(trainer.model)
        layers = list(iter_hydra_layers(model))
        assert layers, "No HYDRA layers in FINETUNE trainer."
        assert all(layer.hydra_mode == HydraMode.FINETUNE for _, layer in layers)
        assert not [n for n, p in model.named_parameters() if "popup_scores" in n and p.requires_grad]
        assert not [n for n in optimizer_parameter_names(trainer) if "popup_scores" in n]

        assert_binary_masks(model)
        assert_layerwise_density(model, KEEP_RATIO)
        enforce_pruned_weights_zero(model)
        assert_pruned_weights_zero(model)

        self.scores_before = snapshot_scores(model)
        self.masks_before = snapshot_masks(model)
        self.weights_before = snapshot_hydra_weights(model)
        self.started = True
        print(f"[PASS] FINETUNE trainer loaded {len(layers)} fixed-mask HYDRA layers.")

    def verify(self, trainer):
        assert self.started
        changed_scores = changed_names(self.scores_before, snapshot_scores(trainer.model))
        changed_masks = changed_names(self.masks_before, snapshot_masks(trainer.model))
        changed_weights = changed_names(self.weights_before, snapshot_hydra_weights(trainer.model))
        assert not changed_scores, "popup_scores changed during FINETUNE."
        assert not changed_masks, "fixed_mask changed during FINETUNE."
        assert changed_weights, "No HYDRA weights changed during FINETUNE."
        assert_pruned_weights_zero(trainer.model)
        print(f"[PASS] FINETUNE changed {len(changed_weights)} weight tensors; scores/masks stayed fixed.")
        print("[PASS] All pruned weights remain exactly zero.")


def common_overrides():
    """Ultralytics/HYDRA options shared by both smoke-test stages."""
    return {
        "model": MODEL_PATH,
        "data": DATA,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "nbs": BATCH,
        "amp": False,
        "workers": WORKERS,
        "device": DEVICE,
        "project": PROJECT,
        "seed": 42,
        "deterministic": True,
        "plots": False,
        "compile": False,
        "warmup_epochs": 0.0,
        "hydra": True,
        "hydra_keep_ratio": KEEP_RATIO,
        "hydra_score_init": "scaled",
        "hydra_global_pruning": False,
        "hydra_use_ema": False,
    }


def build_score_trainer():
    opts = common_overrides()
    opts.update({"epochs": SCORE_EPOCHS, "name": SCORE_NAME, "hydra_stage": "score", "resume": False})
    return HydraDetectionTrainer(overrides=opts)


def build_finetune_trainer(score_checkpoint):
    opts = common_overrides()
    opts.update({
        "model": str(score_checkpoint),
        "epochs": FINETUNE_EPOCHS,
        "name": FINETUNE_NAME,
        "hydra_stage": "finetune",
        "resume": False,
    })
    return HydraDetectionTrainer(overrides=opts)


def run_score_stage():
    print("\n" + "=" * 76)
    print("STAGE A: REAL HYDRA SCORE SEARCH (coco8)")
    print("=" * 76)

    trainer = build_score_trainer()
    audit = ScoreAudit()
    trainer.add_callback("on_train_start", audit.on_train_start)
    trainer.add_callback("on_train_batch_start", audit.on_train_batch_start)
    trainer.train()
    audit.verify(trainer)

    model = unwrap_model(trainer.model)
    set_hydra_keep_ratio(model, KEEP_RATIO)
    report = freeze_hydra_masks(model)
    assert_binary_masks(model)
    assert_layerwise_density(model, KEEP_RATIO)

    if report is not None and hasattr(report, "sparsity"):
        expected = 1.0 - KEEP_RATIO
        print(f"Global reported sparsity: {report.sparsity:.6f}")
        assert abs(report.sparsity - expected) < 1e-3

    print(f"[PASS] Frozen masks retain approximately {KEEP_RATIO:.2%} per selected layer.")

    # Save AFTER masks are frozen, so a fresh FINETUNE trainer must reconstruct
    # the real score/mask state from disk rather than reusing an in-memory model.
    trainer.save_model()
    ckpt = Path(trainer.last)
    assert ckpt.exists(), f"Score checkpoint not found: {ckpt}"
    print(f"[PASS] Score/mask checkpoint: {ckpt}")
    return ckpt


def run_finetune_stage(score_checkpoint):
    print("\n" + "=" * 76)
    print("STAGE B: REAL FIXED-MASK FINETUNING (coco8)")
    print("=" * 76)

    trainer = build_finetune_trainer(score_checkpoint)
    audit = FinetuneAudit()
    trainer.add_callback("on_train_start", audit.on_train_start)
    trainer.train()
    audit.verify(trainer)

    assert Path(trainer.last).exists(), f"Finetune checkpoint not found: {trainer.last}"
    print(f"[PASS] Finetune checkpoint: {trainer.last}")
    return trainer


def run_materialize_and_val(finetune_trainer):
    print("\n" + "=" * 76)
    print("STAGE C: MATERIALIZE + STANDARD ULTRALYTICS VALIDATION")
    print("=" * 76)

    hydra_model = unwrap_model(finetune_trainer.model).eval()
    device = next(hydra_model.parameters()).device

    torch.manual_seed(123)
    x = torch.randn(1, 3, IMGSZ, IMGSZ, device=device)
    with torch.no_grad():
        y_hydra = hydra_model(x)

    materialized = materialize_hydra_model(hydra_model, inplace=False).to(device).eval()
    assert_no_hydra_state(materialized)

    with torch.no_grad():
        y_dense = materialized(x)

    diff = compute_max_diff(y_hydra, y_dense)
    assert diff < OUTPUT_TOLERANCE, f"Materialization changed output: max_diff={diff}"
    print(f"[PASS] Materialization equivalence max_diff={diff:.3e}")
    print("[PASS] No HYDRA-only state remains after materialization.")

    # Exercise the ordinary Ultralytics validation path with the materialized
    # plain DetectionModel. This test is about compatibility, not final mAP.
    wrapper = YOLO(MODEL_PATH)
    wrapper.model = materialized
    metrics = wrapper.val(
        data=DATA,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        plots=False,
        verbose=False,
    )

    print("[PASS] Standard YOLO.val() completed successfully.")
    if hasattr(metrics, "box"):
        if hasattr(metrics.box, "map50"):
            print(f"mAP50:    {float(metrics.box.map50):.6f}")
        if hasattr(metrics.box, "map"):
            print(f"mAP50-95: {float(metrics.box.map):.6f}")


def main():
    print("=" * 76)
    print("HYDRA MILESTONE-2 TRAINER-LEVEL SMOKE TEST")
    print("=" * 76)
    print(f"Model:        {MODEL_PATH}")
    print(f"Dataset:      {DATA}")
    print(f"Device:       {DEVICE}")
    print(f"Image size:   {IMGSZ}")
    print(f"Batch size:   {BATCH}")
    print(f"Keep ratio:   {KEEP_RATIO:.2f}")
    print(f"Prune ratio:  {1.0 - KEEP_RATIO:.2f}")
    print(f"Score epochs: {SCORE_EPOCHS}")
    print(f"FT epochs:    {FINETUNE_EPOCHS}")

    if CLEAN_PREVIOUS_RUNS and Path(PROJECT).exists():
        print(f"Removing previous test directory: {PROJECT}")
        shutil.rmtree(PROJECT)

    score_checkpoint = run_score_stage()
    finetune_trainer = run_finetune_stage(score_checkpoint)
    run_materialize_and_val(finetune_trainer)

    print("\n" + "=" * 76)
    print("ALL MILESTONE-2 TRAINER-LEVEL SMOKE TESTS PASSED")
    print("=" * 76)
    print(
        "Verified real trainer lifecycle, real YOLO detection loss, score-only "
        "optimization, BN freeze during SCORE, fixed masks, checkpoint handoff, "
        "masked fine-tuning, zero enforcement, materialization, and standard val."
    )


if __name__ == "__main__":
    main()


"""
SHARCNET example
----------------
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 opencv/4.11.0
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:$PYTHONPATH
source .yolo_env/bin/activate
python m2_trainer_test.py

Notes
-----
1. The first run may download coco8 if it is not cached.
2. This is a correctness smoke test, not a pruning-performance benchmark.
3. If your HydraDetectionTrainer config keys differ, edit only common_overrides(),
   build_score_trainer(), and build_finetune_trainer().
4. Do not weaken the parameter/mask assertions just to make the test pass.
"""
