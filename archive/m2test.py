import copy
import io
import torch
from torch import nn
from ultralytics import YOLO

from ultralytics.nn.pruning import (
    convert_to_hydra,
    initialize_hydra_scores,
    iter_hydra_layers,
    HydraMode,
)
from ultralytics.nn.pruning.hydra_mask import (
    configure_score_search,
    configure_masked_finetune,
    enforce_pruned_weights_zero,
    freeze_hydra_masks,
    set_hydra_keep_ratio,
)
from ultralytics.nn.pruning.hydra_export import materialize_hydra_model


# ============================================================
# Milestone 2 standalone correctness test
#
# Purpose:
#   Verify the core HYDRA Milestone-2 mechanics before integrating
#   adversarial PGD/BIM/MIM training in Milestone 3.
#
# This test checks:
#   1. Score-search trains popup_scores only.
#   2. Dense weights and BatchNorm stay frozen during score search.
#   3. Frozen masks achieve the requested layer-wise keep ratio.
#   4. Fixed masks do not change during fine-tuning.
#   5. popup_scores do not change during fine-tuning.
#   6. Retained weights can update during fine-tuning.
#   7. Pruned weights stay exactly zero.
#   8. HYDRA state_dict can restore scores/masks exactly.
#   9. A materialized plain-Ultralytics model matches the masked HYDRA model.
#
# Important:
#   This script tests the model/pruning mechanics directly with a synthetic
#   differentiable loss. It does NOT replace a later trainer-level smoke test
#   on coco8 or your real dataset.
# ============================================================


# -----------------------------
# User-adjustable configuration
# -----------------------------
MODEL_PATH = "yolo12l.pt"

# 0.50 means KEEP 50% and PRUNE 50%.
KEEP_RATIO = 0.50

# 320 is enough for a mechanics test and is faster than 640.
# Change to 640 if you want to match the normal training resolution.
TEST_IMGSZ = 320

TOLERANCE = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_max_diff(out1, out2):
    """Recursively compute the maximum absolute difference between model outputs."""
    if isinstance(out1, torch.Tensor):
        return (out1 - out2).abs().max().item()

    if isinstance(out1, (list, tuple)):
        assert len(out1) == len(out2), "Output sequence lengths do not match."
        if len(out1) == 0:
            return 0.0
        return max(compute_max_diff(a, b) for a, b in zip(out1, out2))

    if isinstance(out1, dict):
        assert out1.keys() == out2.keys(), "Output dictionary keys do not match."
        if len(out1) == 0:
            return 0.0
        return max(compute_max_diff(out1[k], out2[k]) for k in out1)

    return 0.0


def differentiable_scalar(output):
    """Build a small synthetic scalar loss from all differentiable output tensors.

    The goal is not to reproduce the YOLO detection loss here. The goal is to
    verify HYDRA parameter-update semantics without needing labels/dataloader.
    """
    tensors = []

    def collect(obj):
        if isinstance(obj, torch.Tensor):
            if obj.is_floating_point() and obj.requires_grad:
                tensors.append(obj)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                collect(x)
        elif isinstance(obj, dict):
            for x in obj.values():
                collect(x)

    collect(output)

    if not tensors:
        raise RuntimeError(
            "No differentiable output tensor was found. "
            "Check the model forward path and HYDRA graph."
        )

    # Mean keeps the scalar numerically small while preserving gradients.
    return sum(t.float().mean() for t in tensors)


def snapshot_named_parameters(model, include_scores=True):
    """Clone named parameters for exact before/after comparison."""
    result = {}
    for name, p in model.named_parameters():
        if include_scores or "popup_scores" not in name:
            result[name] = p.detach().cpu().clone()
    return result


def snapshot_scores(model):
    """Clone all popup_scores tensors."""
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if "popup_scores" in name
    }


def snapshot_masks(model):
    """Clone all fixed HYDRA masks."""
    return {
        name: layer.fixed_mask.detach().cpu().clone()
        for name, layer in iter_hydra_layers(model)
    }


def count_changed(before, after):
    """Return how many same-named tensors changed exactly."""
    changed = []
    for name, tensor_before in before.items():
        tensor_after = after[name]
        if not torch.equal(tensor_before, tensor_after):
            changed.append(name)
    return changed


def print_optimizer_membership(model, optimizer):
    """Print and return names of parameters included in optimizer.param_groups."""
    optimizer_ids = {
        id(p)
        for group in optimizer.param_groups
        for p in group["params"]
    }

    names = [
        name
        for name, p in model.named_parameters()
        if id(p) in optimizer_ids
    ]

    print(f"Optimizer contains {len(names)} named parameters.")
    for name in names[:10]:
        print(f"  {name}")
    if len(names) > 10:
        print(f"  ... ({len(names) - 10} more)")

    return names


# ============================================================
# 1. Load and convert YOLO
# ============================================================
print(f"Device: {DEVICE}")
print(f"Loading model weights: {MODEL_PATH} ...")

yolo = YOLO(MODEL_PATH)
hydra_model = copy.deepcopy(yolo.model).to(DEVICE).eval()

print("\nConverting selected nn.Conv2d leaves to HydraConv2d ...")
convert_to_hydra(hydra_model, keep_ratio=KEEP_RATIO)
initialize_hydra_scores(hydra_model)
set_hydra_keep_ratio(hydra_model, KEEP_RATIO)

hydra_layers = list(iter_hydra_layers(hydra_model))
assert len(hydra_layers) > 0, "[FAIL] No HydraConv2d layer was found."
print(f"[PASS] Found {len(hydra_layers)} HydraConv2d layers.")

torch.manual_seed(42)
dummy_img = torch.randn(
    1, 3, TEST_IMGSZ, TEST_IMGSZ,
    device=DEVICE,
)


# ============================================================
# 2. SCORE SEARCH: verify only popup_scores are trainable/updated
# ============================================================
print("\n============================================================")
print("TEST 1: SCORE SEARCH PARAMETER FREEZING")
print("============================================================")

configure_score_search(hydra_model)

for _, layer in hydra_layers:
    assert layer.hydra_mode == HydraMode.SCORE, \
        "[FAIL] A HYDRA layer is not in SCORE mode."

trainable_names = [
    name for name, p in hydra_model.named_parameters()
    if p.requires_grad
]

assert len(trainable_names) > 0, "[FAIL] No trainable parameter exists in SCORE mode."
assert all("popup_scores" in name for name in trainable_names), (
    "[FAIL] SCORE mode has trainable non-score parameters:\n"
    + "\n".join(name for name in trainable_names if "popup_scores" not in name)
)

print(f"[PASS] All {len(trainable_names)} trainable parameters are popup_scores only.")


# ============================================================
# 3. SCORE SEARCH: verify BatchNorm is frozen
# ============================================================
print("\n============================================================")
print("TEST 2: BATCHNORM FREEZE POLICY")
print("============================================================")

bn_modules = [
    (name, m)
    for name, m in hydra_model.named_modules()
    if isinstance(m, nn.modules.batchnorm._BatchNorm)
]

bad_bn = []
for name, bn in bn_modules:
    affine_trainable = (
        (bn.weight is not None and bn.weight.requires_grad)
        or (bn.bias is not None and bn.bias.requires_grad)
    )
    running_training = bn.training

    if affine_trainable or running_training:
        bad_bn.append((name, affine_trainable, running_training))

assert not bad_bn, (
    "[FAIL] Some BatchNorm layers are not fully frozen in SCORE mode:\n"
    + "\n".join(str(x) for x in bad_bn[:20])
)

print(f"[PASS] {len(bn_modules)} BatchNorm layers have frozen affine parameters and eval-mode statistics.")


# ============================================================
# 4. SCORE SEARCH: optimizer must contain popup_scores only
# ============================================================
print("\n============================================================")
print("TEST 3: SCORE OPTIMIZER MEMBERSHIP")
print("============================================================")

score_params = [
    p
    for name, p in hydra_model.named_parameters()
    if "popup_scores" in name
]

assert len(score_params) > 0, "[FAIL] No popup_scores parameters were found."

# This direct optimizer intentionally tests the pruning mechanics.
# Your HydraDetectionTrainer.build_optimizer() should satisfy the same invariant.
score_optimizer = torch.optim.SGD(score_params, lr=1e-2)

optimizer_names = print_optimizer_membership(hydra_model, score_optimizer)

assert optimizer_names, "[FAIL] SCORE optimizer is empty."
assert all("popup_scores" in name for name in optimizer_names), (
    "[FAIL] SCORE optimizer contains normal YOLO parameters."
)

print("[PASS] SCORE optimizer contains popup_scores only.")


# ============================================================
# 5. SCORE SEARCH: one real backward/optimizer step
# ============================================================
print("\n============================================================")
print("TEST 4: SCORE SEARCH ONE-STEP UPDATE")
print("============================================================")

dense_before = {
    name: p.detach().cpu().clone()
    for name, p in hydra_model.named_parameters()
    if "popup_scores" not in name
}
scores_before = snapshot_scores(hydra_model)

score_optimizer.zero_grad(set_to_none=True)

output = hydra_model(dummy_img)
loss = differentiable_scalar(output)

print(f"Synthetic score-search loss: {loss.item():.6f}")

loss.backward()

# Gradient sanity check before stepping.
missing_grad = []
zero_grad = []
nonzero_grad = []

for name, p in hydra_model.named_parameters():
    if "popup_scores" not in name:
        continue

    if p.grad is None:
        missing_grad.append(name)
    elif torch.count_nonzero(p.grad).item() == 0:
        zero_grad.append(name)
    else:
        nonzero_grad.append(name)

assert len(nonzero_grad) > 0, (
    "[FAIL] No popup_scores tensor received a non-zero gradient. "
    "Check the STE backward path."
)

print(
    f"popup_scores gradients: "
    f"{len(nonzero_grad)} non-zero, "
    f"{len(zero_grad)} all-zero, "
    f"{len(missing_grad)} missing"
)

score_optimizer.step()

dense_after = {
    name: p.detach().cpu().clone()
    for name, p in hydra_model.named_parameters()
    if "popup_scores" not in name
}
scores_after = snapshot_scores(hydra_model)

changed_dense = count_changed(dense_before, dense_after)
changed_scores = count_changed(scores_before, scores_after)

assert len(changed_dense) == 0, (
    "[FAIL] Dense YOLO parameters changed during SCORE search:\n"
    + "\n".join(changed_dense[:20])
)

assert len(changed_scores) > 0, (
    "[FAIL] popup_scores did not change after SCORE optimizer.step()."
)

print("[PASS] Dense YOLO parameters remained bit-exact unchanged.")
print(f"[PASS] {len(changed_scores)} popup_scores tensors changed after one optimization step.")


# ============================================================
# 6. Freeze masks and verify requested layer-wise sparsity
# ============================================================
print("\n============================================================")
print("TEST 5: FIXED MASK AND LAYER-WISE SPARSITY")
print("============================================================")

set_hydra_keep_ratio(hydra_model, KEEP_RATIO)
report = freeze_hydra_masks(hydra_model)

layer_density_errors = []

for name, layer in hydra_layers:
    mask = layer.fixed_mask.detach()

    assert not mask.requires_grad, f"[FAIL] fixed_mask requires gradients: {name}"

    unique_values = torch.unique(mask).detach().cpu()
    allowed = set(unique_values.tolist())

    assert allowed.issubset({0.0, 1.0}), (
        f"[FAIL] Mask is not binary in {name}: values={allowed}"
    )

    total = mask.numel()
    kept = int(mask.sum().item())
    density = kept / total

    # Because top-k uses an integer count, a layer may differ from the requested
    # density by at most approximately one element.
    max_integer_rounding_error = 1.0 / total + 1e-12

    if abs(density - KEEP_RATIO) > max_integer_rounding_error:
        layer_density_errors.append(
            (name, total, kept, density, KEEP_RATIO)
        )

assert not layer_density_errors, (
    "[FAIL] Some layers do not match the requested layer-wise keep ratio:\n"
    + "\n".join(str(x) for x in layer_density_errors[:20])
)

print(
    f"[PASS] Every HYDRA layer has binary layer-wise density ≈ {KEEP_RATIO:.2f}."
)

if report is not None and hasattr(report, "sparsity"):
    print(f"Global reported mask sparsity: {report.sparsity:.6f}")
    expected_sparsity = 1.0 - KEEP_RATIO
    assert abs(report.sparsity - expected_sparsity) < 1e-3, (
        f"[FAIL] Global sparsity mismatch: "
        f"actual={report.sparsity}, expected≈{expected_sparsity}"
    )
    print(f"[PASS] Global sparsity ≈ {expected_sparsity:.2f}.")


# ============================================================
# 7. FINETUNE mode: masks/scores frozen, retained weights update
# ============================================================
print("\n============================================================")
print("TEST 6: MASKED FINE-TUNING ONE-STEP UPDATE")
print("============================================================")

configure_masked_finetune(hydra_model)

for _, layer in hydra_layers:
    assert layer.hydra_mode == HydraMode.FINETUNE, \
        "[FAIL] A HYDRA layer is not in FINETUNE mode."
    assert not layer.popup_scores.requires_grad, \
        "[FAIL] popup_scores is still trainable in FINETUNE mode."

# Make the checkpoint state physically sparse before taking snapshots.
enforce_pruned_weights_zero(hydra_model)

masks_before_ft = snapshot_masks(hydra_model)
scores_before_ft = snapshot_scores(hydra_model)

hydra_weight_before_ft = {
    name: layer.weight.detach().cpu().clone()
    for name, layer in hydra_layers
}

finetune_params = [
    p for name, p in hydra_model.named_parameters()
    if p.requires_grad and "popup_scores" not in name
]

assert finetune_params, "[FAIL] No normal parameter is trainable in FINETUNE mode."

finetune_optimizer = torch.optim.SGD(finetune_params, lr=1e-4)

finetune_optimizer.zero_grad(set_to_none=True)

output_ft = hydra_model(dummy_img)
loss_ft = differentiable_scalar(output_ft)

print(f"Synthetic fine-tuning loss: {loss_ft.item():.6f}")

loss_ft.backward()
finetune_optimizer.step()

# Required Milestone-2 invariant.
enforce_pruned_weights_zero(hydra_model)

masks_after_ft = snapshot_masks(hydra_model)
scores_after_ft = snapshot_scores(hydra_model)

hydra_weight_after_ft = {
    name: layer.weight.detach().cpu().clone()
    for name, layer in hydra_layers
}

changed_masks = count_changed(masks_before_ft, masks_after_ft)
changed_scores_ft = count_changed(scores_before_ft, scores_after_ft)
changed_weights_ft = count_changed(
    hydra_weight_before_ft,
    hydra_weight_after_ft,
)

assert len(changed_masks) == 0, (
    "[FAIL] fixed_mask changed during fine-tuning:\n"
    + "\n".join(changed_masks[:20])
)

assert len(changed_scores_ft) == 0, (
    "[FAIL] popup_scores changed during fine-tuning:\n"
    + "\n".join(changed_scores_ft[:20])
)

assert len(changed_weights_ft) > 0, (
    "[FAIL] No HydraConv2d weight changed during fine-tuning."
)

nonzero_pruned_failures = []

for name, layer in hydra_layers:
    pruned = layer.weight.detach()[layer.fixed_mask == 0]
    if pruned.numel() > 0 and torch.count_nonzero(pruned).item() != 0:
        nonzero_pruned_failures.append(name)

assert not nonzero_pruned_failures, (
    "[FAIL] Some pruned weights became non-zero after fine-tuning:\n"
    + "\n".join(nonzero_pruned_failures[:20])
)

print("[PASS] fixed_mask stayed bit-exact unchanged.")
print("[PASS] popup_scores stayed bit-exact unchanged.")
print(f"[PASS] {len(changed_weights_ft)} HydraConv2d weight tensors changed during fine-tuning.")
print("[PASS] All pruned weights remain exactly zero.")


# ============================================================
# 8. State-dict serialization: scores and masks must survive reload
# ============================================================
print("\n============================================================")
print("TEST 7: HYDRA STATE_DICT SERIALIZATION")
print("============================================================")

# Save to RAM so this mechanics test does not leave temporary files.
buffer = io.BytesIO()
torch.save(hydra_model.state_dict(), buffer)
buffer.seek(0)

restored_model = copy.deepcopy(yolo.model).to(DEVICE).eval()
convert_to_hydra(restored_model, keep_ratio=KEEP_RATIO)
initialize_hydra_scores(restored_model)
set_hydra_keep_ratio(restored_model, KEEP_RATIO)

state = torch.load(buffer, map_location=DEVICE)
missing, unexpected = restored_model.load_state_dict(state, strict=False)

assert not missing, f"[FAIL] Missing keys while restoring HYDRA state: {missing[:20]}"
assert not unexpected, f"[FAIL] Unexpected keys while restoring HYDRA state: {unexpected[:20]}"

# state_dict restores tensors. Runtime stage/ratio may be metadata in your
# trainer checkpoint, so set the expected execution mode explicitly here.
for _, layer in iter_hydra_layers(restored_model):
    layer.set_mode(HydraMode.FINETUNE)

restored_scores = snapshot_scores(restored_model)
restored_masks = snapshot_masks(restored_model)

assert count_changed(scores_after_ft, restored_scores) == [], \
    "[FAIL] popup_scores changed across state_dict save/reload."

assert count_changed(masks_after_ft, restored_masks) == [], \
    "[FAIL] fixed_mask changed across state_dict save/reload."

print("[PASS] popup_scores survived state_dict save/reload exactly.")
print("[PASS] fixed_mask survived state_dict save/reload exactly.")


# ============================================================
# 9. Materialization equivalence
# ============================================================
print("\n============================================================")
print("TEST 8: MATERIALIZATION EQUIVALENCE")
print("============================================================")

hydra_model.eval()

with torch.no_grad():
    out_hydra = hydra_model(dummy_img)

materialized_model = materialize_hydra_model(
    hydra_model,
    inplace=False,
).to(DEVICE).eval()

remaining_hydra_layers = list(iter_hydra_layers(materialized_model))
assert len(remaining_hydra_layers) == 0, (
    f"[FAIL] Materialized model still contains "
    f"{len(remaining_hydra_layers)} HydraConv2d layers."
)

hydra_only_parameter_names = [
    name
    for name, _ in materialized_model.named_parameters()
    if "popup_scores" in name
]

hydra_only_buffer_names = [
    name
    for name, _ in materialized_model.named_buffers()
    if "fixed_mask" in name
]

assert not hydra_only_parameter_names, (
    "[FAIL] Materialized model still contains popup_scores."
)
assert not hydra_only_buffer_names, (
    "[FAIL] Materialized model still contains fixed_mask buffers."
)

with torch.no_grad():
    out_materialized = materialized_model(dummy_img)

max_diff = compute_max_diff(out_hydra, out_materialized)

assert max_diff < TOLERANCE, (
    f"[FAIL] Materialized model output differs from masked HYDRA model. "
    f"Max difference: {max_diff}"
)

print(f"[PASS] No HydraConv2d remains after materialization.")
print(f"[PASS] No popup_scores/fixed_mask remains after materialization.")
print(
    f"[PASS] Masked HYDRA vs materialized model max difference: "
    f"{max_diff:.2e} (< {TOLERANCE})"
)


# ============================================================
# Final result
# ============================================================
print("\n============================================================")
print("ALL MILESTONE-2 CORE MECHANICS TESTS PASSED")
print("============================================================")
print(
    "Milestone 2 model-level pruning mechanics appear correct.\n"
    "Before Milestone 3, also run a small trainer-level smoke test "
    "(for example coco8, 1-2 score-search epochs + 1-2 fine-tune epochs) "
    "to verify HydraDetectionTrainer lifecycle, optimizer construction, "
    "checkpoint/resume, validation, and best/last checkpoint behavior."
)

"""
Example SHARCNET environment:

module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 opencv/4.11.0
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:$PYTHONPATH
source .yolo_env/bin/activate

python m2test.py
"""
