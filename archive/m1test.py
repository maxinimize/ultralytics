import copy
import torch
from ultralytics import YOLO
from ultralytics.nn.pruning import convert_to_hydra, initialize_hydra_scores, iter_hydra_layers, HydraMode

# 1. Load YOLOv12 weights
model_path = "yolo12l.pt"
print(f"Loading model weights: {model_path} ...")
yolo = YOLO(model_path)

# Deepcopy to keep original dense model and HYDRA converted model
dense_model = copy.deepcopy(yolo.model).eval()
hydra_model = copy.deepcopy(yolo.model).eval()

# 2. Convert model to HYDRA architecture (keep_ratio=1.0)
convert_to_hydra(hydra_model, keep_ratio=1.0)
initialize_hydra_scores(hydra_model)

hydra_layers = list(iter_hydra_layers(hydra_model))
print(f"Successfully replaced {len(hydra_layers)} nn.Conv2d layers with HydraConv2d!\n")

# 3. Construct identical random input image
torch.manual_seed(42)
dummy_img = torch.randn(1, 3, 640, 640)

# Obtain inference output of the original dense model
with torch.no_grad():
    out_dense = dense_model(dummy_img)

# 4. Recursively compare numerical consistency between dense model and 3 HYDRA modes
def compute_max_diff(out1, out2):
    """Recursively calculate maximum absolute difference between two model outputs."""
    if isinstance(out1, torch.Tensor):
        return (out1 - out2).abs().max().item()
    elif isinstance(out1, (list, tuple)):
        return max([compute_max_diff(i1, i2) for i1, i2 in zip(out1, out2)])
    elif isinstance(out1, dict):
        return max([compute_max_diff(out1[k], out2[k]) for k in out1])
    else:
        return 0.0

# Floating-point error threshold (~6e-4 accumulated across 205 deep conv layers)
TOLERANCE = 1e-3

print("=== Starting numerical consistency test (keep_ratio=1.0) ===")
for mode in [HydraMode.DENSE, HydraMode.SCORE, HydraMode.FINETUNE]:
    for _, layer in hydra_layers:
        layer.set_mode(mode)
    with torch.no_grad():
        out_hydra = hydra_model(dummy_img)
    
    max_diff = compute_max_diff(out_dense, out_hydra)
    assert max_diff < TOLERANCE, f"[FAIL] Mode {mode.value} mismatch with original model! Max diff: {max_diff}"
    print(f"[PASS] HYDRA mode [{mode.value:<8}] max diff from original dense model: {max_diff:.2e} (< {TOLERANCE})")

print("\nOriginal model and converted HYDRA model are numerically identical at keep_ratio=1.0 (within float precision)!")

"""
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 opencv/4.11.0
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:$PYTHONPATH
source .yolo_env/bin/activate
python m1test.py
"""