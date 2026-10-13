#!/usr/bin/env bash
# Local Ubuntu + NVIDIA GPU setup for maxinimize/ultralytics.
# Run this script from (or copy it into) the repository root.
# Prerequisites: working NVIDIA driver (nvidia-smi), git, curl, and Ubuntu system libraries.
# This script deliberately does NOT use SHARCNET environment modules or alter pyproject.toml.
set -Eeuo pipefail

info() { printf '\n[INFO] %s\n' "$*"; }
fail() { printf '\n[ERROR] %s\n' "$*" >&2; exit 1; }

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$PROJECT_DIR"

[[ -f pyproject.toml && -d ultralytics ]] || fail \
    "Place setup_env_local.sh in the ultralytics repository root (next to pyproject.toml)."

# The SHARCNET-specific pyproject disables pip's OpenCV/Polars dependencies.
if ! grep -Eq '^[[:space:]]*"opencv-python[>=]' pyproject.toml || \
   ! grep -Eq '^[[:space:]]*"polars[>=]' pyproject.toml; then
    fail "pyproject.toml looks like the SHARCNET variant. Restore the original pyproject.toml before local installation."
fi

command -v nvidia-smi >/dev/null 2>&1 || fail \
    "nvidia-smi not found. Install/enable the NVIDIA driver first."
nvidia-smi -L || fail "NVIDIA driver cannot access the GPU. Fix it before installing the environment."

# Customize these values through environment variables if your driver/GPU requires a different supported wheel.
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.9.1}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
ART_VERSION="${ART_VERSION:-1.20.1}"
VENV_PATH="$PROJECT_DIR/.yolo_env"
VENV_PYTHON="$VENV_PATH/bin/python"

info "Project: $PROJECT_DIR"
info "Environment: $VENV_PATH"
info "PyTorch index: $PYTORCH_INDEX_URL"

if [[ ! -e "$VENV_PATH" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        fail "uv is required to create Python 3.11 here. Install it using: curl -LsSf https://astral.sh/uv/install.sh | sh (then reopen your shell)."
    fi
    info "Installing Python 3.11 and creating .yolo_env ..."
    uv python install 3.11
    uv venv --python 3.11 --seed "$VENV_PATH"
else
    info "Existing .yolo_env detected; reusing it without deleting data."
fi

[[ -x "$VENV_PYTHON" ]] || fail \
    "Existing .yolo_env is not a valid Python environment. Rename it, then rerun this script."
"$VENV_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11, got {sys.version}"' || \
    fail "Existing .yolo_env does not use Python 3.11. Rename it, then rerun this script."

info "Updating pip and build tools ..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

info "Installing CUDA-enabled PyTorch ..."
"$VENV_PYTHON" -m pip install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "$PYTORCH_INDEX_URL"

info "Installing this Ultralytics fork in editable mode, Pandas, and ART ..."
"$VENV_PYTHON" -m pip install -e . \
    'pandas>=2.0,<3.0' \
    "adversarial-robustness-toolbox==$ART_VERSION"

info "Checking dependency consistency ..."
"$VENV_PYTHON" -m pip check

info "Checking local imports and NVIDIA GPU acceleration ..."
"$VENV_PYTHON" - <<'PY'
import sys
import torch
import torchvision
import cv2
import pandas
import polars
import art
import ultralytics

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("OpenCV:", cv2.__version__)
print("Pandas:", pandas.__version__)
print("Polars:", polars.__version__)
print("ART:", art.__version__)
print("Ultralytics module:", ultralytics.__file__)

# Verify the fork's adversarial-training and FGD modules can be imported.
from ultralytics.models.yolo.detect.train_adv_test import DetectionTrainer  # noqa: F401
from ultralytics.defenses.hgd_fgd import FGDModel  # noqa: F401

if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch cannot access CUDA. Check NVIDIA driver compatibility with your selected CUDA wheel."
    )
print("GPU:", torch.cuda.get_device_name(0))
a = torch.ones((32, 32), device="cuda")
b = a @ a
print("CUDA matrix multiplication OK; result:", b[0, 0].item())
PY

info "Setup complete. Activate it with: source .yolo_env/bin/activate"
