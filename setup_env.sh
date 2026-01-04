#!/bin/bash
set -e # exit on error

#####################################################################
# replace pyproject.toml with pyproject_shardnet.toml if on SHARCNET 
#####################################################################

# Load necessary modules on SHARCNET
module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load cuda/12.2
module load opencv/4.11.0
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:$PYTHONPATH

VENV_PATH="$PWD/.yolo_env" # virtual environment path

# Check if virtual environment exists
if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment exists: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    echo "Virtual environment activated."
    python -c "import torch, ultralytics; print('Torch:', torch.__version__); print('Ultralytics ok')" || echo '⚠️ Environment issue, please check'
    exit 0
fi

# Create virtual environment
echo "Virtual environment does not exist, creating..."
python -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install "adversarial-robustness-toolbox==1.20.1"

# Verify installation
echo "Environment setup complete. Checking versions:"
python - << 'EOF'
import torch
from ultralytics import YOLO

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

model = YOLO("yolov8n.pt")
print("Loaded model:", model)
EOF
