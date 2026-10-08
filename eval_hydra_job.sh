#!/bin/bash
#SBATCH --job-name=hydra_eval
# #SBATCH --account=def-rsolisob
#SBATCH --time=0-02:59
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail

# ==============================================================================
# 0. Environment and module loading (Compute Canada / Slurm)
# ==============================================================================
module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load cuda/12.2
module load opencv/4.11.0

# OpenCV Python path
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:${PYTHONPATH:-}

# Activate Python virtual environment
source .yolo_env/bin/activate

# Threading and CUDA memory allocation optimization
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export PYTHONUNBUFFERED=1

# ==============================================================================
# 1. Evaluation Configuration
#    Usage:
#      sbatch eval_hydra_job.sh [MODEL_PATH] [DATA_YAML] [ATTACKS] [BATCH] [IMGSZ] [DEVICE] [CLASSES]
# ==============================================================================
DEFAULT_MODEL="/project/6000635/czhan295/ultralytics/runs/hydra_adv_pipe_train_current_coco_train_traffic5k_ep100_pgd_0.6_best_coco_train_traffic5k_k0.5_ep20_30_pgd_0.6-62352076/pruned_materialized.pt"

MODEL="${1:-"runs/train_current_coco_train_traffic5k_ep100_pgd_0.6/weights/best.pt"}"
DATA="${2:-"coco_train_traffic5k.yaml"}"
ATTACKS="${3:-"clean pgd bim mim"}"
BATCH="${4:-"16"}"
IMGSZ="${5:-"640"}"
DEVICE="${6:-"0"}"
CLASSES="${7:-"0 1 2 3 5 6 7 9 11 12"}"

mkdir -p logs

echo "========================================================================"
echo "Starting HYDRA Evaluation Job"
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Host:           $(hostname)"
echo "Model:          ${MODEL}"
echo "Dataset YAML:   ${DATA}"
echo "Attacks:        ${ATTACKS}"
echo "Batch Size:     ${BATCH}"
echo "Image Size:     ${IMGSZ}"
echo "Device:         ${DEVICE}"
echo "Classes:        ${CLASSES}"
echo "CUDA Device:    $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo "========================================================================"

if [ ! -f "${MODEL}" ]; then
    echo "Error: Model file does not exist at: ${MODEL}"
    exit 1
fi

python -u tools/hydra_eval.py \
    --model "${MODEL}" \
    --data "${DATA}" \
    --batch "${BATCH}" \
    --imgsz "${IMGSZ}" \
    --device "${DEVICE}" \
    --attacks ${ATTACKS} \
    ${CLASSES:+--classes ${CLASSES}}

echo -e "\n========================================================================"
echo "HYDRA Evaluation completed successfully!"
echo "========================================================================"
