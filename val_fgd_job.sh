#!/bin/bash
#SBATCH --job-name=fgd_val
# #SBATCH --account=def-rsolisob
#SBATCH --time=0-2:59
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

# ==============================================================================
# FGD Robustness Validation Script
#
# Usage:
#   sbatch val_fgd_job.sh [FGD_WEIGHTS] [DETECTOR_MODEL] [ATTACK_NAME] [DATA] [ATTACK_WEIGHTS]
#
# Examples:
#   1. Clean Utility (Clean -> FGD -> YOLO):
#      sbatch val_fgd_job.sh "" yolo12l.pt raw
#
#   2. White-Box PGD Defense (PGD(YOLO) -> FGD -> YOLO):
#      sbatch val_fgd_job.sh "" yolo12l.pt pgd
#
#   3. Black-Box / Transfer Defense (PGD(Surrogate) -> FGD -> YOLO):
#      sbatch val_fgd_job.sh "" runs/trained/best.pt pgd coco_train_traffic5k_test.yaml yolo12l.pt
#
#   4. BIM Defense (BIM -> FGD -> YOLO):
#      sbatch val_fgd_job.sh "" yolo12l.pt bim
#
#   5. MIM Defense (MIM -> FGD -> YOLO):
#      sbatch val_fgd_job.sh "" yolo12l.pt mim
#
#   6. Undefended Baseline (e.g. PGD -> YOLO without FGD):
#      sbatch val_fgd_job.sh none yolo12l.pt pgd
# ==============================================================================

FGD_WEIGHTS="${1:-"fgd_yolo12l.pt"}"
MODEL="${2:-"runs/train_fgd_phase2_fgdphase2_current_coco_train_traffic5k_ep100_mim_0.6/weights/best.pt"}"
ATTACK_NAME="${3:-"raw"}"
DATA="${4:-"coco_train_traffic5k_test.yaml"}"
ATTACK_WEIGHTS="${5:-"current"}"

CLASSES=(0 1 2 3 5 6 7 9 11 12)
BATCH=16
DEVICE=0
PROJECT="runs"
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL_TAG="${MODEL##*/}"
MODEL_TAG="${MODEL_TAG%.pt}"
DATA_TAG="${DATA##*/}"
DATA_TAG="${DATA_TAG%.yaml}"
ATTACK_TAG="${ATTACK_NAME// /_}"

if [[ -n "$FGD_WEIGHTS" && "$FGD_WEIGHTS" != "none" ]]; then
  FGD_TAG="fgd"
else
  FGD_TAG="nodef"
fi

if [[ -z "$ATTACK_WEIGHTS" || "$ATTACK_WEIGHTS" == "current" || "$ATTACK_WEIGHTS" == "none" ]]; then
  ATT_SRC_TAG="wb"
else
  ATT_SRC_MODEL="${ATTACK_WEIGHTS##*/}"
  ATT_SRC_TAG="bb_${ATT_SRC_MODEL%.pt}"
fi

CLASS_TAG="${CLASSES[*]}"
CLASS_TAG="${CLASS_TAG// /_}"

NAME="${SLURM_JOB_NAME:-fgd_val}_${MODEL_TAG}_${FGD_TAG}_${DATA_TAG}_${ATTACK_TAG}_${ATT_SRC_TAG}_cls${CLASS_TAG}"

mkdir -p logs
LOG_FILE="logs/${NAME}-${SLURM_JOB_ID:-local}.out"
exec > "$LOG_FILE" 2>&1

echo "=========================================================="
echo "Experiment:    $NAME"
echo "Log file:      $LOG_FILE"
echo "Detector:      $MODEL"
echo "FGD:           $FGD_WEIGHTS"
echo "Attack:        $ATTACK_NAME"
echo "Attack Source: $ATTACK_WEIGHTS (${ATT_SRC_TAG})"
echo "Dataset:       $DATA"
echo "Classes:       ${CLASSES[*]}"
echo "=========================================================="

module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load cuda/12.2
module load opencv/4.11.0

export PYTHONPATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:${PYTHONPATH:-}"
source .yolo_env/bin/activate

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export PYTHONUNBUFFERED=1

# === Stage dataset to $SLURM_TMPDIR if tarball exists (Fast Local NVMe) ===
DATA_ARG="${DATA}"
DATASET_TAR="/project/6000635/czhan295/datasets/coco_traffic5k.tar"

if [ -n "${SLURM_TMPDIR:-}" ] && [ -f "${DATASET_TAR}" ]; then
  echo "==> Detected $DATASET_TAR, staging to local SSD: $SLURM_TMPDIR..."
  LOCAL_DATA_DIR="${SLURM_TMPDIR}/datasets/coco"
  mkdir -p "${LOCAL_DATA_DIR}"
  tar -xf "${DATASET_TAR}" -C "${LOCAL_DATA_DIR}"

  ORIGINAL_YAML="ultralytics/cfg/datasets/${DATA}"
  if [ ! -f "${ORIGINAL_YAML}" ]; then
    ORIGINAL_YAML="${DATA}"
  fi

  if [ -f "${ORIGINAL_YAML}" ]; then
    TMP_DATA_YAML="${SLURM_TMPDIR}/local_${DATA_TAG}.yaml"
    sed "s|^path:.*|path: ${LOCAL_DATA_DIR}|g" "${ORIGINAL_YAML}" > "${TMP_DATA_YAML}"
    DATA_ARG="${TMP_DATA_YAML}"
    echo "==> Using local dataset YAML: ${DATA_ARG}"
  fi
else
  echo "==> No local tarball detected. Using default dataset path from ${DATA}"
fi

VAL_ARGS=(
  --fgd_weights "$FGD_WEIGHTS"
  --weights "$MODEL"
  --data "$DATA_ARG"
  --attack_name "$ATTACK_NAME"
  --attack_weights "$ATTACK_WEIGHTS"
  --classes "${CLASSES[@]}"
  --batch "$BATCH"
  --device "$DEVICE"
  --workers "$WORKERS"
  --project "$PROJECT"
  --name "$NAME"
)

python -u val_fgd_run.py "${VAL_ARGS[@]}"
