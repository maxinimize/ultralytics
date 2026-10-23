#!/usr/bin/env bash
#SBATCH --job-name=stn_train
#SBATCH --time=0-23:59
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Training configuration
MODEL="${MODEL:-yolo12l.pt}"
DATA="${DATA:-coco_train_traffic5k.yaml}"
CLASSES=(0 1 2 3 5 6 7 9 11 12)
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-640}"
DEGREES="${DEGREES:-30.0}"
TRANSLATE="${TRANSLATE:-0.10}"
WORSTK_K="${WORSTK_K:-10}"
LOSS_GEOM="${LOSS_GEOM:-1.0}"
LOSS_RECON="${LOSS_RECON:-1.0}"
LOSS_DET="${LOSS_DET:-0.0}"
LR0="${LR0:-0.001}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))}"
PROJECT="${PROJECT:-runs/train_stn}"
RESUME="${RESUME:-}"
CACHE="${CACHE:-ram}"

MODEL_TAG="${MODEL##*/}"
MODEL_TAG="${MODEL_TAG%.pt}"
DATA_TAG="${DATA##*/}"
DATA_TAG="${DATA_TAG%.yaml}"
NAME="${SLURM_JOB_NAME:-stn_train}_${MODEL_TAG}_${DATA_TAG}_ep${EPOCHS}_b${BATCH}_k${WORSTK_K}_deg${DEGREES}_tr${TRANSLATE}"

mkdir -p logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/${NAME}-${SLURM_JOB_ID:-local}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== STN Training Experiment: $NAME ==="
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"

# Load environment if modules exist (e.g. SLURM cluster)
if command -v module &>/dev/null; then
  module load StdEnv/2023 || true
  module load gcc/12.3 || true
  module load python/3.11 || true
  module load cuda/12.2 || true
  module load opencv/4.11.0 || true
fi

if [[ -f "$ROOT/.yolo_env/bin/activate" ]]; then
  source "$ROOT/.yolo_env/bin/activate"
fi

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export PYTHONUNBUFFERED=1

# Resolve dataset path / local SSD staging
DATA_ARG="${DATA}"
DATASET_TAR="/project/6000635/czhan295/datasets/coco_traffic5k.tar"

if [[ -n "${SLURM_TMPDIR:-}" ]] && [[ -f "${DATASET_TAR}" ]]; then
  echo "==> Staging dataset to local SSD: $SLURM_TMPDIR..."
  LOCAL_DATA_DIR="${SLURM_TMPDIR}/datasets/coco"
  mkdir -p "${LOCAL_DATA_DIR}"
  tar -xf "${DATASET_TAR}" -C "${LOCAL_DATA_DIR}"

  ORIGINAL_YAML="ultralytics/cfg/datasets/${DATA}"
  if [[ ! -f "${ORIGINAL_YAML}" ]]; then
    ORIGINAL_YAML="${DATA}"
  fi

  if [[ -f "${ORIGINAL_YAML}" ]]; then
    TMP_DATA_YAML="${SLURM_TMPDIR}/local_${DATA_TAG}.yaml"
    sed "s|^path:.*|path: ${LOCAL_DATA_DIR}|g" "${ORIGINAL_YAML}" > "${TMP_DATA_YAML}"
    DATA_ARG="${TMP_DATA_YAML}"
    echo "==> Using local SSD dataset YAML: ${DATA_ARG}"
  fi
fi

TRAIN_ARGS=(
  --model "$MODEL"
  --data "$DATA_ARG"
  --classes "${CLASSES[@]}"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --imgsz "$IMGSZ"
  --degrees "$DEGREES"
  --translate "$TRANSLATE"
  --worstk_k "$WORSTK_K"
  --loss_geom_weight "$LOSS_GEOM"
  --loss_recon_weight "$LOSS_RECON"
  --loss_det_weight "$LOSS_DET"
  --lr0 "$LR0"
  --device "$DEVICE"
  --workers "$WORKERS"
  --project "$PROJECT"
  --name "$NAME"
)

if [[ -n "$CACHE" ]]; then
  TRAIN_ARGS+=(--cache "$CACHE")
fi

if [[ -n "$RESUME" ]]; then
  TRAIN_ARGS+=(--resume "$RESUME")
fi

echo "==> Running command: python -u train_stn_run.py ${TRAIN_ARGS[*]}"
python -u train_stn_run.py "${TRAIN_ARGS[@]}"
echo "=== STN Training Finished at: $(date) ==="
