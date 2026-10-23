#!/usr/bin/env bash
#SBATCH --job-name=val_stn
#SBATCH --time=0-3:59
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Validation configuration
# Allows overriding via CLI arguments: sbatch val_stn_job.sh [STN_WEIGHTS] [MODEL] [DATA]
STN_WEIGHTS="${1:-"${STN_WEIGHTS:-runs/detect/runs/train_stn/stn_train_yolo12l_coco_train_traffic5k_ep50_b8_k10_deg30.0_tr0.10/weights/best.pt}"}" 
MODEL="${2:-"${MODEL:-yolo12l.pt}"}"
DATA="${3:-"${DATA:-coco_train_traffic5k_test.yaml}"}"
SPLIT="${SPLIT:-val}"
CLASSES=(0 1 2 3 5 6 7 9 11 12)
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
DEGREES="${DEGREES:-30.0}"
TRANSLATE="${TRANSLATE:-0.10}"
WORSTK_K="${WORSTK_K:-10}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))}"
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
PROJECT="${PROJECT:-runs/val_stn}"
MODES="${MODES:-clean_baseline worstk_baseline clean_stn worstk_stn}"
SEED="${SEED:-0}"

# If data file with double underscore is specified and missing, fallback to single underscore
if [[ "$DATA" == *"coco_train_traffic5k__test.yaml"* ]] && [[ ! -f "$DATA" ]] && [[ ! -f "ultralytics/cfg/datasets/$DATA" ]]; then
  CORRECTED_DATA="${DATA//__test.yaml/_test.yaml}"
  if [[ -f "$CORRECTED_DATA" ]] || [[ -f "ultralytics/cfg/datasets/$CORRECTED_DATA" ]]; then
    echo "==> Notice: '$DATA' not found, using '$CORRECTED_DATA'"
    DATA="$CORRECTED_DATA"
  fi
fi

MODEL_TAG="${MODEL##*/}"
MODEL_TAG="${MODEL_TAG%.pt}"
DATA_TAG="${DATA##*/}"
DATA_TAG="${DATA_TAG%.yaml}"
NAME="${SLURM_JOB_NAME:-val_stn}_${MODEL_TAG}_${DATA_TAG}_k${WORSTK_K}_deg${DEGREES}_tr${TRANSLATE}"

mkdir -p logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/${NAME}-${SLURM_JOB_ID:-local}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== STN Validation Experiment: $NAME ==="
echo "STN weights: $STN_WEIGHTS"
echo "Detector weights: $MODEL"
echo "Data config: $DATA"
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

VAL_ARGS=(
  --weights "$MODEL"
  --stn_weights "$STN_WEIGHTS"
  --data "$DATA_ARG"
  --split "$SPLIT"
  --classes "${CLASSES[@]}"
  --modes $MODES
  --batch "$BATCH"
  --imgsz "$IMGSZ"
  --degrees "$DEGREES"
  --translate "$TRANSLATE"
  --worstk_k "$WORSTK_K"
  --seed "$SEED"
  --device "$DEVICE"
  --workers "$WORKERS"
  --project "$PROJECT"
  --name "$NAME"
)

echo "==> Running command: python -u val_stn_run.py ${VAL_ARGS[*]}"
python -u val_stn_run.py "${VAL_ARGS[@]}"
echo "=== STN Validation Finished at: $(date) ==="
