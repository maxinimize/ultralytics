#!/usr/bin/env bash
# Local Ubuntu launcher for train_fgd_run.py (single NVIDIA GPU).
# Run inside tmux. This file does not depend on SLURM.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f "$ROOT/train_fgd_run.py" || ! -f "$ROOT/.yolo_env/bin/activate" ]]; then
  echo "ERROR: Place this script in the Ultralytics project root (with .yolo_env)." >&2
  exit 1
fi
source "$ROOT/.yolo_env/bin/activate"

# Model checkpoint: check candidate paths or default to yolo12l.pt
DEFAULT_MODEL="runs/local_adv_adaptive_current_coco_train_traffic5k_ep100_pgd_bim_mim_worstk_0.10_0.10_0.40_0.15/weights/best.pt"
if [[ ! -f "$DEFAULT_MODEL" && -f "$ROOT/yolo12l.pt" ]]; then
  DEFAULT_MODEL="yolo12l.pt"
fi

# Original job-script positionals, with environment variable overrides:
#   bash train_fgd_local.sh [model] [data_yaml] [epochs] [batch] [attack_names] [attack_ratios]
MODEL="${MODEL:-${1:-$DEFAULT_MODEL}}"
DATA="${DATA:-${2:-coco_train_traffic5k.yaml}}"
EPOCHS="${EPOCHS:-${3:-50}}"
BATCH="${BATCH:-${4:-4}}"
ATTACK_NAME="${ATTACK_NAME:-${ATTACK_NAMES:-${5:-pgd bim mim}}}"
ATTACK_RATIO="${ATTACK_RATIO:-${ATTACK_RATIOS:-${6:-}}}"

# Additional parameters
ATTACK_WEIGHTS="${ATTACK_WEIGHTS:-}"
CLASSES="${CLASSES:-0 1 2 3 5 6 7 9 11 12}"
CACHE="${CACHE:-ram}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-4}"
LR0="${LR0:-0.0001}"
SEED="${SEED:-0}"
MULTI_SCALE="${MULTI_SCALE:-0.0}"
SAVE_PERIOD="${SAVE_PERIOD:--1}"
PROJECT="${PROJECT:-$ROOT/runs}"
JOB_NAME="${JOB_NAME:-local_fgd_train}"
RESUME="${RESUME:-}"

export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"

# Resolve the dataset YAML.
if [[ -f "$DATA" ]]; then
  DATA_ARG="$(realpath "$DATA")"
elif [[ -f "$ROOT/ultralytics/cfg/datasets/$DATA" ]]; then
  DATA_ARG="$ROOT/ultralytics/cfg/datasets/$DATA"
else
  echo "ERROR: Dataset YAML not found: $DATA" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: Model checkpoint not found: $MODEL" >&2
  exit 1
fi

if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
  echo "ERROR: Resume checkpoint not found: $RESUME" >&2
  exit 1
fi

read -r -a ATTACK_NAMES <<< "$ATTACK_NAME"
read -r -a CLASS_IDS <<< "$CLASSES"
if (( ${#ATTACK_NAMES[@]} == 0 )); then
  echo 'ERROR: ATTACK_NAME must be non-empty.' >&2
  exit 1
fi

read -r -a ATTACK_RATIOS <<< "$ATTACK_RATIO"

MODEL_TAG="$(basename "$MODEL" .pt)"
DATA_TAG="$(basename "$DATA" .yaml)"
ATTACK_TAG="${ATTACK_NAMES[*]}"
ATTACK_TAG="${ATTACK_TAG// /_}"
CLASS_TAG="${CLASS_IDS[*]}"
CLASS_TAG="${CLASS_TAG// /_}"

NAME="${NAME:-${JOB_NAME}_${MODEL_TAG}_${DATA_TAG}_ep${EPOCHS}_b${BATCH}_${ATTACK_TAG}_cls${CLASS_TAG}}"

mkdir -p "$ROOT/logs" "$PROJECT"
LOG_FILE="$ROOT/logs/${NAME}-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[local FGD train] $(date -Is)"
echo "Project root: $ROOT"
echo "Experiment: $NAME"
echo "Model: $MODEL"
echo "Dataset YAML: $DATA_ARG"
echo "GPU: $DEVICE  Batch: $BATCH  Workers: $WORKERS  Cache: $CACHE"
echo "Log: $LOG_FILE"

python - <<'PY'
import torch
print('PyTorch:', torch.__version__, '| torch CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA is not available in .yolo_env'
print('GPU:', torch.cuda.get_device_name(0), '| BF16:', torch.cuda.is_bf16_supported())
PY

ARGS=(
  --model "$MODEL"
  --data "$DATA_ARG"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --device "$DEVICE"
  --workers "$WORKERS"
  --project "$PROJECT"
  --name "$NAME"
  --imgsz "$IMGSZ"
  --lr0 "$LR0"
  --seed "$SEED"
  --multi_scale "$MULTI_SCALE"
  --save_period "$SAVE_PERIOD"
  --attack_name "${ATTACK_NAMES[@]}"
)

if [[ -n "$CACHE" && "$CACHE" != "none" && "$CACHE" != "None" ]]; then
  ARGS+=(--cache "$CACHE")
fi

if [[ -n "$ATTACK_WEIGHTS" ]]; then
  ARGS+=(--attack_weights "$ATTACK_WEIGHTS")
fi

if (( ${#ATTACK_RATIOS[@]} > 0 )); then
  ARGS+=(--attack_ratio "${ATTACK_RATIOS[@]}")
fi

if (( ${#CLASS_IDS[@]} > 0 )); then
  ARGS+=(--classes "${CLASS_IDS[@]}")
fi

if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi

printf 'Command: python -u train_fgd_run.py'; printf ' %q' "${ARGS[@]}"; echo
python -u train_fgd_run.py "${ARGS[@]}"
echo "Training finished: $(date -Is)"
