#!/usr/bin/env bash
# Local Ubuntu launcher for train_adv_fgd_stn_adaptive_run.py (Single NVIDIA GPU).
# Combined FGD + STN Defense with Adaptive Adversarial Training.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f "$ROOT/train_adv_fgd_stn_adaptive_run.py" || ! -f "$ROOT/.yolo_env/bin/activate" ]]; then
  echo "ERROR: Place this script in the Ultralytics project root (with .yolo_env)." >&2
  exit 1
fi
source "$ROOT/.yolo_env/bin/activate"

# Configuration options with environment variable overrides:
PREPROCESS_MODE="${PREPROCESS_MODE:-${1:-stn_fgd}}"
ADAPTIVE_RATIO="${ADAPTIVE_RATIO:-${2:-true}}"
FGD_WEIGHTS="${FGD_WEIGHTS:-fgd_yolo12l.pt}"
STN_WEIGHTS="${STN_WEIGHTS:-runs/detect/runs/train_stn/stn_train_yolo12l_coco_train_traffic5k_ep50_b8_k10_deg30.0_tr0.10/weights/best.pt}"
DATA="${DATA:-coco_train_traffic5k.yaml}"
EPOCHS="${EPOCHS:-100}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolo12l.pt}"
ATTACK_WEIGHTS="${ATTACK_WEIGHTS:-current}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-4}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
CACHE="${CACHE:-True}"
AMP="${AMP:-bf16}"
FRACTION="${FRACTION:-1.0}"
ATTACK_THROUGH_PREPROCESS="${ATTACK_THROUGH_PREPROCESS:-true}"
P_CLEAN="${P_CLEAN:-0.25}"
TAU="${TAU:-5.0}"
UPDATE_INTERVAL="${UPDATE_INTERVAL:-5}"
ATTACK_NAMES="${ATTACK_NAMES:-pgd bim mim worstk}"
ATTACK_RATIOS="${ATTACK_RATIOS:-0.10 0.10 0.40 0.15}"
EVAL_WEIGHTS="${EVAL_WEIGHTS:-0.25 0.25 0.25 0.25}"
PROJECT="${PROJECT:-$ROOT/runs/train_adv_fgd_stn_adaptive}"
JOB_NAME="${JOB_NAME:-local_adv_fgd_stn_adaptive}"
RESUME="${RESUME:-}"
CLASSES="${CLASSES:-0 1 2 3 5 6 7 9 11 12}"

export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"

# Resolve dataset YAML
if [[ -f "$DATA" ]]; then
  DATA_ARG="$(realpath "$DATA")"
elif [[ -f "$ROOT/ultralytics/cfg/datasets/$DATA" ]]; then
  DATA_ARG="$ROOT/ultralytics/cfg/datasets/$DATA"
else
  echo "ERROR: Dataset YAML not found: $DATA" >&2
  exit 1
fi

read -r -a ATTACK_NAME_ARR <<< "$ATTACK_NAMES"
read -r -a ATTACK_RATIO_ARR <<< "$ATTACK_RATIOS"
read -r -a EVAL_WEIGHT_ARR <<< "$EVAL_WEIGHTS"
read -r -a CLASS_IDS <<< "$CLASSES"

WEIGHTS_TAG="$(basename "$ATTACK_WEIGHTS" .pt)"
DATA_TAG="$(basename "$DATA" .yaml)"
NAME="${NAME:-${JOB_NAME}_${PREPROCESS_MODE}_adapt${ADAPTIVE_RATIO}_${WEIGHTS_TAG}_${DATA_TAG}_ep${EPOCHS}}"
mkdir -p "$ROOT/logs" "$PROJECT"
LOG_FILE="$ROOT/logs/${NAME}-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[local FGD+STN Adaptive AT] $(date -Is)"
echo "Project root: $ROOT"
echo "Experiment: $NAME"
echo "Mode: $PREPROCESS_MODE | Adaptive: $ADAPTIVE_RATIO | Attack Through Preprocess: $ATTACK_THROUGH_PREPROCESS"
echo "FGD: $FGD_WEIGHTS | STN: $STN_WEIGHTS | YOLO: $YOLO_WEIGHTS"
echo "Dataset YAML: $DATA_ARG"
echo "GPU: $DEVICE | Batch: $BATCH | Workers: $WORKERS | AMP: $AMP | Cache: $CACHE"
echo "Clean: $P_CLEAN | Tau: $TAU | Interval: $UPDATE_INTERVAL"
echo "Log: $LOG_FILE"

python - <<'PY'
import torch
print('PyTorch:', torch.__version__, '| torch CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA is not available in .yolo_env'
print('GPU:', torch.cuda.get_device_name(0), '| BF16:', torch.cuda.is_bf16_supported())
PY

ARGS=(
  --model "$YOLO_WEIGHTS"
  --data "$DATA_ARG"
  --preprocess_mode "$PREPROCESS_MODE"
  --fgd_weights "$FGD_WEIGHTS"
  --stn_weights "$STN_WEIGHTS"
  --adaptive_ratio "$ADAPTIVE_RATIO"
  --attack_through_preprocess "$ATTACK_THROUGH_PREPROCESS"
  --p_clean "$P_CLEAN"
  --tau "$TAU"
  --update_interval "$UPDATE_INTERVAL"
  --attack_weights "$ATTACK_WEIGHTS"
  --imgsz "$IMGSZ"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --device "$DEVICE"
  --workers "$WORKERS"
  --cache "$CACHE"
  --amp "$AMP"
  --fraction "$FRACTION"
  --attack_name "${ATTACK_NAME_ARR[@]}"
  --attack_ratio "${ATTACK_RATIO_ARR[@]}"
  --eval_weights "${EVAL_WEIGHT_ARR[@]}"
  --project "$PROJECT"
  --name "$NAME"
)
if (( ${#CLASS_IDS[@]} > 0 )); then ARGS+=(--classes "${CLASS_IDS[@]}"); fi
if [[ -n "$RESUME" ]]; then ARGS+=(--resume "$RESUME"); fi

printf 'Command: python -u train_adv_fgd_stn_adaptive_run.py'; printf ' %q' "${ARGS[@]}"; echo
python -u train_adv_fgd_stn_adaptive_run.py "${ARGS[@]}"
echo "Training finished: $(date -Is)"
