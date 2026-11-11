#!/usr/bin/env bash
# Local Ubuntu single GPU (RTX 5090 32GB) launch script: FGD and STN joint defense training
set -euo pipefail

# Get script root directory
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f "$ROOT/train_fgd_stn_joint_run.py" || ! -f "$ROOT/.yolo_env/bin/activate" ]]; then
  echo "Error: Please run this script from the project root and ensure the .yolo_env virtual environment exists." >&2
  exit 1
fi
source "$ROOT/.yolo_env/bin/activate"

# Configuration parameters and environment variables
PREPROCESS_MODE="${PREPROCESS_MODE:-${1:-stn_fgd}}"
# INIT="${INIT:-${2:-scratch}}"
# FGD_WEIGHTS="${FGD_WEIGHTS:-none}"
# STN_WEIGHTS="${STN_WEIGHTS:-none}"
INIT="${INIT:-${2:-pretrained}}"
FGD_WEIGHTS="${FGD_WEIGHTS:-fgd_yolo12l.pt}"
STN_WEIGHTS="${STN_WEIGHTS:-runs/detect/runs/train_stn/stn_train_yolo12l_coco_train_traffic5k_ep50_b8_k10_deg30.0_tr0.10/weights/best.pt}"
MODEL="${MODEL:-yolo12l.pt}"
DATA="${DATA:-coco_train_traffic5k.yaml}"
CLASSES="${CLASSES:-0 1 2 3 5 6 7 9 11 12}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-4}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-4}"
CACHE="${CACHE:-True}"
AMP="${AMP:-bf16}"
ATTACK_THROUGH_PREPROCESS="${ATTACK_THROUGH_PREPROCESS:-true}"
ATTACK_WEIGHTS="${ATTACK_WEIGHTS:-current}"
P_CLEAN="${P_CLEAN:-0.25}"
ATTACK_NAMES="${ATTACK_NAMES:-pgd bim mim worstk}"
ATTACK_RATIOS="${ATTACK_RATIOS:-0.10 0.10 0.20 0.35}"
WORSTK_K="${WORSTK_K:-10}"
DEGREES="${DEGREES:-30.0}"
TRANSLATE="${TRANSLATE:-0.10}"
LR_FGD="${LR_FGD:-0.00001}"
LR_STN="${LR_STN:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
LAMBDA_FGD="${LAMBDA_FGD:-1.0}"
LAMBDA_GEOM="${LAMBDA_GEOM:-1.0}"
LAMBDA_RECON="${LAMBDA_RECON:-1.0}"
LAMBDA_DET="${LAMBDA_DET:-0.0}"
LAMBDA_IDENTITY="${LAMBDA_IDENTITY:-0.1}"
VAL_INTERVAL="${VAL_INTERVAL:-5}"
PROJECT="${PROJECT:-$ROOT/runs/train_fgd_stn_joint}"
JOB_NAME="${JOB_NAME:-local_fgd_stn_joint}"
RESUME="${RESUME:-}"
SEED="${SEED:-0}"

# Thread and GPU memory environment variables
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"

# Resolve dataset configuration path
if [[ -f "$DATA" ]]; then
  DATA_ARG="$(realpath "$DATA")"
elif [[ -f "$ROOT/ultralytics/cfg/datasets/$DATA" ]]; then
  DATA_ARG="$ROOT/ultralytics/cfg/datasets/$DATA"
else
  echo "Error: Dataset configuration file not found: $DATA" >&2
  exit 1
fi

read -r -a ATTACK_NAME_ARR <<< "$ATTACK_NAMES"
read -r -a ATTACK_RATIO_ARR <<< "$ATTACK_RATIOS"
read -r -a CLASS_IDS <<< "$CLASSES"

WEIGHTS_TAG="$(basename "$MODEL" .pt)"
DATA_TAG="$(basename "$DATA" .yaml)"
NAME="${NAME:-${JOB_NAME}_${PREPROCESS_MODE}_${INIT}_${WEIGHTS_TAG}_${DATA_TAG}_ep${EPOCHS}_b${BATCH}}"

mkdir -p "$ROOT/logs" "$PROJECT"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$ROOT/logs/${NAME}-${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================================="
echo "=== Local Ubuntu FGD + STN Joint Defense Training: $NAME ==="
echo "=================================================================="
echo "Start Time: $(date -Is)"
echo "Project Path: $ROOT"
echo "Log File: $LOG_FILE"
echo "Defense Order: $PREPROCESS_MODE | Init Mode: $INIT"
echo "Model Config: YOLO=$MODEL | FGD=$FGD_WEIGHTS | STN=$STN_WEIGHTS"
echo "Hardware: GPU=$DEVICE | Batch Size=$BATCH | Workers=$WORKERS | AMP=$AMP | Cache=$CACHE"
echo "Sample Split: Clean Ratio=$P_CLEAN | Attack Names=$ATTACK_NAMES | Attack Ratios=$ATTACK_RATIOS"
echo "Learning Rate: FGD=$LR_FGD, STN=$LR_STN | Weight Decay=$WEIGHT_DECAY"
echo "Loss Weights: fgd=$LAMBDA_FGD, geom=$LAMBDA_GEOM, recon=$LAMBDA_RECON, det=$LAMBDA_DET, identity=$LAMBDA_IDENTITY"
echo "Attack Through Preprocess: $ATTACK_THROUGH_PREPROCESS"
echo "=================================================================="

# Verify CUDA status and GPU info
python - <<'PY'
import torch
print('PyTorch Version:', torch.__version__, '| CUDA Support:', torch.version.cuda)
assert torch.cuda.is_available(), 'Error: CUDA is not available in current virtual environment!'
print('GPU Device:', torch.cuda.get_device_name(0), '| BF16 Support:', torch.cuda.is_bf16_supported())
PY

ARGS=(
  --model "$MODEL"
  --data "$DATA_ARG"
  --preprocess_mode "$PREPROCESS_MODE"
  --init "$INIT"
  --attack_through_preprocess "$ATTACK_THROUGH_PREPROCESS"
  --attack_weights "$ATTACK_WEIGHTS"
  --p_clean "$P_CLEAN"
  --attack_name "${ATTACK_NAME_ARR[@]}"
  --attack_ratio "${ATTACK_RATIO_ARR[@]}"
  --worstk_k "$WORSTK_K"
  --degrees "$DEGREES"
  --translate "$TRANSLATE"
  --lambda_fgd "$LAMBDA_FGD"
  --lambda_geom "$LAMBDA_GEOM"
  --lambda_recon "$LAMBDA_RECON"
  --lambda_det "$LAMBDA_DET"
  --lambda_identity "$LAMBDA_IDENTITY"
  --lr_fgd "$LR_FGD"
  --lr_stn "$LR_STN"
  --weight_decay "$WEIGHT_DECAY"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --imgsz "$IMGSZ"
  --device "$DEVICE"
  --workers "$WORKERS"
  --cache "$CACHE"
  --amp "$AMP"
  --seed "$SEED"
  --val_interval "$VAL_INTERVAL"
  --project "$PROJECT"
  --name "$NAME"
)

if [[ "$INIT" == "pretrained" ]]; then
  ARGS+=(--fgd_weights "$FGD_WEIGHTS" --stn_weights "$STN_WEIGHTS")
fi

if (( ${#CLASS_IDS[@]} > 0 )); then
  ARGS+=(--classes "${CLASS_IDS[@]}")
fi

if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi

printf 'Executing training command: python -u train_fgd_stn_joint_run.py'; printf ' %q' "${ARGS[@]}"; echo
python -u train_fgd_stn_joint_run.py "${ARGS[@]}"
echo "Training finished normally: $(date -Is)"

