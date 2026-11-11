#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Default paths and parameters
FGD_WEIGHTS="${1:-"${FGD_WEIGHTS:-fgd_yolo12l.pt}"}"
STN_WEIGHTS="${2:-"${STN_WEIGHTS:-runs/detect/runs/train_stn/stn_train_yolo12l_coco_train_traffic5k_ep50_b8_k10_deg30.0_tr0.10/weights/best.pt}"}"
# FGD_WEIGHTS="${1:-"${FGD_WEIGHTS:-runs/train_fgd_stn_joint/local_fgd_stn_joint_stn_fgd_pretrained_yolo12l_coco_train_traffic5k_ep20_b4/weights/best_fgd.pt}"}"
# STN_WEIGHTS="${2:-"${STN_WEIGHTS:-runs/train_fgd_stn_joint/local_fgd_stn_joint_stn_fgd_pretrained_yolo12l_coco_train_traffic5k_ep20_b4/weights/best_stn.pt}"}"d
#STN_WEIGHTS="${2:-"${STN_WEIGHTS:-runs/detect/runs/train_stn/stn_train_best_coco_train_traffic5k_ep50_b8_k10_deg30.0_tr0.10/weights/best.pt}"}"
# MODEL="${3:-"${MODEL:-runs/local_adv_adaptive_current_coco_train_traffic5k_ep100_pgd_bim_mim_worstk_0.10_0.10_0.40_0.15/weights/best.pt}"}"
MODEL="${3:-"${MODEL:-yolo12l.pt}"}"
# MODEL="${3:-"${MODEL:-runs/train_adv_fgd_stn_adaptive/local_adv_fgd_stn_adaptive_stn_fgd_adapttrue_current_coco_train_traffic5k_ep100-2/weights/best.pt}"}"
DATA="${4:-"${DATA:-coco_train_traffic5k_test.yaml}"}"
ATTACK_WEIGHTS="${ATTACK_WEIGHTS:-"current"}"
MODES="${MODES:-all}"
ATTACKS="${ATTACKS:-all}"
SPLIT="${SPLIT:-val}"
CLASSES=(0 1 2 3 5 6 7 9 11 12)
BATCH="${BATCH:-4}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-4}"
PROJECT="${PROJECT:-runs/val_fgd_stn}"
BASE_NAME="${NAME:-"exp"}"

# Automatically increment NAME if directory already exists to avoid overwriting
NAME="$BASE_NAME"
if [[ -d "$PROJECT/$NAME" ]]; then
  i=2
  while [[ -d "$PROJECT/${BASE_NAME}${i}" ]]; do
    ((i++))
  done
  NAME="${BASE_NAME}${i}"
fi
SEED="${SEED:-0}"

# Setup logging to logs/ directory
mkdir -p logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/val_fgd_stn_local_${NAME}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Unified FGD + STN Defense Validation (Local) ==="
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"

# Activate virtualenv if present
if [[ -f "$ROOT/.yolo_env/bin/activate" ]]; then
  source "$ROOT/.yolo_env/bin/activate"
fi

VAL_ARGS=(
  --weights "$MODEL"
  --fgd_weights "$FGD_WEIGHTS"
  --stn_weights "$STN_WEIGHTS"
  --data "$DATA"
  --split "$SPLIT"
  --classes "${CLASSES[@]}"
  --modes $MODES
  --attacks $ATTACKS
  --batch "$BATCH"
  --imgsz "$IMGSZ"
  --seed "$SEED"
  --device "$DEVICE"
  --workers "$WORKERS"
  --project "$PROJECT"
  --name "$NAME"
)

if [[ -n "$ATTACK_WEIGHTS" ]]; then
  VAL_ARGS+=(--attack_weights "$ATTACK_WEIGHTS")
fi

echo "==> Running local Unified FGD + STN Defense validation:"
echo "python val_fgd_stn_run.py ${VAL_ARGS[*]}"
python val_fgd_stn_run.py "${VAL_ARGS[@]}"
echo "=== Validation Finished at: $(date) ==="
