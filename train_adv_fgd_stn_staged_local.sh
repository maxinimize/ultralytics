#!/usr/bin/env bash
# Local launcher for train_adv_fgd_stn_staged_run.py (Single NVIDIA GPU).
# Staged Adaptive Adversarial Training with Frozen FGD and STN for YOLO12l.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f "$ROOT/train_adv_fgd_stn_staged_run.py" || ! -f "$ROOT/.yolo_env/bin/activate" ]]; then
  echo "ERROR: Place this script in the Ultralytics project root (with .yolo_env)." >&2
  exit 1
fi
source "$ROOT/.yolo_env/bin/activate"

# Stage parameter: 1 (Clean Fine-tuning) or 2 (Adaptive AT)
STAGE="${STAGE:-${1:-2}}"
STAGE1_WEIGHTS="${STAGE1_WEIGHTS:-${2:-}}"
# STAGE1_WEIGHTS="${STAGE1_WEIGHTS:-${2:-runs/train_adv_fgd_stn_staged/stage1_clean_finetune_stn_fgd_ep5/weights/best.pt}}"

# Shared configuration
PREPROCESS_MODE="${PREPROCESS_MODE:-stn_fgd}"
FGD_WEIGHTS="${FGD_WEIGHTS:-fgd_yolo12l.pt}"
STN_WEIGHTS="${STN_WEIGHTS:-runs/detect/runs/train_stn/stn_train_yolo12l_coco_train_traffic5k_ep50_b8_k10_deg30.0_tr0.10/weights/best.pt}"
DATA="${DATA:-coco_train_traffic5k.yaml}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolo12l.pt}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-4}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
LR0="${LR0:-1e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0.0}"
WARMUP_BIAS_LR="${WARMUP_BIAS_LR:-0.0}"
TRAIN_HEAD_ONLY="${TRAIN_HEAD_ONLY:-true}"
FREEZE_BN="${FREEZE_BN:-true}"
ATTACK_THROUGH_PREPROCESS="${ATTACK_THROUGH_PREPROCESS:-false}"
ATTACK_WEIGHTS="${ATTACK_WEIGHTS:-current}"
STRICT_ATTACKS="${STRICT_ATTACKS:-true}"
CACHE="${CACHE:-True}"
AMP="${AMP:-bf16}"
FRACTION="${FRACTION:-1.0}"
CLASSES="${CLASSES:-0 1 2 3 5 6 7 9 11 12}"
PROJECT="${PROJECT:-$ROOT/runs/train_adv_fgd_stn_staged}"
RESUME="${RESUME:-}"
SEED="${SEED:-0}"

# Stage-specific settings
if [[ "$STAGE" -eq 1 ]]; then
  EPOCHS="${EPOCHS:-5}"
  P_CLEAN="${P_CLEAN:-1.0}"
  ADAPTIVE_RATIO="${ADAPTIVE_RATIO:-false}"
  JOB_NAME="${JOB_NAME:-stage1_clean_finetune}"
  NAME="${NAME:-${JOB_NAME}_${PREPROCESS_MODE}_ep${EPOCHS}}"
elif [[ "$STAGE" -eq 2 ]]; then
  EPOCHS="${EPOCHS:-20}"
  P_CLEAN="${P_CLEAN:-0.65}"
  ADAPTIVE_RATIO="${ADAPTIVE_RATIO:-true}"
  TAU="${TAU:-5.0}"
  UPDATE_INTERVAL="${UPDATE_INTERVAL:-5}"
  ATTACK_NAMES="${ATTACK_NAMES:-pgd bim mim worstk}"
  ATTACK_RATIOS="${ATTACK_RATIOS:-0.12 0.05 0.10 0.08}"
  EVAL_WEIGHTS="${EVAL_WEIGHTS:-0.20 0.20 0.20 0.20 0.20}"
  JOB_NAME="${JOB_NAME:-stage2_adaptive_at}"
  NAME="${NAME:-${JOB_NAME}_${PREPROCESS_MODE}_ep${EPOCHS}}"

  # If stage1_weights was not explicitly provided, search default Stage 1 location
  if [[ -z "$STAGE1_WEIGHTS" ]]; then
    DEFAULT_S1="$PROJECT/stage1_clean_finetune/weights/best.pt"
    if [[ -f "$DEFAULT_S1" ]]; then
      STAGE1_WEIGHTS="$DEFAULT_S1"
      echo "==> Using detected Stage 1 best.pt: $STAGE1_WEIGHTS"
    else
      echo "WARNING: STAGE1_WEIGHTS was not specified and $DEFAULT_S1 was not found." >&2
      echo "Stage 2 will initialize from --model: $YOLO_WEIGHTS." >&2
    fi
  fi
else
  echo "ERROR: Invalid stage '$STAGE'. Supported stages are 1 or 2." >&2
  exit 1
fi

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

# Verify FGD and STN checkpoints
if [[ "$PREPROCESS_MODE" =~ (fgd|stn_fgd|fgd_stn) && ! -f "$FGD_WEIGHTS" ]]; then
  echo "ERROR: FGD checkpoint not found: $FGD_WEIGHTS. Cannot silently fall back to identity." >&2
  exit 1
fi
if [[ "$PREPROCESS_MODE" =~ (stn|stn_fgd|fgd_stn) && ! -f "$STN_WEIGHTS" ]]; then
  echo "ERROR: STN checkpoint not found: $STN_WEIGHTS. Cannot silently fall back to identity." >&2
  exit 1
fi

read -r -a CLASS_IDS <<< "$CLASSES"

mkdir -p "$ROOT/logs" "$PROJECT"
LOG_FILE="$ROOT/logs/${NAME}-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================="
echo "[Staged FGD+STN Training] $(date -Is)"
echo "Stage: $STAGE ($([[ "$STAGE" -eq 1 ]] && echo "Clean Fine-tuning" || echo "Adaptive AT"))"
echo "Project root: $ROOT"
echo "Experiment: $NAME"
echo "Preprocess Mode: $PREPROCESS_MODE | Train Head Only: $TRAIN_HEAD_ONLY | Freeze BN: $FREEZE_BN"
echo "FGD: $FGD_WEIGHTS | STN: $STN_WEIGHTS"
if [[ "$STAGE" -eq 2 && -n "$STAGE1_WEIGHTS" ]]; then
  echo "Stage 1 Best Weights: $STAGE1_WEIGHTS"
fi
echo "Dataset YAML: $DATA_ARG"
echo "Epochs: $EPOCHS | Batch: $BATCH | Imgsz: $IMGSZ | LR0: $LR0 | Optimizer: $OPTIMIZER"
echo "GPU Device: $DEVICE | Workers: $WORKERS | AMP: $AMP | Cache: $CACHE"
echo "Log File: $LOG_FILE"
echo "================================================================="

ARGS=(
  --stage "$STAGE"
  --preprocess_mode "$PREPROCESS_MODE"
  --fgd_weights "$FGD_WEIGHTS"
  --stn_weights "$STN_WEIGHTS"
  --train_head_only "$TRAIN_HEAD_ONLY"
  --freeze_bn "$FREEZE_BN"
  --data "$DATA_ARG"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --imgsz "$IMGSZ"
  --optimizer "$OPTIMIZER"
  --lr0 "$LR0"
  --warmup_epochs "$WARMUP_EPOCHS"
  --warmup_bias_lr "$WARMUP_BIAS_LR"
  --device "$DEVICE"
  --workers "$WORKERS"
  --seed "$SEED"
  --amp "$AMP"
  --cache "$CACHE"
  --fraction "$FRACTION"
  --project "$PROJECT"
  --name "$NAME"
  --strict_attacks "$STRICT_ATTACKS"
)

if [[ -n "${CLASS_IDS[*]:-}" ]]; then
  ARGS+=(--classes "${CLASS_IDS[@]}")
fi

if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi

if [[ "$STAGE" -eq 1 ]]; then
  ARGS+=(
    --model "$YOLO_WEIGHTS"
    --p_clean 1.0
    --adaptive_ratio false
  )
elif [[ "$STAGE" -eq 2 ]]; then
  if [[ -n "$STAGE1_WEIGHTS" ]]; then
    ARGS+=(--stage1_weights "$STAGE1_WEIGHTS")
  else
    ARGS+=(--model "$YOLO_WEIGHTS")
  fi

  read -r -a ATTACK_NAME_ARR <<< "$ATTACK_NAMES"
  read -r -a ATTACK_RATIO_ARR <<< "$ATTACK_RATIOS"
  read -r -a EVAL_WEIGHT_ARR <<< "$EVAL_WEIGHTS"

  ARGS+=(
    --p_clean "$P_CLEAN"
    --adaptive_ratio "$ADAPTIVE_RATIO"
    --tau "$TAU"
    --update_interval "$UPDATE_INTERVAL"
    --attack_weights "$ATTACK_WEIGHTS"
    --attack_through_preprocess "$ATTACK_THROUGH_PREPROCESS"
    --attack_name "${ATTACK_NAME_ARR[@]}"
    --attack_ratio "${ATTACK_RATIO_ARR[@]}"
    --eval_weights "${EVAL_WEIGHT_ARR[@]}"
  )
fi

echo "Executing: python train_adv_fgd_stn_staged_run.py ${ARGS[*]}"
python train_adv_fgd_stn_staged_run.py "${ARGS[@]}"

