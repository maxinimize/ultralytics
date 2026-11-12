#!/bin/bash
#SBATCH --job-name=staged_at
#SBATCH --time=0-23:59
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# SLURM Job Script for SHARCNET / Compute Canada Clusters.
# Staged Adaptive Adversarial Training with Frozen FGD and STN for YOLO12l.
set -euo pipefail

# === Modules & Virtual Environment ===
module load StdEnv/2023 || true
module load gcc/12.3 || true
module load python/3.11 || true
module load cuda/12.2 || true
module load opencv/4.11.0 || true

# OpenCV Python bindings on Compute Canada
if [ -d "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages" ]; then
  export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:${PYTHONPATH:-}
fi

if [ -f ".yolo_env/bin/activate" ]; then
  source .yolo_env/bin/activate
elif [ -f "${HOME}/.yolo_env/bin/activate" ]; then
  source "${HOME}/.yolo_env/bin/activate"
fi

# BLAS & Threading Guards
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# PyTorch Allocator & NCCL Guards
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export TORCH_NCCL_ASYNC_HANDLING=1
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0

# Determine processes and worker threads
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC_PER_NODE=${SLURM_GPUS_ON_NODE}
else
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || true)
  NPROC_PER_NODE=${NGPU:-1}
fi

CPUS_ALLOC="${SLURM_CPUS_PER_TASK:-8}"
THREADS_PER_PROC=$(( CPUS_ALLOC / NPROC_PER_NODE ))
if [ "$THREADS_PER_PROC" -lt 1 ]; then THREADS_PER_PROC=1; fi
export OMP_NUM_THREADS=$THREADS_PER_PROC
export OMP_THREAD_LIMIT=$THREADS_PER_PROC

NUM_WORKERS=$(( THREADS_PER_PROC - 1 ))
if [ "$NUM_WORKERS" -lt 1 ]; then NUM_WORKERS=1; fi

# === Configuration & Arguments ===
STAGE="${1:-${STAGE:-1}}"
STAGE1_WEIGHTS="${2:-${STAGE1_WEIGHTS:-}}"

PREPROCESS_MODE="${PREPROCESS_MODE:-stn_fgd}"
FGD_WEIGHTS="${FGD_WEIGHTS:-fgd_yolo12l.pt}"
STN_WEIGHTS="${STN_WEIGHTS:-runs/detect/runs/train_stn/stn_train_yolo12l_coco_train_traffic5k_ep50_b8_k10_deg30.0_tr0.10/weights/best.pt}"
DATA="${DATA:-coco_train_traffic5k.yaml}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolo12l.pt}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
LR0="${LR0:-1e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0.0}"
WARMUP_BIAS_LR="${WARMUP_BIAS_LR:-0.0}"
TRAIN_HEAD_ONLY="${TRAIN_HEAD_ONLY:-true}"
FREEZE_BN="${FREEZE_BN:-true}"
ATTACK_WEIGHTS="${ATTACK_WEIGHTS:-current}"
ATTACK_THROUGH_PREPROCESS="${ATTACK_THROUGH_PREPROCESS:-true}"
STRICT_ATTACKS="${STRICT_ATTACKS:-true}"
CACHE="${CACHE:-True}"
AMP="${AMP:-bf16}"
FRACTION="${FRACTION:-1.0}"
CLASSES="${CLASSES:-0 1 2 3 5 6 7 9 11 12}"
PROJECT="${PROJECT:-${PWD}/runs/train_adv_fgd_stn_staged}"
JOB_ID="${SLURM_JOB_ID:-local}"

if [ "$STAGE" -eq 1 ]; then
  EPOCHS="${EPOCHS:-5}"
  P_CLEAN="${P_CLEAN:-1.0}"
  ADAPTIVE_RATIO="${ADAPTIVE_RATIO:-false}"
  NAME="stage1_clean_finetune_${PREPROCESS_MODE}_ep${EPOCHS}"
elif [ "$STAGE" -eq 2 ]; then
  EPOCHS="${EPOCHS:-20}"
  P_CLEAN="${P_CLEAN:-0.65}"
  ADAPTIVE_RATIO="${ADAPTIVE_RATIO:-true}"
  TAU="${TAU:-5.0}"
  UPDATE_INTERVAL="${UPDATE_INTERVAL:-5}"
  ATTACK_NAMES="${ATTACK_NAMES:-pgd bim mim worstk}"
  ATTACK_RATIOS="${ATTACK_RATIOS:-0.12 0.05 0.10 0.08}"
  EVAL_WEIGHTS="${EVAL_WEIGHTS:-0.20 0.20 0.15 0.15 0.30}"
  NAME="stage2_adaptive_at_${PREPROCESS_MODE}_ep${EPOCHS}"

  if [ -z "$STAGE1_WEIGHTS" ]; then
    DEFAULT_S1="$PROJECT/stage1_clean_finetune/weights/best.pt"
    if [ -f "$DEFAULT_S1" ]; then
      STAGE1_WEIGHTS="$DEFAULT_S1"
      echo "==> Auto-detected Stage 1 best.pt: $STAGE1_WEIGHTS"
    fi
  fi
else
  echo "ERROR: Invalid stage '$STAGE'. Supported: 1 or 2." >&2
  exit 1
fi

mkdir -p logs "${PROJECT}"
LOG_FILE="logs/${NAME}-${JOB_ID}.out"
exec > "${LOG_FILE}" 2>&1

echo "==================== Environment ===================="
echo "Host: $(hostname)"
echo "Stage: $STAGE ($([ "$STAGE" -eq 1 ] && echo "Clean Fine-tuning" || echo "Adaptive AT"))"
echo "Experiment: ${NAME}"
echo "GPUs: ${NPROC_PER_NODE} | Workers: ${NUM_WORKERS} | OMP Threads: ${OMP_NUM_THREADS}"
echo "Preprocess Mode: ${PREPROCESS_MODE} | Train Head Only: ${TRAIN_HEAD_ONLY} | Freeze BN: ${FREEZE_BN}"
echo "FGD: ${FGD_WEIGHTS} | STN: ${STN_WEIGHTS}"
if [ "$STAGE" -eq 2 ] && [ -n "$STAGE1_WEIGHTS" ]; then
  echo "Stage 1 Weights: ${STAGE1_WEIGHTS}"
fi
echo "Batch: ${BATCH} | Epochs: ${EPOCHS} | LR0: ${LR0} | Optimizer: ${OPTIMIZER}"
echo "Log: ${LOG_FILE}"
echo "====================================================="

# === Optional NVMe Dataset Staging ($SLURM_TMPDIR) ===
DATA_ARG="${DATA}"
DATASET_TAR="/project/6000635/czhan295/datasets/coco_traffic5k.tar"

if [ -n "${SLURM_TMPDIR:-}" ] && [ -f "${DATASET_TAR}" ]; then
  echo "==> Staging dataset tarball to local NVMe: $SLURM_TMPDIR..."
  LOCAL_DATA_DIR="${SLURM_TMPDIR}/datasets/coco"
  mkdir -p "${LOCAL_DATA_DIR}"
  tar -xf "${DATASET_TAR}" -C "${LOCAL_DATA_DIR}"

  ORIGINAL_YAML="ultralytics/cfg/datasets/${DATA}"
  if [ ! -f "${ORIGINAL_YAML}" ]; then
    ORIGINAL_YAML="${DATA}"
  fi

  if [ -f "${ORIGINAL_YAML}" ]; then
    CLEAN_DATA="$(basename "${DATA}" .yaml)"
    TMP_DATA_YAML="${SLURM_TMPDIR}/local_${CLEAN_DATA}.yaml"
    sed "s|^path:.*|path: ${LOCAL_DATA_DIR}|g" "${ORIGINAL_YAML}" > "${TMP_DATA_YAML}"
    DATA_ARG="${TMP_DATA_YAML}"
    echo "==> Using staged dataset YAML: ${DATA_ARG}"
  fi
fi

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
  --device 0
  --workers "$NUM_WORKERS"
  --amp "$AMP"
  --cache "$CACHE"
  --fraction "$FRACTION"
  --project "$PROJECT"
  --name "$NAME"
  --strict_attacks "$STRICT_ATTACKS"
)

if [ -n "${CLASSES}" ]; then
  read -r -a CLASS_IDS <<< "$CLASSES"
  ARGS+=(--classes "${CLASS_IDS[@]}")
fi

if [ "$STAGE" -eq 1 ]; then
  ARGS+=(
    --model "$YOLO_WEIGHTS"
    --p_clean 1.0
    --adaptive_ratio false
  )
elif [ "$STAGE" -eq 2 ]; then
  if [ -n "$STAGE1_WEIGHTS" ]; then
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

echo "Running command: python train_adv_fgd_stn_staged_run.py ${ARGS[*]}"
python train_adv_fgd_stn_staged_run.py "${ARGS[@]}"

