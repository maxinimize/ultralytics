#!/bin/bash
#SBATCH --job-name=train_fgd_phase2
# #SBATCH --account=def-rsolisob
#SBATCH --time=0-23:59        
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
# #SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
# #SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
# #SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# #SBATCH --qos=devel

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# === Modules & venv ===
module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load cuda/12.2
module load opencv/4.11.0

# OpenCV for cv2
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:${PYTHONPATH:-}

source .yolo_env/bin/activate  

# BLAS threading guard
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# PyTorch CUDA allocator
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# NCCL
export TORCH_NCCL_ASYNC_HANDLING=1
export NCCL_DEBUG=WARN

export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0

# Determine number of processes (GPUs) and set per-process threads
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  NPROC_PER_NODE=${SLURM_GPUS_ON_NODE}
else
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || true)
  NPROC_PER_NODE=${NGPU:-1}
fi
if [ "$NPROC_PER_NODE" -lt 1 ]; then NPROC_PER_NODE=1; fi

THREADS_PER_PROC=$(( ${SLURM_CPUS_PER_TASK:-8} / NPROC_PER_NODE ))
if [ "$THREADS_PER_PROC" -lt 1 ]; then THREADS_PER_PROC=1; fi
export OMP_NUM_THREADS=$THREADS_PER_PROC
export OMP_THREAD_LIMIT=$THREADS_PER_PROC

NUM_WORKERS=$(( THREADS_PER_PROC - 1 ))
if [ "$NUM_WORKERS" -lt 1 ]; then NUM_WORKERS=1; fi

# Global batch size
GLOBAL_BATCH="${BATCH:-32}"

# === Experiment & Log configuration ===
JOB_NAME="${SLURM_JOB_NAME:-train}"
JOB_ID="${SLURM_JOB_ID:-local}"

# Phase 2 settings; arguments/environment can override them without editing original scripts.
FGD_WEIGHTS="${FGD_WEIGHTS:-${1:-${PWD}/fgd_yolo12l.pt}}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-${PWD}/yolo12l.pt}"
ATTACK_WEIGHTS="${ATTACK_WEIGHTS:-current}"
DATA="${DATA:-${2:-coco_train_traffic5k.yaml}}"
EPOCHS="${EPOCHS:-${3:-100}}"
ATTACK_NAME="${ATTACK_NAME:-${4:-mim}}"
ATTACK_RATIO="${ATTACK_RATIO:-${5:-0.6}}"
USE_FGD="${USE_FGD:-true}"
ATTACK_THROUGH_FGD="${ATTACK_THROUGH_FGD:-true}"
EPSILON="${EPSILON:-0.031372549}"
STEPS="${STEPS:-5}"
STEP_SIZE="${STEP_SIZE:-0.00784313725}"
IMGSZ="${IMGSZ:-640}"
FRACTION="${FRACTION:-1.0}"
AMP="${AMP:-bf16}"
DEVICE="${DEVICE:-0}"
CACHE="${CACHE:-True}"
PROJECT="${PROJECT:-${PWD}/runs}"
# RESUME="${RESUME:-}"
RESUME="${RESUME:-runs/train_fgd_phase2_fgdphase2_current_coco_train_traffic5k_ep100_mim_0.6/weights/last.pt}"


# Sanitize components for clean file/folder names (remove extensions, replace spaces with underscores)
CLEAN_WEIGHTS="$(basename "${ATTACK_WEIGHTS}" .pt)"
CLEAN_DATA="$(basename "${DATA}" .yaml)"
CLEAN_ATTACK_NAME="${ATTACK_NAME// /_}"
CLEAN_ATTACK_RATIO="${ATTACK_RATIO// /_}"

# Concatenate NAME: job_name + attack_weights + data + epochs + attack_name + attack_ratio
NAME="${JOB_NAME}_fgdphase2_${CLEAN_WEIGHTS}_${CLEAN_DATA}_ep${EPOCHS}_${CLEAN_ATTACK_NAME}_${CLEAN_ATTACK_RATIO}"

mkdir -p logs
LOG_FILE="logs/${NAME}-${JOB_ID}.out"
exec > "${LOG_FILE}" 2>&1

# === Debugging info ===
echo "===== debug env ====="
echo "Host: $(hostname)"
echo "GPUs on node: ${SLURM_GPUS_ON_NODE:-<unset>}"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "OMP_NUM_THREADS (per-proc): $OMP_NUM_THREADS"
echo "DataLoader workers (per-proc): $NUM_WORKERS"
echo "Experiment Name: ${NAME}"
echo "Resume Checkpoint: ${RESUME:-<none>}"
echo "Log File: ${LOG_FILE}"
echo "====================="

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
    TMP_DATA_YAML="${SLURM_TMPDIR}/local_${CLEAN_DATA}.yaml"
    sed "s|^path:.*|path: ${LOCAL_DATA_DIR}|g" "${ORIGINAL_YAML}" > "${TMP_DATA_YAML}"
    DATA_ARG="${TMP_DATA_YAML}"
    echo "==> Using local dataset YAML: ${DATA_ARG}"
  fi
else
  echo "==> No local tarball detected. Using default dataset path from ${DATA}"
fi

TRAIN_ARGS=(
  --model="${YOLO_WEIGHTS}"
  --fgd_weights="${FGD_WEIGHTS}"
  --use_fgd="${USE_FGD}"
  --attack_through_fgd="${ATTACK_THROUGH_FGD}"
  --attack_epsilon="${EPSILON}"
  --attack_steps="${STEPS}"
  --attack_step_size="${STEP_SIZE}"
  --attack_weights="${ATTACK_WEIGHTS}"
  --data="${DATA_ARG}"
  --classes 0 1 2 3 5 6 7 9 11 12
  --imgsz="${IMGSZ}"
  --epochs="${EPOCHS}"
  --batch="${GLOBAL_BATCH}"
  --device="${DEVICE}"
  --workers="${NUM_WORKERS}"
  --cache="${CACHE}"
  --amp="${AMP}"
  --fraction="${FRACTION}"
  --attack_name="${ATTACK_NAME}"
  --attack_ratio="${ATTACK_RATIO}"
  --project="${PROJECT}"
  --name="${NAME}"
)

if [ -n "${RESUME}" ]; then
  if [ ! -f "${RESUME}" ]; then
    echo "ERROR: Resume checkpoint not found: ${RESUME}"
    exit 1
  fi
  echo "==> Resuming training from checkpoint: ${RESUME}"
  TRAIN_ARGS+=(--resume="${RESUME}")
fi

python -u train_adv_fgd_phase2_run.py "${TRAIN_ARGS[@]}"
