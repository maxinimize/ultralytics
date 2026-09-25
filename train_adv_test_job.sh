#!/bin/bash
#SBATCH --job-name=train
# #SBATCH --account=def-rsolisob
#SBATCH --time=0-2:59        
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
# #SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
# #SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
# #SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# #SBATCH --qos=devel
#SBATCH --exclude=rg21806

set -euo pipefail

# === Modules & venv ===
module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load cuda/12.2
module load opencv/4.11.0

# OpenCV for cv2
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:$PYTHONPATH

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

THREADS_PER_PROC=$(( SLURM_CPUS_PER_TASK / NPROC_PER_NODE ))
if [ "$THREADS_PER_PROC" -lt 1 ]; then THREADS_PER_PROC=1; fi
export OMP_NUM_THREADS=$THREADS_PER_PROC
export OMP_THREAD_LIMIT=$THREADS_PER_PROC

NUM_WORKERS=$(( THREADS_PER_PROC - 1 ))
if [ "$NUM_WORKERS" -lt 1 ]; then NUM_WORKERS=1; fi

# Global batch size
GLOBAL_BATCH=16

# === Experiment & Log configuration ===
JOB_NAME="${SLURM_JOB_NAME:-train}"
JOB_ID="${SLURM_JOB_ID:-local}"

# === MANUAL SETTINGS ===
ATTACK_WEIGHTS="${1:-"current"}"
DATA="${2:-"coco_train_traffic5k.yaml"}"
EPOCHS="${3:-"100"}"
ATTACK_NAME="${4:-"mim worstk"}"
ATTACK_RATIO="${5:-"0.4 0.5"}"

# Dynamically calculate attack_num from attack_name
read -ra ATK_ARR <<< "${ATTACK_NAME}"
ATTACK_NUM="${#ATK_ARR[@]}"

# Sanitize components for clean file/folder names (remove extensions, replace spaces with underscores)
CLEAN_WEIGHTS="$(basename "${ATTACK_WEIGHTS}" .pt)"
CLEAN_DATA="$(basename "${DATA}" .yaml)"
CLEAN_ATTACK_NAME="${ATTACK_NAME// /_}"
CLEAN_ATTACK_RATIO="${ATTACK_RATIO// /_}"

# Concatenate NAME: job_name + attack_weights + data + epochs + attack_name + attack_ratio
NAME="${JOB_NAME}_${CLEAN_WEIGHTS}_${CLEAN_DATA}_ep${EPOCHS}_${CLEAN_ATTACK_NAME}_${CLEAN_ATTACK_RATIO}"

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
echo "Log File: ${LOG_FILE}"
echo "====================="

# Train
# python train_adv_test_run.py \
#   --model=runs/train_dp_5ep/weights/last.pt \
#   --attack_weights=yolo12l.pt \
#   --data=coco_train.yaml \
#   --imgsz=640 \
#   --epochs=2 \
#   --batch=${GLOBAL_BATCH} \
#   --device=0 \
#   --workers=${NUM_WORKERS} \
#   --attack_name=dp \
#   --project=runs \
#   --name=train_dp_5ep \
#   --resume

# python train_adv_test_run.py \
#   --model=yolo12l.pt \
#   --attack_weights=yolo12l.pt \
#   --data=coco_train_traffic.yaml \
#   --classes 1 2 3 5 7 \
#   --imgsz=640 \
#   --epochs=50 \
#   --batch=${GLOBAL_BATCH} \
#   --device=0 \
#   --workers=${NUM_WORKERS} \
#   --attack_num=3 \
#   --attack_name="pgd bim mim" \
#   --attack_ratio="0.08" \
#   --project=runs_new \
#   --name=train_online_pgd_bim_mim_traffic_class_mod_0.08_new \

# python train_adv_test_run.py \
#   --model=runs_new/train_online_pgd_bim_mim_traffic_class_mod_0.25_new/weights/last.pt \
#   --attack_weights=yolo12l.pt \
#   --data=coco_train_traffic.yaml \
#   --classes 1 2 3 5 7 \
#   --classes 0 1 2 3 5 6 7 9 11 12 \
#   --imgsz=640 \
#   --epochs=50 \
#   --batch=${GLOBAL_BATCH} \
#   --device=0 \
#   --workers=${NUM_WORKERS} \
#   --attack_num=3 \
#   --attack_name="pgd bim mim" \
#   --attack_ratio="0.25 0.25 0.25" \
#   --project=runs_new \
#   --name=train_online_pgd_mim_traffic_class_mod_0.75_new \
#   --resume=runs_new/train_online_pgd_bim_mim_traffic_class_mod_0.25_new/weights/last.pt

# python train_adv_test_run.py \
#   --model=yolo12l.pt \
#   --attack_weights="${ATTACK_WEIGHTS}" \
#   --data="${DATA}" \
#   --classes 0 1 2 3 5 6 7 9 11 12 \
#   --imgsz=640 \
#   --epochs="${EPOCHS}" \
#   --batch=${GLOBAL_BATCH} \
#   --device=0 \
#   --workers=${NUM_WORKERS} \
#   --attack_num="${ATTACK_NUM}" \
#   --attack_name="${ATTACK_NAME}" \
#   --attack_ratio="${ATTACK_RATIO}" \
#   --project=runs \
#   --name="${NAME}" \
#   # --resume="runs/${NAME}/weights/last.pt" \

python train_adv_test_run.py \
  --device=0 \
  --batch=${GLOBAL_BATCH} \
  --workers=${NUM_WORKERS} \
  --resume="runs/train_current_coco_train_traffic5k_ep100_mim_worstk_0.4_0.5/weights/last.pt"

# 2>&1 | grep --line-buffered -v "expandable_segments: memory mapping failed"