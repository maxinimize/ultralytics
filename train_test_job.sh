#!/bin/bash
#SBATCH --job-name=yolov12_train_test
# #SBATCH --account=def-rsolisob
#SBATCH --time=0-11:59        
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/%x-%j.out

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

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

# NUM_WORKERS=$(( THREADS_PER_PROC - 1 ))
NUM_WORKERS=$(( THREADS_PER_PROC / 2 ))  # 4 / 2 = 2
if [ "$NUM_WORKERS" -lt 1 ]; then NUM_WORKERS=1; fi

# Global batch size
GLOBAL_BATCH=16

mkdir -p logs

# # Run training with ratio parameter
# python train_test_run.py \
#   --model=runs/train_m1_s2/weights/last.pt \
#   --attack_weights=yolo12l.pt \
#   --data=coco_train_temp.yaml \
#   --imgsz=640 \
#   --epochs=25 \
#   --batch=${GLOBAL_BATCH} \
#   --device=0 \
#   --workers=${NUM_WORKERS} \
#   --attack_name="pgd bim mim" \
#   --ratio="0.3 0.2 0.1" \
#   --project=runs \
#   --name=train_m1_s3

python train_test_run.py \
  --model=yolo12l.pt \
  --attack_weights=yolo12l.pt \
  --data=coco_train_temp.yaml \
  --imgsz=640 \
  --epochs=300 \
  --batch=${GLOBAL_BATCH} \
  --device=0 \
  --workers=${NUM_WORKERS} \
  --attack_name="pgd bim mim" \
  --ratio="0.25 0.25 0.4" \
  --project=runs \
  --name=train_pregen_pgd_bim_mim_252540_300eps


# # Resume training with ratio parameter from the last checkpoint
# python train_test_run.py \
#   --resume=runs/train_pgd_bim_mim_252540/weights/last.pt \
#   --device=0 \
#   --batch=${GLOBAL_BATCH} \
#   --workers=${NUM_WORKERS}

