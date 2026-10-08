#!/bin/bash
#SBATCH --job-name=fgd_train
#SBATCH --time=0-23:59
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

# Training settings: edit here, then submit with `sbatch train_fgd_run.sh`.
MODEL="runs/train_current_coco_train_traffic5k_ep100_pgd_0.6/weights/best.pt"
DATA="coco_train_traffic5k.yaml"
CLASSES=(0 1 2 3 5 6 7 9 11 12)
ATTACK_NAMES=(pgd bim mim)
EPOCHS=50
BATCH=4
CACHE="ram"
# ATTACK_WEIGHTS="$MODEL"
# ATTACK_RATIOS=(0.3333333333333333 0.3333333333333333 0.3333333333333333)
# IMGSZ=640
# LR0="0.0001"
# SEED=0
# MULTI_SCALE=0.0
# SAVE_PERIOD=-1
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))
PROJECT="runs"
RESUME=""  # Set to a weights/last.pt path to continue training.

cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL_TAG="${MODEL##*/}"
MODEL_TAG="${MODEL_TAG%.pt}"
# ATTACK_WEIGHTS_TAG="${ATTACK_WEIGHTS##*/}"
# ATTACK_WEIGHTS_TAG="${ATTACK_WEIGHTS_TAG%.pt}"
DATA_TAG="${DATA##*/}"
DATA_TAG="${DATA_TAG%.yaml}"
ATTACK_TAG="${ATTACK_NAMES[*]}"
ATTACK_TAG="${ATTACK_TAG// /_}"
# RATIO_TAG="${ATTACK_RATIOS[*]}"
# RATIO_TAG="${RATIO_TAG// /_}"
CLASS_TAG="${CLASSES[*]}"
CLASS_TAG="${CLASS_TAG// /_}"
NAME="${SLURM_JOB_NAME:-fgd_train}_${MODEL_TAG}_${DATA_TAG}_ep${EPOCHS}_b${BATCH}_${ATTACK_TAG}_cls${CLASS_TAG}"

mkdir -p logs
LOG_FILE="logs/${NAME}-${SLURM_JOB_ID:-local}.out"
exec > "$LOG_FILE" 2>&1
echo "Experiment: $NAME"
echo "Log: $LOG_FILE"

module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load cuda/12.2
module load opencv/4.11.0

export PYTHONPATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:${PYTHONPATH:-}"
source .yolo_env/bin/activate

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export PYTHONUNBUFFERED=1

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
    TMP_DATA_YAML="${SLURM_TMPDIR}/local_${DATA_TAG}.yaml"
    sed "s|^path:.*|path: ${LOCAL_DATA_DIR}|g" "${ORIGINAL_YAML}" > "${TMP_DATA_YAML}"
    DATA_ARG="${TMP_DATA_YAML}"
    echo "==> Using local dataset YAML: ${DATA_ARG}"
  fi
else
  echo "==> No local tarball detected. Using default dataset path from ${DATA}"
fi

TRAIN_ARGS=(
  --model "$MODEL"
  --data "$DATA_ARG"
  --classes "${CLASSES[@]}"
  --cache "$CACHE"
  --attack_name "${ATTACK_NAMES[@]}"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --device 0
  --workers "$WORKERS"
  --project "$PROJECT"
  --name "$NAME"
  # --attack_weights "$ATTACK_WEIGHTS"
  # --attack_ratio "${ATTACK_RATIOS[@]}"
  # --imgsz "$IMGSZ"
  # --lr0 "$LR0"
  # --seed "$SEED"
  # --multi_scale "$MULTI_SCALE"
  # --save_period "$SAVE_PERIOD"
)
if [[ -n "$RESUME" ]]; then
  TRAIN_ARGS+=(--resume "$RESUME")
fi

python -u train_fgd_run.py "${TRAIN_ARGS[@]}"
