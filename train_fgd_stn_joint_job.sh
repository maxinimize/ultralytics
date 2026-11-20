#!/usr/bin/env bash
#SBATCH --job-name=joint_fgd_stn
#SBATCH --time=0-23:59
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

# Get script root directory
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Capture whether runtime parameters were explicitly set before applying defaults
USER_DEVICE_SET=0; [[ -v DEVICE ]] && USER_DEVICE_SET=1
USER_WORKERS_SET=0; [[ -v WORKERS ]] && USER_WORKERS_SET=1
USER_EPOCHS_SET=0; [[ -v EPOCHS ]] && USER_EPOCHS_SET=1
USER_BATCH_SET=0; [[ -v BATCH ]] && USER_BATCH_SET=1
USER_SEED_SET=0; [[ -v SEED ]] && USER_SEED_SET=1
USER_DEVICE="${DEVICE:-}"
USER_WORKERS="${WORKERS:-}"
USER_EPOCHS="${EPOCHS:-}"
USER_BATCH="${BATCH:-}"
USER_SEED="${SEED:-}"

RESUME="${RESUME:-}"
if [[ "${1:-}" == "--resume" && -n "${2:-}" ]]; then
  RESUME="$2"
elif [[ "${1:-}" =~ ^--resume=(.+) ]]; then
  RESUME="${BASH_REMATCH[1]}"
fi

# Training configuration options and environment variable defaults
MODEL="${MODEL:-yolo12l.pt}"
DATA="${DATA:-coco_train_traffic5k.yaml}"
CLASSES=(0 1 2 3 5 6 7 9 11 12)
PREPROCESS_MODE="${PREPROCESS_MODE:-stn_fgd}"
INIT="${INIT:-pretrained}"
FGD_WEIGHTS="${FGD_WEIGHTS:-fgd_yolo12l.pt}"
STN_WEIGHTS="${STN_WEIGHTS:-runs/detect/runs/train_stn/stn_train_yolo12l_coco_train_traffic5k_ep50_b8_k10_deg30.0_tr0.10/weights/best.pt}"
ATTACK_WEIGHTS="${ATTACK_WEIGHTS:-current}"
ATTACK_THROUGH_PREPROCESS="${ATTACK_THROUGH_PREPROCESS:-true}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-4}"
IMGSZ="${IMGSZ:-640}"
LR_FGD="${LR_FGD:-0.00001}"
LR_STN="${LR_STN:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
LAMBDA_FGD="${LAMBDA_FGD:-1.0}"
LAMBDA_GEOM="${LAMBDA_GEOM:-1.0}"
LAMBDA_RECON="${LAMBDA_RECON:-1.0}"
LAMBDA_DET="${LAMBDA_DET:-0.0}"
LAMBDA_IDENTITY="${LAMBDA_IDENTITY:-0.1}"
P_CLEAN="${P_CLEAN:-0.25}"
ATTACK_NAMES="${ATTACK_NAMES:-pgd bim mim worstk}"
ATTACK_RATIOS="${ATTACK_RATIOS:-0.10 0.10 0.40 0.15}"
WORSTK_K="${WORSTK_K:-10}"
DEGREES="${DEGREES:-30.0}"
TRANSLATE="${TRANSLATE:-0.10}"
VAL_INTERVAL="${VAL_INTERVAL:-5}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))}"
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
CACHE="${CACHE:-ram}"
AMP="${AMP:-bf16}"
PROJECT="${PROJECT:-runs/train_fgd_stn_joint}"
RESUME="${RESUME:-}"
# Reproducibility seed (leave unset to inherit default cfg.seed; set explicitly e.g. SEED=0 to override)
SEED="${SEED:-}"

# Generate unique experiment name
MODEL_TAG="${MODEL##*/}"
MODEL_TAG="${MODEL_TAG%.pt}"
DATA_TAG="${DATA##*/}"
DATA_TAG="${DATA_TAG%.yaml}"
NAME="${SLURM_JOB_NAME:-joint_fgd_stn}_${PREPROCESS_MODE}_${INIT}_${MODEL_TAG}_${DATA_TAG}_ep${EPOCHS}_b${BATCH}"

# Initialize log output
mkdir -p "$ROOT/logs" "$ROOT/$PROJECT"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$ROOT/logs/${NAME}-${SLURM_JOB_ID:-local}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================================="
echo "=== SHARCNET FGD + STN Joint Defense Training: $NAME ==="
echo "=================================================================="
echo "Start Time: $(date)"
echo "Project Root: $ROOT"
echo "Log File: $LOG_FILE"
echo "Defense Order: $PREPROCESS_MODE | Init Mode: $INIT"
echo "FGD Weights: $FGD_WEIGHTS | STN Weights: $STN_WEIGHTS | YOLO Model: $MODEL"
echo "Batch Size: $BATCH | Image Size: $IMGSZ | Epochs: $EPOCHS | Mixed Precision: $AMP"
echo "Learning Rate: FGD=$LR_FGD, STN=$LR_STN | Weight Decay: $WEIGHT_DECAY"
echo "Loss Weights: fgd=$LAMBDA_FGD, geom=$LAMBDA_GEOM, recon=$LAMBDA_RECON, det=$LAMBDA_DET, identity=$LAMBDA_IDENTITY"
echo "Attack Through Defense Pipeline: $ATTACK_THROUGH_PREPROCESS | Clean Ratio: $P_CLEAN"
echo "Attack Types: $ATTACK_NAMES | Attack Ratios: $ATTACK_RATIOS"
echo "Worst-of-K Settings: k=$WORSTK_K, degrees=$DEGREES, translate=$TRANSLATE"
echo "=================================================================="

# Load cluster software environment modules
if command -v module &>/dev/null; then
  module load StdEnv/2023 || true
  module load gcc/12.3 || true
  module load python/3.11 || true
  module load cuda/12.2 || true
  module load opencv/4.11.0 || true
fi

# Activate Python virtual environment
if [[ -f "$ROOT/.yolo_env/bin/activate" ]]; then
  source "$ROOT/.yolo_env/bin/activate"
fi

# Set thread and memory allocation environment variables
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export PYTHONUNBUFFERED=1

# Check Python and PyTorch CUDA status
python - <<'PY'
import torch
print('PyTorch Version:', torch.__version__, '| CUDA Support:', torch.version.cuda)
assert torch.cuda.is_available(), 'Error: No available CUDA GPU detected in current environment!'
print('GPU Device:', torch.cuda.get_device_name(0))
PY

# Direct resume branch: use dataset/model from checkpoint, forward only explicitly set runtime options
if [[ -n "$RESUME" ]]; then
  if [[ ! -f "$RESUME" ]]; then
    echo "ERROR: Resume checkpoint not found: $RESUME" >&2
    exit 1
  fi

  LOG_PROJECT="${PROJECT:-runs/train_fgd_stn_joint}"
  LOG_NAME="${SLURM_JOB_NAME:-joint_fgd_stn}_resume_$(basename "$RESUME" .pt)"
  mkdir -p "$ROOT/logs" "$ROOT/$LOG_PROJECT"
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="$ROOT/logs/${LOG_NAME}-${SLURM_JOB_ID:-local}_${TIMESTAMP}.log"
  exec > >(tee -a "$LOG_FILE") 2>&1

  echo "=================================================================="
  echo "=== SHARCNET FGD + STN Joint Training Resume: $LOG_NAME ==="
  echo "=================================================================="
  echo "Start Time: $(date)"
  echo "Project Root: $ROOT"
  echo "Log File: $LOG_FILE"
  echo "Resume Checkpoint: $RESUME"
  echo "=================================================================="

  TRAIN_ARGS=(--resume "$RESUME")
  if (( USER_DEVICE_SET )) && [[ -n "$USER_DEVICE" ]]; then TRAIN_ARGS+=(--device "$USER_DEVICE"); fi
  if (( USER_WORKERS_SET )) && [[ -n "$USER_WORKERS" ]]; then TRAIN_ARGS+=(--workers "$USER_WORKERS"); fi
  if (( USER_EPOCHS_SET )) && [[ -n "$USER_EPOCHS" ]]; then TRAIN_ARGS+=(--epochs "$USER_EPOCHS"); fi
  if (( USER_BATCH_SET )) && [[ -n "$USER_BATCH" ]]; then TRAIN_ARGS+=(--batch "$USER_BATCH"); fi
  if (( USER_SEED_SET )) && [[ -n "$USER_SEED" ]]; then TRAIN_ARGS+=(--seed "$USER_SEED"); fi

  printf 'Executing resume training command: python -u train_fgd_stn_joint_run.py'; printf ' %q' "${TRAIN_ARGS[@]}"; echo
  python -u train_fgd_stn_joint_run.py "${TRAIN_ARGS[@]}"

  echo "Training finished at: $(date)"
  exit 0
fi

# Resolve dataset path with support for SLURM local fast SSD staging
DATA_ARG="${DATA}"
DATASET_TAR="/project/6000635/czhan295/datasets/coco_traffic5k.tar"

if [[ -n "${SLURM_TMPDIR:-}" ]] && [[ -f "${DATASET_TAR}" ]]; then
  echo "==> Extracting dataset to SLURM local fast temporary drive: $SLURM_TMPDIR..."
  LOCAL_DATA_DIR="${SLURM_TMPDIR}/datasets/coco"
  mkdir -p "${LOCAL_DATA_DIR}"
  tar -xf "${DATASET_TAR}" -C "${LOCAL_DATA_DIR}"

  ORIGINAL_YAML="$ROOT/ultralytics/cfg/datasets/${DATA}"
  if [[ ! -f "${ORIGINAL_YAML}" ]]; then
    ORIGINAL_YAML="${DATA}"
  fi

  if [[ -f "${ORIGINAL_YAML}" ]]; then
    TMP_DATA_YAML="${SLURM_TMPDIR}/local_${DATA_TAG}.yaml"
    sed "s|^path:.*|path: ${LOCAL_DATA_DIR}|g" "${ORIGINAL_YAML}" > "${TMP_DATA_YAML}"
    DATA_ARG="${TMP_DATA_YAML}"
    echo "==> Using local temporary drive dataset configuration: ${DATA_ARG}"
  fi
elif [[ -f "$ROOT/ultralytics/cfg/datasets/$DATA" ]]; then
  DATA_ARG="$ROOT/ultralytics/cfg/datasets/$DATA"
fi

# Parse attack parameters and arrays
read -r -a ATTACK_NAME_ARR <<< "$ATTACK_NAMES"
read -r -a ATTACK_RATIO_ARR <<< "$ATTACK_RATIOS"

TRAIN_ARGS=(
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
  --val_interval "$VAL_INTERVAL"
  --project "$PROJECT"
  --name "$NAME"
)

if [[ -n "${SEED:-}" ]]; then
  TRAIN_ARGS+=(--seed "$SEED")
fi

if [[ "$INIT" == "pretrained" ]]; then
  TRAIN_ARGS+=(--fgd_weights "$FGD_WEIGHTS" --stn_weights "$STN_WEIGHTS")
fi

if (( ${#CLASSES[@]} > 0 )); then
  TRAIN_ARGS+=(--classes "${CLASSES[@]}")
fi

if [[ -n "$RESUME" ]]; then
  TRAIN_ARGS+=(--resume "$RESUME")
fi

printf 'Executing training command: python -u train_fgd_stn_joint_run.py'; printf ' %q' "${TRAIN_ARGS[@]}"; echo
python -u train_fgd_stn_joint_run.py "${TRAIN_ARGS[@]}"

echo "Training finished at: $(date)"

