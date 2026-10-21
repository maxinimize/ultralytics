#!/usr/bin/env bash
# Local Ubuntu launcher for train_adv_test_run.py (single NVIDIA GPU).
# Run inside tmux. This file does not depend on SLURM.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f "$ROOT/train_adv_test_run.py" || ! -f "$ROOT/.yolo_env/bin/activate" ]]; then
  echo "ERROR: Place this script in the Ultralytics project root (with .yolo_env)." >&2
  exit 1
fi
source "$ROOT/.yolo_env/bin/activate"

# Original job-script positionals, with environment variable overrides:
#   bash train_adv_test_local.sh [attack_weights] [data_yaml] [epochs] [attack_names] [attack_ratios]
ATTACK_WEIGHTS="${ATTACK_WEIGHTS:-${1:-current}}"
DATA="${DATA:-${2:-coco_train_traffic5k.yaml}}"
EPOCHS="${EPOCHS:-${3:-10}}"
ATTACK_NAME="${ATTACK_NAME:-${4:-mim}}"
ATTACK_RATIO="${ATTACK_RATIO:-${5:-0.6}}"

# RTX 5090 (32 GiB): conservative starting point; adjust after checking VRAM.
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolo12l.pt}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-8}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
CACHE="${CACHE:-True}"  # RAM cache can be enabled explicitly after checking system RAM
PROJECT="${PROJECT:-$ROOT/runs}"
JOB_NAME="${JOB_NAME:-local_adv}"
RESUME="${RESUME:-}"     # Must be an existing runs/.../weights/last.pt for a resume
CLASSES="${CLASSES:-0 1 2 3 5 6 7 9 11 12}"

export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-4}"
export PYTHONUNBUFFERED=1
# Optional: override this variable externally if the allocator causes a CUDA issue.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"

# Resolve the dataset YAML.
if [[ -f "$DATA" ]]; then
  DATA_ARG="$(realpath "$DATA")"
elif [[ -f "$ROOT/ultralytics/cfg/datasets/$DATA" ]]; then
  DATA_ARG="$ROOT/ultralytics/cfg/datasets/$DATA"
else
  echo "ERROR: Dataset YAML not found: $DATA" >&2
  exit 1
fi

if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
  echo "ERROR: Resume checkpoint not found: $RESUME" >&2
  exit 1
fi
read -r -a ATTACK_NAMES <<< "$ATTACK_NAME"
read -r -a ATTACK_RATIOS <<< "$ATTACK_RATIO"
read -r -a CLASS_IDS <<< "$CLASSES"
if (( ${#ATTACK_NAMES[@]} == 0 || ${#ATTACK_RATIOS[@]} == 0 )); then
  echo 'ERROR: ATTACK_NAME and ATTACK_RATIO must be non-empty.' >&2
  exit 1
fi

# Preserve the original job naming scheme and log each launch separately.
WEIGHTS_TAG="$(basename "$ATTACK_WEIGHTS" .pt)"
DATA_TAG="$(basename "$DATA" .yaml)"
NAME="${NAME:-${JOB_NAME}_${WEIGHTS_TAG}_${DATA_TAG}_ep${EPOCHS}_${ATTACK_NAME// /_}_${ATTACK_RATIO// /_}}"
mkdir -p "$ROOT/logs" "$PROJECT"
LOG_FILE="$ROOT/logs/${NAME}-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[local adversarial train] $(date -Is)"
echo "Project root: $ROOT"
echo "Experiment: $NAME"
echo "Dataset YAML: $DATA_ARG"
echo "GPU: $DEVICE  Batch: $BATCH  Workers: $WORKERS  Cache: $CACHE"
echo "Log: $LOG_FILE"
python - <<'PY'
import torch
print('PyTorch:', torch.__version__, '| torch CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA is not available in .yolo_env'
print('GPU:', torch.cuda.get_device_name(0))
PY

ARGS=(
  --model "$YOLO_WEIGHTS"
  --attack_weights "$ATTACK_WEIGHTS"
  --data "$DATA_ARG"
  --imgsz "$IMGSZ"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --device "$DEVICE"
  --workers "$WORKERS"
  --cache "$CACHE"
  --attack_name "${ATTACK_NAMES[@]}"
  --attack_ratio "${ATTACK_RATIOS[@]}"
  --project "$PROJECT"
  --name "$NAME"
)
if (( ${#CLASS_IDS[@]} > 0 )); then ARGS+=(--classes "${CLASS_IDS[@]}"); fi
if [[ -n "$RESUME" ]]; then ARGS+=(--resume "$RESUME"); fi

printf 'Command: python -u train_adv_test_run.py'; printf ' %q' "${ARGS[@]}"; echo
python -u train_adv_test_run.py "${ARGS[@]}"
echo "Training finished: $(date -Is)"
