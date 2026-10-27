#!/usr/bin/env bash
# ==============================================================================
# Local Ubuntu Launcher for HYDRA Adversarial Pruning Pipeline
# Supports Unstructured Pruning and Plan A (Kernel-level Structured Pruning)
# Designed for interactive or background execution inside tmux (No SLURM required)
# ==============================================================================

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ==============================================================================
# 0. Local Python Environment Activation & Thread/CUDA Optimizations
# ==============================================================================
if [[ -f "$ROOT/.yolo_env/bin/activate" ]]; then
    source "$ROOT/.yolo_env/bin/activate"
fi

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"

# ==============================================================================
# 1. Pipeline Arguments & Environment Variable Overrides
#    Usage:
#      bash run_hydra_adv_pipeline_local.sh \
#         [input_model] [data_yaml] [keep_ratio] [attack_name] [attack_ratio] \
#         [score_epochs] [finetune_epochs] [batch] [imgsz] [device] [pruning_mode]
# ==============================================================================
DEFAULT_MODEL="runs/local_adv_adaptive_current_coco_train_traffic5k_ep100_pgd_bim_mim_worstk_0.10_0.10_0.40_0.15/weights/best.pt"
if [[ ! -f "$DEFAULT_MODEL" && -f "$ROOT/yolo12l.pt" ]]; then
    DEFAULT_MODEL="yolo12l.pt"
fi

INPUT_MODEL="${1:-"${INPUT_MODEL:-$DEFAULT_MODEL}"}"
DATA="${2:-"${DATA:-"coco_train_traffic5k.yaml"}"}"
EVAL_DATA="${EVAL_DATA:-"coco_train_traffic5k_test.yaml"}"
KEEP_RATIO="${3:-"${KEEP_RATIO:-"0.5"}"}"
ATTACK_NAME="${4:-"${ATTACK_NAME:-"pgd bim mim worstk"}"}"
ATTACK_RATIO="${5:-"${ATTACK_RATIO:-"0.20 0.275 0.17 0.105"}"}"
SCORE_EPOCHS="${6:-"${SCORE_EPOCHS:-"20"}"}"
FINETUNE_EPOCHS="${7:-"${FINETUNE_EPOCHS:-"30"}"}"
BATCH="${8:-"${BATCH:-"16"}"}"
IMGSZ="${9:-"${IMGSZ:-"640"}"}"
DEVICE="${10:-"${DEVICE:-"0"}"}"
PRUNING_MODE="${11:-"${PRUNING_MODE:-"kernel"}"}"
WORKERS="${WORKERS:-4}"
CLASSES="${CLASSES:-"0 1 2 3 5 6 7 9 11 12"}"

# Stage switches (1: execute, 0: skip)
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_STAGE3="${RUN_STAGE3:-1}"
RUN_STAGE4="${RUN_STAGE4:-1}"

# ==============================================================================
# 2. Experiment Naming and Live Console + File Logging (tee)
# ==============================================================================
JOB_NAME="local_hydra"
JOB_ID="local_$(date +%Y%m%d_%H%M%S)"

if [[ "${INPUT_MODEL}" == *"/weights/"* ]]; then
    CLEAN_MODEL="$(basename "$(dirname "$(dirname "${INPUT_MODEL}")")")_$(basename "${INPUT_MODEL}" .pt)"
else
    CLEAN_MODEL="$(basename "${INPUT_MODEL}" .pt)"
fi

CLEAN_DATA="$(basename "${DATA}" .yaml)"
CLEAN_ATK="${ATTACK_NAME// /_}"
CLEAN_ATK_RATIO="${ATTACK_RATIO// /_}"

NAME="${JOB_NAME}_${CLEAN_MODEL}_${CLEAN_DATA}_mode${PRUNING_MODE}_k${KEEP_RATIO}_ep${SCORE_EPOCHS}_${FINETUNE_EPOCHS}_${CLEAN_ATK}_${CLEAN_ATK_RATIO}"

mkdir -p "$ROOT/logs"
LOG_FILE="$ROOT/logs/${NAME}-${JOB_ID}.log"

# In tmux/terminal, tee output simultaneously to console and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

PIPELINE_DIR="$ROOT/runs/${NAME}-${JOB_ID}"
mkdir -p "${PIPELINE_DIR}"

echo "========================================================================"
echo "Starting HYDRA Adversarial Pruning Pipeline (Local tmux Mode)"
echo "Timestamp:      $(date +'%Y-%m-%d %H:%M:%S')"
echo "Host:           $(hostname)"
echo "Experiment:     ${NAME}"
echo "Log File:       ${LOG_FILE}"
echo "Pipeline Dir:   ${PIPELINE_DIR}"
echo "Pruning Mode:   ${PRUNING_MODE}"
echo "Input Model:    ${INPUT_MODEL}"
echo "Dataset YAML:   ${DATA}"
echo "Eval YAML:      ${EVAL_DATA}"
echo "Keep Ratio:     ${KEEP_RATIO} (Target Sparsity: $(awk "BEGIN {print 1 - ${KEEP_RATIO}}"))"
echo "Attack Config:  ${ATTACK_NAME} (ratio: ${ATTACK_RATIO})"
echo "Epochs:         Score: ${SCORE_EPOCHS}, Finetune: ${FINETUNE_EPOCHS}"
echo "Batch Size:     ${BATCH}, Imgsz: ${IMGSZ}"
echo "Device:         ${DEVICE}, Workers: ${WORKERS}"
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} (Count: {torch.cuda.device_count()})")
PY
echo "========================================================================"

STAGE1_BEST="${PIPELINE_DIR}/stage1_score/weights/best.pt"
STAGE2_BEST="${PIPELINE_DIR}/stage2_finetune/weights/best.pt"
MATERIALIZED_MODEL="${PIPELINE_DIR}/pruned_materialized.pt"

# ==============================================================================
# Stage 1: Robust Score Search
# ==============================================================================
if [ "${RUN_STAGE1}" -eq 1 ]; then
    if [ ! -f "${INPUT_MODEL}" ]; then
        echo "Error: Input model '${INPUT_MODEL}' not found!" >&2
        exit 1
    fi

    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Starting Stage 1: Robust Score Search (${PRUNING_MODE}) <<<<<"
    python -u tools/hydra_prune.py \
        --model "${INPUT_MODEL}" \
        --data "${DATA}" \
        --keep-ratio "${KEEP_RATIO}" \
        --pruning-mode "${PRUNING_MODE}" \
        --epochs "${SCORE_EPOCHS}" \
        --batch "${BATCH}" \
        --imgsz "${IMGSZ}" \
        --workers "${WORKERS}" \
        --device "${DEVICE}" \
        --attack_name ${ATTACK_NAME} \
        --attack_ratio ${ATTACK_RATIO} \
        ${CLASSES:+--classes ${CLASSES}} \
        --project "${PIPELINE_DIR}" \
        --name "stage1_score"

    if [ ! -f "${STAGE1_BEST}" ]; then
        echo "Error: Stage 1 failed to generate expected model ${STAGE1_BEST}" >&2
        exit 1
    fi
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Stage 1 completed! Score weights saved to: ${STAGE1_BEST} <<<<<"
else
    echo ">>>>> Skipping Stage 1 (RUN_STAGE1=${RUN_STAGE1}) <<<<<"
fi

# ==============================================================================
# Stage 2: Robust Fixed-Mask Fine-tuning
# ==============================================================================
if [ "${RUN_STAGE2}" -eq 1 ]; then
    FINETUNE_INIT="${STAGE1_BEST}"
    if [ "${RUN_STAGE1}" -eq 0 ]; then
        FINETUNE_INIT="${INPUT_MODEL}"
    fi

    if [ ! -f "${FINETUNE_INIT}" ]; then
        echo "Error: Stage 2 initialization model '${FINETUNE_INIT}' not found!" >&2
        exit 1
    fi

    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Starting Stage 2: Fixed-Mask Fine-tuning (${PRUNING_MODE}) <<<<<"
    python -u tools/hydra_finetune.py \
        --model "${FINETUNE_INIT}" \
        --data "${DATA}" \
        --keep-ratio "${KEEP_RATIO}" \
        --pruning-mode "${PRUNING_MODE}" \
        --epochs "${FINETUNE_EPOCHS}" \
        --batch "${BATCH}" \
        --imgsz "${IMGSZ}" \
        --workers "${WORKERS}" \
        --device "${DEVICE}" \
        --attack_name ${ATTACK_NAME} \
        --attack_ratio ${ATTACK_RATIO} \
        ${CLASSES:+--classes ${CLASSES}} \
        --project "${PIPELINE_DIR}" \
        --name "stage2_finetune"

    if [ ! -f "${STAGE2_BEST}" ]; then
        echo "Error: Stage 2 failed to generate expected model ${STAGE2_BEST}" >&2
        exit 1
    fi
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Stage 2 completed! Fine-tuned weights saved to: ${STAGE2_BEST} <<<<<"
else
    echo ">>>>> Skipping Stage 2 (RUN_STAGE2=${RUN_STAGE2}) <<<<<"
fi

# ==============================================================================
# Stage 3: Materialization to Standard YOLO
# ==============================================================================
if [ "${RUN_STAGE3}" -eq 1 ]; then
    EXPORT_INIT="${STAGE2_BEST}"
    if [ "${RUN_STAGE2}" -eq 0 ]; then
        EXPORT_INIT="${STAGE1_BEST:-"${INPUT_MODEL}"}"
    fi

    if [ ! -f "${EXPORT_INIT}" ]; then
        echo "Error: Stage 3 input model '${EXPORT_INIT}' not found!" >&2
        exit 1
    fi

    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Starting Stage 3: Model Materialization (${PRUNING_MODE}) <<<<<"
    python -u tools/hydra_export.py \
        --model "${EXPORT_INIT}" \
        --output "${MATERIALIZED_MODEL}" \
        --pruning-mode "${PRUNING_MODE}" \
        --imgsz "${IMGSZ}" \
        --device "${DEVICE}" \
        --verify

    if [ ! -f "${MATERIALIZED_MODEL}" ]; then
        echo "Error: Stage 3 failed to generate materialized model ${MATERIALIZED_MODEL}" >&2
        exit 1
    fi
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Stage 3 completed! Materialized weights saved to: ${MATERIALIZED_MODEL} <<<<<"
else
    echo ">>>>> Skipping Stage 3 (RUN_STAGE3=${RUN_STAGE3}) <<<<<"
fi

# ==============================================================================
# Stage 4: Clean and Adversarial Evaluation & Efficiency Benchmark
# ==============================================================================
if [ "${RUN_STAGE4}" -eq 1 ]; then
    EVAL_TARGET="${MATERIALIZED_MODEL}"
    if [ "${RUN_STAGE3}" -eq 0 ]; then
        EVAL_TARGET="${STAGE2_BEST:-"${INPUT_MODEL}"}"
    fi

    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Starting Stage 4: Evaluation & Benchmark (${EVAL_DATA}) <<<<<"
    python -u tools/hydra_eval.py \
        --model "${EVAL_TARGET}" \
        --data "${EVAL_DATA}" \
        --batch "${BATCH}" \
        --imgsz "${IMGSZ}" \
        --device "${DEVICE}" \
        --attacks clean pgd bim mim \
        --benchmark \
        ${CLASSES:+--classes ${CLASSES}}

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Stage 4 evaluation completed! <<<<<"
else
    echo ">>>>> Skipping Stage 4 (RUN_STAGE4=${RUN_STAGE4}) <<<<<"
fi

echo -e "\n========================================================================"
echo "HYDRA pipeline completed successfully!"
echo "Final materialized model: ${MATERIALIZED_MODEL}"
echo "Pipeline artifacts dir:   ${PIPELINE_DIR}"
echo "Log file:                 ${LOG_FILE}"
echo "========================================================================"
