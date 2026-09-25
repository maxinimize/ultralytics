#!/bin/bash
#SBATCH --job-name=hydra_adv_pipe
# #SBATCH --account=def-rsolisob
#SBATCH --time=0-11:59
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
# #SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

# ==============================================================================
# 0. Environment and module loading (Compute Canada / Slurm)
# ==============================================================================
module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load cuda/12.2
module load opencv/4.11.0

# OpenCV Python path
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:${PYTHONPATH:-}

# Activate Python virtual environment
source .yolo_env/bin/activate

# Threading and CUDA memory allocation optimization
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export PYTHONUNBUFFERED=1

# ==============================================================================
# 1. Experiment configuration (supports command-line arguments or environment variable override)
#    Usage example:
#      sbatch run_hydra_adv_pipeline_job.sh \
#         runs/your_adv_model/weights/best.pt \
#         coco_train_traffic5k.yaml \
#         0.5 \
#         "mim worstk" \
#         "0.4 0.5"
# ==============================================================================
INPUT_MODEL="${1:-"runs/train_current_coco_train_traffic5k_ep100_pgd_0.6/weights/best.pt"}"
DATA="${2:-"coco_train_traffic5k.yaml"}"
EVAL_DATA="${EVAL_DATA:-"coco_train_traffic5k_test.yaml"}" # Dedicated evaluation dataset YAML for Stage 4
KEEP_RATIO="${3:-"0.5"}"                   # 0.5 retains 50% weights (prunes 50%)
ATTACK_NAME="${4:-"pgd"}"           # Attack types (space-separated)
ATTACK_RATIO="${5:-"0.6"}"      # Corresponding adversarial sample ratio(s)
SCORE_EPOCHS="${6:-"20"}"                  # Stage 1 score search epochs
FINETUNE_EPOCHS="${7:-"30"}"               # Stage 2 fixed-mask fine-tuning epochs
BATCH="${8:-"16"}"                         # Batch size
IMGSZ="${9:-"640"}"                        # Image size
DEVICE="${10:-"0"}"                        # GPU device ID
WORKERS="${WORKERS:-4}"                    # DataLoader worker count
CLASSES="${CLASSES:-"0 1 2 3 5 6 7 9 11 12"}"  # Target class indices (e.g., 10 traffic categories)

# Stage execution switches (1: run, 0: skip)
RUN_STAGE1="${RUN_STAGE1:-1}"              # Stage 1: Robust score search
RUN_STAGE2="${RUN_STAGE2:-1}"              # Stage 2: Fixed-mask fine-tuning
RUN_STAGE3="${RUN_STAGE3:-1}"              # Stage 3: Model weight materialization
RUN_STAGE4="${RUN_STAGE4:-1}"              # Stage 4: Clean and adversarial evaluation

# ==============================================================================
# Dynamic logging and experiment directory configuration (merged stdout & stderr into single .out)
# ==============================================================================
JOB_NAME="${SLURM_JOB_NAME:-hydra_adv_pipe}"
JOB_ID="${SLURM_JOB_ID:-local}"

if [[ "${INPUT_MODEL}" == *"/weights/"* ]]; then
    CLEAN_MODEL="$(basename "$(dirname "$(dirname "${INPUT_MODEL}")")")_$(basename "${INPUT_MODEL}" .pt)"
else
    CLEAN_MODEL="$(basename "${INPUT_MODEL}" .pt)"
fi

CLEAN_DATA="$(basename "${DATA}" .yaml)"
CLEAN_ATK="${ATTACK_NAME// /_}"
CLEAN_ATK_RATIO="${ATTACK_RATIO// /_}"

# Construct experiment name
NAME="${JOB_NAME}_${CLEAN_MODEL}_${CLEAN_DATA}_k${KEEP_RATIO}_ep${SCORE_EPOCHS}_${FINETUNE_EPOCHS}_${CLEAN_ATK}_${CLEAN_ATK_RATIO}"

# Unified output redirection (stdout and stderr written to single .out file)
mkdir -p logs
LOG_FILE="logs/${NAME}-${JOB_ID}.out"
exec > "${LOG_FILE}" 2>&1

PIPELINE_DIR="$(pwd)/runs/${NAME}-${JOB_ID}"
mkdir -p "${PIPELINE_DIR}"

echo "========================================================================"
echo "Starting HYDRA Adversarial Pruning Pipeline"
echo "Job ID:         ${JOB_ID}"
echo "Host:           $(hostname)"
echo "Experiment:     ${NAME}"
echo "Log File:       ${LOG_FILE}"
echo "Pipeline Dir:   ${PIPELINE_DIR}"
echo "Input Model:    ${INPUT_MODEL}"
echo "Dataset YAML:   ${DATA}"
echo "Eval Test YAML: ${EVAL_DATA}"
echo "Classes:        ${CLASSES}"
echo "Keep Ratio:     ${KEEP_RATIO} (Sparsity: $(awk "BEGIN {print 1 - ${KEEP_RATIO}}"))"
echo "Attack Names:   ${ATTACK_NAME}"
echo "Attack Ratios:  ${ATTACK_RATIO}"
echo "Score Epochs:   ${SCORE_EPOCHS}"
echo "Finetune Epochs:${FINETUNE_EPOCHS}"
echo "Batch Size:     ${BATCH}"
echo "Workers:        ${WORKERS}"
echo "CUDA Device:    $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo "========================================================================"

# Stage 1 intermediate output path
STAGE1_BEST="${PIPELINE_DIR}/stage1_score/weights/best.pt"
# Stage 2 intermediate output path
STAGE2_BEST="${PIPELINE_DIR}/stage2_finetune/weights/best.pt"
# Stage 3 materialized output path
MATERIALIZED_MODEL="${PIPELINE_DIR}/pruned_materialized.pt"

# ==============================================================================
# Stage 1: Robust Score Search
# ==============================================================================
if [ "${RUN_STAGE1}" -eq 1 ]; then
    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Starting Stage 1: Robust Score Search <<<<<"
    python -u tools/hydra_prune.py \
        --model "${INPUT_MODEL}" \
        --data "${DATA}" \
        --keep-ratio "${KEEP_RATIO}" \
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
        echo "Error: Stage 1 failed to generate expected model ${STAGE1_BEST}"
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
    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Starting Stage 2: Fixed-Mask Fine-tuning <<<<<"
    python -u tools/hydra_finetune.py \
        --model "${STAGE1_BEST}" \
        --data "${DATA}" \
        --keep-ratio "${KEEP_RATIO}" \
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
        echo "Error: Stage 2 failed to generate expected model ${STAGE2_BEST}"
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
    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Starting Stage 3: Model Materialization (HydraConv2d -> nn.Conv2d) <<<<<"
    python -u tools/hydra_export.py \
        --model "${STAGE2_BEST}" \
        --output "${MATERIALIZED_MODEL}" \
        --imgsz "${IMGSZ}" \
        --device "${DEVICE}" \
        --verify

    if [ ! -f "${MATERIALIZED_MODEL}" ]; then
        echo "Error: Stage 3 failed to generate materialized model ${MATERIALIZED_MODEL}"
        exit 1
    fi
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Stage 3 completed! Standard YOLO weights exported to: ${MATERIALIZED_MODEL} <<<<<"
else
    echo ">>>>> Skipping Stage 3 (RUN_STAGE3=${RUN_STAGE3}) <<<<<"
fi

# ==============================================================================
# Stage 4: Clean and Adversarial Evaluation
# ==============================================================================
if [ "${RUN_STAGE4}" -eq 1 ]; then
    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Starting Stage 4: Comprehensive Evaluation (Clean + Adversarial Attacks, data=${EVAL_DATA}) <<<<<"
    python -u tools/hydra_eval.py \
        --model "${MATERIALIZED_MODEL}" \
        --data "${EVAL_DATA}" \
        --batch "${BATCH}" \
        --imgsz "${IMGSZ}" \
        --device "${DEVICE}" \
        --attacks clean pgd bim mim \
        ${CLASSES:+--classes ${CLASSES}}

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] >>>>> Stage 4 evaluation completed! <<<<<"
else
    echo ">>>>> Skipping Stage 4 (RUN_STAGE4=${RUN_STAGE4}) <<<<<"
fi

echo -e "\n========================================================================"
echo "HYDRA pipeline completed successfully!"
echo "Final materialized pruned model saved to: ${MATERIALIZED_MODEL}"
echo "Pipeline directory: ${PIPELINE_DIR}"
echo "========================================================================"
