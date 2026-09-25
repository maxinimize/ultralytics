#!/bin/bash
#SBATCH --job-name=hydra_m2_test
#SBATCH --account=def-rsolisob_gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# To use a full H100 GPU:
# #SBATCH --gres=gpu:h100:1

# To use MIG, replace the line above with the MIG slice supported by the cluster, e.g.:
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1

#SBATCH --output=hydra_m2_test_%j.out
#SBATCH --error=hydra_m2_test_%j.err


echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "========================================"


# Navigate to the Ultralytics project directory
cd /lustre09/project/6000635/czhan295/ultralytics || exit 1


# Load required modules and environment
module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load cuda/12.2
module load opencv/4.11.0


export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:$PYTHONPATH


# Activate the YOLO virtual environment
source .yolo_env/bin/activate


echo
echo "Python:"
which python
python --version

echo
echo "CUDA:"
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY


echo
echo "========================================"
echo "Starting HYDRA Milestone-2 trainer test"
echo "========================================"

python m2_trainer_test.py

EXIT_CODE=$?

echo
echo "========================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE