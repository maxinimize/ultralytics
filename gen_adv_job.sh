#!/bin/bash
#SBATCH --job-name=yolov12_gen
# #SBATCH --account=def-rsolisob
#SBATCH --time=0-2:59        
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
# #SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
# #SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=logs/%x-%j.out  
# #SBATCH --qos=devel

set -euo pipefail

# Load necessary modules
module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load cuda/12.2
module load opencv/4.11.0

# set OpenCV path for cv2
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:$PYTHONPATH

# activate virtual environment
source .yolo_env/bin/activate

#run generation
python generate_pregenerated_adv.py \
  --model yolo12l.pt \
  --data ultralytics/cfg/datasets/coco_train_traffic.yaml \
  --attack_name mim \
  --imgsz 640 \
  --device 0 \
  --batch-size 16 \
  --split val