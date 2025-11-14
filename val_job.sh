#!/bin/bash
#SBATCH --job-name=yolov8_val
# #SBATCH --account=def-rsolisob
#SBATCH --time=0-1:00        
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --output=logs/%x-%j.out  
# #SBATCH --qos=devel

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

# run validation
yolo detect val model=yolov8x.pt data=coco128.yaml imgsz=640