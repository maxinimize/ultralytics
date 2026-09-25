#!/bin/bash
#SBATCH --job-name=yolov8_val
# #SBATCH --account=def-rsolisob
#SBATCH --time=0-2:59        
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
# #SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
# #SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
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

# run validation
# python val_adv_run.py \
#   --weights=runs/yolo12l.pt \
#   --data=coco_val.yaml \
#   --attack_weights=yolo12l.pt \
#   --attack_name=dp \
#   --project=runs \
#   --name=val_dp_on_raw

# python val_adv_run.py \
#   --weights=yolo12l.pt \
#   --data=coco_val.yaml \
#   --project=runs \
#   --name=val_raw_on_raw


# python val_adv_run.py \
#   --weights=runs/train_yolo12l_coco_train_traffic5h_ep50_worstk_0.6/weights/best.pt \
#   --data=coco_train_traffic5h_test.yaml \
#   --classes 1 2 3 5 7 \
#   --attack_weights=yolo12l.pt \
#   --attack_name=worstk \
#   --project=runs \
#   --name=val_yolo12l_coco_train_traffic5k_ep50_worstk_0.6_worstk

# python val_adv_run.py \
#   --weights=runs/train_yolo12l_coco_train_traffic5h_ep100_mim_worstk_0.4_0.5/weights/best.pt \
#   --data=coco_train_traffic5h_test.yaml \
#   --classes 1 2 3 5 7 \
#   --attack_weights=yolo12l.pt \
#   --attack_name=worstk \
#   --worstk_k=10 \
#   --batch=16 \
#   --project=runs \
#   --name=val_yolo12l_coco_train_traffic5h_ep100_mim_worstk_0.4_0.5_worstk

# python val_adv_run.py \
#   --weights=runs/train_yolo12l_coco_train_traffic5k_ep50_pgd_bim_mim_0.25_0.25_0.25/weights/best.pt \
#   --data=coco_train_traffic5k_test.yaml \
#   --classes 0 1 2 3 5 6 7 9 11 12 \
#   --project=runs \
#   --name=val_yolo12l_coco_train_traffic5k_ep50_pgd_bim_mim_0.25_0.25_0.25_raw

# python val_adv_run.py \
#   --weights=yolo12l.pt \
#   --data=coco_train_traffic5k_test.yaml \
#   --classes 0 1 2 3 5 6 7 9 11 12 \
#   --project=runs \
#   --name=val_yolo12l_coco_train_traffic5k_raw


python val_adv_run.py \
  --weights=yolo12l.pt \
  --data=coco_train_traffic5k_test.yaml \
  --classes 0 1 2 3 5 6 7 9 11 12 \
  --project=runs \
  --name=val_current_coco_train_traffic5k_ep100_pgd_0.6_pgd_bb


