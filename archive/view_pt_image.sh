module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load cuda/12.2
module load opencv/4.11.0

export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/CUDA/gcc12/cuda12.2/opencv/4.11.0/lib/python3.11/site-packages:$PYTHONPATH

source .yolo_env/bin/activate

python view_pt_image.py ../datasets/coco/images/val2017/000000006723_pgd.pt
