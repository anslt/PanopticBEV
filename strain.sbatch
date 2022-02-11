#!/bin/bash
#SBATCH --job-name="panoptic_bev"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2,VRAM:48G
#SBATCH --mem=16G
#SBATCH --time=5-0:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/shil/nuscenes/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/shil/nuscenes/logs/slurm-%j.out

module load cuda/11.1 cudnn

CUDA_VISIBLE_DEVICES=0,1 \
CUDA_LAUNCH_BLOCKING=1 \
python3 -m torch.distributed.launch --nproc_per_node=2 --port=29503 scripts/train_panoptic_bev.py \
    --run_name=nuscenes_`date +"%Y_%m_%d_%H_%M_%S"` \
    --project_root_dir=/usr/stud/shil/storage/user/shil/PanopticBEV_mine/PanopticBEV \
    --seam_root_dir=/usr/stud/shil/storage/slurm/shil/nuScenes_panopticbev \
    --dataset_root_dir=/usr/stud/shil/storage/slurm/shil/nuscenes/trainval \
    --mode=train \
    --train_dataset=nuScenes \
    --val_dataset=nuScenes \
    --config=nuscenes.ini