#!/bin/bash
#SBATCH --job-name="panoptic_bev"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/shil/nuscenes/example_logs/slurm-%j.out
#SBATCH --error=/storage/slurm/shil/nuscenes/example_logs/slurm-%j.out

module load cuda/11.1 cudnn

CUDA_VISIBLE_DEVICES=0 \
python3 scripts/train_panoptic_bev.py \
    --run_name=nuscenes_`date +"%Y_%m_%d_%H_%M_%S"` \
    --project_root_dir=/usr/stud/shil/storage/user/shil/PanopticBEV_mine/PanopticBEV \
    --seam_root_dir=/usr/stud/shil/storage/user/shil/PanopticBEV_mine/PanopticBEV/nuScenes_panopticbev_mini2 \
    --dataset_root_dir=/usr/stud/shil/storage/user/shil/PanopticBEV_mine/PanopticBEV/nuScenes_panopticbev_mini2 \
    --mode=train \
    --train_dataset=nuScenes \
    --val_dataset=nuScenes \
    --config=nuscenes_example.ini 


