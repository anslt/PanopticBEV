#!/bin/bash
#SBATCH --job-name="panoptic_bev"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4,VRAM:48G
#SBATCH --mem=64G
#SBATCH --time=16:30:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/shil/kitti360/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/shil/kitti360/logs/slurm-%j.out

module load cuda/11.1 cudnn

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29303 scripts/train_panoptic_bev.py \
    --run_name=kitti360_`date +"%Y_%m_%d_%H_%M_%S"` \
    --project_root_dir=$(pwd) \
    --seam_root_dir=/usr/stud/shil/storage/slurm/shil/kitti360/kitti360_panopticbev \
    --dataset_root_dir=/usr/stud/shil/storage/slurm/shil/kitti360/kitti360_panopticbev \
    --mode=train \
    --train_dataset=Kitti360 \
    --val_dataset=Kitti360 \
    --config=kitti1.ini \
    --resume=/usr/stud/shil/storage/user/shil/PanopticBEV_org/PanopticBEV/experiments/bev_train_kitti360_2022_04_06_00_09_48/saved_models/model_latest.pth

