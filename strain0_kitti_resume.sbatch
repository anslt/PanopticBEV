#!/bin/bash
#SBATCH --job-name="panoptic_bev"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=16G
#SBATCH --time=4-0:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/shil/kitti360/ipm_logs/slurm-%j.out
#SBATCH --error=/storage/slurm/shil/kitti360/ipm_logs/slurm-%j.out

module load cuda/11.1 cudnn

CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --master_port=29611 scripts/train_panoptic_bev.py \
    --run_name=kitti360_ipm_`date +"%Y_%m_%d_%H_%M_%S"` \
    --project_root_dir=/usr/stud/shil/storage/user/shil/PanopticBEV_mine/PanopticBEV \
    --seam_root_dir=/usr/stud/shil/storage/slurm/shil/kitti360/kitti360_panopticbev \
    --dataset_root_dir=/usr/stud/shil/storage/slurm/shil/kitti360/kitti360_panopticbev \
    --mode=train \
    --train_dataset=Kitti360 \
    --val_dataset=Kitti360 \
    --config=kitti0.ini \
    --resume=/usr/stud/shil/storage/user/shil/PanopticBEV_ipm/PanopticBEV/experiments/bev_train_kitti360_ipm_2022_03_05_02_08_48/saved_models/model_latest.pth


