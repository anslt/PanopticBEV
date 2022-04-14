#!/bin/bash
#SBATCH --job-name="panoptic_bev"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:2,VRAM:48G
#SBATCH --mem=64G
#SBATCH --time=5-0:30:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/shil/kitti360/pon_logs/slurm-%j.out
#SBATCH --error=/storage/slurm/shil/kitti360/pon_logs/slurm-%j.out

module load cuda/11.1 cudnn

CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29702 scripts/train_panoptic_bev.py \
    --run_name=kitti360_pon_`date +"%Y_%m_%d_%H_%M_%S"` \
    --project_root_dir=$(pwd)  \
    --seam_root_dir=/usr/stud/shil/storage/slurm/shil/kitti360/kitti360_panopticbev \
    --dataset_root_dir=/usr/stud/shil/storage/slurm/shil/kitti360/kitti360_panopticbev \
    --mode=train \
    --train_dataset=Kitti360 \
    --val_dataset=Kitti360 \
    --config=kitti1.ini
