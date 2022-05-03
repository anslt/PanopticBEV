#!/bin/bash
#SBATCH --job-name="panoptic_bev"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4,VRAM:48G
#SBATCH --mem=32G
#SBATCH --time=5-0:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/shil/nuscenes/ipm_logs/slurm-%j.out
#SBATCH --error=/storage/slurm/shil/nuscenes/ipm_logs/slurm-%j.out

module load cuda/11.1 cudnn

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29626 train_panoptic_bev_ipm_eps.py \
    --run_name=nuscenes_ipm_eps_`date +"%Y_%m_%d_%H_%M_%S"` \
    --project_root_dir=$(pwd) \
    --seam_root_dir=/usr/stud/shil/storage/slurm/shil/nuScenes_panopticbev \
    --dataset_root_dir=/usr/stud/shil/storage/slurm/shil/nuscenes/trainval \
    --mode=train \
    --train_dataset=nuScenes \
    --val_dataset=nuScenes \
    --config=nuscenes_eps.ini

