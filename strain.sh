#!/bin/bash
#SBATCH --job-name="panoptic_bev"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4,VRAM:48G
#SBATCH --mem=64G
#SBATCH --time=7-0:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/shil/nuscenes/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/shil/nuscenes/logs/slurm-%j.out

# module load cuda/11.1 cudnn

CUDA_VISIBLE_DEVICES=0,1,2 \
python3 -m torch.distributed.launch --nproc_per_node=3 --master_port=29520 scripts/train_panoptic_bev.py \
    --run_name=nuscenes_`date +"%Y_%m_%d_%H_%M_%S"` \
    --project_root_dir=$(pwd) \
    --seam_root_dir=/home/data2/nuscenes/nuScenes_panopticbev/ \
    --dataset_root_dir=/home/data2/nuscenes \
    --mode=train \
    --train_dataset=nuScenes \
    --val_dataset=nuScenes \
    --config=nuscenes_server.ini
