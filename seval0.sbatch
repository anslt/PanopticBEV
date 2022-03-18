#!/bin/bash
#SBATCH --job-name="panoptic_bev"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=64G
#SBATCH --time=5-0:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/shil/nuscenes/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/shil/nuscenes/logs/slurm-%j.out

module load cuda/11.1 cudnn

CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --master_port=29550 scripts/eval_panoptic_bev.py \
    --run_name=nuscenes_`date +"%Y_%m_%d_%H_%M_%S"` \
    --project_root_dir=$(pwd)\
    --seam_root_dir=/usr/stud/shil/storage/slurm/shil/nuScenes_panopticbev \
    --dataset_root_dir=/usr/stud/shil/storage/slurm/shil/nuscenes/trainval \
    --mode=test \
    --test_dataset=nuScenes \
    --config=nuscenes0.ini \
    --resume=/usr/stud/shil/storage/user/shil/PanopticBEV_mine/PanopticBEV/experiments/bev_train_nuscenes_2022_02_06_02_17_36/saved_models/model_best.pth


