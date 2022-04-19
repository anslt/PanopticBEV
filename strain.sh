#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1,2 \
python3 -m torch.distributed.launch --nproc_per_node=3 --master_port=29720 scripts/train_panoptic_bev.py \
    --run_name=nuscenes_pon_`date +"%Y_%m_%d_%H_%M_%S"` \
    --project_root_dir=$(pwd)  \
    --seam_root_dir=/home/data2/nuscenes/nuScenes_panopticbev/ \
    --dataset_root_dir=/home/data2/nuscenes \
    --mode=train \
    --train_dataset=nuScenes \
    --val_dataset=nuScenes \
    --config=nuscenes.ini

