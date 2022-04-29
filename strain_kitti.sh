#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 \
python3 -m torch.distributed.launch --nproc_per_node=3 --master_port=29621 scripts/train_panoptic_bev.py \
    --run_name=kitti360_ipm_`date +"%Y_%m_%d_%H_%M_%S"` \
    --project_root_dir=$(pwd) \
    --seam_root_dir=/home/data2/kitti360/kitti360_panopticbev \
    --dataset_root_dir=/home/data2/kitti360/KITTI-360 \
    --mode=train \
    --train_dataset=Kitti360 \
    --val_dataset=Kitti360 \
    --config=kitti1.ini

