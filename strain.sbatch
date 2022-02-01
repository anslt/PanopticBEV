
CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch train_panoptic_bev.py \
    --run_name=nuscenes_`date +"%Y_%m_%d_%H_%M_%S"` \
    --project_root_dir=/usr/stud/shil/storage/user/shil/PanopticBEV \
    --seam_root_dir=/usr/stud/shil/storage/slurm/shil/nuScenes_panopticbev \
    --dataset_root_dir=/usr/stud/shil/storage/slurm/shil/nuscenes/trainval \
    --mode=train \
    --train_dataset=nuScenes \
    --val_dataset=nuScenes \
    --config=nuscenes.ini

