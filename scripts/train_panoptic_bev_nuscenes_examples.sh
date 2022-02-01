CUDA_VISIBLE_DEVICES=0 \
python3 -m train_panoptic_bev.py \
    --run_name=nuscenes_`date +"%Y_%m_%d_%H_%M_%S"` \
    --project_root_dir=/usr/stud/shil/storage/user/shil/PanopticBEV_mine/PanopticBEV \
    --seam_root_dir=/usr/stud/shil/storage/user/shil/PanopticBEV_mine/PanopticBEV/nuScenes_panopticbev_mini2 \
    --dataset_root_dir=/usr/stud/shil/storage/user/shil/PanopticBEV_mine/PanopticBEV/nuScenes_panopticbev_mini2 \
    --mode=train \
    --train_dataset=nuScenes \
    --val_dataset=nuScenes \
    --config=nuscenes.ini \
    --debug=True \
    --local_rank=0
