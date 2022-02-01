python -m script/train_panoptic_bev.py \
    --run_name=nuscenes_`date +"%Y_%m_%d_%H_%M_%S"` \
    --project_root_dir="D:/MASTER/TUM/2021_2_fall/guided_research2/old_code/PanopticBEV" \
    --seam_root_dir="D:/MASTER/TUM/2021_2_fall/guided_research2/data/nuScenes_panopticbev_mini2" \
    --dataset_root_dir="D:/MASTER/TUM/2021_2_fall/guided_research2/data/nuScenes_panopticbev_mini2" \
    --mode=train \
    --train_dataset=nuScenes \
    --val_dataset=nuScenes \
    --config=nuscenes_example.ini
