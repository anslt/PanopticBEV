import numpy as np
import cv2
from PIL import Image
import umsgpack

with open("/usr/stud/shil/storage/slurm/shil/kitti360/kitti360_panopticbev/metadata_ortho.bin", "rb") as fid:
    metadata = umsgpack.unpack(fid, encoding="utf-8")

with open("/usr/stud/shil/storage/slurm/shil/kitti360/kitti360_panopticbev/split/train.txt", "r") as fid:
    lst = fid.readlines()
    lst = [line.strip() for line in lst]

front_msk_frames = os.listdir("/usr/stud/shil/storage/slurm/shil/kitti360/kitti360_panopticbev/front_msk_seam")
front_msk_frames = [frame.split(".")[0] for frame in front_msk_frames]
lst = [entry for entry in lst if entry in front_msk_frames]
lst = set(lst)  # Remove any potential duplicates

meta = metadata["meta"]
images = [img_desc for img_desc in metadata["images"] if img_desc["id"] in lst]

print(len(images))
freq = np.zeros(11)
count = 0
for image in images:
    bev_msk_file = os.path.join("/usr/stud/shil/storage/slurm/shil/kitti360/kitti360_panopticbev/bev_msk/bev_ortho", "{}.png".format(image['id']))
    bev_msk = np.array(Image.open(bev_msk_file), dtype=int32)
    cat = image["cat"]
    size = 768 * 704
    for idx, cat_c in enumerate(cat):
        if cat_c != 255:
            freq[cat_c] += np.sum(bev_msk == idx) / size
    if (count + 1) % 100 == 0:
        print(count)
