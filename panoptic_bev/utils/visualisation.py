import os
import cv2
import numpy as np
import torch
import random

from panoptic_bev.utils.sequence import pad_packed_images
from panoptic_bev.utils.city_labels_mc import labels as cs_labels
from panoptic_bev.utils.nuscenes_labels_mc import labels as nuscenes_labels

STUFF_COLOURS = [(128, 64,128), (244, 35,232), ( 70, 70, 70), (107, 142, 35), (152, 251, 152), (140, 140, 140)]
THING_COLOURS = [(56, 60, 200), (168, 240, 104), (24, 20, 140), (41, 102, 116),
                 (101, 224, 99), (51, 39, 96), (5, 72, 78), (4, 236, 154),
                 (220, 20, 60), (70, 130, 180), (119, 11, 32), (81, 0, 81)]


def visualise_bev(img, bev_gt, bev_pred, **varargs):
    vis_list = []

    img_unpack, _ = pad_packed_images(img)

    if img_unpack.size(0) > 1:
        img_unpack = img_unpack[0].unsqueeze(0)
    img_unpack = torch.nn.functional.interpolate(img_unpack, scale_factor = 2, mode = 'bilinear')

    for b in range(len(bev_gt)):
        vis = []
        bev_gt_unpack = get_panoptic_mask(bev_gt[b], varargs['num_stuff']).unsqueeze(0).cpu()
        bev_pred_unpack = get_panoptic_mask(bev_pred[b]['po_pred'], varargs['num_stuff']).unsqueeze(0).cpu()

        semantic_gt_unpack = get_semantic_mask(bev_gt[b], varargs['num_stuff']).unsqueeze(0).cpu()
        semantic_pred_unpack = get_semantic_mask(bev_pred[b]['po_pred'], varargs['num_stuff']).unsqueeze(0).cpu()

        # Visualise BEV as RGB
        for img in img_unpack:
            vis.append((recover_image(img.cpu(), varargs["rgb_mean"], varargs["rgb_std"]) * 255).type(torch.IntTensor))


        if bev_gt_unpack.shape[2] < img_unpack[0].shape[2]:
            vis_bev_pred = visualise_panoptic_mask_trainid(bev_pred_unpack, varargs['dataset'])
            vis_bev_gt = visualise_panoptic_mask_trainid(bev_gt_unpack, varargs['dataset'])

            # Add the error map and the masked output. The error map is only a semantic error map.
            vis_bev_pred_masked = vis_bev_pred.clone()
            vis_bev_pred_masked[:, bev_gt_unpack.squeeze(0) == 255] = 0  # Set invalid areas to 0

            semantic_bev_pred = visualise_panoptic_mask_trainid(semantic_pred_unpack, varargs['dataset'])
            semantic_bev_gt = visualise_panoptic_mask_trainid(semantic_gt_unpack, varargs['dataset'])

            semantic_bev_pred_masked = semantic_bev_pred.clone()
            semantic_bev_pred_masked[:, semantic_gt_unpack.squeeze(0) == 255] = 0  # Set invalid areas to 0

            error_map = torch.zeros_like(vis_bev_pred_masked)
            bev_pred_sem = bev_pred_unpack.clone()
            bev_gt_sem = bev_gt_unpack.clone()
            bev_pred_sem[bev_pred_sem > 1000] = bev_pred_sem[bev_pred_sem > 1000] // 1000
            bev_gt_sem[bev_gt_sem > 1000] = bev_gt_sem[bev_gt_sem > 1000] // 1000

            instance_bev_pred = vis_bev_pred.clone()
            instance_bev_gt = vis_bev_gt.clone()
            instance_bev_pred[:, bev_pred_sem < 1000] = 0
            instance_bev_gt[:, bev_gt_sem < 1000] = 0

            instance_bev_pred_masked = instance_bev_pred.clone()
            instance_bev_pred_masked[:, bev_gt_unpack.squeeze(0) == 255] = 0  # Set invalid areas to 0

            error_region = (bev_gt_sem != bev_pred_sem).squeeze(0)
            error_map[:, error_region] = 255
            error_map[:, bev_gt_unpack.squeeze(0) == 255] = 0

            # Row 1 --> FV img and error map for semantic
            vis[0] = torch.cat([vis[0], error_map], dim=2)
            # Row 2 --> semantic Pred, Masked Pred and GT
            vis.append(torch.cat([semantic_bev_pred, semantic_bev_pred_masked, semantic_bev_gt], dim=2))
            # Row 3 --> instance panoptic Pred, Masked Pred and GT
            vis.append(torch.cat([instance_bev_pred, instance_bev_pred_masked, instance_bev_gt], dim=2))
            # Row 4 --> panoptic Pred, Masked Pred and GT
            vis.append(torch.cat([vis_bev_pred, vis_bev_pred_masked, vis_bev_gt], dim=2))


        else:
            vis_bev_pred = visualise_panoptic_mask_trainid(bev_pred_unpack, varargs['dataset'])
            vis_bev_gt = visualise_panoptic_mask_trainid(bev_gt_unpack, varargs['dataset'])

            vis.append(vis_bev_gt)
            vis.append(vis_bev_pred)

            bev_pred_sem = bev_pred_unpack.clone()
            bev_gt_sem = bev_gt_unpack.clone()
            bev_pred_sem[bev_pred_sem > 1000] = bev_pred_sem[bev_pred_sem > 1000] // 1000
            bev_gt_sem[bev_gt_sem > 1000] = bev_gt_sem[bev_gt_sem > 1000] // 1000
            error_region = (bev_gt_sem != bev_pred_sem).squeeze(0)

            vis_bev_pred_masked = vis_bev_pred.clone()
            vis_bev_pred_masked[:, bev_gt_unpack.squeeze(0) == 255] = 0  # Set invalid areas to 0
            vis.append(vis_bev_pred_masked)

            # Add the error map and the masked output
            error_map = torch.zeros_like(vis_bev_pred_masked)
            error_map[:, error_region] = 255
            error_map[:, bev_gt_unpack.squeeze(0) == 255] = 0
            vis.append(error_map)

        # Append all the images together
        vis = torch.cat(vis, dim=1)
        vis_list.append(vis)

    return vis_list


def get_panoptic_mask(panoptic_pred, num_stuff):
    canvas = torch.ones((panoptic_pred[0].shape)).type(torch.long).to(panoptic_pred[0].device) * 255
    thing_list = []
    for idd, pred in enumerate(list(panoptic_pred[1])):
        if pred == 255:
            continue
        if panoptic_pred[3][idd] == 0:
            # If not iscrowd
            if pred < num_stuff:
                canvas[panoptic_pred[0] == idd] = pred
            else:
                canvas[panoptic_pred[0] == idd] = pred * 1000 + thing_list.count(pred)
                thing_list.append(pred)
    return canvas

def get_semantic_mask(panoptic_pred, num_stuff):
    canvas = torch.ones((panoptic_pred[0].shape)).type(torch.long).to(panoptic_pred[0].device) * 255
    for idd, pred in enumerate(list(panoptic_pred[1])):
        if pred == 255:
            continue
        if panoptic_pred[3][idd] == 0:
            # If not iscrowd
            if pred < num_stuff:
                canvas[panoptic_pred[0] == idd] = pred
            else:
                canvas[panoptic_pred[0] == idd] = pred * 1000
    return canvas



def recover_image(img, rgb_mean, rgb_std):
    ## z = img.new(rgb_std).view(-1, 1, 1)
    ## print(z.size())
    img = img * img.new(rgb_std).view(-1, 1, 1)
    img = img + img.new(rgb_mean).view(-1, 1, 1)
    return img


def visualise_panoptic_mask_trainid(bev_panoptic, dataset):
    if dataset == "Kitti360":
        stuff_colours_trainid = {label.trainId: label.color for label in cs_labels}
    elif dataset == "nuScenes":
        stuff_colours_trainid = {label.trainId: label.color for label in nuscenes_labels}

    po_vis = torch.zeros((3, bev_panoptic.shape[1], bev_panoptic.shape[2]), dtype=torch.int32).to(bev_panoptic.device)

    # Colour the stuff
    stuff_mask = bev_panoptic <= 1000
    classes = torch.unique(bev_panoptic[stuff_mask])
    for stuff_label in classes:
        po_vis[:, (bev_panoptic == stuff_label).squeeze()] = torch.tensor(stuff_colours_trainid[stuff_label.item()],
                                                                          dtype=torch.int32).unsqueeze(1).to(bev_panoptic.device)

    # Colour the things
    thing_mask = (bev_panoptic > 1000)
    if torch.sum(thing_mask) > 0:
        colors = [list(range(len(THING_COLOURS)))]
        for thing_label in torch.unique(bev_panoptic[thing_mask]):
            if len(colors) == 0:
                colors = [list(range(len(THING_COLOURS)))]
            color_idx = random.choice(colors)
            colors.remove(color_idx)
            po_vis[:, (bev_panoptic == thing_label).squeeze()] = torch.tensor(THING_COLOURS[color_idx],
                                                                              dtype=torch.int32).unsqueeze(1).to(bev_panoptic.device)

    return po_vis


def save_panoptic_output(sample, sample_category, save_tuple, **varargs):
    if save_tuple is None:
        return

    save_path, sample_name = save_tuple[0], save_tuple[1]

    # Check if the directory exists. If not create it
    cam_name = varargs['cam_name'] if "cam_name" in varargs.keys() else None
    if cam_name is not None:
        # save_dir_rgb = os.path.join(save_path, cam_name, "{}_rgb".format(sample_category))
        save_dir = os.path.join(save_path, cam_name, sample_category)
    else:
        # save_dir_rgb = os.path.join(save_path, "{}_rgb".format(sample_category))
        save_dir = os.path.join(save_path, sample_category)

    # if not os.path.exists(save_dir_rgb):
    #    os.makedirs(save_dir_rgb)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # img_name_rgb = os.path.join(save_dir_rgb, "{}.png".format(sample_name))
    img_name = os.path.join(save_dir, "{}.png".format(sample_name))

    # Generate the numpy image and save the image using OpenCV
    # Check if there are multiple elements in the sample. Then you'll have to decode it using generatePOMask function
    if len(sample) > 1:
        po_mask = get_panoptic_mask(sample, varargs["num_stuff"]).unsqueeze(0)
    else:
        po_mask = sample[0].unsqueeze(0)

    # Save the raw version of the mask
    po_mask = po_mask.squeeze(0) # add new line
    po_mask_orig = po_mask.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    row, col, _ = po_mask_orig.shape
    col = col // 3
    row = row // 4

    fv_img = po_mask_orig[:row, :2*col]
    error_img = np.rot90(po_mask_orig[:row, 2*col:])
    sem_pred = np.rot90(po_mask_orig[row:2*row, :col])
    sem_masked_pred = np.rot90(po_mask_orig[row:2 * row, col:2*col])
    sem_gt = np.rot90(po_mask_orig[row:2 * row, 2*col:])
    inst_pred = np.rot90(po_mask_orig[2*row:3 * row, :col])
    inst_masked_pred = np.rot90(po_mask_orig[2*row:3 * row, col:2 * col])
    inst_gt = np.rot90(po_mask_orig[2*row:3 * row, 2 * col:])
    pano_pred = np.rot90(po_mask_orig[3 * row:, :col])
    pano_masked_pred = np.rot90(po_mask_orig[3 * row:, col:2 * col])
    pano_gt = np.rot90(po_mask_orig[3 * row:, 2 * col:])

    cv2.imwrite(img_name[:-4] + "_fv_img.png", fv_img)
    cv2.imwrite(img_name[:-4] + "_error_img.png", error_img)

    cv2.imwrite(img_name[:-4] + "_sem_pred.png", sem_pred)
    cv2.imwrite(img_name[:-4] + "_sem_masked_pred.png", sem_masked_pred)
    cv2.imwrite(img_name[:-4] + "_sem_gt.png", sem_gt)

    cv2.imwrite(img_name[:-4] + "_inst_pred.png", inst_pred)
    cv2.imwrite(img_name[:-4] + "_inst_masked_pred.png", inst_masked_pred)
    cv2.imwrite(img_name[:-4] + "_inst_gt.png", inst_gt)

    cv2.imwrite(img_name[:-4] + "_pano_pred.png", pano_pred)
    cv2.imwrite(img_name[:-4] + "_pano_masked_pred.png", pano_masked_pred)
    cv2.imwrite(img_name[:-4] + "_pano_gt.png", pano_gt)

    # Get the RGB image of the po_pred
    # po_mask_rgb = visualise_panoptic_mask_trainid(po_mask, varargs['dataset'])
    # po_mask_rgb = po_mask_rgb.permute(1, 2, 0).cpu().numpy()
    # cv2.imwrite(img_name_rgb, po_mask_rgb)
