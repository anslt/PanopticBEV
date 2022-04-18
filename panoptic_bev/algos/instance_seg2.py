from math import ceil
import torch
import torch.nn.functional as functional

from panoptic_bev.utils.parallel import PackedSequence
from panoptic_bev.utils.sequence import pack_padded_images

class InstanceSegLoss:
    """instantic segmentation loss

    Parameters
    ----------
    ohem : float or None
        Online hard example mining fraction, or `None` to disable OHEM
    ignore_index : int
        Index of the void class
    """

    def __init__(self, ohem=None, out_shape=(768, 704), ignore_index=255, ignore_labels=None,
                 bev_params=None, extrinsics=None):
        if ohem is not None and (ohem <= 0 or ohem > 1):
            raise ValueError("ohem should be in (0, 1]")
        self.ohem = ohem
        self.ignore_index = ignore_index
        self.ignore_labels = ignore_labels

        resolution = float(bev_params['cam_z']) / float(bev_params['f'])

        rows = torch.arange(0, out_shape[0])
        cols = torch.arange(0, out_shape[1])
        rr, cc = torch.meshgrid(rows, cols)
        idx_mesh = torch.cat([rr.unsqueeze(0), cc.unsqueeze(0)], dim=0)
        ego_position = torch.tensor([out_shape[0] // 2, 0]).view(-1, 1, 1)
        pos_mesh = idx_mesh - ego_position
        self.X, self.Z = pos_mesh[0] * resolution, pos_mesh[1] * resolution
        self.Y = abs(float(extrinsics['translation'][2]))

    def __call__(self, center_logits, offset_logits, center, offset, inst_weights, weights_msk, intrinsics):
        """Compute the instantic segmentation loss

        Parameters
        ----------
        inst_logits : sequence of torch.Tensor
            A sequence of N tensors of segmentation logits with shapes C x H_i x W_i
        inst : sequence of torch.Tensor
            A sequence of N tensors of ground truth instantic segmentations with shapes H_i x W_i

        Returns
        -------
        inst_loss : torch.Tensor
            A scalar tensor with the computed loss
        """
        center_loss = []
        offset_loss = []
        for i, (center_logits_i, offset_logits_i, center_i, offset_i, inst_weights_i) in \
                enumerate(zip(center_logits, offset_logits, center, offset, inst_weights)):
            # if self.ignore_labels is not None:
            #     inst_i[(inst_i == self.ignore_labels).any(-1)] = self.ignore_index  # Remap the ignore_labels to ignore_index

            center_loss_i = functional.mse_loss(inst_weights * center_logits_i, inst_weights * center_i, reduction="none")
            offset_loss_i = functional.l1_loss(inst_weights * offset_logits_i, inst_weights * offset_i,
                                               reduction="none")

            # f_x = intrinsics[i][0][0]
            # f_y = intrinsics[i][1][1]
            # self.X = self.X.to(inst_i.device)
            # self.Z = self.Z.to(inst_i.device)
            #
            # S = torch.sqrt((f_x**2 * self.Z**2) + (f_x*self.X + f_y*self.Y)**2) / (self.Z**2)
            # sensitivity_map = 1 / torch.log(1 + S)
            # sensitivity_map[torch.isnan(sensitivity_map)] = 0.
            # # sensitivity_wt = sensitivity_map * 10
            # sensitivity_wt = sensitivity_map * 10
            #
            # # Distance-based weighting
            # inst_loss_i *= (1 + sensitivity_wt.to(inst_i.device).squeeze(0))
            #
            # # Multiply the instace-based weights mask
            # inst_loss_i *= (wt_msk_i / 10000)
            #
            # inst_loss_i = inst_loss_i.view(-1)
            #
            # if self.ohem is not None and self.ohem != 1:
            #     top_k = int(ceil(inst_loss_i.numel() * self.ohem))
            #     if top_k != inst_loss_i.numel():
            #         inst_loss_i, _ = inst_loss_i.topk(top_k)

            center_loss.append(center_loss_i.mean())
            offset_loss.append(offset_loss_i.mean())

        return sum(center_loss) / len(center_logits), sum(offset_loss) / len(offset_logits)


class InstanceSegAlgo:
    """instantic segmentation algorithm

    Parameters
    ----------
    loss : instanticSegLoss
    num_classes : int
        Number of classes
    """

    def __init__(self, loss, num_classes, ignore_index=255):
        self.loss = loss
        self.num_classes = num_classes
        self.ignore_index = ignore_index


    @staticmethod
    def _pack_logits(inst_logits, valid_size, img_size):
        inst_logits = functional.interpolate(inst_logits, size=img_size, mode="bilinear", align_corners=False)
        return pack_padded_images(inst_logits, valid_size)

    @staticmethod
    def _logits2(head, x):
        center_logits, offset_logits = head(x)
        return center_logits, offset_logits

    @staticmethod
    def _logits(head, x, img_size, roi):
        inst_feat, _, _ = head(x, None, img_size, roi)
        return inst_feat

    def processing(self, head, head1, x, center, offset, valid_size, img_size, inst_weights, weights_msk, intrinsics):
        """Given input features and ground truth compute instantic segmentation loss, confusion matrix and prediction

        Parameters
        ----------
        head : torch.nn.Module
            Module to compute instantic segmentation logits given an input feature map. Must be callable as `head(x)`
        x : torch.Tensor
            A tensor of image features with shape N x C x H x W
        inst : sequence of torch.Tensor
            A sequence of N tensors of ground truth instantic segmentations with shapes H_i x W_i
        valid_size : list of tuple of int
            List of valid image sizes in input coordinates
        img_size : tuple of int
            Spatial size of the, possibly padded, image tensor used as input to the network that calculates x

        Returns
        -------
        inst_loss : torch.Tensor
            A scalar tensor with the computed loss
        conf_mat : torch.Tensor
            A confusion matrix tensor with shape M x M, where M is the number of classes
        inst_pred : PackedSequence
            A sequence of N tensors of instantic segmentations with shapes H_i x W_i
        """
        # Compute logits and prediction
        inst_feat = self._logits(head, x, img_size, False)
        # inst_logits = self._pack_logits(inst_logits_, valid_size, img_size)
        inst_feat = functional.interpolate(inst_feat, size=img_size, mode="bilinear", align_corners=False)
        center_logits, offset_logits = self._logits2(head1, inst_feat)

        # inst_pred = PackedSequence([inst_logits_i.max(dim=0)[1] for inst_logits_i in inst_logits])
        print(offset_logits.shape)
        print(center_logits.shape)
        print(pad_packed_images(center).shape)
        print(offset.shape)

        # Compute loss and confusion matrix
        center_loss, offset_loss = self.loss(center_logits, offset_logits, center, offset, inst_weights,
                                             weights_msk, intrinsics=intrinsics)
        # conf_mat = self._confusion_matrix(inst_pred, inst)

        return center_loss, offset_loss, center_logits, offset_logits


