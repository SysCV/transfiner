# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
import cv2
import numpy as np
import copy
import math

from detectron2.layers.roi_align import ROIAlign
from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

__all__ = [
    "BaseMaskRCNNHead",
    "MaskRCNNConvUpsampleHead",
    "build_mask_head",
    "ROI_MASK_HEAD_REGISTRY",
]


ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def pos_embed(x, temperature=10000, scale=2 * math.pi, normalize=True):
    """
    This is a more standard version of the position embedding, very similar to
    the one used by the Attention is all you need paper, generalized to work on
    images.
    """
    batch_size, channel, height, width = x.size()
    mask = x.new_ones((batch_size, height, width))
    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    num_pos_feats = channel // 2
    assert num_pos_feats * 2 == channel, (
        'The input channel number must be an even number.')
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                         pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                         pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos


def dice_loss_my(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # inputs = inputs.sigmoid()
    # print('input shape:', inputs.shape)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    # print('loss shape:', loss.shape)
    return loss.sum() / num_boxes


def get_incoherent_mask(input_masks, sfact):
    mask = input_masks.float()
    w = input_masks.shape[-1]
    h = input_masks.shape[-2]
    # print('mask 1 shape:', input_masks)
    mask_small = F.interpolate(mask, (h//sfact, w//sfact), mode='bilinear')
    # print('mask small shape:', mask_small.shape)
    mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
    # print('mask recover shape:', mask_recover.shape)
    # # print('mask orii shape:', mask.shape)
    mask_residue = (mask - mask_recover)
    # print('mask residue shape:', mask_residue.shape)
    mask_uncertain = F.interpolate(
        mask_residue, (h//sfact, w//sfact), mode='bilinear')
    mask_uncertain[mask_uncertain >= 0.01] = 1.

    return mask_uncertain


def crop_and_resize_my(bit_masks, boxes: torch.Tensor, mask_size: int, sfact: int):
    """
    Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
    This can be used to prepare training targets for Mask R-CNN.
    It has less reconstruction error compared to rasterization with polygons.
    However we observe no difference in accuracy,
    but BitMasks requires more memory to store all the masks.

    Args:
        boxes (Tensor): Nx4 tensor storing the boxes for each mask
        mask_size (int): the size of the rasterized mask.

    Returns:
        Tensor:
            A bool tensor of shape (N, mask_size, mask_size), where
            N is the number of predicted boxes for this image.
    """
    assert len(boxes) == len(bit_masks), "{} != {}".format(
        len(boxes), len(self))
    device = bit_masks.device
    # print('bit masks shape:', bit_masks.shape)
    # print('boxes shape:', boxes.shape)
    batch_inds = torch.arange(len(boxes), device=device).to(
        dtype=boxes.dtype)[:, None]
    # print('ori boxes ori:', boxes[0])
    boxes = boxes / float(sfact)
    # print('ori boxes now:', boxes[0])
    rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5

    bit_masks = bit_masks.to(dtype=torch.float32)
    rois = rois.to(device=device)
    output = (
        ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
        .forward(bit_masks[:, None, :, :], rois)
        .squeeze(1)
    )
    output = output >= 0.05
    return output


@torch.jit.unused
def mask_rcnn_loss(pred_mask_logits: torch.Tensor, pred_mask_logits_uncertain: torch.Tensor, x_hr: torch.Tensor, x_hr_l: torch.Tensor, x_c: torch.Tensor, transfomer_encoder: torch.nn.Module, instances: List[Instances], vis_period: int = 0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(
        3), "Mask prediction must be square!"
    #print('enter mask loss1')
    gt_classes = []
    gt_masks = []
    gt_masks_s = []
    gt_masks_l = []
    gt_masks_uncertain = []

    for index, instances_per_image in enumerate(instances):
        # if index >= 1:
        #     continue
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(
                dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        # print('instances_per_image.gt_masks:', instances_per_image.gt_masks)
        # print('instances_per_image.gt_masks bit:',
        #       instances_per_image.gt_masks_bit)
        # print('instances_per_image.gt_masks bit shape:', instances_per_image.gt_masks_bit.tensor.shape)
        # print('instances_per_image.gt_masks bit max:', instances_per_image.gt_masks_bit.tensor.max())
        sfact = 2
        mask_uncertain = get_incoherent_mask(
            instances_per_image.gt_masks_bit.tensor.unsqueeze(1), sfact)
        # print('mask_uncertain shape:', mask_uncertain.shape)
        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        gt_masks_per_image_l = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len * 2
        ).to(device=pred_mask_logits.device)
        # print('gt_masks_per_image max:', gt_masks_per_image.max())
        gt_masks_per_image_s = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, int(mask_side_len * 0.5)
        ).to(device=pred_mask_logits.device)

        gt_masks_per_image_uncertain = crop_and_resize_my(mask_uncertain.squeeze(
            1), instances_per_image.proposal_boxes.tensor, mask_side_len, sfact).to(device=pred_mask_logits.device)
        # print('gt_masks_per_image_uncertain shape:',
        #       gt_masks_per_image_uncertain.shape)
        # print('gt_masks_per_image_uncertain max:', gt_masks_per_image_uncertain.max())
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len

        gt_masks.append(gt_masks_per_image)
        gt_masks_s.append(gt_masks_per_image_s)
        gt_masks_l.append(gt_masks_per_image_l)
        gt_masks_uncertain.append(gt_masks_per_image_uncertain)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0 + pred_mask_logits_uncertain.sum() * 0

    gt_masks = cat(gt_masks, dim=0)
    gt_masks_s = cat(gt_masks_s, dim=0)
    gt_masks_l = cat(gt_masks_l, dim=0)
    gt_masks_uncertain = cat(gt_masks_uncertain, dim=0)
    # print('out gtmasks:', gt_masks.shape)
    # print('out gt_masks_uncertain:', gt_masks_uncertain.shape)
    # print('out gt_masks_uncertain max:', gt_masks_uncertain.max())

    # for i in range(gt_masks.shape[0]):
    #     cv2.imwrite('./masks_uncertain_check_new8/mask_'+str(i)+'_gt_img.png', gt_masks[i].cpu().numpy() * 255)
    #     cv2.imwrite('./masks_uncertain_check_new8/mask_'+str(i)+'_uncertain_img.png', gt_masks_uncertain[i].cpu().numpy() * 255)

    # return
    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    pred_mask_logits_uncertain = pred_mask_logits_uncertain[:, 0]
    pred_mask_logits_uncertain_lg = F.interpolate(pred_mask_logits_uncertain.unsqueeze(1), (56, 56))
    #print('pred_mask_logits_uncertain:', pred_mask_logits_uncertain.shape)
    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5

    gt_masks = gt_masks.to(dtype=torch.float32)
    gt_masks_uncertain = gt_masks_uncertain.to(dtype=torch.float32)
    gt_masks_s = gt_masks_s.to(dtype=torch.float32)
    gt_masks_l = gt_masks_l.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() /
                         max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum(
    ).item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)
    # print('pred_mask_logits:', pred_mask_logits.shape)
    # print('gt masks:', gt_masks.shape)
    # print('gt masks uncertain:', gt_masks_uncertain.shape)
    # for i in range(gt_masks_uncertain.shape[0]):
    #      cv2.imwrite('./masks_uncertain10/mask_'+str(i)+'_u_img.png', gt_masks_uncertain.squeeze(1).cpu().numpy()[i] * 255)
    #      cv2.imwrite('./masks_uncertain10/mask_'+str(i)+'_g_img.png', gt_masks.squeeze(1).cpu().numpy()[i] * 255)

    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks, reduction="mean")

    mask_loss_uncertain = dice_loss_my(
        pred_mask_logits_uncertain, gt_masks_uncertain, pred_mask_logits_uncertain.shape[0]) + F.binary_cross_entropy(pred_mask_logits_uncertain, gt_masks_uncertain, reduction="mean")

    mask_uncertain_bool = (pred_mask_logits_uncertain.detach() >= 0.5)
    mask_uncertain_bool_lg = (pred_mask_logits_uncertain_lg.detach() >= 0.75)
    # (F.sigmoid(pred_mask_logits.detach()) >= 0.5)
    pred_mask_logits_bool = F.sigmoid(pred_mask_logits.detach())
    # print('pred_mask_logits_bool:', pred_mask_logits_bool.shape)

    pred_mask_logits_bool_small = F.interpolate(
        pred_mask_logits_bool.float().unsqueeze(1), (14, 14), mode='bilinear')

    pred_mask_logits_bool_large = F.interpolate(
        pred_mask_logits_bool.float().unsqueeze(1), (56, 56), mode='bilinear').squeeze(1)

    uncertain_pos = torch.nonzero(
        mask_uncertain_bool.squeeze(1), as_tuple=True)
    uncertain_pos_lg = torch.nonzero(
        mask_uncertain_bool_lg.squeeze(1), as_tuple=True)

    # print('target masks shape:', gt_masks.shape)
    # print('target masks large:', gt_masks_l.shape)
    # print('pred_mask_logits shape:', pred_mask_logits.shape)

    # pred_mask_logits_new = pred_mask_logits.detach().clone()
    # print('x_hr shape:', x_hr.shape)
    uncertain_feats = x_hr.permute(0, 2, 3, 1)[uncertain_pos]
    uncertain_feats_l = x_hr_l.permute(0, 2, 3, 1)[uncertain_pos_lg]

    x_hr_pos = pos_embed(x_hr)
    x_hr_pos_l = pos_embed(x_hr_l)

    uncertain_feats_pos = x_hr_pos.permute(0, 2, 3, 1)[uncertain_pos]
    uncertain_feats_pos_l = x_hr_pos_l.permute(0, 2, 3, 1)[uncertain_pos_lg]
    # print('x_hr pos shape:', x_hr_pos.shape)
    # print('x_hr shape:', x_hr.shape)
    uncertain_labels = gt_masks[uncertain_pos].unsqueeze(-1)
    uncertain_labels_l = gt_masks_l[uncertain_pos_lg].unsqueeze(-1)

    pred_coarse_labels = pred_mask_logits_bool[uncertain_pos]
    pred_coarse_labels_l = pred_mask_logits_bool_large[uncertain_pos_lg]

    # print('pred coarse lables:', pred_coarse_labels.shape)
    # print('uncertain feats:', uncertain_feats.shape)
    # print('uncertain labels:', uncertain_labels.shape)
    # print('gt_masks_s shape:', gt_masks_s.shape)
    gt_masks_s = gt_masks_s.flatten(1)

    number_pts = [(uncertain_pos[0] == ci).sum().item()
                  for ci in range(pred_mask_logits.shape[0])]
    number_pts_l = [(uncertain_pos_lg[0] == ci).sum().item()
                  for ci in range(pred_mask_logits.shape[0])]

    number_pts = torch.cumsum(torch.tensor(number_pts), dim=0)
    number_pts_l = torch.cumsum(torch.tensor(number_pts_l), dim=0)

    SAMPLE_NUM = 50
    uncertain_feats_box_list = []
    uncertain_feats_box_list_pos = []
    select_gt_box_list = []
    select_coarse_labels_list = []

    valid_box_pos = [True for i in range(len(number_pts))]

    for box_i in range(len(number_pts)):
        if box_i == 0:
            uncertain_feats_s = uncertain_feats[0: number_pts[box_i]]
            uncertain_feats_s_l = uncertain_feats_l[0: number_pts_l[box_i]]

            uncertain_feats_pos_s = uncertain_feats_pos[0: number_pts[box_i]]
            uncertain_feats_pos_s_l = uncertain_feats_pos_l[0: number_pts_l[box_i]]

            uncertain_labels_s = uncertain_labels[0: number_pts[box_i]]
            uncertain_labels_s_l = uncertain_labels_l[0: number_pts_l[box_i]]

            pred_coarse_labels_s = pred_coarse_labels[0: number_pts[box_i]]
            pred_coarse_labels_s_l = pred_coarse_labels_l[0: number_pts_l[box_i]]
        else:
            uncertain_feats_s = uncertain_feats[number_pts[box_i-1]:number_pts[box_i]]
            uncertain_feats_s_l = uncertain_feats_l[number_pts_l[box_i-1]:number_pts_l[box_i]]

            uncertain_feats_pos_s = uncertain_feats_pos[number_pts[box_i-1]:number_pts[box_i]]
            uncertain_feats_pos_s_l = uncertain_feats_pos_l[number_pts_l[box_i-1]:number_pts_l[box_i]]

            uncertain_labels_s = uncertain_labels[number_pts[box_i-1]:number_pts[box_i]]
            uncertain_labels_s_l = uncertain_labels_l[number_pts_l[box_i-1]:number_pts_l[box_i]]

            pred_coarse_labels_s = pred_coarse_labels[number_pts[box_i-1]:number_pts[box_i]]
            pred_coarse_labels_s_l = pred_coarse_labels_l[number_pts_l[box_i-1]:number_pts_l[box_i]]


        if uncertain_feats_s.size()[0] < 5 or uncertain_feats_s_l.size()[0] < 5:
            valid_box_pos[box_i] = False
            continue

        rand_inx = torch.randperm(uncertain_feats_s.size()[0])
        if len(rand_inx) < SAMPLE_NUM:
            rand_inx = rand_inx.repeat(10)[:SAMPLE_NUM]
        
        rand_inx_l = torch.randperm(uncertain_feats_s_l.size()[0])
        if len(rand_inx_l) < SAMPLE_NUM:
            rand_inx_l = rand_inx_l.repeat(10)[:SAMPLE_NUM]

        # .permute(1, 0)
        uncertain_feats_s = uncertain_feats_s[rand_inx][:SAMPLE_NUM]
        uncertain_feats_s_l = uncertain_feats_s_l[rand_inx_l][:SAMPLE_NUM]
        uncertain_feats_s = torch.cat((uncertain_feats_s, uncertain_feats_s_l), dim=0)

        uncertain_feats_pos_s = uncertain_feats_pos_s[rand_inx][:SAMPLE_NUM]
        uncertain_feats_pos_s_l = uncertain_feats_pos_s_l[rand_inx_l][:SAMPLE_NUM]
        uncertain_feats_pos_s = torch.cat((uncertain_feats_pos_s, uncertain_feats_pos_s_l), dim=0)
        
        uncertain_labels_s = uncertain_labels_s[rand_inx][:SAMPLE_NUM]
        uncertain_labels_s_l = uncertain_labels_s_l[rand_inx_l][:SAMPLE_NUM]
        uncertain_labels_s = torch.cat((uncertain_labels_s, uncertain_labels_s_l), dim=0)

        pred_coarse_labels_s = pred_coarse_labels_s[rand_inx][:SAMPLE_NUM]
        pred_coarse_labels_s_l = pred_coarse_labels_s_l[rand_inx_l][:SAMPLE_NUM]
        pred_coarse_labels_s = torch.cat((pred_coarse_labels_s, pred_coarse_labels_s_l), dim=0)

        uncertain_feats_box_list.append(uncertain_feats_s)
        uncertain_feats_box_list_pos.append(uncertain_feats_pos_s)
        select_gt_box_list.append(uncertain_labels_s)
        select_coarse_labels_list.append(pred_coarse_labels_s)

    x_c_pos = pos_embed(x_c).flatten(2)[valid_box_pos].permute(2, 0, 1)
    x_c = x_c.flatten(2)
    # print('x_c_pos:', x_c_pos.shape)
    # print('x_c:', x_c.shape)

    pred_mask_logits_bool_small = pred_mask_logits_bool_small.flatten(2)
    x_c_cat = torch.cat((x_c, pred_mask_logits_bool_small), dim=1).unsqueeze(-1)

    if len(uncertain_feats_box_list) == 0:
        encoded_feats = transfomer_encoder(
            x_c_cat).permute(1, 2, 0).unsqueeze(-1)
        selected_pred = transfomer_encoder.conv_r1(
            encoded_feats).squeeze(1).squeeze(-1)
        # print('gt_masks_s dytype:', gt_masks_s.dtype)
        mask_loss_refine = F.l1_loss(selected_pred, gt_masks_s)
        return mask_loss, mask_loss_uncertain, mask_loss_refine

    select_box_feats = torch.stack(uncertain_feats_box_list)
    select_box_feats_pos = torch.stack(uncertain_feats_box_list_pos)
    select_gt_boxs_labels = torch.stack(select_gt_box_list).squeeze(-1)
    select_coarse_labels = torch.stack(select_coarse_labels_list).unsqueeze(-1)
    select_box_feats_cat = torch.cat(
        (select_box_feats, select_coarse_labels), dim=2)

    select_box_feats_cat = select_box_feats_cat.unsqueeze(
        -1).permute(0, 2, 1, 3)
    select_box_feats_pos = select_box_feats_pos.permute(1, 0, 2)
    # print('select_box_feats_pos shape:', select_box_feats_pos.shape)
    select_box_feats_cat_pos = torch.cat(
        (x_c_pos, select_box_feats_pos), dim=0)
    # print('select_box_feats_cat_pos shape:', select_box_feats_cat_pos.shape)
    select_box_feats_cat = torch.cat(
        (x_c_cat[valid_box_pos], select_box_feats_cat), dim=2)
    select_gt_boxs_labels = torch.cat(
        (gt_masks_s[valid_box_pos], select_gt_boxs_labels), dim=1)

    encoded_feats = transfomer_encoder(
        select_box_feats_cat, select_box_feats_cat_pos).permute(1, 2, 0).unsqueeze(-1)
    selected_pred = transfomer_encoder.conv_r1(
        encoded_feats).squeeze(1).squeeze(-1)
    mask_loss_refine = F.l1_loss(selected_pred, select_gt_boxs_labels)

    return mask_loss, mask_loss_uncertain, mask_loss_refine


def mask_rcnn_inference(pred_mask_logits: torch.Tensor, pred_instances: List[Instances]):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        #print('enter cls agno')
        mask_probs_pred = pred_mask_logits + 0.2
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices,
                                           class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


class BaseMaskRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic described in :paper:`Mask R-CNN`
    """

    @configurable
    def __init__(self, *, loss_weight: float = 1.0, vis_period: int = 0):
        """
        NOTE: this interface is experimental.

        Args:
            loss_weight (float): multiplier of the loss
            vis_period (int): visualization period
        """
        super().__init__()
        self.vis_period = vis_period
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {"vis_period": cfg.VIS_PERIOD}

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x, x_uncertain, x_hr, x_hr_l, x_c, encoder = self.layers(x)

        if self.training:
            loss_masks, loss_mask_uncertains, loss_mask_refine = mask_rcnn_loss(
                x, x_uncertain, x_hr, x_hr_l, x_c, encoder, instances, self.vis_period)
            return {"loss_mask": loss_masks * self.loss_weight, "loss_mask_uncertain": loss_mask_uncertains * self.loss_weight * 0.5, "loss_mask_refine": loss_mask_refine}
        else:
            pred_mask_logits_uncertain = x_uncertain[:, 0]
            pred_mask_logits_uncertain_lg = F.interpolate(pred_mask_logits_uncertain.unsqueeze(1), (56, 56))
            pred_mask_logits = x

            num_masks = pred_mask_logits.shape[0]
            print('len instances before:', len(instances))
            class_pred = cat([i.pred_classes for i in instances])
            indices = torch.arange(num_masks, device=class_pred.device)
            mask_probs_pred = pred_mask_logits[indices,
                                           class_pred][:, None].sigmoid()
            
            mask_uncertain_bool = (pred_mask_logits_uncertain.detach() >= 0.5)

            pred_mask_logits_bool = mask_probs_pred
            # print('pred_mask_logits_bool:', pred_mask_logits_bool.shape)
            pred_mask_logits_bool_small = F.interpolate(
                pred_mask_logits_bool.float(), (14, 14), mode='bilinear')

            # print('mask_uncertain_bool.squeeze(1) shape:', mask_uncertain_bool.shape)

            uncertain_pos = torch.nonzero(mask_uncertain_bool, as_tuple=True)
            # print('mask_uncertain_bool:', mask_uncertain_bool.shape)
            # print('uncertain_pos1:', uncertain_pos)
            # print('uncertain_pos1:', uncertain_pos[0].requires_grad)
            # print('x_hr shape:', x_hr.shape)
            # print('x_hr shape:', x_hr.device)
            uncertain_feats = x_hr.permute(0, 2, 3, 1)[uncertain_pos]
            # print('uncertain_feats shape:', uncertain_feats.shape)
            x_hr_pos = pos_embed(x_hr)
            # print('x_hr pos shape:', x_hr_pos.device)
            # print('x_hr pos shape:', x_hr_pos.shape)
            uncertain_feats_pos = x_hr_pos.permute(0, 2, 3, 1)[uncertain_pos]
            # print('uncertain_feats_pos shape:', uncertain_feats_pos.shape)
            pred_coarse_labels = pred_mask_logits_bool.squeeze(1)[uncertain_pos]
            #print('uncertain_pos after:', uncertain_pos)
            
            number_pts = [(uncertain_pos[0] == ci).sum().item() for ci in range(pred_mask_logits.shape[0])]
            number_pts = torch.cumsum(torch.tensor(number_pts), dim=0)

            # uncertain_feats_box_list = []
            # uncertain_feats_box_list_pos = []
            # select_coarse_labels_list = []

            # valid_box_pos = [True for i in range(len(number_pts))]
            selected_pred_list = []
            for box_i in range(len(number_pts)):
                if box_i == 0:
                    uncertain_feats_s = uncertain_feats[0: number_pts[box_i]]
                    uncertain_feats_pos_s = uncertain_feats_pos[0: number_pts[box_i]]
                    pred_coarse_labels_s = pred_coarse_labels[0: number_pts[box_i]]
                else:
                    uncertain_feats_s = uncertain_feats[number_pts[box_i-1]:number_pts[box_i]]
                    uncertain_feats_pos_s = uncertain_feats_pos[number_pts[box_i-1]:number_pts[box_i]]
                    pred_coarse_labels_s = pred_coarse_labels[number_pts[box_i-1]:number_pts[box_i]]

                # print('uncertain_feats_s:', uncertain_feats_s.shape)

                x_c_pos_i = pos_embed(x_c).flatten(2)[box_i:box_i+1].permute(2, 0, 1)
                x_c_i = x_c.flatten(2)[box_i:box_i+1]

                # print('x_c_pos_i:', x_c_pos_i.shape)
                # print('x_c_i:', x_c_i.shape)

                pred_mask_logits_bool_small_i = pred_mask_logits_bool_small[box_i:box_i+1].flatten(2)
                # print('pred_mask_logits_bool_small_i shape:', pred_mask_logits_bool_small_i.shape)
                x_c_cat_i = torch.cat((x_c_i, pred_mask_logits_bool_small_i), dim=1).unsqueeze(-1)
                # print('x_c_cat_i:', x_c_cat_i.shape)
                select_box_feats = uncertain_feats_s.unsqueeze(0)
                select_box_feats_pos = uncertain_feats_pos_s.unsqueeze(0)
                select_coarse_labels = pred_coarse_labels_s.unsqueeze(0).unsqueeze(-1)
                # print('select_box_feats:', select_box_feats.shape)
                # print('select_coarse_labels:', select_coarse_labels.shape)
                select_box_feats_cat = torch.cat((select_box_feats, select_coarse_labels), dim=2)
                # print('select_box_feats_cat:', select_box_feats_cat.shape)
                select_box_feats_cat = select_box_feats_cat.unsqueeze(-1).permute(0, 2, 1, 3)
                select_box_feats_pos = select_box_feats_pos.permute(1, 0, 2)
                select_box_feats_cat_pos = torch.cat((x_c_pos_i, select_box_feats_pos), dim=0)
                # print('select_box_feats_cat_pos:', select_box_feats_cat_pos.shape)
                select_box_feats_cat = torch.cat((x_c_cat_i, select_box_feats_cat), dim=2)
                # print('select_box_feats_cat:', select_box_feats_cat.shape)
                encoded_feats = encoder(select_box_feats_cat, select_box_feats_cat_pos).permute(1, 2, 0).unsqueeze(-1)
                selected_pred = encoder.conv_r1(encoded_feats).flatten()[x_c_cat_i.shape[2]:]
                # print('selected_pred:', selected_pred.shape)

                # print('selected pred:', selected_pred.shape)
                selected_pred_list.append(selected_pred)

            select_num = 0
            for sel_p in selected_pred_list:
                select_num += sel_p.shape[0]

            if select_num > 0: # switch for modification
                selected_pred_list_cat = torch.cat(selected_pred_list)
                print('selected_pred_list_cat:', selected_pred_list_cat.shape)
                print('selected_pred_list_cat max:', selected_pred_list_cat.max())
                print('pred_mask_logits_bool.squeeze(1)[uncertain_pos] max:', pred_mask_logits_bool.squeeze(1)[uncertain_pos].max())
                pred_mask_logits_bool.squeeze(1)[uncertain_pos] = selected_pred_list_cat
            
            num_boxes_per_image = [len(i) for i in instances]
            print('pred_mask_logits_bool shape:', pred_mask_logits_bool.shape)
            pred_mask_logits_bool_l = F.interpolate(pred_mask_logits_bool, (56, 56), mode='bilinear', align_corners=True)
            print('pred_mask_logits_bool_l:', pred_mask_logits_bool_l.shape)
            print('num_boxes_per_image:', num_boxes_per_image)
            mask_probs_pred = pred_mask_logits_bool.split(num_boxes_per_image, dim=0)
            print('len instances:', len(instances))
            print('len mask_probs_pred:', len(mask_probs_pred))
            for prob, ins in zip(mask_probs_pred, instances):
                print('prob shape:', prob.shape)
                ins.pred_masks = prob  # (1, Hmask, Wmask)

            # print('x shape:', x.shape)
            # mask_rcnn_inference(x, instances)

            # mask_rcnn_inference(x_uncertain, instances) # for vis incohernt
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(BaseMaskRCNNHead):
    # class MaskRCNNConvUpsampleHead(BaseMaskRCNNHead, nn.Sequential):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of foreground classes (i.e. background is not
                included). 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []
        self.conv_norm_relus_uncertain = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes,
                                kernel_size=1, stride=1, padding=0)
        # self.predictor_for_inc = Conv2d(num_classes, 1,
        #                         kernel_size=1, stride=1, padding=0)

        encoder_layer = TransformerEncoderLayer(d_model=256, nhead=4)
        # used for the b4 and b4 correct; nice_light
        self.encoder = TransformerEncoder(encoder_layer, num_layers=3)

        for k, conv_dim in enumerate(conv_dims[:-1]):
            print('k:', k)
            # if k == 0:
            #     cur_channels += 1
            if k == 3:
                conv_dim = 128
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("mask_fcn_uncertain{}".format(k + 1), conv)
            self.conv_norm_relus_uncertain.append(conv)
            cur_channels = conv_dim

        self.deconv_uncertain = ConvTranspose2d(
            cur_channels, cur_channels, kernel_size=2, stride=2, padding=0
        )

        self.predictor_uncertain = Conv2d(cur_channels, 1,
                                          kernel_size=1, stride=1, padding=0)

        self.sig = nn.Sigmoid()

        for layer in self.conv_norm_relus + [self.deconv] + self.conv_norm_relus_uncertain + [self.deconv_uncertain]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.normal_(self.predictor_uncertain.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)
            nn.init.constant_(self.predictor_uncertain.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    # def layers(self, x):
    #     print('x:', x.shape)
    #     for layer in self:
    #         x = layer(x)
    #     print('x out shape:', x.shape)
    #     return x

    def layers(self, x_list):
        x = x_list[0]
        x_c = x.clone()
        x_hr = x_list[1]
        x_hr_l = x_list[2]
        # print('x_hr_l:', x_hr_l.shape)
        B, C, H, W = x.size()
        x_uncertain = x.clone()

        for cnt, layer in enumerate(self.conv_norm_relus):
            x = layer(x)

        x_uncertain += x
        # print('x_uncertain:', x_uncertain.shape)
        # print('mask shape:', mask.shape)
        #mask_inc = self.predictor_for_inc(self.sig(mask.detach()))
        #x_uncertain = torch.cat((x_uncertain, F.interpolate(mask_inc, (x_uncertain.shape[-1], x_uncertain.shape[-1]))), dim = 1)

        for cnt, layer in enumerate(self.conv_norm_relus_uncertain):
            x_uncertain = layer(x_uncertain)

        x = F.relu(self.deconv(x))
        mask = self.predictor(x)

        #x_uncertain = F.relu(self.deconv_uncertain(x_uncertain))
        x_uncertain = self.deconv_uncertain(x_uncertain)
        mask_uncertain = self.sig(self.predictor_uncertain(x_uncertain))
        return mask, mask_uncertain, x_hr, x_hr_l, x_c, self.encoder


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos):
        q = k = self.with_pos_embed(src, pos)
        # q = k = src
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.conv_fuse = nn.Conv2d(257, 256, 1, 1)
        self.conv_r1 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1, 1),
            nn.Sigmoid())

    def forward(self, src, pos=None):
        src = self.conv_fuse(src).squeeze(-1)
        src = src.permute(2, 0, 1)
        # print('src shape:', src.shape)
        # print('pos shape:', pos.shape)
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
