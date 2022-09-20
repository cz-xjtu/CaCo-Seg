#!/usr/bin/env python3
# encoding: utf-8
"""
@Project: ConsistencyCAM
@author: Zane-Chen
@contact: 1900938761@qq.com
@file: transformer.py
@time: 2022/3/17 11:43
@IDE: PyCharm
@desc: define python functions to be called by cpp
"""
import torch

from tools import trans_utils
import torch.nn.functional as F


def scale_trans(model, images, labels, H, W, scale_factor, tag, with_cam=False):
    trans_images_scale = trans_utils.scale_features(images, scale_factor)
    trans_logits_scale, trans_features_scale = model(trans_images_scale, with_cam=with_cam)
    if "ss" in tag:
        s_cls_loss = F.binary_cross_entropy_with_logits(trans_logits_scale, labels)  # scale class loss
    else:
        s_cls_loss = torch.zeros(1)

    if "sc" in tag:
        detrans_features_scale = trans_utils.descale_features(trans_features_scale, (H, W))
        # detrans_logits_scale = F.adaptive_avg_pool2d(detrans_features_scale, (1,1))
        detrans_cams_scale = torchutils.make_cam(detrans_features_scale) * mask
        # s_cons_loss = torch.mean(torch.abs(cams - detrans_cams_scale) * mask)
        s_cons_loss = torch.mean(
            torch.abs(cams - detrans_cams_scale) * mask, dim=2, keepdim=True).mean(
            dim=3, keepdim=True).sum() / (mask.sum() + 1e-5)  # scale consistency losses
    else:
        s_cons_loss = torch.tensor([0.])