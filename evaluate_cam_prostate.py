#!/usr/bin/env python3
# encoding: utf-8
"""
@Project: ConsistencyCAM
@author: Zane-Chen
@contact: 1900938761@qq.com
@file: evaluate_cam.py
@time: 2022/3/2 12:49
@IDE: PyCharm
@desc: define python functions to be called by cpp
"""
import os
import sys

import imageio
import numpy as np
import argparse
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", default="./experiments/prostate", type=str)
# parser.add_argument("--work_dir", default="./experiments/acdc", type=str)
parser.add_argument("--cam_dir", default="cam", type=str)
parser.add_argument('--cam_method', default='msf', type=str, help="[normal, msf]")
parser.add_argument('--save_png', default='cam_png', type=str)
parser.add_argument('--gt_dir', default='/data2/CZ/data/prostate_MR/DL_Label', type=str)
# parser.add_argument('--gt_dir', default='/data2/CZ/data/CZ_ACDC/DL_Label', type=str)
parser.add_argument("--num_classes", default=2, type=int)  # 1
parser.add_argument("--weights", default="alpha_10_parr_base_ss_sc_cs_cc320_metric2_cls_resnet50_ConsCAM_lr0.001_bs16_epoch15_unfixed_10epoch.pth", type=str)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':
    args = parser.parse_args()
    cam_dir = os.path.join(args.work_dir, args.cam_dir, args.weights[0:-4], args.cam_method)
    cam_list = os.listdir(cam_dir)
    threshold = 0.
    best_thr = 0.
    best_dice = 0.
    while threshold < 1.:
        n_cl = args.num_classes
        hist = np.zeros((n_cl, n_cl))
        for cam_id in cam_list:
            cam_dict = np.load(os.path.join(cam_dir, cam_id), allow_pickle=True).item()
            cam = cam_dict[0]
            # cam = np.load(os.path.join(cam_dir, cam_id), allow_pickle=True)

            cam = np.pad(cam[np.newaxis, ...], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            cam_png = np.argmax(cam, axis=0)
            # compute hist
            cam_png[cam_png > 0] = 1
            label_id = cam_id.replace('Image', 'Label')[0:-4]
            # label_id = '1.3.6.1.4.1.14519.5.2.1.6279.6001.' + cam_id[0:-4]
            gt = Image.open(os.path.join(args.gt_dir, label_id + '.png'))
            gt = np.array(gt)
            gt[gt > 0] = 1
            # gt[gt == 5] = 1
            # gt[gt > 1] = 0
            hh, ww = np.shape(gt)
            hist += fast_hist(gt.flatten(), cam_png.flatten(), n_cl)
        # per-class IoU
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        # per-class Dice
        dice = 2 * np.diag(hist) / (hist.sum(1) + hist.sum(0))
        if dice[1] > best_dice:
            best_thr = threshold
            best_dice = dice[1]
        print('Threshold:%.2f' % threshold,
              'IoU:%.2f' % (iou[1] * 100),
              'Dice:%.2f' % (dice[1] * 100))
        threshold += 0.05

    print('Best Threshold:%.2f' % best_thr,
          'Best IoU:%.2f' % (100*best_dice/(2-best_dice)),
          'Best Dice:%.2f' % (best_dice*100))
    if args.save_png:
        print('Saving cam_png...')
        save_dir = create_directory(os.path.join(args.work_dir, args.save_png, args.weights[0:-4], args.cam_method))
        for cam_id in cam_list:
            cam_dict = np.load(os.path.join(cam_dir, cam_id), allow_pickle=True).item()
            cam = cam_dict[0]
            # cam = np.load(os.path.join(cam_dir, cam_id), allow_pickle=True)

            cam = np.pad(cam[np.newaxis, ...], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=best_thr)
            cam_png = np.argmax(cam, axis=0)
            imageio.imsave(os.path.join(save_dir, cam_id[0:-4] + '.png'), (cam_png * 255).astype(np.uint8))
        print('Saved!')
