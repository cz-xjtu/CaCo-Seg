#!/usr/bin/env python3
# encoding: utf-8
"""
@Project: ConsistencyCAM
@author: Zane-Chen
@contact: 1900938761@qq.com
@file: vis_cam.py
@time: 2022/3/7 14:51
@IDE: PyCharm
@desc: define python functions to be called by cpp
"""
import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", default='/data2/CZ/data/CZ_ACDC/', type=str)
# parser.add_argument("--data_root", default='/data5/chenz/LUNA16/', type=str)
parser.add_argument("--work_dir", default="experiments/acdc/", type=str)
# parser.add_argument("--work_dir", default="./experiments/luna16_subset0/5", type=str)
parser.add_argument("--cam_dir", default="cam/parr_base_cs_cc_new512_alpha0.01_metric2_cls_resnet50_ConsCAM_lr0.01_bs16_epoch20_unfixed/msf", type=str)
# parser.add_argument("--cam_dir", default="cam/base_metric2_cls_resnet50_ConsCAM_lr0.01_bs16_epoch15_unfixed/msf", type=str)


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


if __name__ == "__main__":
    args = parser.parse_args()
    input_cam_dir = os.path.join(args.work_dir, args.cam_dir)
    save_vis_dir = create_directory(os.path.join(args.work_dir, 'vis', args.cam_dir))
    cam_list = os.listdir(input_cam_dir)
    for cam_name in cam_list:
        cam_name = cam_name[0:-4]
        print(cam_name)
        # cam_dict = np.load(os.path.join(input_cam_dir, cam_name + '.npy'), allow_pickle=True)
        cam_dict = np.load(os.path.join(input_cam_dir, cam_name + '.npy'), allow_pickle=True).item()
        cam = cam_dict[0]
        # cam = cam_dict
        heatmap = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_vis_dir, cam_name + '.jpg'), heatmap)
