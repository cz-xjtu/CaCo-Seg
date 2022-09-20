"""several kinds of scoring metrics"""
import numpy as np
from PIL import Image
import os
import glob


def score(prediction_dir, ground_truth_dir, class_num=2):
    hist = compute_hist(prediction_dir, ground_truth_dir, class_num)
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print('>>>', 'overall accuracy', acc)
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print('>>>', 'mean accuracy', np.nanmean(acc))
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('>>>', 'per class IOU', iu)
    mean_iou = np.nanmean(iu)
    print('>>>', 'mean IU', mean_iou)
    dice = 2 * np.diag(hist) / (hist.sum(1) + hist.sum(0))
    print('>>>', 'Dice', dice)
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    print('>>>', 'fwavacc', fwavacc)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def compute_hist(prediction_dir, ground_truth_dir, class_num=2):
    n_cl = class_num
    hist = np.zeros((n_cl, n_cl))
    for pred_img in glob.glob(os.path.join(prediction_dir, '*.png')):
        pred = Image.open(pred_img).convert("L")
        #pred = pred.resize((256, 256), Image.NEAREST)
        pred = np.array(pred)
        pred[pred > 0] = 1
        id = pred_img.split('/')[-1]
        #id = id[0:-4]
        #label_name = str(int(id)+1) + '.png'
        #label_name = pred_img.split('\\')[-1].replace('_fake_B', '')
        #label_name = pred_img[-5:]
        # label_name = id
        label_name = id.replace('Image', 'Label')
        #print(label_name)
        #label_path = os.path.join(ground_truth_dir, 'DL_Label' + label_name + '.png')
        label_path = os.path.join(ground_truth_dir, label_name)
        gt = Image.open(label_path)
        #gt = gt.resize((256, 256), Image.NEAREST)
        gt = np.array(gt)
        gt[gt > 0] = 1
        hist += fast_hist(gt.flatten(),pred.flatten(),n_cl)
    return hist


if __name__ == '__main__':
    pred_dir = '/data5/chenz/code/ConsistencyCAM/experiments/prostate/cam_png/parr_base_ss_sc_cs_cc_metric2_cls_resnet50_ConsCAM_lr0.01_bs16_epoch15_unfixed_10epoch/msf_post'
    # pred_dir = '/data5/chenz/code/ConsistencyCAM/experiments/acdc/cam_png/parr_base_cs_cc_new512_alpha0.01_metric2_cls_resnet50_ConsCAM_lr0.01_bs16_epoch20_unfixed/msf_post'
    gt_dir = '/data2/CZ/data/promise12/test_gt'
    # gt_dir = '/data2/CZ/data/CZ_ACDC/DL_Label'
    # gt_dir = '/data2/CZ/data/prostate_MR/DL_Label'

    score(pred_dir, gt_dir)