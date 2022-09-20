#!/usr/bin/env python3
# encoding: utf-8
"""
@Project: ConsistencyCAM
@author: Zane-Chen
@contact: 1900938761@qq.com
@file: train_cls.py
@time: 2022/2/24 下午5:24
@IDE: PyCharm
@desc: define python functions to be called by cpp
"""
import os
import random
import sys
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import importlib
import argparse
from tools import pyutils, imutils, torchutils, trans_utils, evaluate_utils
from data import prostate_dataset
import torch.nn.functional as F
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser()
###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--train_list", default="data/train_all_prostate.txt", type=str)
parser.add_argument("--val_list", default="data/train_valid_prostate.txt", type=str)
parser.add_argument("--data_root", default='/data2/CZ/data/prostate_MR/', type=str)  # for Tian_lab1
# parser.add_argument("--data_root", default='/data1/cz/dataset/prostate_MR/', type=str)  # for Tian_lab4
# parser.add_argument("--data_root", default='/home/data10/chenz/dataset/prostate_MR/', type=str)  # for Tian_lab5

###############################################################################
# Network
###############################################################################
parser.add_argument("--network", default="network.resnet50_ConsCAM", type=str)
# parser.add_argument("--weights", default="pretrained_model/ilsvrc-cls_rna-a1_cls1000_ep-0001.params", type=str)
parser.add_argument("--weights", default="", type=str)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument("--crop_size", default=512, type=int)
parser.add_argument("--second_crop_size", default=320, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--max_epoches", default=15, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--wt_dec", default=5e-4, type=float)
parser.add_argument('--num_pieces', default=4, type=int)
parser.add_argument('--alpha1', default=10, type=int, help="weight for s_cons_loss")
parser.add_argument('--alpha2', default=10, type=int, help="weight for p_cons_loss")
parser.add_argument('--alpha_schedule', default="unfixed", type=str, help="fixed, unfixed")
parser.add_argument("--metric", default="metric2", type=str, help="metric1: evaluate on downsampling size,"
                                                                  "metric2: evaluate on original size")
parser.add_argument('--with_cam', default=False, type=bool, help="global avg first or later")

###############################################################################
# Save
###############################################################################
parser.add_argument("--work_dir", default="./experiments/prostate", type=str)
parser.add_argument('--print_ratio', default=0.01, type=float)
parser.add_argument("--tag_prefix", default="alpha_10_parr_base_ss_sc_cs_cc320", type=str, help="[parr, base, ss, ps,psv3 sc, pc]")


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    print(vars(args))
    torchutils.set_seed(args.seed)
    tag = f'{args.tag_prefix}_{args.metric}_cls_{args.network[8:]}_lr{args.lr}_bs{args.batch_size}' \
          f'_epoch{args.max_epoches}_{args.alpha_schedule}'
    model_dir = create_directory(f'{args.work_dir}/model/')
    model_path = model_dir + tag + '.pth'
    model_path_10epoch = model_dir + tag + '_10epoch.pth'
    tensorboard_dir = create_directory(f'{args.work_dir}/tensorboards/{tag}/')

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        imutils.RandomResizeLong(args.min_image_size, args.max_image_size),
        transforms.RandomHorizontalFlip(),
        torchutils.Normalize(mean=data_mean, std=data_std),
        imutils.RandomCrop(args.crop_size),
        imutils.HWC_to_CHW,
        torch.from_numpy
    ])
    test_transform = transforms.Compose([
        torchutils.NormalizeForSegmentation(data_mean, data_std),
        imutils.RandomCropForSegmentation(args.crop_size),
        torchutils.TransposeForSegmentation()
    ])
    train_dataset = prostate_dataset.ProstateClsDataset(args.train_list, data_root=args.data_root,
                                                        transform=train_transforms)
    train_dataset_for_seg = prostate_dataset.ProstateDatasetForTestingCAM(args.train_list, args.data_root,
                                                                          test_transform)
    valid_dataset_for_seg = prostate_dataset.ProstateDatasetForTestingCAM(args.val_list, args.data_root, test_transform)


    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)


    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)
    train_loader_for_seg = DataLoader(train_dataset_for_seg, batch_size=args.batch_size, num_workers=1, drop_last=True)
    valid_loader_for_seg = DataLoader(valid_dataset_for_seg, batch_size=args.batch_size, num_workers=1, drop_last=True)

    max_iteration = len(train_dataset) // args.batch_size * args.max_epoches
    val_iteration = len(train_dataset) // args.batch_size
    log_iteration = int(val_iteration * args.print_ratio)

    ###################################################################################
    # Network
    ###################################################################################
    model = getattr(importlib.import_module(args.network), 'NetWithCAM')()
    param_groups = model.get_parameter_groups()
    # model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    # if the_number_of_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    print(model)

    ###################################################################################
    # Optimizer
    ###################################################################################
    # optimizer for resnet38
    # optimizer = torchutils.PolyOptimizer([
    #     {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
    #     {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
    #     {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
    #     {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    # ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_iteration)

    # optimizer for resnet50
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_iteration)

    if args.weights:
        if args.weights[-7:] == '.params':
            import network.resnet38d
            assert 'resnet38' in args.network
            weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
        else:
            weights_dict = torch.load(args.weights)
        model.load_state_dict(weights_dict, strict=False)

    #################################################################################################
    # Train
    #################################################################################################
    best_train_mIoU = -1
    thresholds = list(np.arange(0.0, 1.0, 0.05))


    def evaluate(loader):
        #################################################################################################
        # Evaluator
        #################################################################################################
        model.eval()

        meter_dic = {th: evaluate_utils.Calculator_For_mIoU('./data/prostate.json') for th in thresholds}

        with torch.no_grad():
            length = len(loader)
            for step, (images, labels, gt_masks) in enumerate(loader):
                # images = images.cuda()
                images = images
                labels = labels.cuda(non_blocking=True)

                _, features = model(images)

                mask = labels.unsqueeze(2).unsqueeze(3)
                cams = (torchutils.make_cam(features) * mask)

                # for visualization
                if step == 0:
                    obj_cams = cams.max(dim=1)[0]

                    for b in range(4):
                        image = images[b].cpu().detach().numpy()
                        cam = obj_cams[b].cpu().detach().numpy()

                        image = image.transpose((1, 2, 0))
                        image = (image * data_std) + data_mean
                        image *= 255
                        image = image.astype(np.uint8)
                        image = image[..., ::-1]
                        h, w, c = image.shape

                        cam = (cam * 255).astype(np.uint8)
                        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
                        cam = cv2.applyColorMap(cam, colormap=cv2.COLORMAP_JET)

                        image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
                        image = image.astype(np.float32) / 255.

                        writer.add_image('CAM/{}'.format(b + 1), image, optimizer.global_step - 1, dataformats='HWC')

                for batch_index in range(images.size()[0]):
                    # c, h, w -> h, w, c
                    cam = cams[batch_index].cpu().detach().numpy().transpose((1, 2, 0))
                    gt_mask = gt_masks[batch_index].cpu().detach().numpy()

                    if args.metric == "metric1":
                        h, w, c = cam.shape
                        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        h, w = gt_mask.shape
                        cam_s = cv2.resize(cam[:, :, 0], (w, h), interpolation=cv2.INTER_LINEAR)
                        cam = cam_s[..., np.newaxis]

                    for th in thresholds:
                        bg = np.ones_like(cam[:, :, 0]) * th
                        pred_mask = np.argmax(np.concatenate([bg[..., np.newaxis], cam], axis=-1), axis=-1)
                        meter_dic[th].add(pred_mask, gt_mask)

                # break
                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        print(' ')
        model.train()

        best_th = 0.0
        best_mIoU = 0.0

        for th in thresholds:
            mIoU, mIoU_foreground = meter_dic[th].get(clear=True)
            if best_mIoU < mIoU_foreground:
                best_th = th
                best_mIoU = mIoU_foreground

        return best_th, best_mIoU


    # avg_meter = pyutils.AverageMeter('loss', 'b_cls_loss', 's_cls_loss', 'c_cls_loss', 's_cons_loss', 'c_cons_loss')
    # avg_meter = pyutils.AverageMeter('loss', 'b_cls_loss', 's_cls_loss', 'r_cls_loss', 's_cons_loss', 'r_cons_loss')
    avg_meter = pyutils.AverageMeter('loss', 'b_cls_loss', 's_cls_loss', 'c_cls_loss', 'r_cls_loss', 's_cons_loss', 'c_cons_loss', 'r_cons_loss')
    timer = pyutils.Timer("Session started: ")
    writer = SummaryWriter(tensorboard_dir)

    for ep in range(args.max_epoches):
        for iter, pack in enumerate(train_data_loader):
            images = pack[1]
            labels = pack[2]
            # images = images.cuda()
            labels = labels.cuda()
            mask = labels.unsqueeze(2).unsqueeze(3)

            #################################################################################################
            # Base-Branch
            #################################################################################################
            logits, features = model(images, with_cam=args.with_cam)
            cams = (torchutils.make_cam(features) * mask)
            _, _, ih, iw = images.size()
            N, C, H, W = features.size()
            cls_loss = 0
            cls_terms = 0
            cons_loss = 0
            cons_terms = 0
            s_cons_loss = 0
            p_cons_loss = 0
            cls_loss_dic = {}
            cons_loss_dic = {}
            b_cls_loss = F.binary_cross_entropy_with_logits(logits, labels)  # base class loss
            cls_loss_dic['b_cls_loss'] = b_cls_loss
            cls_loss += b_cls_loss
            cls_terms += 1

            #################################################################################################
            # Transformer
            #################################################################################################
            if ("ss" in args.tag_prefix) or ("sc" in args.tag_prefix):
                scale_factor = random.choice([0.5, 0.75, 1.5, 1.75])
                trans_images_scale = trans_utils.scale_features(images, scale_factor)
                trans_logits_scale, trans_features_scale = model(trans_images_scale, with_cam=args.with_cam)
                if "ss" in args.tag_prefix:
                    s_cls_loss = F.binary_cross_entropy_with_logits(trans_logits_scale, labels)  # scale class loss
                    cls_loss += s_cls_loss
                    cls_terms += 1
                else:
                    s_cls_loss = torch.tensor([0.])

                if "sc" in args.tag_prefix:
                    detrans_features_scale = trans_utils.descale_features(trans_features_scale, (H, W))
                    # detrans_logits_scale = F.adaptive_avg_pool2d(detrans_features_scale, (1,1))
                    detrans_cams_scale = torchutils.make_cam(detrans_features_scale) * mask
                    # s_cons_loss = torch.mean(torch.abs(cams - detrans_cams_scale) * mask)
                    s_cons_loss = torch.mean(
                        torch.abs(cams - detrans_cams_scale) * mask, dim=2, keepdim=True).mean(
                        dim=3, keepdim=True).sum() / (mask.sum() + 1e-5)  # scale consistency losses
                else:
                    s_cons_loss = torch.tensor([0.])
            else:
                s_cls_loss = torch.tensor([0.])
                s_cons_loss = torch.tensor([0.])

            if ("ps" in args.tag_prefix) or ("pc" in args.tag_prefix):
                trans_images_patch = trans_utils.tile_features(images, args.num_pieces)
                trans_logits_patch, trans_features_patch = model(trans_images_patch, with_cam=args.with_cam)
                if "ps" in args.tag_prefix:
                    # trans_logits_patch = trans_logits_patch.view(N, C*args.num_pieces).mean(dim=1, keepdim=True)
                    # detrans_features_patch = trans_utils.merge_features(trans_features_patch, args.num_pieces,
                    #                                                     args.batch_size)
                    # detrans_logits_patch = F.adaptive_avg_pool2d(detrans_features_patch, (1,1))
                    p_cls_loss = F.binary_cross_entropy_with_logits(trans_logits_patch, labels)  # patch class loss
                    # p_cls_loss = F.binary_cross_entropy_with_logits(detrans_logits_patch.view(-1, 1), labels)  # patch class loss
                    cls_loss += (p_cls_loss / 1)
                    cls_terms += 1
                else:
                    p_cls_loss = torch.tensor([0.])

                if "pc" in args.tag_prefix:
                    detrans_features_patch = trans_utils.merge_features(trans_features_patch, args.num_pieces,
                                                                        args.batch_size)
                    # detrans_logits_patch = F.adaptive_avg_pool2d(detrans_features_patch, (1,1))
                    detrans_cams_patch = torchutils.make_cam(detrans_features_patch) * mask
                    # p_cons_loss = torch.mean(torch.abs(features - detrans_features_patch) * class_mask)
                    p_cons_loss = torch.mean(
                        torch.abs(cams - detrans_cams_patch) * mask, dim=2, keepdim=True).mean(
                        dim=3, keepdim=True).sum() / (mask.sum() + 1e-5)
                else:
                    p_cons_loss = torch.tensor([0.])
            else:
                p_cls_loss = torch.tensor([0.])
                p_cons_loss = torch.tensor([0.])

            if ("cs" in args.tag_prefix) or ("cc" in args.tag_prefix):
                random_range = args.crop_size - args.second_crop_size
                start_hh = random.randrange(random_range)
                start_ww = random.randrange(random_range)
                trans_images_crop = trans_utils.crop_features(images, start_hh, start_ww, args.second_crop_size)
                trans_logits_crop, trans_features_crop = model(trans_images_crop, with_cam=args.with_cam)
                if "cs" in args.tag_prefix:
                    # trans_logits_patch = trans_logits_patch.view(N, C*args.num_pieces).mean(dim=1, keepdim=True)
                    # detrans_features_patch = trans_utils.merge_features(trans_features_patch, args.num_pieces,
                    #                                                     args.batch_size)
                    # detrans_logits_patch = F.adaptive_avg_pool2d(detrans_features_patch, (1,1))
                    c_cls_loss = F.binary_cross_entropy_with_logits(trans_logits_crop, labels)  # crop class loss
                    # p_cls_loss = F.binary_cross_entropy_with_logits(detrans_logits_patch.view(-1, 1), labels)  # patch class loss
                    cls_loss += (c_cls_loss / 1)
                    cls_terms += 1
                else:
                    c_cls_loss = torch.tensor([0.])

                if "cc" in args.tag_prefix:
                    _, _, ch, cw = trans_features_crop.size()
                    trans_cams_crop = torchutils.make_cam(trans_features_crop) * mask
                    trans_cams = F.interpolate(cams, size=(ih, iw), mode='bilinear', align_corners=True)
                    trans_cams = trans_utils.crop_features(trans_cams, start_hh, start_ww, args.second_crop_size)
                    trans_cams_crop = F.interpolate(trans_cams_crop, size=(args.second_crop_size, args.second_crop_size), mode='bilinear', align_corners=True)
                    # p_cons_loss = torch.mean(torch.abs(features - detrans_features_patch) * class_mask)
                    c_cons_loss = torch.mean(
                        torch.abs(trans_cams - trans_cams_crop) * mask, dim=2, keepdim=True).mean(
                        dim=3, keepdim=True).sum() / (mask.sum() + 1e-5)
                else:
                    c_cons_loss = torch.tensor([0.])
            else:
                c_cls_loss = torch.tensor([0.])
                c_cons_loss = torch.tensor([0.])

            if ("rs" in args.tag_prefix) or ("rc" in args.tag_prefix):
                random_degree = random.randint(-90, 90)
                trans_images_rotate = trans_utils.rotate_features(images, random_degree)
                trans_logits_rotate, trans_features_rotate = model(trans_images_rotate, with_cam=args.with_cam)
                if "rs" in args.tag_prefix:
                    # trans_logits_patch = trans_logits_patch.view(N, C*args.num_pieces).mean(dim=1, keepdim=True)
                    # detrans_features_patch = trans_utils.merge_features(trans_features_patch, args.num_pieces,
                    #                                                     args.batch_size)
                    # detrans_logits_patch = F.adaptive_avg_pool2d(detrans_features_patch, (1,1))
                    r_cls_loss = F.binary_cross_entropy_with_logits(trans_logits_rotate, labels)  # crop class loss
                    # p_cls_loss = F.binary_cross_entropy_with_logits(detrans_logits_patch.view(-1, 1), labels)  # patch class loss
                    cls_loss += (r_cls_loss / 1)
                    cls_terms += 1
                else:
                    r_cls_loss = torch.tensor([0.])

                if "rc" in args.tag_prefix:
                    _, _, ch, cw = trans_features_rotate.size()
                    trans_cams_rotate = torchutils.make_cam(trans_features_rotate) * mask
                    trans_cams = F.interpolate(cams, size=(ih, iw), mode='bilinear', align_corners=True)
                    trans_cams = trans_utils.rotate_features(trans_cams, random_degree)
                    trans_cams_rotate = F.interpolate(trans_cams_rotate, size=(ih, iw), mode='bilinear', align_corners=True)
                    # p_cons_loss = torch.mean(torch.abs(features - detrans_features_patch) * class_mask)
                    r_cons_loss = torch.mean(
                        torch.abs(trans_cams - trans_cams_rotate) * mask, dim=2, keepdim=True).mean(
                        dim=3, keepdim=True).sum() / (mask.sum() + 1e-5)
                else:
                    r_cons_loss = torch.tensor([0.])
            else:
                r_cls_loss = torch.tensor([0.])
                r_cons_loss = torch.tensor([0.])

            # test = trans_utils.gaussian_noise(images)

            # cls_loss = (b_cls_loss + s_cls_loss + p_cls_loss) / 3
            cls_loss = cls_loss / cls_terms

            if args.alpha_schedule == "fixed":
                alpha1 = args.alpha1 / 10
                alpha2 = args.alpha2 / 10
            else:
                alpha1 = min(args.alpha1 * optimizer.global_step / (max_iteration * 1.), args.alpha1)
                alpha2 = min(args.alpha2 * optimizer.global_step / (max_iteration * 1.), args.alpha2)

            # loss = cls_loss + alpha1 * s_cons_loss + alpha1 * c_cons_loss + alpha2 * r_cons_loss
            loss = cls_loss + alpha1*s_cons_loss + alpha2*c_cons_loss
            # loss = cls_loss + alpha1 * s_cons_loss
            # loss = cls_loss + alpha2 * c_cons_loss
            # loss = cls_loss + alpha2 * r_cons_loss
            # loss = cls_loss
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # avg_meter.add({'loss': loss.item(), 'b_cls_loss': b_cls_loss.item(), 's_cls_loss': s_cls_loss.item(),
            #                'c_cls_loss': c_cls_loss.item(), 's_cons_loss': s_cons_loss.item(),
            #                'c_cons_loss': c_cons_loss.item()})
            # avg_meter.add({'loss': loss.item(), 'b_cls_loss': b_cls_loss.item(), 's_cls_loss': s_cls_loss.item(),
            #                'r_cls_loss': r_cls_loss.item(), 's_cons_loss': s_cons_loss.item(),
            #                'r_cons_loss': r_cons_loss.item()})
            avg_meter.add({'loss': loss.item(), 'b_cls_loss': b_cls_loss.item(), 's_cls_loss': s_cls_loss.item(),
                           'c_cls_loss': c_cls_loss.item(), 'r_cls_loss': r_cls_loss.item(), 's_cons_loss': s_cons_loss.item(),
                           'c_cons_loss': c_cons_loss.item(), 'r_cons_loss': r_cons_loss.item()})

            #################################################################################################
            # For Log
            #################################################################################################
            if (optimizer.global_step - 1) % log_iteration == 0:
                timer.update_progress(optimizer.global_step / max_iteration)
                # print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_iteration),
                #       'loss:%.4f %.4f %.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'b_cls_loss', 's_cls_loss',
                #                                                            'c_cls_loss', 's_cons_loss', 'c_cons_loss'),
                #       'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                #       'Fin:%s' % (timer.str_est_finish()),
                #       'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_iteration),
                      'loss:%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'b_cls_loss', 's_cls_loss',
                                                                           'c_cls_loss', 'r_cls_loss', 's_cons_loss', 'c_cons_loss', 'r_cons_loss'),
                      'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
                writer.add_scalar('Train/loss', loss, optimizer.global_step - 1)
                writer.add_scalar('Train/base_class_loss', b_cls_loss, optimizer.global_step - 1)
                writer.add_scalar('Train/scale_class_loss', s_cls_loss, optimizer.global_step - 1)
                # writer.add_scalar('Train/patch_class_loss', p_cls_loss, optimizer.global_step - 1)
                writer.add_scalar('Train/crop_class_loss', c_cls_loss, optimizer.global_step - 1)
                writer.add_scalar('Train/rotate_class_loss', r_cls_loss, optimizer.global_step - 1)
                writer.add_scalar('Train/scale_consistency_loss', s_cons_loss, optimizer.global_step - 1)
                # writer.add_scalar('Train/patch_consistency_loss', p_cons_loss, optimizer.global_step - 1)
                writer.add_scalar('Train/crop_consistency_loss', c_cons_loss, optimizer.global_step - 1)
                writer.add_scalar('Train/rotate_consistency_loss', r_cons_loss, optimizer.global_step - 1)
                writer.add_scalar('Train/learning_rate', optimizer.param_groups[0]['lr'], optimizer.global_step - 1)
                avg_meter.pop()

            # if optimizer.global_step - 1 and (optimizer.global_step - 1) % val_iteration == 1:

        else:
            print('')
            #################################################################################################
            # Evaluation
            #################################################################################################
            threshold, mIoU = evaluate(valid_loader_for_seg)

            if best_train_mIoU == -1 or best_train_mIoU < mIoU:
                best_train_mIoU = mIoU

                if ep <= 10:
                    torchutils.save_model(model, model_path=model_path_10epoch, parallel=the_number_of_gpu > 1)
                else:
                    torchutils.save_model(model, model_path=model_path, parallel=the_number_of_gpu > 1)

                print('save model')

            print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_iteration),
                  'threshold:%.2f' % threshold,
                  'train_mIoU:%.2f' % mIoU,
                  'best_train_mIoU:%.2f' % best_train_mIoU, flush=True)

            writer.add_scalar('Evaluation/threshold', threshold, optimizer.global_step - 1)
            writer.add_scalar('Evaluation/train_mIoU', mIoU, optimizer.global_step - 1)
            writer.add_scalar('Evaluation/best_train_mIoU', best_train_mIoU, optimizer.global_step - 1)
            timer.reset_stage()
