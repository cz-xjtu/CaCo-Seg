#!/usr/bin/env python3
# encoding: utf-8
"""
@Project: ConsistencyCAM
@author: Zane-Chen
@contact: 1900938761@qq.com
@file: resnet50_ConsCAM.py
@time: 2022/3/4 17:00
@IDE: PyCharm
@desc: define python functions to be called by cpp
"""
import torch.nn as nn
import torch.nn.functional as F
from tools import torchutils, trans_utils
from network import resnet50
import torch



class NetWithCAM(nn.Module):

    def __init__(self):
        super(NetWithCAM, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 1, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])
        # self.initialize(self.newly_added)  # if with_cam=True

    def forward(self, x, with_cam=False):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        if with_cam:
            cams = self.classifier(x)
            # cams = F.relu(cams)  # in this condition, relu is not used
            logits = torchutils.gap2d(cams, keepdims=True)
            logits = logits.view(-1, 1)
        else:
            cams = F.conv2d(x, self.classifier.weight)
            cams = F.relu(cams)
            if x.shape[0] > 16:
                x = trans_utils.merge_features(x, 4, 8)
            logits = torchutils.gap2d(x, keepdims=True)
            logits = self.classifier(logits)
            logits = logits.view(-1, 1)

        return logits, cams

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def get_parameter_groups(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




