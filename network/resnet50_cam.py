import random

import torch.nn as nn
import torch.nn.functional as F
from tools import torchutils, trans_utils
from network import resnet50
import torch


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        # self.classifier = nn.Conv2d(2048, 20, 1, bias=False)
        self.classifier = nn.Conv2d(2048, 1, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])
        self.initialize(self.newly_added)

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()
        # x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)

        cams = F.conv2d(x, self.classifier.weight)
        cams = F.relu(cams)
        logits = torchutils.gap2d(x, keepdims=True)
        logits = self.classifier(logits)
        # x = x.view(-1, 20)
        logits = logits.view(-1, 1)

        return logits, cams

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def get_parameter_groups(self):

        return list(self.backbone.parameters()), list(self.newly_added.parameters())

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x

