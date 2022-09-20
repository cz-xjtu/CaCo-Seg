import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)

import network.resnet38d
from tools import pyutils

class Net(network.resnet38d.Net):
    def __init__(self, num_classes=1):
        super(Net, self).__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)

        # self.fc8 = nn.Conv2d(4096, 21, 1, bias=False)
        self.fc8 = nn.Conv2d(4096, num_classes, 1, bias=False)

        # self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        # self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        # self.f9 = torch.nn.Conv2d(192+3, 192, 1, bias=False)
        
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        # torch.nn.init.kaiming_normal_(self.f8_3.weight)
        # torch.nn.init.kaiming_normal_(self.f8_4.weight)
        # torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)
        # self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9, self.fc8]
        self.from_scratch_layers = [self.fc8]
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, x):
        d = super().forward_as_dict(x)
        cam = self.fc8(self.dropout7(d['conv6']))
        logits = F.adaptive_avg_pool2d(cam, (1, 1))

        # cam = F.interpolate(cam, (H,W), mode='bilinear', align_corners=True)
        return logits, cam

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups

