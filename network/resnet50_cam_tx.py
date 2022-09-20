import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
import torch
from tools import m_t_modi
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class Net(nn.Module):

    def __init__(self, cfg):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=False, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.causal_predictor = m_t_modi.CausalPredictor(cfg, 513*513)

        # self.classifier = nn.Conv2d(2049, 20, 1, bias=False)
        self.classifier = nn.Conv2d(2049, 1, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def concate(self, x, mt):
        xm_resize = F.interpolate(mt.unsqueeze(1), [x.shape[2], x.shape[3]], mode='bilinear', align_corners=False)
        return torch.cat((x, xm_resize), 1)

    def forward(self, x, xm):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        mt = self.causal_predictor(x, xm.view(xm.shape[0], -1))
        x = self.concate(x, mt)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        # x = x.view(-1, 20)
        x = x.view(-1, 1)

        return x

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self, cfg):
        super(CAM, self).__init__(cfg)

    def forward(self, x, xm):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        mt = self.causal_predictor(x, xm.view(xm.shape[0], -1))
        x = self.concate(x, mt)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf
    CONFIG = OmegaConf.load('/home/tzq-cz/code/code_rep/CONTA/pseudo_mask/voc12/confounder.yaml')
    model = CAM(CONFIG).to(torch.device("cuda"))

    model.eval()
    image = torch.randn(2, 3, 513, 513).to(torch.device("cuda"))
    seg_pred = torch.randn(2, 1, 513, 513).to(torch.device("cuda"))

    print(model)
    print("input:", image.shape)
    print("output:", model(image, seg_pred).shape)