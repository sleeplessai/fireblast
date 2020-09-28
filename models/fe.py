import torch
import torch.nn as nn
from torchvision.models import resnet50, resnext50_32x4d

gpu = torch.device("cuda:0")
backbones = {
  "res50": resnet50(pretrained=True).to(gpu),
  "resx50" : resnext50_32x4d(pretrained=True).to(gpu)
}

class Backbone(nn.Module):
  def __init__(self,
               backbone_t="resx50",
               backbone_1d=512):
    super(Backbone, self).__init__()
    self.backbone = backbones[backbone_t]
    assert 0 < backbone_1d <= 2048
    if backbone_1d == 2048:
      self.use_conv1x1 = False
    else:
      self.use_conv1x1 = True
      self.conv1x1 = nn.Sequential(
        nn.Conv2d(2048, backbone_1d, kernel_size=1),
        nn.BatchNorm2d(backbone_1d, eps=1e-05, momentum=0.1),
        nn.ReLU()
      )

  def forward(self, x):
    x = self.backbone.conv1(x)
    x = self.backbone.bn1(x)
    x = self.backbone.relu(x)
    x = self.backbone.maxpool(x)

    x = self.backbone.layer1(x)
    x = self.backbone.layer2(x)
    x = self.backbone.layer3(x)
    x = self.backbone.layer4(x)

    if self.use_conv1x1: x = self.conv1x1(x)

    x = self.backbone.avgpool(x)
    x = torch.flatten(x, 1)
    return x
