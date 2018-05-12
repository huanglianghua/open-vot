from __future__ import absolute_import

import torch.nn as nn

from ..utils import initialize_weights


class CaffeNet(nn.Module):

    def __init__(self):
        super(CaffeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, padding=1, groups=2),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, padding=1, groups=2),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        initialize_weights(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x
