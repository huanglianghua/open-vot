from __future__ import absolute_import

import torch.nn as nn
import torch

from .caffenet import CaffeNet
from ..utils import initialize_weights


class GOTURN(nn.Module):

    def __init__(self):
        super(GOTURN, self).__init__()
        self.branch_z = CaffeNet()
        self.branch_x = CaffeNet()

        self.fc6 = nn.Sequential(
            nn.Linear(6 * 6 * 256 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))
        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))
        self.fc7b = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))
        self.fc8 = nn.Sequential(
            nn.Linear(4096, 4))
        initialize_weights(self)

    def forward(self, z, x):
        assert z.size(-1) == x.size(-1) == 256
        z = self.branch_z(z)
        x = self.branch_x(x)
        out = torch.cat((z, x), dim=1)
        out = out.view(out.size(0), -1)

        out = self.fc6(out)
        out = self.fc7(out)
        out = self.fc7b(out)
        out = self.fc8(out)

        return out
