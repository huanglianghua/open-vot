from __future__ import absolute_import

import torch.nn as nn

from .submodules import Adjust2d, XCorr


class SiameseNet(nn.Module):

    def __init__(self, branch, norm='bn'):
        super(SiameseNet, self).__init__()
        self.branch = branch
        self.norm = Adjust2d(norm=norm)
        self.xcorr = XCorr()

    def forward(self, z, x):
        assert z.size()[:2] == x.size()[:2]
        z = self.branch(z)
        x = self.branch(x)
        out = self.xcorr(z, x)
        out = self.norm(out, z, x)

        return out
