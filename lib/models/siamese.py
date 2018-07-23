from __future__ import absolute_import

import torch.nn as nn

from .submodules import Adjust2d, XCorr


class SiameseNet(nn.Module):

    def __init__(self, branch, norm='bn'):
        super(SiameseNet, self).__init__()
        self.branch = branch
        self.norm_type = norm
        if norm != 'cosine_similarity':
            self.norm = Adjust2d(norm=norm)
            self.xcorr = XCorr()
        else:
            self.norm = Adjust2d(norm=norm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z, x):
        assert z.size()[:2] == x.size()[:2]
        z = self.branch(z)
        x = self.branch(x)
        if self.norm_type == 'cosine_similarity':
            out = self.norm(None, z, x)
        else:
            out = self.xcorr(z, x)
            out = self.norm(out, z, x)
        return out
