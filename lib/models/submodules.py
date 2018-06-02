from __future__ import absolute_import, division

import torch.nn as nn
import torch
import torch.nn.functional as F


class XCorr(nn.Module):

    def __init__(self):
        super(XCorr, self).__init__()

    def forward(self, z, x):
        out = []
        for i in range(z.size(0)):
            out.append(F.conv2d(x[i, :].unsqueeze(0),
                                z[i, :].unsqueeze(0)))

        return torch.cat(out, dim=0)


class Adjust2d(nn.Module):

    def __init__(self, norm='bn'):
        super(Adjust2d, self).__init__()
        assert norm in [None, 'bn', 'cosine', 'euclidean', 'linear']
        self.norm = norm
        if norm == 'bn':
            self.bn = nn.BatchNorm2d(1)
        elif norm == 'linear':
            self.linear = nn.Conv2d(1, 1, 1, bias=True)
        self._initialize_weights()

    def forward(self, out, z=None, x=None):
        if self.norm == 'bn':
            out = self.bn(out)
        elif self.norm == 'linear':
            out = self.linear(out)
        elif self.norm == 'cosine':
            n, k = out.size(0), z.size(-1)
            norm_z = torch.sqrt(
                torch.pow(z, 2).view(n, -1).sum(1)).view(n, 1, 1, 1)
            norm_x = torch.sqrt(
                k * k * F.avg_pool2d(torch.pow(x, 2), k, 1).sum(1, keepdim=True))
            out = out / (norm_z * norm_x + 1e-32)
        elif self.norm == 'euclidean':
            n, k = out.size(0), z.size(-1)
            sqr_z = torch.pow(z, 2).view(n, -1).sum(1).view(n, 1, 1, 1)
            sqr_x = k * k * \
                F.avg_pool2d(torch.pow(x, 2), k, 1).sum(1, keepdim=True)
            out = out + sqr_z + sqr_x
            out = out.clamp(min=1e-32).sqrt()
        elif self.norm == None:
            out = out

        return out

    def _initialize_weights(self):
        if self.norm == 'bn':
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        elif self.norm == 'linear':
            self.linear.weight.data.fill_(1e-3)
            self.linear.bias.data.zero_()
