from __future__ import absolute_import

import torch
import torch.nn as nn
from ..utils.complex import *

class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)

class DCFNetFeaturePadding(nn.Module):
    def __init__(self):
        super(DCFNetFeaturePadding, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)

class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        self.yf = config.yf.clone()
        self.config = config

    def forward(self, z, x):
        z = self.feature(z)
        x = self.feature(x)

        zf = torch.rfft(z, signal_ndim=2)
        xf = torch.rfft(x, signal_ndim=2)

        kzzf = torch.sum(tensor_complex_mulconj(zf,zf), dim=1, keepdim=True)

        kzyf = tensor_complex_mulconj(zf, self.yf.to(device=z.device))

        solution =  tensor_complex_division(kzyf, kzzf + self.config.lambda0)

        response = torch.irfft(torch.sum(tensor_complex_mulconj(xf, solution), dim=1, keepdim=True), signal_ndim=2)

        return response

class DCFNetOnline(nn.Module):
    def __init__(self, config=None):
        super(DCFNetOnline, self).__init__()
        self.feature = DCFNetFeaturePadding()
        self.config = config

    def forward(self, x):
        x = self.feature(x)

        x =  x * self.config.cos_window

        xf = torch.rfft(x, signal_ndim=2)
        solution =  tensor_complex_division(self.model_alphaf, self.model_betaf + self.config.lambda0)
        response = torch.irfft(torch.sum(tensor_complex_mulconj(xf, solution), dim=1, keepdim=True), signal_ndim=2)
        r_max = torch.max(response)

        return response

    def update(self, z, lr=1.):
        z = self.feature(z)

        z = z * self.config.cos_window
        zf = torch.rfft(z, signal_ndim=2)

        kzzf = torch.sum(tensor_complex_mulconj(zf,zf), dim=1, keepdim=True)
        kzyf = tensor_complex_mulconj(zf,self.config.yf_online.to(device=z.device))

        if lr > 0.99:
            self.model_alphaf = kzyf
            self.model_betaf = kzzf
        else:
            self.model_alphaf = (1 - lr) * self.model_alphaf.data + lr * kzyf.data
            self.model_betaf = (1 - lr) * self.model_betaf.data + lr * kzzf.data
