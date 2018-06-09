from __future__ import absolute_import, division

import numpy as np
import cv2

from . import Tracker
from ..utils import dict2tuple


class TrackerCSK(Tracker):

    def __init__(self, **kargs):
        self.parse_args(**kargs)

    def parse_args(self, **kargs):
        self.cfg = {
            'padding': 1,
            'output_sigma_factor': 1/16,
            'sigma': 0.2,
            'lambda_': 1e-2,
            'interp_factor': 0.075}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = dict2tuple(self.cfg)

    def init(self, image, init_rect):
        # initialize parameters
        self.center = init_rect[:2] + init_rect[2:] / 2
        self.target_sz = init_rect[2:]
        if np.sqrt(np.prod(self.target_sz)) > 100:
            self.resize_image = True
            self.center = np.floor(self.center / 2)
            self.target_sz = np.floor(self.target_sz / 2)
        else:
            self.resize_image = False
        self.sz = np.floor(self.target_sz * (1 + self.cfg.padding))

        output_sigma = np.sqrt(np.prod(self.target_sz)) * \
            self.cfg.output_sigma_factor
        [rs, cs] = np.meshgrid(np.arange(self.sz[1]) - np.floor(self.sz[1] / 2),
                               np.arange(self.sz[0]) - np.floor(self.sz[0] / 2))
        y = np.exp(-0.5 / output_sigma ** 2 * (rs.T ** 2 + cs.T ** 2))
        self.yf = np.fft.fft2(y)
        self.cos_window = np.outer(np.hanning(self.sz[1]),
                                   np.hanning(self.sz[0]))
        
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.resize_image:
            image = cv2.resize(image, (int(np.floor(image.shape[1] / 2)), int(np.floor(image.shape[0] / 2))), 0.5, 0.5)
        self.z = self.get_subwindow(image, self.center, self.sz, self.cos_window)
        k = self.dense_gauss_kernel(self.cfg.sigma, self.z)
        self.alphaf = self.yf / (np.fft.fft2(k) + self.cfg.lambda_)

    def update(self, image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.resize_image:
            image = cv2.resize(image, (int(np.floor(image.shape[1] / 2)), int(np.floor(image.shape[0] / 2))), 0.5, 0.5)
        
        # locate target
        x = self.get_subwindow(image, self.center, self.sz,self.cos_window)
        k = self.dense_gauss_kernel(self.cfg.sigma, x, self.z)
        response = np.fft.ifft2(self.alphaf * np.fft.fft2(k)).real
        max_pos = np.unravel_index(response.argmax(), response.shape)
        self.center = self.center - np.floor(self.sz / 2) + max_pos[::-1]
        
        # update model
        x = self.get_subwindow(image, self.center, self.sz,self.cos_window)
        k = self.dense_gauss_kernel(self.cfg.sigma, x)
        new_alphaf = self.yf / (np.fft.fft2(k) + self.cfg.lambda_)
        new_z = x
        self.alphaf = (1 - self.cfg.interp_factor) * self.alphaf + \
            self.cfg.interp_factor * new_alphaf
        self.z = (1 - self.cfg.interp_factor) * self.z + \
            self.cfg.interp_factor * new_z
        
        bndbox = np.concatenate([
            self.center - self.target_sz / 2, self.target_sz])
        if self.resize_image:
            bndbox = bndbox * 2
        
        return bndbox
    
    def get_subwindow(self, image, center, sz, cos_window):
        xs = np.floor(self.center[0]) + np.arange(self.sz[0]) - np.floor(self.sz[0] / 2)
        ys = np.floor(self.center[1]) + np.arange(self.sz[1]) - np.floor(self.sz[1] / 2)

        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs >= image.shape[1]] = image.shape[1] - 1
        ys[ys >= image.shape[0]] = image.shape[0] - 1

        rs, cs = np.meshgrid(ys.astype(int), xs.astype(int))
        out = image[rs.T, cs.T]
        out = out / 255.0 - 0.5
        out = self.cos_window * out

        return out

    def dense_gauss_kernel(self, sigma, x, y=None):
        xf = np.fft.fft2(x)
        xx = np.sum(x * x)

        if y is None:
            yf = xf
            yy = xx
        else:
            yf = np.fft.fft2(y)
            yy = np.sum(y * y)
        
        xyf = xf * np.conj(yf)
        center = np.floor(np.asarray(x.shape)[::-1] / 2).astype(int)
        xy = np.roll(np.fft.ifft2(xyf), center)
        k = np.exp(-1 / self.cfg.sigma / 2 * np.maximum(0, (xx + yy - 2 * xy) / x.size))

        return k
