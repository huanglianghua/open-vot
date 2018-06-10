from __future__ import absolute_import, division

import numpy as np
import cv2

from . import Tracker
from ..utils import dict2tuple
from ..utils.complex import real, fft, ifft, complex_mul, complex_div, circ_shift


class TrackerCSK(Tracker):

    def __init__(self, **kargs):
        self.parse_args(**kargs)

    def parse_args(self, **kargs):
        self.cfg = {
            'padding': 1,
            'output_sigma_factor': 0.0625,
            'sigma': 0.2,
            'lambda_': 1e-2,
            'interp_factor': 0.075}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = dict2tuple(self.cfg)

    def init(self, image, init_rect):
        # intialize parameters
        self.resize_image = False
        if np.sqrt(init_rect[2:].prod()) > 100:
            self.resize_image = True
            init_rect = init_rect / 2
        self.t_center = init_rect[:2] + init_rect[2:] / 2
        self.t_sz = init_rect[2:]
        self.padded_sz = self.t_sz * (1 + self.cfg.padding)
        self.padded_sz = np.floor(self.padded_sz).astype(int)

        # create gaussian labels
        output_sigma = np.sqrt(self.t_sz.prod()) * \
            self.cfg.output_sigma_factor
        rs, cs = np.ogrid[:self.padded_sz[1], :self.padded_sz[0]]
        rs, cs = rs - self.padded_sz[1] / 2, cs - self.padded_sz[0] / 2
        y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
        self.yf = fft(y)

        # initialize hanning window
        self.hann_window = np.outer(
            np.hanning(self.padded_sz[1]),
            np.hanning(self.padded_sz[0])).astype(np.float32)

        # crop padded target and train the classifier
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.resize_image:
            size = (int(image.shape[1] / 2), int(image.shape[0] / 2))
            image = cv2.resize(image, size)
        self.z = self._crop(image, self.t_center, self.padded_sz)
        self.z = self.hann_window * (np.float32(self.z) / 255 - 0.5)
        k = self._gaussian_correlation(self.z, self.z)
        self.alphaf = complex_div(self.yf, fft(k) + self.cfg.lambda_)

    def update(self, image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.resize_image:
            size = (int(image.shape[1] / 2), int(image.shape[0] / 2))
            image = cv2.resize(image, size)

        # locate target
        x = self._crop(image, self.t_center, self.padded_sz)
        x = self.hann_window * (np.float32(x) / 255 - 0.5)
        k = self._gaussian_correlation(x, self.z)
        score = real(ifft(complex_mul(self.alphaf, fft(k))))
        _, _, _, max_loc = cv2.minMaxLoc(score)
        self.t_center = self.t_center - np.floor(self.padded_sz / 2) + max_loc
        # limit the estimated bounding box to be overlapped with the image
        self.t_center = np.clip(
            self.t_center, -self.t_sz / 2 + 1,
            image.shape[1::-1] + self.t_sz / 2 - 2)

        # update model
        new_z = self._crop(image, self.t_center, self.padded_sz)
        new_z = self.hann_window * (np.float32(new_z) / 255 - 0.5)
        k = self._gaussian_correlation(new_z, new_z)
        new_alphaf = complex_div(self.yf, fft(k) + self.cfg.lambda_)
        self.alphaf = (1 - self.cfg.interp_factor) * self.alphaf + \
            self.cfg.interp_factor * new_alphaf
        self.z = (1 - self.cfg.interp_factor) * self.z + \
            self.cfg.interp_factor * new_z

        bndbox = np.concatenate([
            self.t_center - self.t_sz / 2, self.t_sz])
        if self.resize_image:
            bndbox = bndbox * 2

        return bndbox

    def _crop(self, image, center, size):
        corners = np.zeros(4, dtype=int)
        corners[:2] = np.floor(center - size / 2).astype(int)
        corners[2:] = corners[:2] + size
        pads = np.concatenate(
            (-corners[:2], corners[2:] - image.shape[1::-1]))
        pads = np.maximum(0, pads)

        if np.any(pads > 0):
            corners = np.concatenate((
                corners[:2] + pads[:2],
                corners[2:] - pads[2:])).astype(int)

        patch = image[corners[1]:corners[3], corners[0]:corners[2]]

        if np.any(pads > 0):
            patch = cv2.copyMakeBorder(
                patch, pads[1], pads[3], pads[0], pads[2],
                borderType=cv2.BORDER_REPLICATE)

        return patch

    def _gaussian_correlation(self, x1, x2):
        xcorr = cv2.mulSpectrums(fft(x1), fft(x2), 0, conjB=True)
        xcorr = circ_shift(real(ifft(xcorr)))

        out = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2 * xcorr) / x1.size
        out = out * (out >= 0)
        out = np.exp(-out / (self.cfg.sigma ** 2))

        return out
