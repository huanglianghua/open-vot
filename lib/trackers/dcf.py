from __future__ import absolute_import, division

import numpy as np
import cv2

from . import Tracker
from ..utils import dict2tuple
from ..utils.complex import real, conj, fft, ifft, complex_mul, complex_div
from ..descriptors.fhog import fast_hog


class TrackerDCF(Tracker):

    def __init__(self, **kargs):
        super(TrackerDCF, self).__init__('DCF')
        self.parse_args(**kargs)

    def parse_args(self, **kargs):
        self.cfg = {
            'lambda_': 1e-4,
            'padding': 1.5,
            'output_sigma_factor': 0.125,
            'interp_factor': 0.012,
            'cell_size': 4,
            'kernel_type': 'gaussian'}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = dict2tuple(self.cfg)

    def init(self, image, init_rect):
        # initialize parameters
        self.resize_image = False
        if np.sqrt(init_rect[2:].prod()) > 100:
            self.resize_image = True
            init_rect = init_rect / 2
        self.t_center = init_rect[:2] + init_rect[2:] / 2
        self.t_sz = init_rect[2:]
        mod = self.cfg.cell_size * 2
        self.padded_sz = self.t_sz * (1 + self.cfg.padding)
        self.padded_sz = self.padded_sz.astype(int) // mod * mod + mod

        # get feature size and initialize hanning window
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if self.resize_image:
            size = (int(image.shape[1] / 2), int(image.shape[0] / 2))
            image = cv2.resize(image, size)
        z = self._crop(
            image, self.t_center, self.padded_sz)
        z = fast_hog(z, self.cfg.cell_size)
        self.feat_sz = z.shape
        self.hann_window = np.outer(
            np.hanning(self.feat_sz[0]),
            np.hanning(self.feat_sz[1])).astype(np.float32)
        self.hann_window = self.hann_window[:, :, np.newaxis]
        self.zf = fft(z * self.hann_window)

        # create gaussian labels
        output_sigma = self.cfg.output_sigma_factor * \
            np.sqrt(np.prod(self.feat_sz[:2])) / (1 + self.cfg.padding)
        rs, cs = np.ogrid[:self.feat_sz[0], :self.feat_sz[1]]
        rs, cs = rs - self.feat_sz[0] // 2, cs - self.feat_sz[1] // 2
        y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
        self.yf = fft(y)

        # train classifier
        kf = self._linear_correlation(self.zf, self.zf)
        self.alphaf = complex_div(self.yf, kf + self.cfg.lambda_)

    def update(self, image):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if self.resize_image:
            size = (int(image.shape[1] / 2), int(image.shape[0] / 2))
            image = cv2.resize(image, size)

        # locate target
        x = self._crop(image, self.t_center, self.padded_sz)
        x = self.hann_window * fast_hog(x, self.cfg.cell_size)
        kf = self._linear_correlation(fft(x), self.zf)
        score = real(ifft(complex_mul(self.alphaf, kf)))
        offset = self._locate_target(score)
        self.t_center += offset * self.cfg.cell_size
        # limit the estimated bounding box to be overlapped with the image
        self.t_center = np.clip(
            self.t_center, -self.t_sz / 2 + 2,
            image.shape[1::-1] + self.t_sz / 2 - 1)

        # update model
        new_z = self._crop(image, self.t_center, self.padded_sz)
        new_z = fast_hog(new_z, self.cfg.cell_size)
        new_zf = fft(new_z * self.hann_window)
        kf = self._linear_correlation(new_zf, new_zf)
        new_alphaf = complex_div(self.yf, kf + self.cfg.lambda_)
        self.alphaf = (1 - self.cfg.interp_factor) * self.alphaf + \
            self.cfg.interp_factor * new_alphaf
        self.zf = (1 - self.cfg.interp_factor) * self.zf + \
            self.cfg.interp_factor * new_zf

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

    def _linear_correlation(self, x1f, x2f):
        xcorr = complex_mul(x1f, conj(x2f))
        xcorr = np.sum(xcorr, axis=2) / x1f.size

        return xcorr

    def _locate_target(self, score):
        def subpixel_peak(left, center, right):
            divisor = 2 * center - left - right
            if abs(divisor) < 1e-3:
                return 0
            return 0.5 * (right - left) / divisor

        _, _, _, max_loc = cv2.minMaxLoc(score)
        loc = np.float32(max_loc)

        if max_loc[0] in range(1, score.shape[1] - 1):
            loc[0] += subpixel_peak(
                score[max_loc[1], max_loc[0] - 1],
                score[max_loc[1], max_loc[0]],
                score[max_loc[1], max_loc[0] + 1])
        if max_loc[1] in range(1, score.shape[0] - 1):
            loc[1] += subpixel_peak(
                score[max_loc[1] - 1, max_loc[0]],
                score[max_loc[1], max_loc[0]],
                score[max_loc[1] + 1, max_loc[0]])
        offset = loc - np.float32(score.shape[1::-1]) / 2

        return offset
