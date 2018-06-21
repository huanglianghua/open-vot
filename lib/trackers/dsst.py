from __future__ import absolute_import, division

import numpy as np
import cv2

from . import Tracker
from ..utils import dict2tuple
from ..utils.complex import real, conj, fft2, ifft2, fft1, ifft1, complex_mul, complex_div
from ..descriptors.fhog import fast_hog


class TrackerDSST(Tracker):

    def __init__(self, **kargs):
        super(TrackerDSST, self).__init__('DSST')
        self.parse_args(**kargs)

    def parse_args(self, **kargs):
        self.cfg = {
            'padding': 1,
            'output_sigma_factor': 0.0625,
            'scale_sigma_factor': 0.25,
            'lambda_': 1e-2,
            'learning_rate': 0.025,
            'scale_num': 33,
            'scale_step': 1.02,
            'scale_model_max_area': 512}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = dict2tuple(self.cfg)

    def init(self, image, init_rect):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # initialize parameters
        self.t_center = init_rect[:2] + init_rect[2:] / 2
        self.t_sz = init_rect[2:].astype(int)
        self.t_scale = 1.0
        self.padded_sz = self.t_sz * (1 + self.cfg.padding)

        scale_factor = 1.0
        if self.t_sz.prod() > self.cfg.scale_model_max_area:
            scale_factor = np.sqrt(
                self.cfg.scale_model_max_area / self.t_sz.prod())
        self.scale_model_sz = (self.t_sz * scale_factor).astype(int)
        self.scale_model_sz = self.scale_model_sz // 8 * 8
        self.min_scale_factor = self.cfg.scale_step ** np.ceil(
            np.log((5 / self.padded_sz).max()) / np.log(self.cfg.scale_step))
        self.max_scale_factor = self.cfg.scale_step ** np.floor(
            np.log((image.shape[1::-1] / self.t_sz).min()) / np.log(self.cfg.scale_step))

        # create translation gaussian labels
        output_sigma = self.cfg.output_sigma_factor * \
            np.sqrt(np.prod(self.t_sz[:2]))
        rs, cs = np.ogrid[:self.padded_sz[1], :self.padded_sz[0]]
        rs, cs = rs - self.padded_sz[1] // 2, cs - self.padded_sz[0] // 2
        y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
        self.yf = fft2(y)[:, :, np.newaxis, :]

        # create scale gaussian labels
        scale_sigma = self.cfg.scale_sigma_factor * \
            np.sqrt(self.cfg.scale_num)
        ss = np.ogrid[:self.cfg.scale_num] - np.ceil(self.cfg.scale_num / 2)
        ys = np.exp(-0.5 / scale_sigma ** 2 * (ss ** 2))
        self.ysf = fft1(ys)[:, np.newaxis, :]
        self.scale_factors = self.cfg.scale_step ** (-ss)

        # initialize hanning windows
        self.hann_window = np.outer(
            np.hanning(self.padded_sz[1]),
            np.hanning(self.padded_sz[0])).astype(np.float32)
        self.hann_window = self.hann_window[:, :, np.newaxis]
        if self.cfg.scale_num % 2 == 0:
            self.scale_window = np.hanning(
                self.cfg.scale_num + 1).astype(np.float32)
            self.scale_window = self.scale_window[1:]
        else:
            self.scale_window = np.hanning(
                self.cfg.scale_num).astype(np.float32)
        self.scale_window = self.scale_window[:, np.newaxis]

        # train translation filter
        z = self._get_translation_sample(
            image, self.t_center, self.padded_sz, self.t_scale)
        self.hf_num, self.hf_den = self._train_translation_filter(
            fft2(z), self.yf)

        # train scale filter
        zs = self._get_scale_sample(
            image, self.t_center, self.t_sz,
            self.t_scale * self.scale_factors, self.scale_model_sz)
        self.sf_num, self.sf_den = self._train_scale_filter(fft1(zs), self.ysf)

    def update(self, image):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # locate target center
        x = self._get_translation_sample(
            image, self.t_center, self.padded_sz, self.t_scale)
        score = self._calc_translation_score(fft2(x), self.hf_num, self.hf_den)
        _, _, _, max_loc = cv2.minMaxLoc(score)
        self.t_center = self.t_center - self.t_scale * \
            (np.floor(self.padded_sz / 2) - max_loc)
        # limit the estimated bounding box to be overlapped with the image
        self.t_center = np.clip(
            self.t_center, -self.t_sz / 2 + 1,
            image.shape[1::-1] + self.t_sz / 2 - 2)

        # locate target scale
        xs = self._get_scale_sample(
            image, self.t_center, self.t_sz,
            self.t_scale * self.scale_factors, self.scale_model_sz)
        score = self._calc_scale_score(fft1(xs), self.sf_num, self.sf_den)
        scale_id = score.argmax()
        self.t_scale *= self.scale_factors[scale_id]
        self.t_scale = np.clip(
            self.t_scale, self.min_scale_factor, self.max_scale_factor)

        # update translation filter
        z = self._get_translation_sample(
            image, self.t_center, self.padded_sz, self.t_scale)
        hf_num, hf_den = self._train_translation_filter(
            fft2(z), self.yf)
        self.hf_num = (1 - self.cfg.learning_rate) * self.hf_num + \
            self.cfg.learning_rate * hf_num
        self.hf_den = (1 - self.cfg.learning_rate) * self.hf_den + \
            self.cfg.learning_rate * hf_den

        # update scale filter
        zs = self._get_scale_sample(
            image, self.t_center, self.t_sz,
            self.t_scale * self.scale_factors, self.scale_model_sz)
        sf_num, sf_den = self._train_scale_filter(fft1(zs), self.ysf)
        self.sf_num = (1 - self.cfg.learning_rate) * self.sf_num + \
            self.cfg.learning_rate * sf_num
        self.sf_den = (1 - self.cfg.learning_rate) * self.sf_den + \
            self.cfg.learning_rate * sf_den

        bndbox = np.concatenate([
            self.t_center - self.t_sz * self.t_scale / 2,
            self.t_sz * self.t_scale])

        return bndbox

    def _get_translation_sample(self, image, center, size, scale):
        patch_sz = (size * scale).astype(int)
        patch = self._crop(image, center, patch_sz)
        if np.any(patch.shape[1::-1] != size):
            patch = cv2.resize(patch, tuple(size))

        feature = fast_hog(np.float32(patch) / 255.0, 1)[:, :, :27]
        feature = np.pad(feature, ((1, 1), (1, 1), (0, 0)), 'edge')
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        feature = np.concatenate((gray, feature), axis=2)

        return self.hann_window * feature

    def _get_scale_sample(self, image, center, size, scale_factors, scale_model_sz):
        features = []
        for scale in scale_factors:
            patch_sz = size * scale
            patch = self._crop(image, center, patch_sz)
            patch = cv2.resize(patch, tuple(scale_model_sz))
            feature = fast_hog(np.float32(patch) / 255.0, 4, False)
            features.append(feature.reshape(-1))
        features = np.stack(features)

        return self.scale_window * features

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

    def _train_translation_filter(self, zf, yf):
        hf_num = complex_mul(yf, conj(zf))
        hf_den = complex_mul(zf, conj(zf))
        hf_den = np.sum(hf_den, axis=2)

        return hf_num, hf_den

    def _train_scale_filter(self, zsf, ysf):
        sf_num = complex_mul(ysf, conj(zsf))
        sf_den = complex_mul(zsf, conj(zsf))
        sf_den = np.sum(sf_den, axis=1, keepdims=True)

        return sf_num, sf_den

    def _calc_translation_score(self, xf, hf_num, hf_den):
        num = np.sum(complex_mul(hf_num, xf), axis=2)
        den = hf_den + self.cfg.lambda_
        score = real(ifft2(complex_div(num, den)))

        return score

    def _calc_scale_score(self, xsf, sf_num, sf_den):
        num = np.sum(complex_mul(sf_num, xsf), axis=1, keepdims=True)
        den = sf_den + self.cfg.lambda_
        score = real(ifft2(complex_div(num, den)))
        score = score.squeeze(1)

        return score
