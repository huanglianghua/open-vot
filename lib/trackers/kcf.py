from __future__ import absolute_import, division

import numpy as np
import cv2

from . import Tracker
from ..utils import dict2tuple
from ..descriptors import fhog
from ..utils.complex import real, fft, ifft, complex_mul, complex_div, circ_shift


class TrackerKCF(Tracker):

    def __init__(self, **kargs):
        super(TrackerKCF, self).__init__('KCF')
        self.parse_args(**kargs)

    def parse_args(self, **kargs):
        self.cfg = {
            'lambda_': 1e-4,
            'padding': 2.5,
            'output_sigma_factor': 0.125,
            'interp_factor': 0.012,
            'sigma': 0.6,
            'cell_size': 4}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = dict2tuple(self.cfg)

    def init(self, image, init_rect):
        self._roi = [float(t) for t in init_rect]
        assert(init_rect[2] > 0 and init_rect[3] > 0)
        self._tmpl = self.extract_feature(image, 1)
        self._prob = self.create_labels(self.feat_sz[0], self.feat_sz[1])
        self._alphaf = np.zeros(
            (self.feat_sz[0], self.feat_sz[1], 2), np.float32)
        self.train(self._tmpl, 1.0)

    def update(self, image):
        if(self._roi[0]+self._roi[2] <= 0):
            self._roi[0] = -self._roi[2] + 1
        if(self._roi[1]+self._roi[3] <= 0):
            self._roi[1] = -self._roi[2] + 1
        if(self._roi[0] >= image.shape[1]-1):
            self._roi[0] = image.shape[1] - 2
        if(self._roi[1] >= image.shape[0]-1):
            self._roi[1] = image.shape[0] - 2

        cx = self._roi[0] + self._roi[2]/2.
        cy = self._roi[1] + self._roi[3]/2.

        loc, peak_value = self.locate_target(
            self._tmpl, self.extract_feature(image, 0))

        self._roi[0] = cx - self._roi[2]/2.0 + \
            loc[0]*self.cfg.cell_size
        self._roi[1] = cy - self._roi[3]/2.0 + \
            loc[1]*self.cfg.cell_size

        if(self._roi[0] >= image.shape[1]-1):
            self._roi[0] = image.shape[1] - 1
        if(self._roi[1] >= image.shape[0]-1):
            self._roi[1] = image.shape[0] - 1
        if(self._roi[0]+self._roi[2] <= 0):
            self._roi[0] = -self._roi[2] + 2
        if(self._roi[1]+self._roi[3] <= 0):
            self._roi[1] = -self._roi[3] + 2
        assert(self._roi[2] > 0 and self._roi[3] > 0)

        x = self.extract_feature(image, 0)
        self.train(x, self.cfg.interp_factor)

        return self._roi

    def subpixel_peak(self, left, center, right):
        divisor = 2*center - right - left  # float
        return (0 if abs(divisor) < 1e-3 else 0.5*(right-left)/divisor)

    def create_hanning(self):
        hann2d = np.outer(
            np.hanning(self.feat_sz[0]),
            np.hanning(self.feat_sz[1])).astype(np.float32)
        self.hann = hann2d[:, :, np.newaxis]

    def create_labels(self, sizey, sizex):
        syh, sxh = sizey//2, sizex//2
        output_sigma = np.sqrt(sizex*sizey) / \
            self.cfg.padding * self.cfg.output_sigma_factor
        mult = -0.5 / (output_sigma*output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y-syh)**2, (x-sxh)**2
        res = np.exp(mult * (y+x))
        return fft(res)

    def gaussian_correlation(self, x1, x2):
        c = np.zeros((self.feat_sz[0], self.feat_sz[1]), np.float32)
        for i in range(self.feat_sz[2]):
            caux = cv2.mulSpectrums(fft(x1[:, :, i]), fft(x2[:, :, i]), 0, conjB=True)
            caux = real(ifft(caux))
            c += caux
        c = circ_shift(c)

        d = (np.sum(x1*x1) + np.sum(x2*x2) - 2.0*c) / \
            (self.feat_sz[0]*self.feat_sz[1]*self.feat_sz[2])
        d = d * (d >= 0)
        d = np.exp(-d / (self.cfg.sigma*self.cfg.sigma))

        return d

    def extract_feature(self, image, inithann):
        extracted_roi = [0, 0, 0, 0]  # [int,int,int,int]
        cx = self._roi[0] + self._roi[2]/2  # float
        cy = self._roi[1] + self._roi[3]/2  # float

        if(inithann):
            padded_w = self._roi[2] * self.cfg.padding
            padded_h = self._roi[3] * self.cfg.padding
            mod = 2 * self.cfg.cell_size
            self._tmpl_sz = [
                int(padded_w) // mod * mod + mod,
                int(padded_h) // mod * mod + mod]

        extracted_roi[2] = int(self._tmpl_sz[0])
        extracted_roi[3] = int(self._tmpl_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2]/2)
        extracted_roi[1] = int(cy - extracted_roi[3]/2)

        # z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
        z = self._crop(image, extracted_roi)
        if(z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]):
            z = cv2.resize(z, tuple(self._tmpl_sz))

        mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
        mapp = fhog.getFeatureMaps(z, self.cfg.cell_size, mapp)
        mapp = fhog.normalizeAndTruncate(mapp, 0.2)
        mapp = fhog.PCAFeatureMaps(mapp)
        self.feat_sz = [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]
        feature = mapp['map'].reshape(
            (self.feat_sz[0], self.feat_sz[1], self.feat_sz[2]))

        if(inithann):
            self.create_hanning()  # create_hanning need size_patch

        feature = self.hann * feature
        
        return feature

    def _crop(self, image, rect):
        rect = np.array(rect, dtype=int)
        corners = np.zeros(4, dtype=int)
        corners[:2] = rect[:2]
        corners[2:] = rect[:2] + rect[2:]
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

    def locate_target(self, z, x):
        k = self.gaussian_correlation(x, z)
        res = real(ifft(complex_mul(self._alphaf, fft(k))))

        _, pv, _, pi = cv2.minMaxLoc(res)   # pv:float  pi:tuple of int
        p = [float(pi[0]), float(pi[1])]   # cv::Point2f, [x,y]  #[float,float]

        if(pi[0] > 0 and pi[0] < res.shape[1]-1):
            p[0] += self.subpixel_peak(res[pi[1], pi[0]-1],
                                       pv, res[pi[1], pi[0]+1])
        if(pi[1] > 0 and pi[1] < res.shape[0]-1):
            p[1] += self.subpixel_peak(res[pi[1]-1, pi[0]],
                                       pv, res[pi[1]+1, pi[0]])

        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.

        return p, pv

    def train(self, x, train_interp_factor):
        k = self.gaussian_correlation(x, x)
        alphaf = complex_div(self._prob, fft(k)+self.cfg.lambda_)

        self._tmpl = (1-train_interp_factor)*self._tmpl + train_interp_factor*x
        self._alphaf = (1-train_interp_factor)*self._alphaf + \
            train_interp_factor*alphaf
