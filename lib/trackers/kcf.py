from __future__ import absolute_import, division

import numpy as np
import cv2

from . import Tracker
from ..descriptors import fhog
from ..utils.complex import *


# recttools
def x2(rect):
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]


def limit(rect, limit):
    if(rect[0]+rect[2] > limit[0]+limit[2]):
        rect[2] = limit[0]+limit[2]-rect[0]
    if(rect[1]+rect[3] > limit[1]+limit[3]):
        rect[3] = limit[1]+limit[3]-rect[1]
    if(rect[0] < limit[0]):
        rect[2] -= (limit[0]-rect[0])
        rect[0] = limit[0]
    if(rect[1] < limit[1]):
        rect[3] -= (limit[1]-rect[1])
        rect[1] = limit[1]
    if(rect[2] < 0):
        rect[2] = 0
    if(rect[3] < 0):
        rect[3] = 0
    return rect


def getBorder(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert(np.all(np.array(res) >= 0))
    return res


def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])   # modify cutWindow
    assert(cutWindow[2] > 0 and cutWindow[3] > 0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1]+cutWindow[3],
              cutWindow[0]:cutWindow[0]+cutWindow[2]]

    if(border != [0, 0, 0, 0]):
        res = cv2.copyMakeBorder(
            res, border[1], border[3], border[0], border[2], borderType)
    return res


class TrackerKCF(Tracker):

    def __init__(self, fixed_window=True, multiscale=True):
        super(TrackerKCF, self).__init__('KCF')
        self.lambdar = 0.0001   # regularization
        self.padding = 2.5   # extra area surrounding the target
        self.output_sigma_factor = 0.125   # bandwidth of gaussian target

        self.interp_factor = 0.012   # linear interpolation factor for adaptation
        self.sigma = 0.6  # gaussian kernel bandwidth
        # TPAMI   #interp_factor = 0.02   #sigma = 0.5
        self.cell_size = 4   # HOG cell size

        if(multiscale):
            self.template_size = 96   # template size
            self.scale_step = 1.05   # scale step for multi-scale estimation
            # to downweight detection scores of other scales for added stability
            self.scale_weight = 0.96
        elif(fixed_window):
            self.template_size = 96
            self.scale_step = 1
        else:
            self.template_size = 1
            self.scale_step = 1

        self._tmpl_sz = [0, 0]  # cv::Size, [width,height]  #[int,int]
        # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self._roi = [0., 0., 0., 0.]
        self.feat_sz = [0, 0, 0]  # [int,int,int]
        self._scale = 1.   # float
        # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._alphaf = None
        self._prob = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
        self._tmpl = None
        # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
        self.hann = None

    def subpixel_peak(self, left, center, right):
        divisor = 2*center - right - left  # float
        return (0 if abs(divisor) < 1e-3 else 0.5*(right-left)/divisor)

    def create_hanning(self):
        hann2d = np.outer(
            np.hanning(self.feat_sz[0]),
            np.hanning(self.feat_sz[1])).astype(np.float32)

        hann1d = hann2d.reshape(self.feat_sz[0]*self.feat_sz[1])
        self.hann = np.zeros((self.feat_sz[2], 1), np.float32) + hann1d
        self.hann = self.hann.astype(np.float32)

    def create_labels(self, sizey, sizex):
        syh, sxh = sizey//2, sizex//2
        output_sigma = np.sqrt(sizex*sizey) / \
            self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma*output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y-syh)**2, (x-sxh)**2
        res = np.exp(mult * (y+x))
        return fft(res)

    def gaussian_correlation(self, x1, x2):
        c = np.zeros((self.feat_sz[0], self.feat_sz[1]), np.float32)
        for i in range(self.feat_sz[2]):
            x1aux = x1[i, :].reshape((self.feat_sz[0], self.feat_sz[1]))
            x2aux = x2[i, :].reshape((self.feat_sz[0], self.feat_sz[1]))
            caux = cv2.mulSpectrums(fft(x1aux), fft(x2aux), 0, conjB=True)
            caux = real(ifft(caux))
            #caux = circ_shift(caux)
            c += caux
        c = circ_shift(c)

        if(x1.ndim == 3 and x2.ndim == 3):
            d = (np.sum(x1[:, :, 0]*x1[:, :, 0]) + np.sum(x2[:, :, 0]*x2[:, :, 0]
                                                          ) - 2.0*c) / (self.feat_sz[0]*self.feat_sz[1]*self.feat_sz[2])
        elif(x1.ndim == 2 and x2.ndim == 2):
            d = (np.sum(x1*x1) + np.sum(x2*x2) - 2.0*c) / \
                (self.feat_sz[0]*self.feat_sz[1]*self.feat_sz[2])

        d = d * (d >= 0)
        d = np.exp(-d / (self.sigma*self.sigma))

        return d

    def extract_feature(self, image, inithann, scale_adjust=1.0):
        extracted_roi = [0, 0, 0, 0]  # [int,int,int,int]
        cx = self._roi[0] + self._roi[2]/2  # float
        cy = self._roi[1] + self._roi[3]/2  # float

        if(inithann):
            padded_w = self._roi[2] * self.padding
            padded_h = self._roi[3] * self.padding

            if(self.template_size > 1):
                if(padded_w >= padded_h):
                    self._scale = padded_w / float(self.template_size)
                else:
                    self._scale = padded_h / float(self.template_size)
                self._tmpl_sz[0] = int(padded_w / self._scale)
                self._tmpl_sz[1] = int(padded_h / self._scale)
            else:
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._scale = 1.

            self._tmpl_sz[0] = int(
                self._tmpl_sz[0]) // (2*self.cell_size) * 2*self.cell_size + 2*self.cell_size
            self._tmpl_sz[1] = int(
                self._tmpl_sz[1]) // (2*self.cell_size) * 2*self.cell_size + 2*self.cell_size

        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0])
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2]/2)
        extracted_roi[1] = int(cy - extracted_roi[3]/2)

        z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
        if(z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]):
            z = cv2.resize(z, tuple(self._tmpl_sz))

        mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
        mapp = fhog.getFeatureMaps(z, self.cell_size, mapp)
        mapp = fhog.normalizeAndTruncate(mapp, 0.2)
        mapp = fhog.PCAFeatureMaps(mapp)
        self.feat_sz = [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]
        # (size_patch[2], size_patch[0]*size_patch[1])
        FeaturesMap = mapp['map'].reshape(
            (self.feat_sz[0]*self.feat_sz[1], self.feat_sz[2])).T

        if(inithann):
            self.create_hanning()  # create_hanning need size_patch

        FeaturesMap = self.hann * FeaturesMap
        return FeaturesMap

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
        alphaf = complex_div(self._prob, fft(k)+self.lambdar)

        self._tmpl = (1-train_interp_factor)*self._tmpl + train_interp_factor*x
        self._alphaf = (1-train_interp_factor)*self._alphaf + \
            train_interp_factor*alphaf

    def init(self, image, roi):
        self._roi = [float(t) for t in roi]
        assert(roi[2] > 0 and roi[3] > 0)
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
            self._tmpl, self.extract_feature(image, 0, 1.0))

        if(self.scale_step != 1):
            # Test at a smaller _scale
            new_loc1, new_peak_value1 = self.locate_target(
                self._tmpl, self.extract_feature(image, 0, 1.0/self.scale_step))
            # Test at a bigger _scale
            new_loc2, new_peak_value2 = self.locate_target(
                self._tmpl, self.extract_feature(image, 0, self.scale_step))

            if(self.scale_weight*new_peak_value1 > peak_value and new_peak_value1 > new_peak_value2):
                loc = new_loc1
                peak_value = new_peak_value1
                self._scale /= self.scale_step
                self._roi[2] /= self.scale_step
                self._roi[3] /= self.scale_step
            elif(self.scale_weight*new_peak_value2 > peak_value):
                loc = new_loc2
                peak_value = new_peak_value2
                self._scale *= self.scale_step
                self._roi[2] *= self.scale_step
                self._roi[3] *= self.scale_step

        self._roi[0] = cx - self._roi[2]/2.0 + \
            loc[0]*self.cell_size*self._scale
        self._roi[1] = cy - self._roi[3]/2.0 + \
            loc[1]*self.cell_size*self._scale

        if(self._roi[0] >= image.shape[1]-1):
            self._roi[0] = image.shape[1] - 1
        if(self._roi[1] >= image.shape[0]-1):
            self._roi[1] = image.shape[0] - 1
        if(self._roi[0]+self._roi[2] <= 0):
            self._roi[0] = -self._roi[2] + 2
        if(self._roi[1]+self._roi[3] <= 0):
            self._roi[1] = -self._roi[3] + 2
        assert(self._roi[2] > 0 and self._roi[3] > 0)

        x = self.extract_feature(image, 0, 1.0)
        self.train(x, self.interp_factor)

        return self._roi
