from __future__ import absolute_import, division

import numpy as np
import cv2

from . import Tracker
from ..utils.complex import complex_div


class TrackerMOSSE(Tracker):

    def __init__(self):
        super(TrackerMOSSE, self).__init__('MOSSE')
        self.eps = 1e-5

    def init(self, frame, rect):
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rect = rect.astype(int)
        rect[2:] += rect[:2]
        x1, y1, x2, y2 = rect
        w, h = map(cv2.getOptimalDFTSize, [x2 - x1, y2 - y1])
        x1, y1 = (x1 + x2 - w) // 2, (y1 + y2 - h) // 2
        self.t_center = x, y = x1 + 0.5 * (w - 1), y1 + 0.5 * (h - 1)
        self.t_sz = w, h
        img = cv2.getRectSubPix(frame, (w, h), (x, y))

        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h // 2, w // 2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.A = np.zeros_like(self.G)
        self.B = np.zeros_like(self.G)
        for _i in range(128):
            patch = self._preprocess(self._random_warp(img))
            F = cv2.dft(patch, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.A += cv2.mulSpectrums(self.G, F, 0, conjB=True)
            self.B += cv2.mulSpectrums(F, F, 0, conjB=True)
        self._update_kernel()
        self.update(frame)

    def update(self, frame, rate=0.125):
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        (x, y), (w, h) = self.t_center, self.t_sz
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
        img = self._preprocess(img)
        self.last_resp, (dx, dy), self.psr = self._linear_correlation(img)
        self.good = self.psr > 8.0

        if self.good:
            self.t_center = x + dx, y + dy
            self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.t_center)
            img = self._preprocess(img)

            F = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
            A = cv2.mulSpectrums(self.G, F, 0, conjB=True)
            B = cv2.mulSpectrums(F, F, 0, conjB=True)
            self.A = self.A * (1.0-rate) + A * rate
            self.B = self.B * (1.0-rate) + B * rate
            self._update_kernel()

        return np.array([x - 0.5 * (w - 1), y - 0.5 * (h - 1), w, h])

    def _preprocess(self, img):
        img = np.log(np.float32(img) + 1.0)
        img = (img - img.mean()) / (img.std() + self.eps)

        return img * self.win

    def _linear_correlation(self, img):
        C = cv2.mulSpectrums(
            cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx - 5, my - 5), (mx + 5, my + 5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval - smean) / (sstd + self.eps)

        return resp, (mx - w // 2, my - h // 2), psr

    def _random_warp(self, img):
        h, w = img.shape[:2]
        T = np.zeros((2, 3))
        coef = 0.2
        ang = (np.random.rand() - 0.5) * coef
        c, s = np.cos(ang), np.sin(ang)
        T[:2, :2] = [[c, -s], [s, c]]
        T[:2, :2] += (np.random.rand(2, 2) - 0.5) * coef
        c = (w / 2, h / 2)
        T[:, 2] = c - np.dot(T[:2, :2], c)

        return cv2.warpAffine(img, T, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _update_kernel(self):
        self.H = complex_div(self.A, self.B)
        self.H[..., 1] *= -1
