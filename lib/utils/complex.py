from __future__ import absolute_import

import cv2
import numpy as np


def real(img):
    return img[:, :, 0]


def imag(img):
    return img[:, :, 1]


def fft(img):
    return cv2.dft(np.float32(img),
                   flags=cv2.DFT_COMPLEX_OUTPUT)


def ifft(img):
    return cv2.dft(np.float32(img),
                   flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE))


def complex_mul(a, b):
    out = a.copy()
    out[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    out[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]

    return out


def complex_div(a, b):
    out = a.copy()
    divisor = (b[:, :, 0] ** 2 + b[:, :, 1] ** 2)

    out[:, :, 0] = (a[:, :, 0] * b[:, :, 0] +
                    a[:, :, 1] * b[:, :, 1]) / divisor
    out[:, :, 1] = (a[:, :, 1] * b[:, :, 0] +
                    a[:, :, 0] * b[:, :, 1]) / divisor

    return out


def circ_shift(img):
    out = img.copy()
    h, w = img.shape
    cy, cx = h // 2, w // 2

    out[:cy, :cx], out[cy:, cx:] = img[h - cy:, w - cx:], img[:h - cy, :w - cx]
    out[:cy, cx:], out[cy:, :cx] = img[h - cy:, :w - cx], img[:h - cy, w - cx:]

    return out
