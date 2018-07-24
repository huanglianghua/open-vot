from __future__ import absolute_import

import cv2
import numpy as np
import torch


def real(img):
    return img[..., 0]


def imag(img):
    return img[..., 1]


def conj(img):
    img = img.copy()
    img[..., 1] = -img[..., 1]

    return img


def fft2(img):
    img = np.float32(img)
    if img.ndim == 2:
        out = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
    elif img.ndim == 3:
        out = []
        for c in range(img.shape[2]):
            out.append(cv2.dft(
                img[..., c], flags=cv2.DFT_COMPLEX_OUTPUT))
        out = np.stack(out, axis=2)
    else:
        raise Exception('only supports 2 or 3 dimensional array')

    return out


def ifft2(img):
    img = np.float32(img)
    if img.ndim == 3:
        out = cv2.dft(img, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE)
    elif img.ndim == 4:
        out = []
        for c in range(img.shape[2]):
            out.append(cv2.dft(
                img[:, :, c, :], flags=cv2.DFT_INVERSE | cv2.DFT_SCALE))
    else:
        raise Exception('only supports 3 or 4 dimensional array')

    return out


def fft1(img):
    img = np.float32(img)
    if img.ndim == 1:
        img = img[np.newaxis, :]
        out = cv2.dft(img, flags=cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT)
        out = out.squeeze(0)
    elif img.ndim == 2:
        out = cv2.dft(img, flags=cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT)
    else:
        raise Exception('only supports 1 or 2 dimensional array')

    return out


def ifft1(img):
    img = np.float32(img)
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
        out = cv2.dft(img, flags=cv2.DFT_ROWS | cv2.DFT_SCALE)
        out = out.squeeze(0)
    elif img.ndim == 3:
        out = cv2.dft(img, flags=cv2.DFT_ROWS | cv2.DFT_SCALE)
    else:
        raise Exception('only supports 2 or 3 dimensional array')

    return out


def complex_mul(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = a.copy()
    out[..., 0] = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    out[..., 1] = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]

    return out


def complex_div(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = a.copy()
    divisor = b[..., 0] ** 2 + b[..., 1] ** 2

    out[..., 0] = (a[..., 0] * b[..., 0] +
                   a[..., 1] * b[..., 1]) / divisor
    out[..., 1] = (a[..., 1] * b[..., 0] +
                   a[..., 0] * b[..., 1]) / divisor

    return out

def tensor_complex_mul(x, z): # x z
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)

def tensor_complex_mulconj(x, z): #x  z*
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)

def tensor_complex_conj(x): #x*
    out_real = x[..., 0]
    out_imag = -1*x[..., 1]
    return torch.stack((out_real, out_imag), -1)

def tensor_complex_division(x, z):
    denominator = z[..., 0]**2 + z[..., 1]**2
    out_real = (x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]) / denominator
    out_imag = (x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]) / denominator
    return torch.stack((out_real, out_imag), -1)

def fftshift(img):
    out = img.copy()

    if img.ndim == 1:
        w = img.shape[0]
        c = w // 2
        out[:c], out[c:] = img[w - c:], img[:w - c]
    elif img.ndim == 2:
        h, w = img.shape
        cy, cx = h // 2, w // 2
        out[:cy, :cx], out[cy:, cx:] = \
            img[h - cy:, w - cx:], img[:h - cy, :w - cx]
        out[:cy, cx:], out[cy:, :cx] = \
            img[h - cy:, :w - cx], img[:h - cy, w - cx:]
    else:
        raise Exception('only supports 1 or 2 dimensional array')

    return out


def ifftshift(img):
    out = img.copy()

    if img.ndim == 1:
        w = img.shape[0]
        c = w // 2
        out[w - c:], out[:w - c] = img[:c], img[c:]
    elif img.ndim == 2:
        h, w = img.shape
        cy, cx = h // 2, w // 2
        out[h - cy:, w - cx:], out[:h - cy, :w - cx] = \
            img[:cy, :cx], img[cy:, cx:]
        out[h - cy:, :w - cx], out[:h - cy, w - cx:] = \
            img[:cy, cx:], img[cy:, :cx]
    else:
        raise Exception('only supports 1 or 2 dimensional array')

    return out
