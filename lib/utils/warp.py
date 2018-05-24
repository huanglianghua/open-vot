from __future__ import absolute_import, division

import numbers
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageStat, ImageOps


def pad_pil(image, npad, padding='avg'):
    if npad == 0:
        return image

    if padding == 'avg':
        avg_chan = ImageStat.Stat(image).mean
        # PIL doesn't support float RGB image
        avg_chan = tuple(int(round(c)) for c in avg_chan)
        image = ImageOps.expand(image, border=npad, fill=avg_chan)
    else:
        image = ImageOps.expand(image, border=npad, fill=padding)

    return image


def crop_pil(image, center, size, padding='avg', out_size=None):
    # convert bndbox to corners
    size = np.array(size)
    corners = np.concatenate((center - size / 2, center + size / 2))
    corners = np.round(corners).astype(int)

    pads = np.concatenate((-corners[:2], corners[2:] - image.size))
    npad = max(0, int(pads.max()))

    if npad > 0:
        image = pad_pil(image, npad, padding=padding)
    corners = tuple((corners + npad).tolist())
    patch = image.crop(corners)

    if out_size is not None:
        if isinstance(out_size, numbers.Number):
            out_size = (out_size, out_size)
        if not out_size == patch.size:
            patch = patch.resize(out_size, Image.BILINEAR)

    return patch


def pad_array(image, npad, padding='avg'):
    if npad == 0:
        return image

    if padding == 'avg':
        avg_chan = image.mean(axis=(0, 1))
        image = cv2.copyMakeBorder(image, npad, npad, npad, npad,
                                   cv2.BORDER_CONSTANT, value=avg_chan)
    else:
        image = cv2.copyMakeBorder(image, npad, npad, npad, npad,
                                   cv2.BORDER_CONSTANT, value=0)

    return image


def crop_array(image, center, size, padding='avg', out_size=None):
    # convert bndbox to corners
    size = np.array(size)
    corners = np.concatenate((center - size / 2, center + size / 2))
    corners = np.round(corners).astype(int)

    pads = np.concatenate((-corners[:2], corners[2:] - image.shape[1::-1]))
    npad = max(0, int(pads.max()))

    if npad > 0:
        image = pad_array(image, npad, padding=padding)
    corners = tuple((corners + npad).tolist())
    patch = image[corners[1]:corners[3], corners[0]:corners[2]]

    if out_size is not None:
        if isinstance(out_size, numbers.Number):
            out_size = (out_size, out_size)
        if not out_size == patch.shape[1::-1]:
            patch = cv2.resize(patch, out_size, interpolation=cv2.INTER_LINEAR)

    return patch


def encode_theta(center, size, angle, img_sz):
    device = center.device
    sx, sy = size / (img_sz - 1)
    tx, ty = (2 * center - img_sz + 1) / (img_sz - 1)

    theta = torch.FloatTensor([
        sx, 0, tx, 0, sy, ty]).view(-1, 2, 3).to(device)

    return theta


def decode_theta(theta, img_sz):
    device = theta.device
    sx, sy, tx, ty = theta[0, 0], theta[1, 1], theta[0, 2], theta[1, 2]

    center = torch.FloatTensor([tx, ty]).to(device) * (img_sz - 1)
    center = (center + img_sz - 1) / 2
    size = torch.FloatTensor([sx, sy]).to(device) * (img_sz - 1)
    angle = torch.zeros(1).to(device)

    return center, size, angle


def crop_tensor(image, center, size, padding='avg', out_size=None):
    assert out_size is not None
    img_sz = torch.tensor(image.size()[:-3:-1]).to(image.device).float()

    # calculate padding
    corners = torch.cat((center - size / 2, center + size / 2))
    pads = torch.cat((-corners[:2], corners[2:] - img_sz))
    npad = max(0, pads.max().item())

    if npad > 0 and padding == 'avg':
        avg_chan = image.view(3, -1).mean(dim=1).view(1, 3, 1, 1)
        image -= avg_chan

    out_size = torch.Size((1, 1, out_size, out_size))

    theta = encode_theta(center, size, 0, img_sz)
    grid = F.affine_grid(theta, out_size)
    patch = F.grid_sample(image, grid)

    if npad > 0 and padding == 'avg':
        patch += avg_chan

    return patch


def resize_tensor(image, size):
    if isinstance(size, numbers.Number):
        size = torch.Size((1, 1, size, size))

    theta = torch.FloatTensor([1, 0, 0, 0, 1, 0]).to(
        image.device).view(-1, 2, 3).float()
    grid = F.affine_grid(theta, size)

    return F.grid_sample(image, grid)
