from __future__ import absolute_import, division

import numbers
import numpy as np
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
