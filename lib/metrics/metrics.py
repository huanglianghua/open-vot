from __future__ import absolute_import, division

import numpy as np


def center_error(rects1, rects2):
    r"""Center error.
    """
    centers1 = rects1[:, :2] + (rects1[:, 2:] - 1) / 2
    centers2 = rects2[:, :2] + (rects2[:, 2:] - 1) / 2
    ces = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=1))

    return ces


def iou(rects1, rects2):
    r"""Intersection over union.
    """
    rects_inter = _intersection(rects1, rects2)

    if rects1.ndim == 1:
        areas1 = np.prod(rects1[2:])
        areas2 = np.prod(rects2[2:])
        area_inter = np.prod(rects_inter[2:])
    elif rects1.ndim == 2:
        areas1 = np.prod(rects1[:, 2:], axis=1)
        areas2 = np.prod(rects2[:, 2:], axis=1)
        area_inter = np.prod(rects_inter[:, 2:], axis=1)
    else:
        raise Exception('Wrong dimension of rects!')

    area_union = areas1 + areas2 - area_inter
    ious = area_inter / (area_union + 1e-12)
    
    assert np.all(np.logical_and(ious >= 0 - 1e-6, ious <= 1 + 1e-6))
    ious = np.clip(ious, 0, 1)

    return ious


def _intersection(rects1, rects2):
    r"""Rectangle intersection.
    """
    assert rects1.shape == rects2.shape

    if rects1.ndim == 1:
        x1 = max(rects1[0], rects2[0])
        y1 = max(rects1[1], rects2[1])
        x2 = min(rects1[0] + rects1[2], rects2[0] + rects2[2])
        y2 = min(rects1[1] + rects1[3], rects2[1] + rects2[3])

        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        return np.array([x1, y1, w, h])
    elif rects1.ndim == 2:
        x1 = np.maximum(rects1[:, 0], rects2[:, 0])
        y1 = np.maximum(rects1[:, 1], rects2[:, 1])
        x2 = np.minimum(rects1[:, 0] + rects1[:, 2],
                        rects2[:, 0] + rects2[:, 2])
        y2 = np.minimum(rects1[:, 1] + rects1[:, 3],
                        rects2[:, 1] + rects2[:, 3])

        w = np.maximum(x2 - x1, 0)
        h = np.maximum(y2 - y1, 0)

        return np.stack((x1, y1, w, h), axis=1)
