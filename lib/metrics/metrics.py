from __future__ import absolute_import, division

import numpy as np
from shapely.geometry import box, Polygon


def center_error(rects1, rects2):
    r"""Center error.
    """
    centers1 = rects1[..., :2] + (rects1[..., 2:] - 1) / 2
    centers2 = rects2[..., :2] + (rects2[..., 2:] - 1) / 2
    errors = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=-1))

    return errors


def rect_iou(rects1, rects2):
    r"""Intersection over union.
    """
    assert rects1.shape == rects2.shape
    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious


def _intersection(rects1, rects2):
    r"""Rectangle intersection.
    """
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T


def poly_iou(polys1, polys2, bound=None):
    r"""Intersection over union of polygons.
    """
    assert len(polys1) == len(polys2)
    polys1 = _to_poly(polys1)
    polys2 = _to_poly(polys2)
    if bound is not None:
        bound = box(0, 0, bound[0] - 1, bound[1] - 1)
        polys1 = [p.intersection(bound) for p in polys1]
        polys2 = [p.intersection(bound) for p in polys2]

    ious = []
    eps = np.finfo(float).eps
    for poly1, poly2 in zip(polys1, polys2):
        area_inter = poly1.intersection(poly2).area
        area_union = poly1.union(poly2).area
        ious.append(area_inter / (area_union + eps))
    
    return np.clip(np.asarray(ious), 0.0, 1.0)


def _to_poly(arrays):
    r"""Convert 4, 6 or 8 dimensional arrays to Polygons
    """
    def to_poly(array):
        if len(array) == 4:
            return box(array[0], array[1],
                       array[0] + array[2], array[1] + array[3])
        elif len(array) == 8:
            return Polygon([(array[2 * i], array[2 * i + 1])
                            for i in range(4)])
        else:
            raise Exception(
                'Only 4, 6 or 8 dimensional array is supported.')
    
    return [to_poly(t) for t in arrays]
