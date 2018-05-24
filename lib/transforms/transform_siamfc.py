from __future__ import absolute_import, division

import numpy as np
import torchvision.transforms.functional as F
import torch

from ..utils import dict2tuple
from ..utils.ioutil import load_siamfc_stats
from ..utils.warp import crop_pil


class TransformSiamFC(object):

    def __init__(self, stats_path=None, **kargs):
        self.parse_args(**kargs)
        self.stats = None
        if stats_path:
            self.stats = load_siamfc_stats(stats_path)

    def parse_args(self, **kargs):
        # default branch is AlexNetV1
        default_args = {
            'exemplar_sz': 127,
            'search_sz': 255,
            'score_sz': 17,
            'context': 0.5,
            'r_pos': 16,
            'r_neg': 0,
            'total_stride': 8,
            'ignore_label': -100}

        for key, val in default_args.items():
            if key in kargs:
                setattr(self, key, kargs[key])
            else:
                setattr(self, key, val)

    def __call__(self, img_z, img_x, bndbox_z, bndbox_x):
        crop_z = self._crop(img_z, bndbox_z, self.exemplar_sz)
        crop_x = self._crop(img_x, bndbox_x, self.search_sz)
        labels, weights = self._create_labels()

        # data augmentation
        if np.random.rand() > 0.5:
            crop_z = F.hflip(crop_z)
            crop_x = F.hflip(crop_x)

        crop_z = 255.0 * F.to_tensor(crop_z)
        crop_x = 255.0 * F.to_tensor(crop_x)
        labels = torch.from_numpy(labels).float()
        weights = torch.from_numpy(weights).float()

        # color augmentation
        if self.stats:
            offset_z = np.reshape(np.dot(
                self.stats.rgb_variance_z,
                np.random.randn(3, 1)), (3, 1, 1))
            offset_x = np.reshape(np.dot(
                self.stats.rgb_variance_x,
                np.random.randn(3, 1)), (3, 1, 1))
            crop_z += torch.from_numpy(offset_z).float()
            crop_x += torch.from_numpy(offset_x).float()
            crop_z = torch.clamp(crop_z, 0.0, 255.0)
            crop_x = torch.clamp(crop_x, 0.0, 255.0)

        return crop_z, crop_x, labels, weights

    def _crop(self, image, bndbox, out_size):
        center = bndbox[:2] + bndbox[2:] / 2
        size = bndbox[2:]

        context = self.context * size.sum()
        patch_sz = out_size / self.exemplar_sz * \
            np.sqrt((size + context).prod())

        return crop_pil(image, center, patch_sz, out_size=out_size)

    def _create_labels(self):
        labels = self._create_logisticloss_labels()
        weights = np.zeros_like(labels)

        pos_num = np.sum(labels == 1)
        neg_num = np.sum(labels == 0)
        weights[labels == 1] = 0.5 / pos_num
        weights[labels == 0] = 0.5 / neg_num
        weights *= pos_num + neg_num

        labels = labels[np.newaxis, :]
        weights = weights[np.newaxis, :]

        return labels, weights

    def _create_logisticloss_labels(self):
        label_sz = self.score_sz
        r_pos = self.r_pos / self.total_stride
        r_neg = self.r_neg / self.total_stride
        labels = np.zeros((label_sz, label_sz))

        for r in range(label_sz):
            for c in range(label_sz):
                dist = np.sqrt((r - label_sz // 2) ** 2 +
                               (c - label_sz // 2) ** 2)
                if dist <= r_pos:
                    labels[r, c] = 1
                elif dist <= r_neg:
                    labels[r, c] = self.ignore_label
                else:
                    labels[r, c] = 0

        return labels
