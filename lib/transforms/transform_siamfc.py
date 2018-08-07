from __future__ import absolute_import, division

import numpy as np
import torchvision.transforms.functional as F
import torch
import numbers
from PIL import Image

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
            'ignore_label': -100,
            # augmentation parameters
            'aug_translate': True,
            'max_translate': 4,
            'aug_stretch': True,
            'max_stretch': 0.05,
            'aug_color': True}

        for key, val in default_args.items():
            if key in kargs:
                setattr(self, key, kargs[key])
            else:
                setattr(self, key, val)

    def __call__(self, img_z, img_x, bndbox_z, bndbox_x):
        crop_z = self._crop(img_z, bndbox_z, self.exemplar_sz)
        crop_x = self._crop(img_x, bndbox_x, self.search_sz)
        labels, weights = self._create_labels()

        crop_z = self._acquire_augment(
            crop_z, self.exemplar_sz, self.stats.rgb_variance_z)
        crop_x = self._acquire_augment(
            crop_x, self.search_sz, self.stats.rgb_variance_x)

        crop_z = (255.0 * F.to_tensor(crop_z)).float()
        crop_x = (255.0 * F.to_tensor(crop_x)).float()
        labels = torch.from_numpy(labels).float()
        weights = torch.from_numpy(weights).float()

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

    def _acquire_augment(self, patch, out_size, rgb_variance):
        center = (out_size // 2, out_size // 2)
        patch_sz = np.asarray(patch.size)

        if self.aug_stretch:
            scale = (1 + self.max_stretch * (-1 + 2 * np.random.rand(2)))
            size = np.round(np.minimum(out_size * scale, patch_sz))
        else:
            size = patch_sz

        if self.aug_translate:
            mx, my = np.minimum(
                self.max_translate, np.floor((patch_sz - size) / 2))
            rx = np.random.randint(-mx, mx) if mx > 0 else 0
            ry = np.random.randint(-my, my) if my > 0 else 0
            dx = center[0] - size[0] // 2 + rx
            dy = center[1] - size[1] // 2 + ry
        else:
            dx = center[0] - size[0] // 2
            dy = center[1] - size[1] // 2

        patch = patch.crop((
            int(dx), int(dy),
            int(dx + round(size[0])),
            int(dy + round(size[1]))))
        patch = patch.resize((out_size, out_size), Image.NEAREST)

        if self.aug_color:
            offset = np.reshape(np.dot(
                rgb_variance, np.random.randn(3)), (1, 1, 3))
            out = patch - offset
        else:
            out = patch

        return out
