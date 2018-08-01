from __future__ import absolute_import, division

import numpy as np
import torchvision.transforms.functional as F
import torch

from ..utils import dict2tuple
from ..utils.ioutil import load_siamfc_stats
from ..utils.warp import crop_pil
from lib.utils.viz import show_frame

class TransformDCFNet(object):

    def __init__(self, stats_path=None, **kargs):
        self.parse_args(**kargs)
        self.stats = None
        if stats_path:
            self.stats = load_siamfc_stats(stats_path)

    def parse_args(self, **kargs):

        default_args = {
            'exemplar_sz': 125,
            'padding': 2
            }

        for key, val in default_args.items():
            if key in kargs:
                setattr(self, key, kargs[key])
            else:
                setattr(self, key, val)

    def __call__(self, img_z, img_x, bndbox_z, bndbox_x):

        crop_z = self._crop(img_z, bndbox_z)
        crop_x = self._crop(img_x, bndbox_x)

        # data augmentation
        if np.random.rand() > 0.5:
            crop_z = F.hflip(crop_z)
            crop_x = F.hflip(crop_x)

        crop_z = 255.0 * F.to_tensor(crop_z)
        crop_x = 255.0 * F.to_tensor(crop_x)

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

        return crop_z, crop_x

    def _crop(self, image, bndbox):
        center = bndbox[:2] + bndbox[2:] / 2
        size = bndbox[2:]

        patch_sz = size * (1 + self.padding)

        return crop_pil(image, center, patch_sz, out_size=self.exemplar_sz)
