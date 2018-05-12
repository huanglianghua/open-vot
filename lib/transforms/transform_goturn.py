from __future__ import absolute_import

import torchvision.transforms.functional as F
import torch
import numpy as np
import math

from ..utils.warp import crop


class TransformGOTURN(object):

    def __init__(self, lambda_shift=5, lambda_scale=15,
                 min_scale=-0.4, max_scale=0.4, context=2,
                 out_size=227, label_scale_factor=10,
                 mean_color=[104, 117, 123]):
        self.lambda_shift = lambda_shift
        self.lambda_scale = lambda_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.context = context
        self.out_size = out_size
        self.label_scale_factor = label_scale_factor
        self.mean_color = mean_color

    def __call__(self, img_z, img_x, bndbox_z, bndbox_x):
        rand_bndbox_x = self._rand_shift(bndbox_x, img_x.size)

        # crop image regions
        crop_z = self._crop(img_z, bndbox_z)
        crop_x = self._crop(img_x, rand_bndbox_x)
        labels = self._create_labels(rand_bndbox_x, bndbox_x)

        # convert data to tensors
        crop_z = 255.0 * F.to_tensor(crop_z)
        crop_x = 255.0 * F.to_tensor(crop_x)
        labels = torch.from_numpy(labels)

        # color augmentation
        mean_color = torch.tensor(self.mean_color).float().view(3, 1, 1)
        crop_z -= mean_color
        crop_x -= mean_color

        return crop_z, crop_x, labels

    def _rand_shift(self, bndbox, img_sz):
        def rand_fn(lambda_, min_val=None, max_val=None):
            sign = +1 if np.random.rand() > 0.5 else -1
            rand = math.log(np.random.rand()) / (lambda_ * sign)
            if min_val is not None or max_val is not None:
                rand = np.clip(rand, min_val, max_val)
            return rand

        center = bndbox[:2] + bndbox[2:] / 2
        size = bndbox[2:]

        # randomly rescale the size
        scale_factors = [
            rand_fn(self.lambda_scale, self.min_scale, self.max_scale),
            rand_fn(self.lambda_scale, self.min_scale, self.max_scale)]
        rand_sz = size * (1 + np.array(scale_factors))
        rand_sz = np.clip(rand_sz, 1.0, img_sz)

        # randomly shift the center
        shift_factors = [
            rand_fn(self.lambda_shift),
            rand_fn(self.lambda_shift)]
        rand_center = center + size * shift_factors
        rand_center = np.clip(
            rand_center, rand_sz / 2, img_sz - rand_sz / 2)
        
        rand_bndbox = np.concatenate([
            rand_center - rand_sz / 2, rand_sz])

        return rand_bndbox

    def _crop(self, image, bndbox):
        center = bndbox[:2] + bndbox[2:] / 2
        size = bndbox[2:] * self.context
        patch = crop(image, center, size, padding=0,
                     out_size=self.out_size)

        return patch

    def _create_labels(self, rand_bndbox, bndbox):
        rand_corners = np.concatenate([
            rand_bndbox[:2], rand_bndbox[:2] + rand_bndbox[2:]])
        anchor_corners = np.concatenate([
            bndbox[:2], bndbox[:2] + bndbox[2:]])
        offsets = anchor_corners - rand_corners

        # normalize offsets
        offsets[0::2] /= rand_bndbox[2] * self.context
        offsets[1::2] /= rand_bndbox[3] * self.context

        # normalize corners
        margin = self.out_size * (1 - 1 / self.context) / 2
        corners = np.array([
            margin, margin,
            self.out_size - margin, self.out_size - margin])
        corners = corners / self.out_size + offsets
        corners *= self.label_scale_factor

        return corners
