from __future__ import absolute_import, print_function, division

import unittest
import random
import cv2
import torch
import numpy as np
from PIL import Image

from lib.utils.warp import pad_pil, crop_pil, pad_array, crop_array, crop_tensor, resize_tensor
from lib.utils.viz import show_frame
from lib.datasets import OTB


class TestWarp(unittest.TestCase):

    def setUp(self):
        self.otb_dir = 'data/OTB'

    def tearDown(self):
        pass

    def test_pad_pil(self):
        dataset = OTB(self.otb_dir, download=True)

        npad = random.choice([0, 10, 50])
        padding = random.choice([None, 0, 'avg'])
        print('[PIL-pad] padding:', padding, 'npad:', npad)

        img_files, anno = random.choice(dataset)
        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            image = pad_pil(image, npad, padding=padding)
            show_frame(image, fig_n=1)

    def test_crop_pil(self):
        dataset = OTB(self.otb_dir, download=True)

        padding = random.choice([None, 0, 'avg'])
        out_size = random.choice([None, 255])
        print('[PIL-crop] padding:', padding, 'out_size:', out_size)

        img_files, anno = random.choice(dataset)
        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            bndbox = anno[f, :]
            center = bndbox[:2] + bndbox[2:] / 2
            patch = crop_pil(image, center, bndbox[2:],
                             padding=padding, out_size=out_size)
            show_frame(patch, fig_n=2, pause=0.1)

    def test_pad_array(self):
        dataset = OTB(self.otb_dir, download=True)

        npad = random.choice([0, 10, 50])
        padding = random.choice([None, 0, 'avg'])
        print('[cv2-pad] padding:', padding, 'npad:', npad)

        img_files, anno = random.choice(dataset)
        for f, img_file in enumerate(img_files):
            image = cv2.imread(img_file)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = pad_array(image, npad, padding=padding)
            show_frame(image[:, :, ::-1], fig_n=1)

    def test_crop_array(self):
        dataset = OTB(self.otb_dir, download=True)

        padding = random.choice([None, 0, 'avg'])
        out_size = random.choice([None, 255])
        print('[cv2-crop] padding:', padding, 'out_size:', out_size)

        img_files, anno = random.choice(dataset)
        for f, img_file in enumerate(img_files):
            image = cv2.imread(img_file)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bndbox = anno[f, :]
            center = bndbox[:2] + bndbox[2:] / 2
            patch = crop_array(image, center, bndbox[2:],
                               padding=padding, out_size=out_size)
            show_frame(patch, fig_n=2, pause=0.1)

    def test_crop_tensor(self):
        dataset = OTB(self.otb_dir, download=True)

        padding = random.choice([None, 0, 'avg'])
        out_size = random.choice([255])
        print('[PyTorch-crop] padding:', padding, 'out_size:', out_size)

        img_files, anno = random.choice(dataset)
        for f, img_file in enumerate(img_files):
            image = cv2.imread(img_file)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image).permute(
                2, 0, 1).unsqueeze(0).float()
            bndbox = torch.from_numpy(anno[f, :]).float()
            center = bndbox[:2] + bndbox[2:] / 2
            patch = crop_tensor(image, center, bndbox[2:],
                                padding=padding, out_size=out_size)
            patch = patch.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            show_frame(patch, fig_n=1, pause=0.1)

    def test_resize_tensor(self):
        dataset = OTB(self.otb_dir, download=True)

        out_size = random.choice([30, 100, 255])
        print('[PyTorch-resize]:', out_size)

        img_files, anno = random.choice(dataset)
        for f, img_file in enumerate(img_files):
            image = cv2.imread(img_file)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image).permute(
                2, 0, 1).unsqueeze(0).float()
            image = resize_tensor(image, out_size)
            image = image.squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
            show_frame(image, fig_n=2, pause=0.1)


if __name__ == '__main__':
    unittest.main()
