from __future__ import absolute_import, print_function, division

import unittest
import random
from PIL import Image

from lib.utils.warp import pad, crop
from lib.utils.viz import show_frame
from lib.datasets import OTB


class TestWarp(unittest.TestCase):

    def setUp(self):
        self.otb_dir = 'data/OTB'

    def tearDown(self):
        pass

    def test_pad(self):
        dataset = OTB(self.otb_dir, download=True)

        npad = random.choice([0, 10, 50])
        padding = random.choice([None, 0, 'avg'])
        print('padding:', padding, 'npad:', npad)

        img_files, anno = random.choice(dataset)
        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            image = pad(image, npad, padding=padding)
            show_frame(image, fig_n=1)

    def test_crop(self):
        dataset = OTB(self.otb_dir, download=True)

        padding = random.choice([None, 0, 'avg'])
        out_size = random.choice([None, 255])
        print('padding:', padding, 'out_size:', out_size)

        img_files, anno = random.choice(dataset)
        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            bndbox = anno[f, :]
            center = bndbox[:2] + bndbox[2:] / 2
            patch = crop(image, center, bndbox[2:],
                         padding=padding, out_size=out_size)
            show_frame(patch, fig_n=2, pause=0.1)


if __name__ == '__main__':
    unittest.main()
