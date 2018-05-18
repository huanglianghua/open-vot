from __future__ import absolute_import

import unittest
import random
import os
from PIL import Image

from lib.datasets import ImageNetVID, ImageNetObject
from lib.utils.viz import show_frame


class TestImageNet(unittest.TestCase):

    def setUp(self):
        self.vid_dir = 'data/ILSVRC'
        self.obj_dir = 'data/imagenet'
        self.visualize = True

    def tearDown(self):
        pass

    def test_imagenet_vid(self):
        dataset = ImageNetVID(self.vid_dir, return_rect=True)
        self.assertGreater(len(dataset), 0)

        for i in range(10):
            img_files, anno = random.choice(dataset)
            self.assertGreater(len(img_files), 0)
            self.assertEqual(len(img_files), len(anno))

        if self.visualize:
            img_files, anno = random.choice(dataset)
            for f, img_file in enumerate(img_files):
                image = Image.open(img_file)
                show_frame(image, anno[f, :])

    def test_imagenet_obj(self):
        subset = random.choice(['train', 'val'])
        dataset = ImageNetObject(
            self.obj_dir, subset=subset, return_rect=True)
        self.assertGreater(len(dataset), 0)

        for i in range(10):
            img_file, bndbox = dataset[i]
            self.assertTrue(os.path.isfile(img_file))
            self.assertTrue(len(bndbox) == 4)

            if self.visualize:
                img_file, bndbox = random.choice(dataset)
                image = Image.open(img_file)
                show_frame(image, bndbox, fig_n=1, pause=0.1)


if __name__ == '__main__':
    unittest.main()
