from __future__ import absolute_import

import unittest
import random
from PIL import Image

from lib.datasets import OTB
from lib.utils.viz import show_frame


class TestOTB(unittest.TestCase):

    def setUp(self):
        self.otb_dir = 'data/OTB'
        self.visualize = True

    def tearDown(self):
        pass

    def test_load(self):
        dataset = OTB(self.otb_dir)
        self.assertGreater(len(dataset), 0)

        for img_files, anno in dataset:
            self.assertGreater(len(img_files), 0)
            self.assertEqual(len(img_files), len(anno))

        if self.visualize:
            img_files, anno = random.choice(dataset)
            for f, img_file in enumerate(img_files):
                image = Image.open(img_file)
                show_frame(image, anno[f, :])

    def test_download(self):
        dataset = OTB(self.otb_dir, download=True, version=2015)
        self.assertGreater(len(dataset), 0)


if __name__ == '__main__':
    unittest.main()
