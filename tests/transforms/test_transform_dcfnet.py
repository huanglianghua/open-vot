from __future__ import absolute_import, division

import unittest
import random
import torchvision.transforms.functional as F
import numpy as np

from lib.transforms import TransformDCFNet
from lib.datasets import VOT, Pairwise
from lib.utils.viz import show_frame


class TestTransformDCFNet(unittest.TestCase):

    def setUp(self):
        self.vot_dir = 'data/vot2017'
        self.stats_path = 'pretrained/siamfc/cfnet_ILSVRC2015.stats.mat'
        self.visualize = True

    def tearDown(self):
        pass

    def test_transform_dcfnet(self):
        base_dataset = VOT(self.vot_dir, return_rect=True, download=True)
        transform = TransformDCFNet(stats_path=self.stats_path)
        dataset = Pairwise(
            base_dataset, transform=transform, pairs_per_video=1, subset='train')
        self.assertGreater(len(dataset), 0)

        if self.visualize:
            crop_z, crop_x = random.choice(dataset)
            crop_z = F.to_pil_image(crop_z / 255.0)
            crop_x = F.to_pil_image(crop_x / 255.0)

            show_frame(crop_z, None, fig_n=1, pause=1)
            show_frame(crop_x, None, fig_n=2, pause=1)

if __name__ == '__main__':
    unittest.main()
