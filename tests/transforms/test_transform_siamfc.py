from __future__ import absolute_import, division

import unittest
import random
import torchvision.transforms.functional as F
import numpy as np

from lib.transforms import TransformSiamFC
from lib.datasets import VOT, Pairwise
from lib.utils.viz import show_frame


class TestTransformSiamFC(unittest.TestCase):

    def setUp(self):
        self.vot_dir = 'data/vot2017'
        self.visualize = True

    def tearDown(self):
        pass

    def test_transform_siamfc(self):
        base_dataset = VOT(self.vot_dir, return_bndbox=True)
        transform = TransformSiamFC()
        dataset = Pairwise(
            base_dataset, transform=transform, subset='train')
        self.assertGreater(len(dataset), 0)

        for crop_z, crop_x, labels, weights in dataset:
            self.assertAlmostEqual(
                weights[labels == 1].sum().item(),
                weights[labels == 0].sum().item())
            self.assertAlmostEqual(
                weights.sum().item(), labels[labels >= 0].numel())
            self.assertEqual(
                weights[labels == transform.ignore_label].sum().item(), 0)

        if self.visualize:
            crop_z, crop_x, labels, weights = random.choice(dataset)
            crop_z = F.to_pil_image(crop_z / 255.0)
            crop_x = F.to_pil_image(crop_x / 255.0)
            labels = self._rescale(labels.cpu().squeeze().numpy())
            weights = self._rescale(weights.cpu().squeeze().numpy())

            bndbox_z = np.array([31, 31, 64, 64])
            bndbox_x = np.array([95, 95, 64, 64])

            show_frame(crop_z, bndbox_z, fig_n=1, pause=1)
            show_frame(crop_x, bndbox_x, fig_n=2, pause=1)
            show_frame(labels, fig_n=3, pause=1, cmap='hot')
            show_frame(weights, fig_n=4, pause=5, cmap='hot')

    def _rescale(self, array):
        array -= array.min()
        array /= array.max()

        return array


if __name__ == '__main__':
    unittest.main()
