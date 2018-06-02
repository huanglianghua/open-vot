from __future__ import absolute_import, division

import unittest
import random
import torch
import torchvision.transforms.functional as F
import numpy as np

from lib.transforms import TransformGOTURN
from lib.datasets import VOT, Pairwise
from lib.utils.viz import show_frame


class TestTransformGOTURN(unittest.TestCase):

    def setUp(self):
        self.vot_dir = 'data/vot2017'
        self.visualize = True

    def tearDown(self):
        pass

    def test_transform_goturn(self):
        base_dataset = VOT(self.vot_dir, return_rect=True, download=True)
        transform = TransformGOTURN()
        dataset = Pairwise(
            base_dataset, transform, pairs_per_video=1,
            frame_range=1, causal=True)
        self.assertGreater(len(dataset), 0)

        for crop_z, crop_x, labels in dataset:
            self.assertEqual(crop_z.size(), crop_x.size())

        if self.visualize:
            for t in range(10):
                crop_z, crop_x, labels = random.choice(dataset)
                mean_color = torch.tensor(
                    transform.mean_color).float().view(3, 1, 1)
                crop_z = F.to_pil_image((crop_z + mean_color) / 255.0)
                crop_x = F.to_pil_image((crop_x + mean_color) / 255.0)
                labels = labels.cpu().numpy()
                labels *= transform.out_size / transform.label_scale_factor

                bndbox = np.concatenate([
                    labels[:2], labels[2:] - labels[:2]])
                show_frame(crop_x, bndbox, fig_n=1, pause=1)


if __name__ == '__main__':
    unittest.main()
