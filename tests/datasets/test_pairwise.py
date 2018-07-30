from __future__ import absolute_import, print_function

import unittest
import random

from lib.datasets import Pairwise, OTB
from lib.utils.viz import show_frame


class TestPairwise(unittest.TestCase):

    def setUp(self):
        self.otb_dir = 'data/OTB'
        self.visualize = True

    def tearDown(self):
        pass

    def test_pairwise(self):
        base_dataset = OTB(self.otb_dir, download=True)
        frame_range = random.choice([0, 1, 100])
        causal = random.choice([True, False])
        subset = random.choice(['train', 'val'])
        return_index = random.choice([True, False])
        rand_choice = random.choice([True, False])
        dataset = Pairwise(
            base_dataset, pairs_per_video=1, frame_range=frame_range,
            causal=causal, subset=subset, return_index=return_index,
            rand_choice=rand_choice)
        self.assertGreater(len(dataset), 0)

        for i, item in enumerate(dataset):
            img_z, img_x, bndbox_z, bndbox_x = \
                item[0], item[1], item[2], item[3]
            if return_index:
                print('rand_z:', item[4], '\trand_x:', item[5])
            self.assertEqual(img_z.mode, 'RGB')
            self.assertEqual(img_x.mode, 'RGB')
            self.assertEqual(bndbox_z.shape, (4,))
            self.assertEqual(bndbox_x.shape, (4,))

        if self.visualize:
            item = random.choice(dataset)
            img_z, img_x, bndbox_z, bndbox_x = \
                item[0], item[1], item[2], item[3]
            print(bndbox_z)
            if return_index:
                print('rand_z:', item[4], '\trand_x:', item[5])
            show_frame(img_z, bndbox_z, fig_n=1, pause=1)
            show_frame(img_x, bndbox_x, fig_n=2, pause=1)


if __name__ == '__main__':
    unittest.main()
