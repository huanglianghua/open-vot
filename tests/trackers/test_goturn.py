from __future__ import absolute_import, print_function

import unittest
import random
from torch.utils.data import DataLoader

from lib.trackers import TrackerGOTURN
from lib.datasets import VOT


class TestTrackerGOTURN(unittest.TestCase):

    def setUp(self):
        self.vot_dir = 'data/vot2017'
        self.net_path = 'pretrained/goturn/tracker.pt'

    def tearDown(self):
        pass

    def test_goturn_track(self):
        dataset = VOT(self.vot_dir, return_bndbox=True)
        tracker = TrackerGOTURN(self.net_path)

        img_files, anno = random.choice(dataset)
        rects, speed = tracker.track(img_files, anno[0, :],
                                     visualize=True)
        self.assertEqual(rects.shape, anno.shape)


if __name__ == '__main__':
    unittest.main()
