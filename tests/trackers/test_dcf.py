from __future__ import absolute_import, print_function

import unittest
import random

from lib.trackers import TrackerDCF
from lib.datasets import OTB


class TestTrackerDCF(unittest.TestCase):

    def setUp(self):
        self.otb_dir = 'data/OTB'

    def tearDown(self):
        pass

    def test_dcf(self):
        dataset = OTB(self.otb_dir, download=True)
        tracker = TrackerDCF()

        img_files, anno = random.choice(dataset)
        rects, speed = tracker.track(
            img_files, anno[0, :], visualize=True)
        self.assertEqual(rects.shape, anno.shape)


if __name__ == '__main__':
    unittest.main()
