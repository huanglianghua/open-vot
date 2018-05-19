from __future__ import absolute_import

import unittest
import random

from lib.managers import ManagerGOTURN


class TestManagerGOTURN(unittest.TestCase):

    def setUp(self):
        self.cfg_file = 'config/goturn.json'
        self.vot_dir = 'data/vot2017'
        self.vid_dir = 'data/ILSVRC'
        self.det_dir = 'data/imagenet'
        self.net_path = 'pretrained/goturn/tracker.pth'
        self.manager = ManagerGOTURN(self.cfg_file)

    def tearDown(self):
        pass

    def test_track(self):
        self.manager.track(self.vot_dir, self.net_path)

    def test_train(self):
        self.manager.train(self.vid_dir, self.det_dir, self.vot_dir)


if __name__ == '__main__':
    unittest.main()
