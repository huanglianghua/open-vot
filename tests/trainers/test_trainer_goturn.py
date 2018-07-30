from __future__ import absolute_import

import unittest
import random

from lib.trainers import TrainerGOTURN


class TestTrainerGOTURN(unittest.TestCase):

    def setUp(self):
        self.cfg_file = 'config/goturn.json'
        self.vot_dir = 'data/vot2017'
        self.vid_dir = 'data/ILSVRC'
        self.det_dir = 'data/imagenet'
        self.net_path = 'pretrained/goturn/tracker.pth'
        self.trainer = TrainerGOTURN(self.cfg_file)

    def tearDown(self):
        pass

    def test_train(self):
        self.trainer.train(self.vid_dir, self.det_dir, self.vot_dir)


if __name__ == '__main__':
    unittest.main()
