from __future__ import absolute_import

import unittest
import random

from lib.trainers import TrainerSiamFC


class TestManagerSiamFC(unittest.TestCase):

    def setUp(self):
        self.branch = 'alexv1'
        self.cfg_file = 'config/siamfc.json'
        self.vot_dir = 'data/vot2017'
        self.vid_dir = 'data/ILSVRC'
        self.net_path = 'pretrained/siamfc/2016-08-17.net.mat'
        self.stats_path = 'pretrained/siamfc/cfnet_ILSVRC2015.stats.mat'
        self.trainer = TrainerSiamFC(self.branch,  self.net_path, self.cfg_file)

    def tearDown(self):
        pass

    def test_train(self):
        self.trainer.train(self.vid_dir, self.stats_path, self.vot_dir)


if __name__ == '__main__':
    unittest.main()
