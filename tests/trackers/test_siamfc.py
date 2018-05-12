from __future__ import absolute_import, print_function

import unittest
import random
from torch.utils.data import DataLoader

from lib.trackers import TrackerSiamFC
from lib.datasets import VOT, Pairwise
from lib.transforms import TransformSiamFC


class TestTrackerSiamFC(unittest.TestCase):

    def setUp(self):
        self.vot_dir = 'data/vot2017'
        self.net_v1 = 'pretrained/siamfc/2016-08-17.net.mat'
        self.net_v2 = 'pretrained/siamfc/baseline-conv5_e55.mat'
        self.stats_path = 'pretrained/siamfc/cfnet_ILSVRC2015.stats.mat'

    def tearDown(self):
        pass

    def test_siamfc_track_v1(self):
        dataset = VOT(self.vot_dir, return_bndbox=True)
        tracker = TrackerSiamFC(
            branch='alexv1', net_path=self.net_v1, z_lr=0,
            response_up=16, scale_step=1.0375, window_influence=0.176)

        img_files, anno = random.choice(dataset)
        rects, speed = tracker.track(img_files, anno[0, :],
                                     visualize=True)
        self.assertEqual(rects.shape, anno.shape)

    def test_siamfc_train_v1(self):
        tracker = TrackerSiamFC(branch='alexv1')
        transform = TransformSiamFC(
            score_sz=17, stats_path=self.stats_path)

        base_dataset = VOT(self.vot_dir, return_bndbox=True)
        dataset = Pairwise(base_dataset, transform)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # training step
        for it, batch in enumerate(dataloader):
            update_lr = it == 0
            loss = tracker.step(batch, update_lr=update_lr)
            print('Iter: {} Loss: {:.6f}'.format(it, loss))

        # val step
        for it, batch in enumerate(dataloader):
            loss = tracker.step(batch, backward=False)
            print('Val. Iter: {} Loss: {:.6f}'.format(it, loss))

    def test_siamfc_track_v2(self):
        dataset = VOT(self.vot_dir, return_bndbox=True)
        tracker = TrackerSiamFC(branch='alexv2', net_path=self.net_v2)

        img_files, anno = random.choice(dataset)
        rects, speed = tracker.track(img_files, anno[0, :],
                                     visualize=True)
        self.assertEqual(rects.shape, anno.shape)

    def test_siamfc_train_v2(self):
        tracker = TrackerSiamFC(branch='alexv2')
        transform = TransformSiamFC(
            score_sz=33, stats_path=self.stats_path)

        base_dataset = VOT(self.vot_dir, return_bndbox=True)
        dataset = Pairwise(base_dataset, transform)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # training step
        for it, batch in enumerate(dataloader):
            update_lr = it == 0
            loss = tracker.step(batch, update_lr=update_lr)
            print('Iter: {} Loss: {:.6f}'.format(it, loss))

        # val step
        for it, batch in enumerate(dataloader):
            loss = tracker.step(batch, backward=False)
            print('Val. Iter: {} Loss: {:.6f}'.format(it, loss))


if __name__ == '__main__':
    unittest.main()
