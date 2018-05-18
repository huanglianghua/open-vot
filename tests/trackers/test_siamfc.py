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
        dataset = VOT(self.vot_dir, return_bndbox=True, download=True)
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
            stats_path=self.stats_path, score_sz=17,
            r_pos=16, total_stride=8)

        base_dataset = VOT(self.vot_dir, return_bndbox=True, download=True)
        dataset = Pairwise(base_dataset, transform)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # training loop
        for it, batch in enumerate(dataloader):
            update_lr = it == 0
            loss = tracker.step(batch, backward=True, update_lr=update_lr)
            print('Iter: {} Loss: {:.6f}'.format(it + 1, loss))

        # val loop
        for it, batch in enumerate(dataloader):
            loss = tracker.step(batch, backward=False)
            print('Val. Iter: {} Loss: {:.6f}'.format(it + 1, loss))

    def test_siamfc_track_v2(self):
        dataset = VOT(self.vot_dir, return_bndbox=True, download=True)
        tracker = TrackerSiamFC(
            branch='alexv2', net_path=self.net_v2, z_lr=0.01,
            response_up=8, scale_step=1.0816, window_influence=0.25)

        img_files, anno = random.choice(dataset)
        rects, speed = tracker.track(img_files, anno[0, :],
                                     visualize=True)
        self.assertEqual(rects.shape, anno.shape)

    def test_siamfc_train_v2(self):
        tracker = TrackerSiamFC(branch='alexv2')
        transform = TransformSiamFC(
            stats_path=self.stats_path, score_sz=33,
            r_pos=8, total_stride=4)

        base_dataset = VOT(self.vot_dir, return_bndbox=True, download=True)
        dataset = Pairwise(base_dataset, transform)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # training loop
        for it, batch in enumerate(dataloader):
            update_lr = it == 0
            loss = tracker.step(batch, backward=True, update_lr=update_lr)
            print('Iter: {} Loss: {:.6f}'.format(it + 1, loss))

        # val loop
        for it, batch in enumerate(dataloader):
            loss = tracker.step(batch, backward=False)
            print('Val. Iter: {} Loss: {:.6f}'.format(it + 1, loss))


if __name__ == '__main__':
    unittest.main()
