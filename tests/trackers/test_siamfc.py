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
        self.net_path = 'pretrained/siamfc/baseline-conv5_e55.mat'

    def tearDown(self):
        pass

    def test_siamfc_track(self):
        dataset = VOT(self.vot_dir, return_bndbox=True)
        tracker = TrackerSiamFC(net_path=self.net_path)

        img_files, anno = random.choice(dataset)
        rects, speed = tracker.track(img_files, anno[0, :],
                                     visualize=True)
        self.assertEqual(rects.shape, anno.shape)

    def test_siamfc_train(self):
        tracker = TrackerSiamFC()
        transform = TransformSiamFC()

        base_dataset = VOT(self.vot_dir, return_bndbox=True)
        dataset = Pairwise(base_dataset, transform)
        dataloader = DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4)

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
