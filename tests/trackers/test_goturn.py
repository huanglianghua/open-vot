from __future__ import absolute_import, print_function

import unittest
import random
from torch.utils.data import DataLoader

from lib.trackers import TrackerGOTURN
from lib.datasets import VOT, Pairwise
from lib.transforms import TransformGOTURN


class TestTrackerGOTURN(unittest.TestCase):

    def setUp(self):
        self.vot_dir = 'data/vot2017'
        self.net_path = 'pretrained/goturn/tracker.pt'

    def tearDown(self):
        pass

    def test_goturn_track(self):
        dataset = VOT(self.vot_dir, return_bndbox=True, download=True)
        tracker = TrackerGOTURN(self.net_path)

        img_files, anno = random.choice(dataset)
        rects, speed = tracker.track(img_files, anno[0, :],
                                     visualize=True)
        self.assertEqual(rects.shape, anno.shape)

    def test_goturn_train(self):
        tracker = TrackerGOTURN(net_path=self.net_path)
        transform = TransformGOTURN()

        base_dataset = VOT(self.vot_dir, return_bndbox=True, download=True)
        dataset = Pairwise(
            base_dataset, transform, frame_range=1, causal=True)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # training loop
        for it, batch in enumerate(dataloader):
            update_lr = it == 0
            loss = tracker.step(batch, backward=True, update_lr=update_lr)
            print('Iter: {} Loss: {:.6f}'.format(it + 1, loss))

        # validation loop
        for it, batch in enumerate(dataloader):
            loss = tracker.step(batch, backward=False)
            print('Val. Iter: {} Loss: {:.6f}'.format(it + 1, loss))


if __name__ == '__main__':
    unittest.main()
