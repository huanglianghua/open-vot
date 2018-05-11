from __future__ import absolute_import, print_function

import unittest
import torch
import random
import time

from lib.models import SiameseNet, AlexNet


class TestSiameseNet(unittest.TestCase):

    def setUp(self):
        self.z = torch.randn((2, 3, 127, 127))
        self.x = torch.randn((2, 3, 255, 255))

    def tearDown(self):
        pass

    def test_siamese_net(self):
        for norm in [None, 'bn', 'cosine', 'euclidean', 'linear']:
            net = SiameseNet(AlexNet(), norm=norm)

            with torch.set_grad_enabled(True):
                net.train()
                start = time.time()
                out_train = net(self.z, self.x)
                print('inference time of training: %.3f' %
                      (time.time() - start))
                self.assertEqual(out_train.requires_grad, True)
                self.assertEqual(net.training, True)

            with torch.set_grad_enabled(False):
                net.eval()
                start = time.time()
                out_eval = net(self.z, self.x)
                print('inference time of test: %.3f' % (time.time() - start))
                self.assertEqual(out_eval.requires_grad, False)
                self.assertEqual(net.training, False)
                self.assertNotEqual(out_train.mean(), out_eval.mean())

            if norm == 'cosine':
                self.assertGreaterEqual(out_train.min().item(), -1)
                self.assertLessEqual(out_train.max().item(), 1)
                self.assertGreaterEqual(out_eval.min().item(), -1)
                self.assertLessEqual(out_eval.max().item(), 1)
            elif norm == 'euclidean':
                self.assertGreaterEqual(out_train.min().item(), 0)
                self.assertGreaterEqual(out_eval.min().item(), 0)


if __name__ == '__main__':
    unittest.main()
