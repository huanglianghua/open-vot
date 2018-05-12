from __future__ import absolute_import, print_function

import unittest
import torch
import random
import time

from lib.models import CaffeNet


class TestCaffeNet(unittest.TestCase):

    def setUp(self):
        self.x = torch.randn((2, 3, 256, 256))
        self.net = CaffeNet()

    def tearDown(self):
        pass

    def test_caffenet(self):
        net = CaffeNet()

        with torch.set_grad_enabled(True):
            net.train()
            start = time.time()
            out_train = net(self.x)
            print('inference time of training: %.3f' % (time.time() - start))

            self.assertTrue(out_train.requires_grad)
            self.assertTrue(net.training)

        with torch.set_grad_enabled(False):
            net.eval()
            start = time.time()
            out_eval = net(self.x)
            print('inference time of test: %.3f' % (time.time() - start))

            self.assertFalse(out_eval.requires_grad)
            self.assertFalse(net.training)
            self.assertAlmostEqual(
                out_train.mean().item(), out_eval.mean().item())


if __name__ == '__main__':
    unittest.main()
