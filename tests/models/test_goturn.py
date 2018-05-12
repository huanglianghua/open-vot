from __future__ import absolute_import

import unittest
import torch
import time

from lib.models import GOTURN


class TestGOTURN(unittest.TestCase):

    def setUp(self):
        self.z = torch.randn((2, 3, 227, 227))
        self.x = torch.randn((2, 3, 227, 227))
        self.net = GOTURN()

    def tearDown(self):
        pass

    def test_goturn(self):
        with torch.set_grad_enabled(True):
            self.net.train()
            start = time.time()
            out_train = self.net(self.z, self.x)
            print('inference time of training: %.3f' % (time.time() - start))
            self.assertTrue(self.net.training)
            self.assertTrue(out_train.requires_grad)

        with torch.set_grad_enabled(False):
            self.net.eval()
            start = time.time()
            out_eval = self.net(self.z, self.x)
            print('inference time of test: %.3f' % (time.time() - start))
            self.assertFalse(self.net.training)
            self.assertFalse(out_eval.requires_grad)
            self.assertNotAlmostEqual(
                out_train.mean().item(), out_eval.mean().item())


if __name__ == '__main__':
    unittest.main()
