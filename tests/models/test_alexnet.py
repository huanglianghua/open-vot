from __future__ import absolute_import, print_function

import unittest
import torch
import random
import time

from lib.models import AlexNet


class TestAlexNet(unittest.TestCase):

    def setUp(self):
        self.x = torch.randn((2, 3, 224, 224))
        self.net = AlexNet()

    def tearDown(self):
        pass

    def test_alexnet(self):
        with torch.set_grad_enabled(True):
            self.net.train()
            start = time.time()
            out_train = self.net(self.x)
            print('inference time of training: %.3f' % (time.time() - start))

            self.assertEqual(out_train.requires_grad, True)
            self.assertEqual(self.net.training, True)

        with torch.set_grad_enabled(False):
            self.net.eval()
            start = time.time()
            out_eval = self.net(self.x)
            print('inference time of test: %.3f' % (time.time() - start))

            self.assertEqual(out_eval.requires_grad, False)
            self.assertEqual(self.net.training, False)
            self.assertNotEqual(out_train.mean(), out_eval.mean())


if __name__ == '__main__':
    unittest.main()
