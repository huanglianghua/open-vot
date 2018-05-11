from __future__ import absolute_import, print_function

import unittest
import torch
import random
import time

from lib.models import AlexNetV1, AlexNetV2


class TestAlexNet(unittest.TestCase):

    def setUp(self):
        self.x = torch.randn((2, 3, 224, 224))

    def tearDown(self):
        pass
    
    def test_alexnet_v1(self):
        net = AlexNetV1()

        with torch.set_grad_enabled(True):
            net.train()
            start = time.time()
            out_train = net(self.x)
            print('inference time of training: %.3f' % (time.time() - start))

            self.assertEqual(out_train.requires_grad, True)
            self.assertEqual(net.training, True)

        with torch.set_grad_enabled(False):
            net.eval()
            start = time.time()
            out_eval = net(self.x)
            print('inference time of test: %.3f' % (time.time() - start))

            self.assertEqual(out_eval.requires_grad, False)
            self.assertEqual(net.training, False)
            self.assertNotEqual(out_train.mean(), out_eval.mean())

    def test_alexnet_v2(self):
        net = AlexNetV2()

        with torch.set_grad_enabled(True):
            net.train()
            start = time.time()
            out_train = net(self.x)
            print('inference time of training: %.3f' % (time.time() - start))

            self.assertEqual(out_train.requires_grad, True)
            self.assertEqual(net.training, True)

        with torch.set_grad_enabled(False):
            net.eval()
            start = time.time()
            out_eval = net(self.x)
            print('inference time of test: %.3f' % (time.time() - start))

            self.assertEqual(out_eval.requires_grad, False)
            self.assertEqual(net.training, False)
            self.assertNotEqual(out_train.mean(), out_eval.mean())


if __name__ == '__main__':
    unittest.main()
