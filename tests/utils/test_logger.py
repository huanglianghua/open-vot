from __future__ import absolute_import

import unittest
import numpy as np
import torch.nn as nn

from lib.utils.logger import Logger


class TestLogger(unittest.TestCase):

    def setUp(self):
        self.logger = Logger(log_dir='logs/unittest')
        self.model = nn.Conv2d(3, 32, 3, 1)

    def tearDown(self):
        pass
    
    def test_logger(self):
        for it in range(100):
            self.logger.add_scalar('data/unittest', 3, it)
            self.logger.add_text('unittest', 'iter %d' % (it + 1), it)
            self.logger.add_array('unittest', np.random.rand(5, 5), it)
            self.logger.add_checkpoint('unittest', self.model.state_dict(), it)


if __name__ == '__main__':
    unittest.main()
