from __future__ import absolute_import

import unittest
import time

from lib.utils.logger import Logger


class TestLogger(unittest.TestCase):

    def setUp(self):
        self.log_file = 'logs/unit_test.log'
    
    def tearDown(self):
        pass

    def test_logger(self):
        logger = Logger(self.log_file)
        for i in range(100):
            logger.log('logging index %d' % i)
            time.sleep(0.01)

        logger.flush()
        logger.close()


if __name__ == '__main__':
    unittest.main()
