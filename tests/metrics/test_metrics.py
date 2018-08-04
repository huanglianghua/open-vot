from __future__ import absolute_import

import unittest
import numpy as np

from lib.metrics import rect_iou, center_error


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.rects1 = 100.0 * np.random.rand(100000, 4)
        self.rects2 = 100.0 * np.random.rand(100000, 4)

    def tearDown(self):
        pass

    def test_iou(self):
        ious = rect_iou(self.rects1, self.rects2)
        self.assertGreater(ious.min(), 0 - 1e-12)
        self.assertLess(ious.max(), 1 + 1e-12)

        ious = rect_iou(self.rects1, self.rects1)
        self.assertAlmostEqual(ious.max(), 1)
        self.assertAlmostEqual(ious.min(), 1)

    def test_center_error(self):
        ces = center_error(self.rects1, self.rects2)
        self.assertGreaterEqual(ces.min(), 0)

        ces = center_error(self.rects1, self.rects1)
        self.assertAlmostEqual(ces.max(), 0)
        self.assertAlmostEqual(ces.min(), 0)


if __name__ == '__main__':
    unittest.main()
