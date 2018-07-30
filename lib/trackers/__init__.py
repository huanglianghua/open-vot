from __future__ import absolute_import

import cv2
import numpy as np
import time

from ..utils.viz import show_frame


class Tracker(object):

    def __init__(self, name):
        self.name = name

    def init(self, image, init_rect):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()

    def track(self, img_files, init_rect, visualize=False):
        frame_num = len(img_files)
        bndboxes = np.zeros((frame_num, 4))
        bndboxes[0, :] = init_rect

        elapsed_time = 0
        for f, img_file in enumerate(img_files):
            image = cv2.imread(img_file)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            start_time = time.time()
            if f == 0:
                self.init(image, init_rect)
            else:
                bndboxes[f, :] = self.update(image)
            elapsed_time += time.time() - start_time

            if visualize and f%5==0:
                show_frame(image, bndboxes[f, :], fig_n=1)
        speed_fps = frame_num / elapsed_time

        return bndboxes, speed_fps


from .siamfc import TrackerSiamFC
from .goturn import TrackerGOTURN
from .csk import TrackerCSK
from .kcf import TrackerKCF
from .dcf import TrackerDCF
from .dcfnet import TrackerDCFNet
from .mosse import TrackerMOSSE
from .dsst import TrackerDSST
