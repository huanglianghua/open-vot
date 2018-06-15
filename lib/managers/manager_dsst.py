from __future__ import absolute_import, division

import json
import torch
import multiprocessing
from torch.utils.data import DataLoader

from ..trackers import TrackerDSST
from ..utils.logger import Logger
from ..datasets import OTB
from ..metrics import iou, center_error


class ManagerDSST(object):

    def __init__(self, cfg_file=None):
        cfg = {}
        if cfg_file is not None:
            with open(cfg_file, 'r') as f:
                cfg = json.load(f)

        self.tracker = TrackerDSST(**cfg)
        self.logger = Logger(log_dir='logs/dsst')

    def track(self, otb_dir, visualize=False):
        tracker = self.tracker
        dataset = OTB(otb_dir, download=True)

        tag = 'test'
        avg_iou = 0
        avg_prec = 0
        avg_speed = 0
        total_frames = 0

        for s, (img_files, anno) in enumerate(dataset):
            seq_name = dataset.seq_names[s]

            # tracking loop
            rects, speed_fps = tracker.track(
                img_files, anno[0, :], visualize=visualize)

            mean_iou = iou(rects, anno).mean()
            prec = (center_error(rects, anno) <= 20).sum() / len(anno)

            frame_num = len(img_files)
            avg_iou += mean_iou * frame_num
            avg_prec += prec * frame_num
            avg_speed += speed_fps * frame_num
            total_frames += frame_num

            # logging
            self.logger.add_text(tag, '{}/{} Performance on {}: IoU: {:.3f} Prec: {:.3f} Speed: {:.3f} fps'.format(
                s + 1, len(dataset), seq_name, mean_iou, prec, speed_fps))
            self.logger.add_scalar(tag + '/iou_' + seq_name, mean_iou)
            self.logger.add_scalar(tag + '/prec_' + seq_name, prec)
            self.logger.add_scalar(tag + '/speed_' + seq_name, speed_fps)

        avg_iou /= total_frames
        avg_prec /= total_frames
        avg_speed /= total_frames

        # logging
        self.logger.add_text(tag, 'Overall Performance: IoU: {:.3f} Prec: {:.3f} Speed: {:.3f} fps'.format(
            avg_iou, avg_prec, avg_speed))
        self.logger.add_scalar(tag + '/iou_overall', avg_iou)
        self.logger.add_scalar(tag + '/prec_overall', avg_prec)
        self.logger.add_scalar(tag + '/speed_overall', avg_speed)

        return avg_iou, avg_prec, avg_speed
