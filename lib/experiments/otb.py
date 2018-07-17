from __future__ import absolute_import, print_function

import os
import collections
import numpy as np

from ..datasets import OTB
from ..metrics import iou, center_error


class ExperimentOTB(object):

    def __init__(self, otb_dir, result_dir='results', report_dir='reports'):
        self.dataset = OTB(otb_dir, download=True)
        self.result_dir = os.path.join(result_dir, 'OTB')
        self.report_dir = os.path.join(report_dir, 'OTB')

    def run(self, tracker, visualize=False):
        print('running tracker %s on OTB...' % tracker.name)

        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('sequence:', seq_name)

            # tracking loop
            rects, speed_fps = tracker.track(
                img_files, anno[0, :], visualize=visualize)
            assert len(rects) == len(anno)

            # record results
            self._record(tracker.name, seq_name, rects, speed_fps)

    def report(self, tracker_names):
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        if not isinstance(tracker_names, collections.Container):
            tracker_names = [tracker_names]

        performance = {}
        for name in tracker_names:
            ious = []
            center_errors = []
            seq_wise = {}

            for s, (_, anno) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                record_file = os.path.join(
                    self.result_dir, name, seq_name + '.txt')
                rects = np.loadtxt(record_file, delimiter=',')

                ious_ = iou(rects[1:], anno[1:])
                center_errors_ = center_error(rects[1:], anno[1:])
                seq_wise.update({
                    'mean_iou': np.mean(ious_),
                    'precision_score': (center_errors_ <= 20).sum() / len(anno),
                    'success_rate': (ious_ >= 0.5).sum() / len(ious_)})

                ious.extend(ious_)
                center_errors.extend(center_errors_)

            cdf_iou, cdf_center_error = self._generate_curves(
                ious, center_errors)
            succ_score = np.mean(cdf_iou)
            prec_score = cdf_center_error[21]
            succ_rate = (np.array(ious) >= 0.5).sum() / len(ious)

            performance.update({
                name: {
                    'success_curve': cdf_iou,
                    'precision_curve': cdf_center_error,
                    'success_score': succ_score,
                    'precision_score': prec_score,
                    'success_rate': succ_rate,
                    'seq_wise': seq_wise}})

        return performance

    def _record(self, tracker_name, seq_name, results, speed_fps):
        record_dir = os.path.join(self.result_dir, tracker_name)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)

        record_file = os.path.join(record_dir, '%s.txt' % seq_name)
        np.savetxt(record_file, results, fmt='%.2f', delimiter=',')
        print('  results recorded in', record_file)

    def _generate_curves(self, ious, center_errors):
        assert len(ious) == len(center_errors)
        n = len(ious)
        ious = np.array(ious, np.float32)
        center_errors = np.array(center_errors, np.float32)

        nbins = 51
        step_iou = 1 / nbins

        cdf_iou = []
        cdf_center_error = []
        for i in range(nbins):
            cdf_iou.append(
                np.count_nonzero(ious >= step_iou * (i + 1)) / n)
            cdf_center_error.append(
                np.count_nonzero(center_errors <= i) / n)
        cdf_iou[-1] = 0
        cdf_center_error[0] = 0

        return cdf_iou, cdf_center_error
