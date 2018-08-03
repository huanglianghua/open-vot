from __future__ import absolute_import, division

import os
import numpy as np
import matplotlib.pyplot as plt
import json

from ..datasets.otb import OTB
from ..metrics import iou, center_error


class ExperimentOTB(object):

    def __init__(self, otb_dir, version=2015,
                 result_dir='results', report_dir='reports'):
        super(ExperimentOTB, self).__init__()
        self.dataset = OTB(otb_dir, version, download=True)
        self.result_dir = os.path.join(result_dir, 'OTB-' + str(version))
        self.report_dir = os.path.join(report_dir, 'OTB-' + str(version))
        # as nbins_iou increases, the success score
        # converges to average overlap (AO)
        self.nbins_iou = 101
        self.nbins_ce = 51

    def run(self, tracker, visualize=False):
        print('Running tracker %s on OTB...' % tracker.name)

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))
            # tracking loop
            rects, speed_fps = tracker.track(
                img_files, anno[0, :], visualize=visualize)
            assert len(rects) == len(anno)
            # record results
            self._record(tracker.name, seq_name, rects, speed_fps)

    def report(self, tracker_names):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)

        performance = {}
        for name in tracker_names:
            seq_num = len(self.dataset)
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            prec_curve = np.zeros((seq_num, self.nbins_ce))
            speed_fps = np.zeros(seq_num)

            for s, (_, anno) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                record_file = os.path.join(
                    self.result_dir, name, '%s.txt' % seq_name)
                rects = np.loadtxt(record_file, delimiter=',')
                rects[0] = anno[0]

                if len(rects) > len(anno):
                    rects = rects[:len(anno)]
                assert len(rects) == len(anno)

                ious = iou(rects, anno)
                center_errors = center_error(rects, anno)
                succ_curve[s], prec_curve[s] = self._calc_curves(ious, center_errors)

                # calculate average tracking speed
                speed_file = os.path.join(
                    self.result_dir, name, 'FPS/%s.txt' % seq_name)
                if os.path.isfile(speed_file):
                    speeds = np.loadtxt(speed_file)
                    speed_fps[s] = np.mean(speeds)

            succ_curve = np.mean(succ_curve, axis=0)
            prec_curve = np.mean(prec_curve, axis=0)
            succ_score = np.mean(succ_curve)
            prec_score = prec_curve[20]
            succ_rate = succ_curve[self.nbins_iou // 2]
            if np.count_nonzero(speed_fps) > 0:
                speed_fps = np.sum(speed_fps) / np.count_nonzero(speed_fps)
            else:
                speed_fps = None

            performance.update({name: {
                'success_curve': succ_curve.tolist(),
                'precision_curve': prec_curve.tolist(),
                'success_score': succ_score,
                'precision_score': prec_score,
                'success_rate': succ_rate,
                'speed_fps': speed_fps}})

        # report the performance
        report_file = os.path.join(report_dir, 'performance.json')
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        self._visualize(performance, report_dir)

        return performance

    def _record(self, tracker_name, seq_name, rects, speed_fps):
        # record bounding boxes
        record_dir = os.path.join(self.result_dir, tracker_name)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        record_file = os.path.join(record_dir, '%s.txt' % seq_name)
        np.savetxt(record_file, rects, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        # record speeds
        speed_dir = os.path.join(record_dir, 'FPS')
        if not os.path.isdir(speed_dir):
            os.makedirs(speed_dir)
        speed_file = os.path.join(speed_dir, '%s.txt' % seq_name)
        np.savetxt(speed_file, speed_fps, fmt='%.3f')

    def _calc_curves(self, ious, center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]

        bin_iou = np.greater(ious, thr_iou)
        bin_ce = np.less_equal(center_errors, thr_ce)

        succ_curve = np.mean(bin_iou, axis=0)
        prec_curve = np.mean(bin_ce, axis=0)

        return succ_curve, prec_curve

    def _visualize(self, performance, report_dir):
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        succ_file = os.path.join(report_dir, 'success_plots.png')
        prec_file = os.path.join(report_dir, 'precision_plots.png')

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for name, perf in performance.items():
            line, = ax.plot(thr_iou, perf['success_curve'])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, perf['success_score']))
        ax.legend(lines, legends, loc=1)
        ax.set(xlabel='Overlap threshold', ylabel='Success rate',
               xlim=(0, 1), ylim=(0, None), title='Success plots of OPE')

        fig.savefig(succ_file, dpi=300)
        plt.draw()
        plt.pause(.001)

        # plot precision curves
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for name, perf in performance.items():
            line, = ax.plot(thr_ce, perf['precision_curve'])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, perf['precision_score']))
        ax.legend(lines, legends, loc=2)
        ax.set(xlabel='Location error threshold', ylabel='Precision',
               xlim=(0, 50), ylim=(0, None), title='Precision plots of OPE')

        fig.savefig(prec_file, dpi=300)
        plt.draw()
        plt.pause(1)
