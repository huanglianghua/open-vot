from __future__ import absolute_import

import os
import cv2
import time

from ..datasets import VOT
from ..utils.viz import show_frame


class ExperimentVOT(object):

    def __init__(self, vot_dir,
                 result_dir='results', report_dir='reports',
                 experiments=['baseline', 'unsupervised', 'realtime']):
        self.datsaet = VOT(vot_dir, download=True)
        self.result_dir = os.path.join(result_dir, 'VOT')
        self.report_dir = os.path.join(report_dir, 'VOT')
        self.experiments = self.setup_experiments(experiments)

    def setup_experiments(self, experiments):
        exp = []
        for e in experiments:
            assert e in ['baseline', 'unsupervised', 'realtime']
            if e == 'baseline':
                exp.append(self.run_baseline)
            elif e == 'unsupervised':
                exp.append(self.run_unsupervised)
            elif e == 'realtime':
                exp.append(self.run_realtime)

        return exp

    def run(self, tracker, visualize=False):
        print('running tracker %s on VOT...' % tracker.name)
        for e in self.experiments:
            e(tracker, visualize)

    def run_baseline(self, tracker, visualize=False):
        print('running baseline experiment...')

        for s, (img_files, anno) in enumerate(self.datsaet):
            seq_name = self.datsaet.seq_names[s]
            print('sequence:', seq_name)

            for r in range(15):
                if r == 3 and self._check_deterministic(seq_name):
                    print('detected a deterministic tracker, ' +
                          'skipping remaining trails.')
                    break

                print('  repetition:', r + 1)
                states = []
                times = []
                failed = False
                passed_frames = -1

                # tracking loop
                for f, img_file in enumerate(img_files):
                    image = cv2.imread(img_file)

                    start_time = time.time()
                    if f == 0:
                        tracker.init(image, anno[f])
                        states.append([1])
                    elif not failed:
                        state = tracker.update(image)
                        if iou(state, anno[f]) > 0:
                            states.append(state)
                        else:
                            failed = True
                            passed_frames = 1
                            states.append([2])
                    else:
                        if passed_frames < 5:
                            passed_frames += 1
                            states.append([0])
                            start_time = np.nan
                        else:
                            tracker.init(image, anno[f])
                            failed = False
                            passed_frames = -1
                            states.append([1])
                    times.append(time.time() - start_time)

                    if visualize:
                        init = len(states[-1]) == 1 and states[-1][0] == 1
                        if init:
                            show_frame(image, anno[f])
                        else:
                            show_frame(image, state)

                self._record(tracker.name, seq_name, r, states, times)

    def run_unsupervised(self, tracker, visualize=False):
        print('running unsupervised experiment...')

        for s, (img_files, anno) in enumerate(self.datsaet):
            seq_name = self.datsaet.seq_names[s]
            print('sequences:', seq_name)

            for r in range(1):
                if r == 3 and self._check_deterministic(seq_name):
                    print('detected a deterministic tracker, ' +
                          'skipping remaining trails.')
                    break

                print('  repetition:', r + 1)
                states = []
                times = []

                # tracking loop
                for f, img_file in enumerate(img_files):
                    image = cv2.imread(img_file)

                    start_time = time.time()
                    if f == 0:
                        tracker.init(image, anno[f])
                        states.append([1])
                    else:
                        state = tracker.update(image)
                        states.append(state)
                    times.append(time.time() - start_time)

                    if visualize:
                        show_frame(image, state if f > 0 else anno[f])

                # record tracking results
                self._record(tracker.name, seq_name, r, states, times)

    def run_realtime(self, tracker, visualize=False):
        print('running realtime experiment...')

        for s, (img_files, anno) in enumerate(self.datsaet):
            seq_name = self.datsaet.seq_names[s]
            print('sequences:', seq_name)

            for r in range(1):
                if r == 3 and self._check_deterministic(seq_name):
                    print('detected a deterministic tracker, ' +
                          'skipping remaining trails.')
                    break

                print('  repetition:', r + 1)
                states = []
                times = []
                init_frame = 0

                # tracking loop
                for f, img_file in enumerate(img_files):
                    image = cv2.imread(img_file)

                    start_time = time.time()
                    if f == init_frame:
                        tracker.init(image, anno[f])
                        elapsed_time = time.time() - start_time
                        states.append([1])

                        acc_time = 1. / self.default_fps
                        grace = self.grace - 1
                        failed = False
                    elif not failed:
                        if grace > 0:
                            state = tracker.update(image)

                            elapsed_time = time.time() - start_time
                            acc_time += 1. / self.default_fps
                            grace -= 1
                            if grace == 0:
                                next_frame = init_frame + \
                                    round(np.floor(
                                        (acc_time + max(1. / self.default_fps, elapsed_time)) * self.default_fps))
                        elif f < next_frame:
                            state = state
                            elapsed_time = np.nan
                        else:
                            state = tracker.update(image)
                            elapsed_time = time.time() - start_time
                            acc_time += max(1. / self.default_fps,
                                            elapsed_time)
                            next_frame = init_frame + \
                                round(np.floor(
                                    (acc_time + max(1. / self.default_fps, elapsed_time)) * self.default_fps))

                        if iou(state, anno[f]) > 0:
                            states.append(state)
                        else:
                            failed = True
                            states.append([2])
                            init_frame = next_frame + self.skip_initialize
                    elif failed:
                        states.append([0])
                        elapsed_time = np.nan
                    times.append(elapsed_time)

                    if visualize:
                        if f == init_frame:
                            show_frame(image, anno[f])
                        else:
                            show_frame(image, state)

                self._record(tracker.name, seq_name, r, states, times)

    def report(self, tracker_names):
        pass

    def _record(self, tracker_name, seq_name, repetition, states, times):
        log_dir = os.path.join(self.log_dir, seq_name)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        states_file = os.path.join(
            log_dir, '%s_%03d.txt' % (seq_name, repeat + 1))
        times_file = os.path.join(log_dir, '%s_time.txt' % seq_name)

        # record tracking results
        states_str = []
        for state in states:
            assert len(state) in [1, 4, 6]
            if len(state) == 1:
                states_str.append('%d' % state[0])
            else:
                states_str.append(str.join(',', ['%.4f' % s for s in state]))
        states_str = str.join('\n', states_str)
        with open(states_file, 'w') as f:
            f.write(states_str)

        # record tracking times
        if not os.path.isfile(times_file):
            times_arr = np.zeros((len(times), self.repetitions))
            np.savetxt(times_file, times_arr, fmt='%.6f', delimiter=',')
        else:
            times_arr = np.loadtxt(times_file, delimiter=',')
        if times_arr.ndim == 1:
            times_arr = times_arr[:, np.newaxis]
        times_arr[:, repeat] = times
        np.savetxt(times_file, times_arr, fmt='%.6f', delimiter=',')

    def _check_deterministic(self, seq_name):
        log_dir = os.path.join(self.log_dir, seq_name)
        states_files = sorted(glob.glob(os.path.join(
            log_dir, '%s_[0-9]*.txt' % seq_name)))
        if len(states_files) < 3:
            return False

        states_all = []
        for states_file in states_files:
            with open(states_file, 'r') as f:
                states_all.append(f.read())

        return len(set(states_all)) == 1
