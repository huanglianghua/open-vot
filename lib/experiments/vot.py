from __future__ import absolute_import, division

import os
import cv2
import time
import numpy as np
import glob
import json
import ast

from ..datasets.vot import VOT
from ..utils import dict2tuple
from ..utils.viz import show_frame
from ..metrics import iou


class ExperimentVOT(object):

    def __init__(self, vot_dir, version=2017,
                 result_dir='results', report_dir='reports', **kargs):
        super(ExperimentVOT, self).__init__()
        self.dataset = VOT(vot_dir, version,
                           anno_type='rect', download=True)
        self.result_dir = os.path.join(result_dir, 'vot%d' % version)
        self.report_dir = os.path.join(report_dir, 'vot%d' % version)
        self.parse_args(**kargs)
        # setup experiment functions
        self.setup_experiments()

    def parse_args(self, **kargs):
        self.cfg = {
            'experiments': ['baseline', 'unsupervised', 'realtime'],
            'repetitions': {
                'baseline': 15,
                'unsupervised': 1,
                'realtime': 1},
            'min_repetitions': 3,
            'default_fps': 20,
            'grace': 3,
            'skip_initialize': 5}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = dict2tuple(self.cfg)

    def setup_experiments(self):
        self.experiments = []
        for e in self.cfg.experiments:
            assert e in ('baseline', 'unsupervised', 'realtime')
            if e == 'baseline':
                self.experiments.append(self.run_baseline)
            elif e == 'unsupervised':
                self.experiments.append(self.run_unsupervised)
            elif e == 'realtime':
                self.experiments.append(self.run_realtime)

    def run(self, tracker, visualize=False):
        print('Running tracker %s on VOT%d...' %
              (tracker.name, self.dataset.version))
        for e in self.experiments:
            e(tracker, visualize)

    def run_baseline(self, tracker, visualize=False):
        print('Running baseline experiment...')

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' %
                  (s + 1, len(self.dataset), seq_name))

            # run multiple repetitions for each sequence
            for r in range(self.cfg.repetitions['baseline']):
                # for deterministic tracker, skip repetitions
                if r == self.cfg.min_repetitions and self._check_deterministic('baseline', tracker.name, seq_name):
                    print('  Detected a deterministic tracker, ' +
                          'skipping remaining trails.')
                    break
                print('  Repetition: %d' % (r + 1))

                # tracking loop of reset-based experiment
                states = []
                elapsed_times = []
                failure = False
                skipped_frames = -1
                for f, img_file in enumerate(img_files):
                    img = cv2.imread(img_file)
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    start = time.time()
                    if f == 0:
                        # initial frame
                        tracker.init(img, anno[f])
                        states.append([1])
                    elif not failure:
                        # during success frames
                        state = tracker.update(img)
                        if iou(state, anno[f]) == 0:  # tracking failure
                            failure = True
                            skipped_frames = 1
                            states.append([2])
                        else:
                            states.append(state)
                    else:
                        # during failure frames
                        if skipped_frames == 5:
                            tracker.init(img, anno[f])
                            states.append([1])
                            failure = False
                            skipped_frames = -1
                        else:
                            skipped_frames += 1
                            states.append([0])
                            start = np.NaN
                    elapsed_times.append(time.time() - start)

                    if visualize:
                        if len(states[-1]) == 1:
                            show_frame(img, anno[f], color=(255, 255, 255)
                                       if img.shape[2] == 3 else 255)
                        else:
                            show_frame(img, state, color=(255, 0, 0)
                                       if img.shape[2] == 3 else 255)

                # record results
                self._record('baseline', tracker.name, seq_name, r,
                             states, elapsed_times)

    def run_unsupervised(self, tracker, visualize=False):
        print('Running unsupervised experiment...')

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' %
                  (s + 1, len(self.dataset), seq_name))

            # run multiple repetitions for each sequence
            for r in range(self.cfg.repetitions['unsupervised']):
                # for deterministic tracker, skip repetitions
                if r == self.cfg.min_repetitions and self._check_deterministic('unsupervised', seq_name):
                    print('  Detected a deterministic tracker, ' +
                          'skipping remaining trails.')
                    break
                print('  Repetition: %d' % (r + 1))

                # tracking loop
                states = []
                elapsed_times = []
                for f, img_file in enumerate(img_files):
                    img = cv2.imread(img_file)
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    start = time.time()
                    if f == 0:
                        tracker.init(img, anno[f])
                        states.append([1])
                    else:
                        states.append(tracker.update(img))
                    elapsed_times.append(time.time() - start)

                    if visualize:
                        show_frame(img, states[f] if f > 0 else anno[f])

                # record results
                self._record('unsupervised', tracker.name, seq_name, r,
                             states, elapsed_times)

    def run_realtime(self, tracker, visualize=False):
        print('Running realtime experiment...')

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' %
                  (s + 1, len(self.dataset), seq_name))

            # run multiple repetitions for each sequence
            for r in range(self.cfg.repetitions['realtime']):
                # for deterministic tracker, skip repetitions
                if r == self.cfg.min_repetitions and self._check_deterministic('realtime', tracker.name, seq_name):
                    print('  Detected a deterministic tracker, ' +
                          'skipping remaining trails.')
                    break
                print('  Repetition: %d' % (r + 1))

                # tracking loop of reset-based realtime experiment
                states = []
                elapsed_times = []
                init_frame = 0
                for f, img_file in enumerate(img_files):
                    img = cv2.imread(img_file)
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    start = time.time()
                    if f == init_frame:
                        # initial frame
                        tracker.init(img, anno[f])
                        end = time.time()
                        states.append([1])

                        # initialize parameters
                        accum_time = 1. / self.cfg.default_fps
                        grace = self.cfg.grace - 1
                        failure = False
                        skipped_frames = -1
                    elif not failure:
                        if grace > 0:
                            # during grace frames
                            state = tracker.update(img)
                            end = time.time()

                            accum_time += 1. / self.cfg.default_fps
                            grace -= 1
                            if grace == 0:
                                # calculate the next frame according to the realtime setting
                                next_frame = init_frame + round(np.floor(
                                    (accum_time + max(1. / self.cfg.default_fps, end - start)) * self.cfg.default_fps))
                        elif f < next_frame:
                            # during skipped frames
                            state = state  # assign with the last tracking result
                            end = np.NaN
                        else:
                            # during normal frames
                            state = tracker.update(img)
                            end = time.time()

                            accum_time += max(1. / self.cfg.default_fps, end - start)
                            # calculate the next frame according to the realtime setting
                            next_frame = init_frame + round(np.floor(
                                (accum_time + max(1. / self.cfg.default_fps, end - start)) * self.cfg.default_fps))
                        
                        if iou(state, anno[f]) > 0:
                            states.append(state)
                        else:
                            states.append([2])
                            end = np.NaN

                            failure = True
                            skipped_frames = 1
                    else:
                        # during failure frames
                        if skipped_frames == self.cfg.skip_initialize:
                            # initial frame
                            tracker.init(img, anno[f])
                            end = time.time()
                            states.append([1])

                            # initialize parameters
                            accum_time = 1. / self.cfg.default_fps
                            grace = self.cfg.grace - 1
                            failure = False
                            skipped_frames = -1
                        else:
                            skipped_frames += 1
                            states.append([0])
                            end = np.NaN
                    elapsed_times.append(end - start)

                    if visualize:
                        if len(states[-1]) == 1:
                            show_frame(img, anno[f], color=(255, 255, 255)
                                       if img.shape[2] == 3 else 255)
                        else:
                            show_frame(img, state, color=(255, 0, 0)
                                       if img.shape[2] == 3 else 255)

                # record results
                self._record('baseline', tracker.name, seq_name, r,
                             states, elapsed_times)

    def report(self, tracker_names):
        assert isinstance(tracker_names, (list, tuple))

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        
        performance = {}
        for name in tracker_names:
            seq_num = len(self.dataset)
            ious = []
            failures = []

            for s, (_, anno) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                seq_ious = np.full((15, len(anno)), np.NaN, dtype=float)
                seq_failures = np.full(15, np.NaN, dtype=float)

                record_files = sorted(glob.glob(os.path.join(
                    self.result_dir, name, 'baseline', seq_name,
                    '%s_[0-9]*.txt' % seq_name)))

                for r, record_file in enumerate(record_files):
                    results = self._read_record(record_file)
                    assert(len(results) == len(anno))
                    seq_ious[r, :] = self._iou(results, anno)
                    seq_failures[r] = self._failure(results, anno)
                
                seq_ious = np.nanmean(seq_ious, axis=0)
                seq_failures[np.isnan(seq_failures)] = np.nanmean(seq_failures)

                if len(seq_ious) > 0:
                    ious.append(seq_ious)
                if len(seq_failures) > 0:
                    failures.append(seq_failures)

            failures = np.stack(failures, axis=0)
            failures = failures[~np.isnan(failures[:, 0]), :]
            avg_iou = np.nanmean(np.concatenate(ious))
            avg_failure = np.nanmean(np.sum(failures, axis=0))

            performance.update({name: {
                'accuracy': avg_iou,
                'robustness': avg_failure}})

        # report the performance
        report_file = os.path.join(report_dir, 'performance.json')
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)

        return performance

    def _check_deterministic(self, experiment, tracker_name, seq_name):
        record_dir = os.path.join(
            self.result_dir, tracker_name, experiment, seq_name)
        record_files = sorted(glob.glob(os.path.join(
            record_dir, '%s_[0-9]*.txt' % seq_name)))
        if len(record_files) < self.cfg.min_repetitions:
            return False

        states = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                states.append(f.read())

        return len(set(states)) == 1

    def _record(self, experiment, tracker_name, seq_name, repetition,
                states, elapsed_times):
        record_dir = os.path.join(
            self.result_dir, tracker_name, experiment, seq_name)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)

        # record states
        record_file = os.path.join(
            record_dir, '%s_%03d.txt' % (seq_name, repetition + 1))
        content = []
        for state in states:
            if len(state) == 1:
                content.append('%d' % state[0])
            else:
                content.append(str.join(',', ['%.3f' % s for s in state]))
        content = str.join('\n', content)
        with open(record_file, 'w') as f:
            f.write(content)

        # record elapsed times
        time_file = os.path.join(record_dir, '%s_time.txt' % seq_name)
        if not os.path.isfile(time_file):
            content = np.zeros((
                len(elapsed_times), self.cfg.repetitions[experiment]))
            if content.ndim == 1:
                content = content[:, np.newaxis]
        else:
            content = np.loadtxt(time_file, delimiter=',')
        content[:, repetition] = elapsed_times
        np.savetxt(time_file, content, fmt='%.6f', delimiter=',')

    def _read_record(self, record_file):
        with open(record_file, 'r') as f:
            content = f.read().strip()
        states = [[ast.literal_eval(t) for t in line.split(',')]
                  for line in content.split('\n')]
        
        return states

    def _iou(self, results, anno):
        assert len(results) == len(anno)
        burnin = 10

        if burnin > 0:
            mask = np.asarray([(len(t) == 1 and t[0] == 1) for t in results], dtype=np.uint8)
            se = np.concatenate((np.zeros(burnin - 1), np.ones(burnin))).astype(np.uint8)
            mask = cv2.dilate(mask[::-1], se).squeeze().astype(bool)[::-1]
        else:
            mask = np.zeros(len(results), dtype=bool)
        
        ious = np.full(len(anno), np.NaN, dtype=float)
        for f, state in enumerate(results):
            if not mask[f] and len(state) > 1:
                ious[f] = iou(np.asarray(state), anno[f])
        
        return ious

    def _failure(self, results, anno):
        failures = [t for t in results if len(t) == 1 and t[0] == 2]

        return len(failures)
