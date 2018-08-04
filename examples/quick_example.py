from __future__ import absolute_import

from lib.trackers import *
from lib.experiments import ExperimentVOT


tracker_factory = {
    'MOSSE': TrackerMOSSE(),
    'CSK': TrackerCSK(),
    'KCF': TrackerKCF(),
    'DCF': TrackerDCF()}

vot_dir = 'data/vot2017'
experiment = ExperimentVOT(vot_dir, version=2017)

# for tracker in tracker_factory.values():
#     experiment.run(tracker, visualize=False)

experiment.report(list(tracker_factory.keys()))
