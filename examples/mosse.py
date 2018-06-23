from __future__ import absolute_import

import argparse

from lib.trackers import TrackerMOSSE
from lib.experiments import ExperimentOTB


otb_dir = 'data/OTB'
experiment = ExperimentOTB(otb_dir)

tracker = TrackerMOSSE()
experiment.run(tracker, visualize=True)

performance = experiment.report([tracker.name])
