from __future__ import absolute_import

import argparse

from lib.trackers import TrackerCSK
from lib.experiments import ExperimentOTB


otb_dir = 'data/OTB'
experiment = ExperimentOTB(otb_dir)

tracker = TrackerCSK()
experiment.run(tracker, visualize=True)

experiment.report(tracker.name)
