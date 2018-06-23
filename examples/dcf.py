from __future__ import absolute_import

import argparse

from lib.trackers import TrackerDCF
from lib.experiments import ExperimentOTB


otb_dir = 'data/OTB'
experiment = ExperimentOTB(otb_dir)

tracker = TrackerDCF()
experiment.run(tracker, visualize=True)

experiment.report(tracker.name)
