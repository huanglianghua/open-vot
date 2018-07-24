from __future__ import absolute_import

import argparse

from lib.trackers import TrackerDCFNet
from lib.experiments import ExperimentOTB


otb_dir = 'data/OTB'
experiment = ExperimentOTB(otb_dir)

tracker = TrackerDCFNet(net_path=None)
experiment.run(tracker, visualize=True)

print(experiment.report(tracker.name))
