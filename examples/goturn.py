from __future__ import absolute_import

import argparse

from lib.trackers import TrackerGOTURN
from lib.experiments import ExperimentOTB


otb_dir = 'data/OTB'
experiment = ExperimentOTB(otb_dir)

net_path = 'pretrained/goturn/tracker.pth'
tracker = TrackerGOTURN(net_path=net_path)
experiment.run(tracker, visualize=True)

experiment.report(tracker.name)
