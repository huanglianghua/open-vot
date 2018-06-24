from __future__ import absolute_import

import argparse

from lib.trackers import TrackerSiamFC
from lib.experiments import ExperimentOTB


otb_dir = 'data/OTB'
experiment = ExperimentOTB(otb_dir)

net_path = 'pretrained/siamfc/2016-08-17.net.mat'
tracker = TrackerSiamFC(branch='alexv1', net_path=net_path)
experiment.run(tracker, visualize=True)

experiment.report(tracker.name)
