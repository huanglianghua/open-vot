from __future__ import absolute_import

import argparse

from lib.trackers import TrackerDCFNet
from lib.experiments import ExperimentOTB


otb_dir = 'data/OTB'
experiment = ExperimentOTB(otb_dir)

net_path = 'logs/dcfnet/checkpoints/siamfc_step2.pth'
tracker = TrackerDCFNet(net_path=net_path,online=True)
experiment.run(tracker, visualize=True)

print(experiment.report(tracker.name))
