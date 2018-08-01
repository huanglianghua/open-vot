from __future__ import absolute_import

import argparse

from lib.trackers import *
from lib.experiments import *


tracker_factory = {
    'mosse': TrackerMOSSE,
    'csk': TrackerCSK,
    'kcf': TrackerKCF,
    'dcf': TrackerDCF,
    'dsst': TrackerDSST,
    'goturn': TrackerGOTURN,
    'siamfc': TrackerSiamFC,
    'siamfcv2': TrackerSiamFC,
    'dcfnet': TrackerDCFNet}

experiment_factory = {
    'otb': ExperimentOTB}

# parse arguments
parser = argparse.ArgumentParser(description='tracking experiment')
parser.add_argument('-t', '--tracker', type=str, default='csk')
parser.add_argument('-e', '--experiment', type=str, default='otb')
parser.add_argument('-d', '--dataset-folder', type=str, default='data/OTB')
# for deep trackers
parser.add_argument('-n', '--network-path', type=str,
                    default='pretrained/siamfc/2016-08-17.net.mat')
args = parser.parse_args()

# setup tracker
if not args.tracker in ['goturn', 'siamfc', 'siamfcv2', 'dcfnet']:
    # traditional tracker
    tracker = tracker_factory[args.tracker]()
else:
    # deep tracker
    if args.tracker == 'siamfc':
        tracker = tracker_factory[args.tracker](
            branch='alexv1', net_path=args.network_path)
    elif args.tracker == 'siamfcv2':
        tracker = tracker_factory[args.tracker](
            branch='alexv2', net_path=args.network_path)
    elif args.tracker == 'dcfnet':
        tracker = tracker_factory[args.tracker](
            net_path=args.network_path, online=True)
    else:
        tracker = tracker_factory[args.tracker](
            net_path=args.network_path)

# setup experiment
experiment = experiment_factory[args.experiment](args.dataset_folder)

# run experiment and record results in 'results' folder
experiment.run(tracker, visualize=False)

# report performance in 'reports' folder
experiment.report([tracker.name])
