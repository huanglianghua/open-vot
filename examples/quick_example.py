from __future__ import absolute_import

import json

from lib.trackers import TrackerSiamFC
from lib.experiments import ExperimentOTB


otb_dir = 'data/OTB'
experiment = ExperimentOTB(otb_dir, version=2013)

# branch = 'alexv2'
# config_file = 'config/siamfc.json'
# with open(config_file, 'r') as f:
#     config = json.load(f)[branch]

# net_path = 'pretrained/siamfc/baseline-conv5_e55.mat'
# tracker = TrackerSiamFC(branch=branch, net_path=net_path, **config)
tracker = TrackerSiamFC(net_path='Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained/model.ckpt-0')

experiment.run(tracker, visualize=True)
# experiment.report(['SiamFC'])
