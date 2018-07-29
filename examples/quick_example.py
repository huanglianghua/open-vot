from __future__ import absolute_import

from lib.trackers import TrackerCSK
from lib.experiments import ExperimentOTB


otb_dir = 'data/OTB'
experiment = ExperimentOTB(otb_dir, version=2013)

tracker = TrackerCSK()
experiment.run(tracker, visualize=True)

experiment.report([tracker.name])
