from __future__ import absolute_import

import argparse

from lib.managers import ManagerMOSSE

otb_dir = 'data/OTB'
manager = ManagerMOSSE()
manager.track(otb_dir, visualize=True)
