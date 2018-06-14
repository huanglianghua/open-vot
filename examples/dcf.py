from __future__ import absolute_import

import argparse

from lib.managers import ManagerDCF

otb_dir = 'data/OTB'
manager = ManagerDCF()
manager.track(otb_dir, visualize=True)
