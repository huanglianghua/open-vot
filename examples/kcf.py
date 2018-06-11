from __future__ import absolute_import

import argparse

from lib.managers import ManagerKCF

otb_dir = 'data/OTB'
manager = ManagerKCF()
manager.track(otb_dir, visualize=True)
