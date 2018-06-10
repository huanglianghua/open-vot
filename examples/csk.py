from __future__ import absolute_import

import argparse

from lib.managers import ManagerCSK

otb_dir = 'data/OTB'
manager = ManagerCSK()
manager.track(otb_dir, visualize=True)
