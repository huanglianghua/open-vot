from __future__ import absolute_import

import argparse

from lib.managers import ManagerGOTURN


cfg_file = 'config/goturn.json'
vot_dir = 'data/vot2017'
vid_dir = 'data/ILSVRC'
det_dir = 'data/imagenet'
net_path = 'pretrained/goturn/tracker.pth'
manager = ManagerGOTURN(cfg_file)

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--phase', type=str, default='test',
                    choices=['train', 'test'])
args = parser.parse_args()

if args.phase == 'test':
    manager.track(vot_dir, net_path, visualize=True)
elif args.phase == 'train':
    manager.train(vid_dir, det_dir, vot_dir)
