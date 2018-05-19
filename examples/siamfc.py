from __future__ import absolute_import

import argparse

from lib.managers import ManagerSiamFC


cfg_file = 'config/siamfc.json'
vot_dir = 'data/vot2017'
vid_dir = 'data/ILSVRC'
net_v1 = 'pretrained/siamfc/2016-08-17.net.mat'
net_v2 = 'pretrained/siamfc/baseline-conv5_e55.mat'
stats_path = 'pretrained/siamfc/cfnet_ILSVRC2015.stats.mat'

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--branch', type=str,
                    default='alexv1', choices=['alexv1', 'alexv2'])
parser.add_argument('-p', '--phase', type=str,
                    default='test', choices=['train', 'test'])
args = parser.parse_args()

manager = ManagerSiamFC(args.branch, cfg_file)
net_path = net_v1 if args.branch == 'alexv1' else net_v2

if args.phase == 'test':
    manager.track(vot_dir, net_path, visualize=True)
elif args.phase == 'train':
    manager.train(vid_dir, stats_path, vot_dir)
