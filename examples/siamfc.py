from __future__ import absolute_import

import argparse
import json
import torch
from torch.utils.data import DataLoader

from lib.trackers import TrackerSiamFC
from lib.datasets import VOT, Pairwise
from lib.metrics import iou, center_error
from lib.utils.logger import Logger
from lib.transforms import TransformSiamFC


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--branch', type=str,
                    default='alexv1', choices=['alexv1', 'alexv2'])
parser.add_argument('-p', '--phase', type=str,
                    default='test', choices=['train', 'test'])
args = parser.parse_args()

# paths
config_file = 'config/siamfc.json'
vot_dir = 'data/vot2017'
net_paths = {
    'alexv1': 'pretrained/siamfc/2016-08-17.net.mat',
    'alexv2': 'pretrained/siamfc/baseline-conv5_e55.mat'}
stats_path = 'pretrained/siamfc/cfnet_ILSVRC2015.stats.mat'

with open(config_file) as f:
    config = json.load(f)
config = config[args.branch]
net_path = net_paths[args.branch]

if args.phase == 'test':
    tracker = TrackerSiamFC(args.branch, net_path, **config)
    dataset = VOT(vot_dir, return_bndbox=True)
    logger = Logger('logs/siamfc_%s.log' % args.branch)

    avg_iou = 0
    avg_prec = 0
    avg_speed = 0
    total_frames = 0

    for s, (img_files, anno) in enumerate(dataset):
        seq_name = dataset.seq_names[s]

        # tracking loop
        bndboxes, speed_fps = tracker.track(
            img_files, anno[0, :], visualize=True)

        mean_iou = iou(bndboxes, anno).mean()
        prec = (center_error(bndboxes, anno) <= 20).sum() / len(anno)
        logger.log('- Performance on {}: IoU: {:.3f} Prec: {:.3f} Speed: {:.3f}'.format(
            seq_name, mean_iou, prec, speed_fps))

        frame_num = len(img_files)
        avg_iou += mean_iou * frame_num
        avg_prec += prec * frame_num
        avg_speed += speed_fps * frame_num
        total_frames += frame_num

    avg_iou /= total_frames
    avg_prec /= total_frames
    avg_speed /= total_frames

    logger.log('- Overall Performance: IoU: {:.3f} Prec: {:.3f} Speed: {:.3f}'.format(
        avg_iou, avg_prec, avg_speed))
elif args.phase == 'train':
    tracker = TrackerSiamFC(args.branch, None, **config)
    transform = TransformSiamFC(
        score_sz=config['score_sz'], r_pos=config['r_pos'], total_stride=config['total_stride'], stats_path=stats_path)
    logger = Logger('logs/siamfc_%s_train.log' % args.branch)

    cuda = torch.cuda.is_available()
    epoch_num = config['epoch_num']

    dataset_train = Pairwise(
        VOT(vot_dir, return_bndbox=True), transform=transform, subset='train')
    dataloader_train = DataLoader(
        dataset_train, batch_size=config['batch_size'], shuffle=True,
        pin_memory=cuda, drop_last=True, num_workers=4)
    dataset_val = Pairwise(
        VOT(vot_dir, return_bndbox=True), transform=transform, subset='val')
    dataloader_val = DataLoader(
        dataset_val, batch_size=config['batch_size'], shuffle=True,
        pin_memory=cuda, drop_last=False, num_workers=4)

    iter_num = len(dataloader_train)
    for epoch in range(epoch_num):
        # training loop
        loss_epoch = 0
        for it, batch in enumerate(dataloader_train):
            loss = tracker.step(batch, backward=True, update_lr=(it == 0))
            loss_epoch += loss
            logger.log('Epoch: {}/{} Iter: {}/{} Loss: {:.5f}'.format(
                epoch + 1, epoch_num, it + 1, iter_num, loss))

        loss_epoch /= iter_num

        # validation loop
        loss_val = 0
        for it, batch in enumerate(dataloader_val):
            loss = tracker.step(batch, backward=False, update_lr=False)
            loss_val += loss

        loss_val /= len(dataloader_val)
        logger.log(
            '--Epoch Loss: {:.5f} Val. Loss: {:.5f}'.format(loss_epoch, loss_val))
