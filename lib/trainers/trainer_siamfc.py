from __future__ import absolute_import, division

import json
import torch
import multiprocessing
from torch.utils.data import DataLoader

from ..trackers import TrackerSiamFC
from ..utils.logger import Logger
from ..datasets import VOT, ImageNetVID, Pairwise
from ..metrics import rect_iou, center_error
from ..transforms import TransformSiamFC


class TrainerSiamFC(object):

    def __init__(self, branch='alexv1',net_path=None, cfg_file=None):
        cfg = {}
        if cfg_file is not None:
            with open(cfg_file, 'r') as f:
                cfg = json.load(f)
            cfg = cfg[branch]

        self.branch = branch
        self.tracker = TrackerSiamFC(branch=branch, net_path=net_path, **cfg)
        self.cfg = self.tracker.cfg
        self.logger = Logger(log_dir='logs/siamfc')
        self.cuda = torch.cuda.is_available()

    def train(self, vid_dir, stats_path=None, vot_dir=None):
        tracker = self.tracker
        transform = TransformSiamFC(stats_path, **self.cfg._asdict())

        epoch_num = self.cfg.epoch_num
        cpu_num = multiprocessing.cpu_count()

        if vot_dir is not None:
            vot_dataset = VOT(vot_dir, return_rect=True, download=True)
        base_dataset = ImageNetVID(vid_dir, return_rect=True)

        # training dataset
        dataset_train = Pairwise(
            base_dataset, transform, subset='train')
        dataloader_train = DataLoader(
            dataset_train, batch_size=self.cfg.batch_size, shuffle=True,
            pin_memory=self.cuda, drop_last=True, num_workers=cpu_num)

        # validation dataset
        dataset_val = Pairwise(
            base_dataset, transform, subset='val')
        dataloader_val = DataLoader(
            dataset_val, batch_size=self.cfg.batch_size, shuffle=False,
            pin_memory=self.cuda, drop_last=True, num_workers=cpu_num)

        train_iters = len(dataloader_train)
        val_iters = len(dataloader_val)

        for epoch in range(epoch_num):
            # training loop
            loss_epoch = 0
            tracker.scheduler.step(epoch)
            for it, batch in enumerate(dataloader_train):
                loss = tracker.step(batch)
                loss_epoch += loss

                # logging
                step = epoch * train_iters + it
                self.logger.add_text('train/iter_loss', '--Epoch: {}/{} Iter: {}/{} Loss: {:.6f}'.format(
                    epoch + 1, epoch_num, it + 1, train_iters, loss), step)
                self.logger.add_scalar('train/iter_loss', loss, step)

            loss_epoch /= train_iters

            # logging
            self.logger.add_text('train/epoch_loss', 'Epoch: {}/{} Loss: {:.6f}'.format(
                epoch + 1, epoch_num, loss_epoch), epoch)

            # validation loop
            loss_val = 0

            for it, batch in enumerate(dataloader_val):
                loss = tracker.step(batch, backward=False)
                loss_val += loss

            loss_val /= val_iters

            # logging
            self.logger.add_text('train/val_epoch_loss', 'Epoch: {}/{} Val. Loss: {:.6f}'.format(
                epoch + 1, epoch_num, loss_val), epoch)
            self.logger.add_scalars('epoch_loss', {'epoch_loss': loss_epoch, 
                                                   'val_epoch_loss', loss_val}, epoch)

            # tracking loop if vot_dir is available
            if vot_dir is not None:
                self.track(vot_dir, visualize=False)

            # add checkpoint
            self.logger.add_checkpoint(
                'siamfc', self.tracker.model.module.state_dict(),
                (epoch + 1) // 100 + 1)

        return tracker
