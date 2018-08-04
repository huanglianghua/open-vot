from __future__ import absolute_import

import json
import torch
import multiprocessing
import math
from torch.utils.data import DataLoader

from ..trackers import TrackerGOTURN
from ..utils.logger import Logger
from ..datasets import VOT, ImageNetVID, ImageNetObject, Pairwise
from ..metrics import rect_iou, center_error
from ..utils import initialize_weights
from ..transforms import TransformGOTURN


class TrainerGOTURN(object):

    def __init__(self, cfg_file=None):
        cfg = {}
        if cfg_file is not None:
            with open(cfg_file, 'r') as f:
                cfg = json.load(f)

        self.tracker = TrackerGOTURN(net_path=None, **cfg)
        self.cfg = self.tracker.cfg
        self.logger = Logger(log_dir='logs/goturn')
        self.cuda = torch.cuda.is_available()

    def train(self, vid_dir, det_dir, vot_dir=None):
        tracker = self.tracker
        initialize_weights(tracker.model)
        transform = TransformGOTURN(**self.cfg._asdict())

        epoch_num = self.cfg.epoch_num
        cpu_num = multiprocessing.cpu_count()

        if vot_dir is not None:
            vot_dataset = VOT(vot_dir, return_rect=True, download=True)

        # training dataset
        seq_dataset_train = Pairwise(
            ImageNetVID(vid_dir, return_rect=True), transform,
            pairs_per_video=1, frame_range=1, causal=True, subset='train')
        seq_loader_train = DataLoader(
            seq_dataset_train, batch_size=self.cfg.batch_size, shuffle=True,
            pin_memory=self.cuda, drop_last=True, num_workers=cpu_num)

        img_dataset_train = ImageNetObject(
            det_dir, return_rect=True, subset='train', transform=transform)
        img_loader_train = DataLoader(
            img_dataset_train, batch_size=self.cfg.batch_size, shuffle=True,
            pin_memory=self.cuda, drop_last=True, num_workers=cpu_num)

        # validation dataset
        seq_dataset_val = Pairwise(
            ImageNetVID(vid_dir, return_rect=True), transform,
            pairs_per_video=1, frame_range=1, causal=True, subset='val')
        seq_loader_val = DataLoader(
            seq_dataset_val, batch_size=self.cfg.batch_size, shuffle=True,
            pin_memory=self.cuda, drop_last=True, num_workers=cpu_num)

        img_dataset_val = ImageNetObject(
            det_dir, return_rect=True, subset='val', transform=transform)
        img_loader_val = DataLoader(
            img_dataset_val, batch_size=self.cfg.batch_size, shuffle=True,
            pin_memory=self.cuda, drop_last=True, num_workers=cpu_num)

        train_iters = min(len(seq_loader_train), len(img_loader_train))
        val_iters = min(len(seq_loader_val), len(img_loader_val))
        repeat_num = max(1, math.ceil(epoch_num / train_iters))

        for epoch in range(repeat_num):
            # training loop
            seq_iter = iter(seq_loader_train)
            img_iter = iter(img_loader_train)
            loss_epoch = 0

            for it in range(train_iters):
                loss_seq = tracker.step(next(seq_iter), update_lr=True)
                loss_img = tracker.step(next(img_iter))
                loss = (loss_seq + loss_img) / 2
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
            self.logger.add_scalar('train/epoch_loss', loss_epoch, epoch)

            # validation loop
            seq_iter = iter(seq_loader_val)
            img_iter = iter(img_loader_val)
            loss_val = 0

            for it in range(val_iters):
                loss_seq = tracker.step(next(seq_iter), update_lr=True)
                loss_img = tracker.step(next(img_iter))
                loss = (loss_seq + loss_img) / 2
                loss_val += loss

            loss_val /= val_iters

            # logging
            self.logger.add_text('train/val_epoch_loss', 'Epoch: {}/{} Val. Loss: {:.6f}'.format(
                epoch + 1, epoch_num, loss_val), epoch)
            self.logger.add_scalar('train/val_epoch_loss', loss_val, epoch)

            # tracking loop if vot_dir is available
            if vot_dir is not None:
                self.track(vot_dir, visualize=False)

            # add checkpoint
            self.logger.add_checkpoint(
                'goturn', self.tracker.model.module.state_dict(),
                (epoch + 1) // 100 + 1)

        return tracker
