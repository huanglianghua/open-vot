from __future__ import absolute_import, division

import json
import torch
import multiprocessing
from torch.utils.data import DataLoader

from ..trackers import TrackerSiamFC
from ..utils.logger import Logger
from ..datasets import VOT, ImageNetVID, Pairwise
from ..metrics import iou, center_error
from ..utils import initialize_weights
from ..transforms import TransformSiamFC


class ManagerSiamFC(object):

    def __init__(self, branch='alexv1', cfg_file=None):
        cfg = {}
        if cfg_file is not None:
            with open(cfg_file, 'r') as f:
                cfg = json.load(f)
            cfg = cfg[branch]

        self.branch = branch
        self.tracker = TrackerSiamFC(branch=branch, net_path=None, **cfg)
        self.cfg = self.tracker.cfg
        self.logger = Logger(log_dir='logs/siamfc')
        self.cuda = torch.cuda.is_available()

    def track(self, vot_dir, net_path=None, step=None, visualize=True):
        tracker = self.tracker
        if net_path is not None:
            tracker.setup_model(self.branch, net_path)
        dataset = VOT(vot_dir, return_rect=True, download=True)

        tag = 'test' if step is None else 'test_%d' % (step + 1)

        avg_iou = 0
        avg_prec = 0
        avg_speed = 0
        total_frames = 0

        for s, (img_files, anno) in enumerate(dataset):
            seq_name = dataset.seq_names[s]

            # tracking loop
            rects, speed_fps = tracker.track(
                img_files, anno[0, :], visualize=visualize)

            mean_iou = iou(rects, anno).mean()
            prec = (center_error(rects, anno) <= 20).sum() / len(anno)

            frame_num = len(img_files)
            avg_iou += mean_iou * frame_num
            avg_prec += prec * frame_num
            avg_speed += speed_fps * frame_num
            total_frames += frame_num

            # logging
            self.logger.add_text(tag, '{}/{} Performance on {}: IoU: {:.3f} Prec: {:.3f} Speed: {:.3f} fps'.format(
                s + 1, len(dataset), seq_name, mean_iou, prec, speed_fps), step)
            self.logger.add_scalar(tag + '/iou_' + seq_name, mean_iou, step)
            self.logger.add_scalar(tag + '/prec_' + seq_name, prec, step)
            self.logger.add_scalar(tag + '/speed_' + seq_name, speed_fps, step)

        avg_iou /= total_frames
        avg_prec /= total_frames
        avg_speed /= total_frames

        # logging
        self.logger.add_text(tag, 'Overall Performance: IoU: {:.3f} Prec: {:.3f} Speed: {:.3f} fps'.format(
            mean_iou, prec, speed_fps), step)
        self.logger.add_scalar(tag + '/iou_overall', mean_iou, step)
        self.logger.add_scalar(tag + '/prec_overall', prec, step)
        self.logger.add_scalar(tag + '/speed_overall', speed_fps, step)

        return avg_iou, avg_prec, avg_speed

    def train(self, vid_dir, stats_path=None, vot_dir=None):
        tracker = self.tracker
        initialize_weights(tracker.model)
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

            for it, batch in enumerate(dataloader_train):
                loss = tracker.step(batch, update_lr=(it == 0))
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
            loss_val = 0

            for it, batch in enumerate(dataloader_val):
                loss = tracker.step(batch, backward=False)
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
                'siamfc', self.tracker.model.module.state_dict(),
                (epoch + 1) // 100 + 1)

        return tracker
