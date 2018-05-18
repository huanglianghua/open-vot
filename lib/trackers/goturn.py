from __future__ import absolute_import

import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import torchvision.transforms.functional as F
from torch.optim.lr_scheduler import StepLR
from PIL import Image

from ..models import GOTURN
from ..utils.viz import show_frame
from ..utils.warp import crop


class TrackerGOTURN(object):

    def __init__(self, net_path=None, **kargs):
        self.parse_args(**kargs)
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.setup_model(net_path)
        self.setup_optimizer()

    def parse_args(self, **kargs):
        default_args = {
            'context': 2,
            'scale_factor': 10,
            'input_dim': 227,
            'mean_color': [104, 117, 123],
            'base_lr': 1e-6,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'lr_step_size': 2000,
            'gamma': 0.1,
            'epoch_num': 10000,
            'lr_mult_fc_weight': 10,
            'lr_mult_fc_bias': 20,
            'lr_mult_conv': 0,
            'batch_size': 50 // 2}

        for key in default_args:
            if key in kargs:
                setattr(self, key, kargs[key])
            else:
                setattr(self, key, default_args[key])

    def setup_model(self, net_path=None):
        self.model = GOTURN()
        if net_path is not None:
            ext = os.path.splitext(net_path)[1]
            if ext == '.pth':
                state_dict = torch.load(
                    net_path, map_location=lambda storage, loc: storage)
                self.model.load_state_dict(state_dict)
            else:
                raise Exception('unsupport file extention')

        self.model = nn.DataParallel(self.model).to(self.device)

    def setup_optimizer(self):
        params = []
        for name, param in self.model.named_parameters():
            lr = self.base_lr
            weight_decay = self.weight_decay
            if 'conv' in name:
                if 'weight' in name:
                    lr *= self.lr_mult_conv
                    weight_decay *= 1
                elif 'bias' in name:
                    lr *= self.lr_mult_conv
                    weight_decay *= 0
            elif 'fc' in name:
                if 'weight' in name:
                    lr *= self.lr_mult_fc_weight
                    weight_decay *= 1
                elif 'bias' in name:
                    lr *= self.lr_mult_fc_bias
                    weight_decay *= 0
            params.append({
                'params': param,
                'initial_lr': lr,
                'weight_decay': weight_decay})

        self.optimizer = optim.SGD(
            params, lr=self.base_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay)
        self.scheduler = StepLR(
            self.optimizer, self.lr_step_size, gamma=self.gamma)
        self.criterion = nn.L1Loss().to(self.device)

    def init(self, image, init_rect):
        self.image_prev = image
        self.bndbox_prev = init_rect

    def update(self, image):
        z = self._crop(
            self.image_prev, self.bndbox_prev)
        x, roi = self._crop(
            image, self.bndbox_prev, return_roi=True)

        corners = self._locate_target(z, x)
        corners = corners.squeeze().cpu().numpy()
        corners /= self.scale_factor

        corners = np.clip(corners, 0, 1)
        corners[0::2] *= roi[2]
        corners[1::2] *= roi[3]
        corners[0::2] += roi[0]
        corners[1::2] += roi[1]

        bndbox_curr = np.concatenate([
            corners[:2], corners[2:] - corners[:2]])
        bndbox_curr[2:] = np.clip(bndbox_curr[2:], 1.0, image.size)

        # update
        self.image_prev = image
        self.bndbox_prev = bndbox_curr

        return bndbox_curr

    def track(self, img_files, init_rect, visualize=False):
        frame_num = len(img_files)
        bndboxes = np.zeros((frame_num, 4))
        bndboxes[0, :] = init_rect

        elapsed_time = 0
        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            if image.mode == 'L':
                image = image.convert('RGB')

            start_time = time.time()
            if f == 0:
                self.init(image, init_rect)
            else:
                bndboxes[f, :] = self.update(image)
            elapsed_time += time.time() - start_time

            if visualize:
                show_frame(image, bndboxes[f, :], fig_n=1)
        speed_fps = frame_num / elapsed_time

        return bndboxes, speed_fps

    def step(self, batch, backward=True, update_lr=False):
        if backward:
            if update_lr:
                self.scheduler.step()
            self.model.train()
        else:
            self.model.eval()

        z, x, labels = \
            batch[0].to(self.device), batch[1].to(self.device), \
            batch[2].to(self.device)

        self.optimizer.zero_grad()
        with torch.set_grad_enabled(backward):
            pred = self.model(z, x)
            loss = self.criterion(pred, labels)
            if backward:
                loss.backward()
                self.optimizer.step()

        return loss.item()

    def _crop(self, image, bndbox, return_roi=False):
        center = bndbox[:2] + bndbox[2:] / 2
        center = np.clip(center, 0.0, image.size)
        size = bndbox[2:] * self.context
        size = np.clip(size, 1.0, image.size)

        patch = crop(image, center, size, padding=0,
                     out_size=self.input_dim)

        if return_roi:
            roi = np.concatenate([center - size / 2, size])
            return patch, roi
        else:
            return patch

    def _locate_target(self, z, x):
        mean_color = torch.tensor(self.mean_color).float()
        mean_color = mean_color.view(3, 1, 1).to(self.device)
        z = 255.0 * F.to_tensor(z) - mean_color
        x = 255.0 * F.to_tensor(x) - mean_color
        z, x = z.unsqueeze(0), x.unsqueeze(0)
        with torch.set_grad_enabled(False):
            self.model.eval()
            corners = self.model(z, x)

        return corners
