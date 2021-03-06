from __future__ import absolute_import, division

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from torch.optim.lr_scheduler import LambdaLR

from . import Tracker
from ..utils import dict2tuple
from ..models import DCFNet, DCFNetOnline
from ..utils.warp import warp_cv2

def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g

class TrackerDCFNet(Tracker):

    def __init__(self, net_path=None, online=False, **kargs):
        super(TrackerDCFNet, self).__init__('DCFNet')
        self.online = online
        self.parse_args(**kargs)
        self.setup_model(net_path)
        self.setup_optimizer()

    def parse_args(self, **kargs):
        # default branch is AlexNetV1
        self.cfg = {
            'crop_sz': 125,
            'output_sz': 121,
            'lambda0': 1e-4,
            'padding': 2.0,
            'output_sigma_factor': 0.1,
            'initial_lr': 1e-2,
            'final_lr': 1e-5,
            'epoch_num': 50,
            'weight_decay': 5e-4,
            'batch_size': 32,

            'interp_factor': 0.01,
            'num_scale': 3,
            'scale_step': 1.0275,
            'min_scale_factor': 0.2,
            'max_scale_factor': 5,
            'scale_penalty': 0.9925,
            }

        for key, val in kargs.items():
            self.cfg.update({key: val})

        self.cfg['output_sigma'] = self.cfg['crop_sz'] / (1 + self.cfg['padding']) * self.cfg['output_sigma_factor']
        self.cfg['y'] = gaussian_shaped_labels(self.cfg['output_sigma'], [self.cfg['output_sz'], self.cfg['output_sz']])
        self.cfg['yf'] = torch.rfft(torch.Tensor(self.cfg['y']).view(1, 1, self.cfg['output_sz'], self.cfg['output_sz']).cuda(), signal_ndim=2)
        self.cfg['net_average_image'] = np.array([104, 117, 123]).reshape(1, 1, -1).astype(np.float32)

        self.cfg['scale_factor'] = self.cfg['scale_step'] ** (np.arange(self.cfg['num_scale']) - self.cfg['num_scale'] / 2)
        self.cfg['scale_penalties'] = self.cfg['scale_penalty'] ** (np.abs((np.arange(self.cfg['num_scale']) - self.cfg['num_scale'] / 2)))
        self.cfg['net_input_size'] = [self.cfg['crop_sz'], self.cfg['crop_sz']]
        self.cfg['cos_window'] = torch.Tensor(np.outer(np.hanning(self.cfg['crop_sz']), np.hanning(self.cfg['crop_sz']))).cuda()
        self.cfg['y_online'] = gaussian_shaped_labels(self.cfg['output_sigma'], self.cfg['net_input_size'])
        self.cfg['yf_online'] = torch.rfft(torch.Tensor(self.cfg['y_online']).view(1, 1, self.cfg['crop_sz'], self.cfg['crop_sz']).cuda(), signal_ndim=2)

        self.cfg = dict2tuple(self.cfg)

    def setup_model(self, net_path=None):

        if self.online:
            self.model = DCFNetOnline(config = self.cfg)
        else:
            self.model = DCFNet(config = self.cfg)
        if net_path:
            self.load_param(net_path)
        self.gpu_num = torch.cuda.device_count()
        print('GPU NUM: {:2d}'.format(self.gpu_num))
        if self.gpu_num > 1 and self.online == False:
            self.model = nn.DataParallel(self.model, list(range(self.gpu_num))).cuda()
        else:
            self.model = self.model.cuda()

        self.target = torch.Tensor(self.cfg.y).cuda().unsqueeze(0).unsqueeze(0).repeat(self.cfg.batch_size * self.gpu_num, 1, 1, 1)

    def load_param(self, path='param.pth'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict']
            if 'module' in state_dict.keys()[0]:  # train with nn.DataParallel
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(state_dict)
        else:
            self.model.feature.load_state_dict(checkpoint)

    def setup_optimizer(self):

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay)
        lambda1 = lambda epoch: np.logspace(0, -2, num=self.cfg.epoch_num)[epoch]
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=[lambda1])
        self.criterion = nn.SmoothL1Loss(size_average=False)

    def init(self, image, init_rect):

        self.model.eval()

        self.target_pos = init_rect[:2] + init_rect[2:] / 2 - 1
        self.target_sz = init_rect[2:]

        self.min_sz = np.maximum(self.cfg.min_scale_factor * self.target_sz, 4)
        self.max_sz = np.minimum(image.shape[:2], self.cfg.max_scale_factor * self.target_sz)

        self.padded_sz = self.target_sz * (1 + self.cfg.padding)

        # get feature size and initialize hanning window
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        target = warp_cv2(image, self.target_pos, self.padded_sz,
                          self.cfg.net_input_size, (0, 0, 0))
        target = target - self.cfg.net_average_image
        # cv2.imshow('response', target)
        # cv2.waitKey(0)
        target = torch.from_numpy(target).cuda().permute(2, 0, 1).unsqueeze(0).float()

        self.model.update(target)

        self.patch_crop = torch.zeros(self.cfg.num_scale, target.shape[1], target.shape[2], target.shape[3]).cuda() # buff

    def update(self, image):

        self.model.eval()

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for i in range(self.cfg.num_scale):  # crop multi-scale search region
            window_sz = self.target_sz * (self.cfg.scale_factor[i] * (1 + self.cfg.padding))
            target = warp_cv2(image, self.target_pos, window_sz,
                              self.cfg.net_input_size, (0,0,0))
            target = target - self.cfg.net_average_image
            target = torch.from_numpy(target).cuda().permute(2, 0, 1).unsqueeze(0).float()
            self.patch_crop[i, :] = target

        response = self.model(self.patch_crop)

        peak, idx = torch.max(response.view(self.cfg.num_scale, -1), 1)
        peak = peak.data.cpu().numpy() * self.cfg.scale_penalties
        best_scale = np.argmax(peak)
        r_max, c_max = np.unravel_index(idx[best_scale], self.cfg.net_input_size)

        if r_max > self.cfg.net_input_size[0] / 2:
            r_max = r_max - self.cfg.net_input_size[0]
        if c_max > self.cfg.net_input_size[1] / 2:
            c_max = c_max - self.cfg.net_input_size[1]

        window_sz = self.target_sz * (self.cfg.scale_factor[best_scale] * (1 + self.cfg.padding))

        self.target_pos = self.target_pos + np.array([c_max, r_max]) * window_sz / self.cfg.net_input_size
        self.target_sz = np.minimum(np.maximum(window_sz / (1 + self.cfg.padding), self.min_sz), self.max_sz)

        # model update
        window_sz = self.target_sz * (1 + self.cfg.padding)
        target = warp_cv2(image, self.target_pos, window_sz,
                          self.cfg.net_input_size, (0, 0, 0))
        target = target - self.cfg.net_average_image
        target = torch.from_numpy(target).cuda().permute(2, 0, 1).unsqueeze(0).float()

        self.model.update(target, lr=self.cfg.interp_factor)

        bndbox = np.concatenate([
            self.target_pos - self.target_sz / 2 + 1, self.target_sz])

        return bndbox

    def step(self, batch, backward=True, update_lr=False):
        if backward:
            if update_lr:
                self.scheduler.step()
            self.model.train()
        else:
            self.model.eval()

        template, search = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)

        self.optimizer.zero_grad()
        with torch.set_grad_enabled(backward):
            output = self.model(template, search)
            loss = self.criterion(output, self.target)/template.size(0)
            if backward:
                loss.backward()
                self.optimizer.step()

        return loss.item()
