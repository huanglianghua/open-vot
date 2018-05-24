from __future__ import absolute_import, division

import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from . import Tracker
from ..utils import dict2tuple
from ..models import GOTURN
from ..utils.ioutil import load_goturn_from_caffe
from ..utils.warp import crop_tensor


class TrackerGOTURN(Tracker):

    def __init__(self, net_path=None, **kargs):
        self.parse_args(**kargs)
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.setup_model(net_path)
        self.setup_optimizer()

    def parse_args(self, **kargs):
        self.cfg = {
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

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = dict2tuple(self.cfg)

    def setup_model(self, net_path=None):
        self.model = GOTURN()
        if net_path is not None:
            ext = os.path.splitext(net_path)[1]
            if ext == '.pth':
                state_dict = torch.load(
                    net_path, map_location=lambda storage, loc: storage)
                self.model.load_state_dict(state_dict)
            elif ext == '.caffemodel':
                proto_path = os.path.join(
                    os.path.dirname(net_path), 'tracker.prototxt')
                load_goturn_from_caffe(net_path, proto_path, self.model)
            else:
                raise Exception('unsupport file extention')

        self.model = nn.DataParallel(self.model).to(self.device)

    def setup_optimizer(self):
        params = []
        for name, param in self.model.named_parameters():
            lr = self.cfg.base_lr
            weight_decay = self.cfg.weight_decay
            if 'conv' in name:
                if 'weight' in name:
                    lr *= self.cfg.lr_mult_conv
                    weight_decay *= 1
                elif 'bias' in name:
                    lr *= self.cfg.lr_mult_conv
                    weight_decay *= 0
            elif 'fc' in name:
                if 'weight' in name:
                    lr *= self.cfg.lr_mult_fc_weight
                    weight_decay *= 1
                elif 'bias' in name:
                    lr *= self.cfg.lr_mult_fc_bias
                    weight_decay *= 0
            params.append({
                'params': param,
                'initial_lr': lr,
                'weight_decay': weight_decay})

        self.optimizer = optim.SGD(
            params, lr=self.cfg.base_lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay)
        self.scheduler = StepLR(
            self.optimizer, self.cfg.lr_step_size, gamma=self.cfg.gamma)
        self.criterion = nn.L1Loss().to(self.device)

    def init(self, image, init_rect):
        image = torch.from_numpy(image).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        init_rect = torch.from_numpy(init_rect).to(self.device).float()
        self.image_prev = image
        self.bndbox_prev = init_rect

    def update(self, image):
        image = torch.from_numpy(image).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()

        z, _ = self._crop(self.image_prev, self.bndbox_prev)
        x, roi = self._crop(image, self.bndbox_prev)

        corners = self._locate_target(z, x)
        corners = corners.squeeze() / self.cfg.scale_factor
        corners = corners.clamp_(0, 1)

        corners[0::2] *= roi[2]
        corners[1::2] *= roi[3]
        corners[0::2] += roi[0]
        corners[1::2] += roi[1]

        bndbox_curr = torch.cat((corners[:2], corners[2:] - corners[:2]))
        bndbox_curr[2].clamp_(1.0, image.size(-1))
        bndbox_curr[3].clamp_(1.0, image.size(-2))

        # update
        self.image_prev = image
        self.bndbox_prev = bndbox_curr

        return bndbox_curr.numpy()

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
        center[0].clamp_(0.0, image.size(-1))
        center[1].clamp_(0.0, image.size(-2))

        size = bndbox[2:] * self.cfg.context
        size[0].clamp_(1.0, image.size(-1))
        size[1].clamp_(1.0, image.size(-2))

        patch = crop_tensor(image, center, size, padding=0,
                            out_size=self.cfg.input_dim)
        roi = torch.cat([center - size / 2, size])
        
        return patch, roi

    def _locate_target(self, z, x):
        mean_color = torch.FloatTensor(self.cfg.mean_color)
        mean_color = mean_color.to(self.device).view(1, 3, 1, 1)
        z -= mean_color
        x -= mean_color
        with torch.set_grad_enabled(False):
            self.model.eval()
            corners = self.model(z, x)

        return corners
