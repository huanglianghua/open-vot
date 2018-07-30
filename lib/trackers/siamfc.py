from __future__ import absolute_import, division

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from torch.optim.lr_scheduler import StepLR

from . import Tracker
from ..utils import dict2tuple
from ..models import SiameseNet, AlexNetV1, AlexNetV2
from ..utils.ioutil import load_siamfc_from_matconvnet
from ..utils.warp import warp_cv2


class BCEWeightedLogitsLoss(nn.Module):

    def __init__(self):
        super(BCEWeightedLogitsLoss, self).__init__()

    def forward(self, input, target, weight=None):
        return F.binary_cross_entropy_with_logits(
            input, target, weight, size_average=True)

class BCEWeightedLoss(nn.Module):

    def __init__(self):
        super(BCEWeightedLoss, self).__init__()

    def forward(self, input, target, weight=None):
        return F.binary_cross_entropy(
            input, target, weight, size_average=True)

class TrackerSiamFC(Tracker):

    def __init__(self, branch='alexv1', net_path=None, **kargs):
        super(TrackerSiamFC, self).__init__('SiamFC')
        self.parse_args(**kargs)
        self.norm_type = "cosine_similarity"
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.setup_model(branch, net_path)
        self.setup_optimizer()

    def parse_args(self, **kargs):
        # default branch is AlexNetV1
        self.cfg = {
            'exemplar_sz': 127,
            'search_sz': 255,
            'response_up': 16,
            'context': 0.5,
            'window_influence': 0.176,
            'z_lr': 0,
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_penalty': 0.97,
            'scale_lr': 0.59,
            'r_pos': 16,
            'r_neg': 0,
            'initial_lr': 1e-2,
            'final_lr': 1e-5,
            'step_size': 2,
            'epoch_num': 50,
            'lr_mult_conv_weight': 1,
            'lr_mult_conv_bias': 2,
            'lr_mult_bn_weight': 2,
            'lr_mult_bn_bias': 1,
            'lr_mult_linear_weight': 0,
            'lr_mult_linear_bias': 1,
            'weight_decay': 5e-4,
            'batch_size': 8}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = dict2tuple(self.cfg)

    def setup_model(self, branch='alexv1', net_path=None):
        assert branch in ['alexv1', 'alexv2']
        if branch == 'alexv1':
            self.model = SiameseNet(AlexNetV1(), norm=self.norm_type)
        elif branch == 'alexv2':
            self.model = SiameseNet(AlexNetV2(), norm=self.norm_type)

        if net_path is not None:
            ext = os.path.splitext(net_path)[1]
            if ext == '.mat':
                load_siamfc_from_matconvnet(net_path, self.model)
            elif ext == '.pth':
                state_dict = torch.load(
                    net_path, map_location=lambda storage, loc: storage)
                self.model.load_state_dict(state_dict)
            else:
                raise Exception('unsupport file extension')

        self.branch = nn.DataParallel(self.model.branch).to(self.device)
        self.norm = nn.DataParallel(self.model.norm).to(self.device)
        self.model = nn.DataParallel(self.model).to(self.device)

    def setup_optimizer(self):

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay)
        gamma = (self.cfg.final_lr / self.cfg.initial_lr) ** \
            (1 / (self.cfg.epoch_num // self.cfg.step_size))
        self.scheduler = StepLR(
            self.optimizer, self.cfg.step_size, gamma=gamma)
        if self.norm == "cosine_similarity":
            self.criterion = BCEWeightedLoss().to(self.device)
        else:
            self.criterion = BCEWeightedLogitsLoss().to(self.device)

    def init(self, image, init_rect):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # initialize parameters
        self.center = init_rect[:2] + init_rect[2:] / 2
        self.target_sz = init_rect[2:]
        context = self.cfg.context * self.target_sz.sum()
        self.z_sz = np.sqrt((self.target_sz + context).prod())
        self.x_sz = self.z_sz * self.cfg.search_sz / self.cfg.exemplar_sz
        self.min_x_sz = 0.2 * self.x_sz
        self.max_x_sz = 5.0 * self.x_sz

        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -self.cfg.scale_num // 2,
            self.cfg.scale_num // 2, self.cfg.scale_num)
        self.score_sz, self.total_stride = self._deduce_network_params(
            self.cfg.exemplar_sz, self.cfg.search_sz)
        self.final_score_sz = self.cfg.response_up * (self.score_sz - 1) + 1

        self.penalty = np.outer(
            np.hanning(self.final_score_sz),
            np.hanning(self.final_score_sz))
        self.penalty /= self.penalty.sum()
        self.avg_color = np.mean(image, axis=(0, 1))

        # extract template features
        crop_z = warp_cv2(image, self.center, self.z_sz,
                          self.cfg.exemplar_sz, self.avg_color)
        crop_z = torch.from_numpy(crop_z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        crop_z = (crop_z / 255.0 - 0.5) / 0.5

        with torch.set_grad_enabled(False):
            self.branch.eval()
            self.z = self.branch(crop_z)

    def update(self, image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # update scaled sizes
        scaled_exemplar = self.scale_factors * self.z_sz
        scaled_search_area = self.scale_factors * self.x_sz
        scaled_target = self.scale_factors[:, np.newaxis] * self.target_sz

        # locate target
        crops_x = [warp_cv2(
            image, self.center, size, self.cfg.search_sz, self.avg_color)
            for size in scaled_search_area]
        crops_x = torch.stack([(torch.from_numpy(c).to(
            self.device).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5
            for c in crops_x], dim=0)

        with torch.set_grad_enabled(False):
            self.branch.eval()
            x = self.branch(crops_x)
        print(x.size(),self.z.size())
        score, scale_id = self._calc_score(self.z, x)

        self.x_sz = (1 - self.cfg.scale_lr) * self.x_sz + \
            self.cfg.scale_lr * scaled_search_area[scale_id]
        self.x_sz = np.clip(self.x_sz, self.min_x_sz, self.max_x_sz)
        self.center = self._locate_target(self.center, score, self.final_score_sz,
                                          self.total_stride, self.cfg.search_sz,
                                          self.cfg.response_up, self.x_sz)
        self.target_sz = (1 - self.cfg.scale_lr) * self.target_sz + \
            self.cfg.scale_lr * scaled_target[scale_id]

        # update the template
        # self.z_sz = (1 - self.cfg.scale_lr) * self.z_sz + \
        #     self.cfg.scale_lr * scaled_exemplar[scale_id]
        if self.cfg.z_lr > 0:
            crop_z = warp_cv2(image, self.center, self.z_sz,
                              self.cfg.exemplar_sz, self.avg_color)
            crop_z = torch.from_numpy(crop_z).to(
                self.device).permute(2, 0, 1).unsqueeze(0).float()
            with torch.set_grad_enabled(False):
                self.branch.eval()
                new_z = self.branch(crop_z)
            self.z = (1 - self.cfg.z_lr) * self.z + \
                self.cfg.z_lr * new_z
        self.z_sz = (1 - self.cfg.scale_lr) * self.z_sz + \
            self.cfg.scale_lr * scaled_exemplar[scale_id]

        bndbox = np.concatenate([
            self.center - self.target_sz / 2, self.target_sz])

        return bndbox

    def step(self, batch, backward=True, update_lr=False):
        if backward:
            if update_lr:
                self.scheduler.step()
            self.model.train()
        else:
            self.model.eval()

        z, x, labels, weights = \
            batch[0].to(self.device), batch[1].to(self.device), \
            batch[2].to(self.device), batch[3].to(self.device)

        self.optimizer.zero_grad()
        with torch.set_grad_enabled(backward):
            pred = self.model(z, x)
            loss = self.criterion(pred, labels, weights)
            if backward:
                loss.backward()
                self.optimizer.step()

        return loss.item()

    def _deduce_network_params(self, exemplar_sz, search_sz):
        z = torch.zeros(1, 3, exemplar_sz, exemplar_sz).to(self.device)
        x = torch.zeros(1, 3, search_sz, search_sz).to(self.device)
        with torch.set_grad_enabled(False):
            self.model.eval()
            y = self.model(z, x)
        score_sz = y.size(-1)

        total_stride = 1
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.MaxPool2d)):
                stride = m.stride[0] if isinstance(
                    m.stride, tuple) else m.stride
                total_stride *= stride

        return score_sz, total_stride

    def _calc_score(self, z, x):
        if self.norm_type != "cosine_similarity":
            scores = F.conv2d(x, z)
            print(x.size(),z.size(),scores.size())
            with torch.set_grad_enabled(False):
                self.norm.eval()
                scores = self.norm(scores)
        else:
            with torch.set_grad_enabled(False):
                self.norm.eval()
                scores = self.norm(None, z, x)

        scores[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        scores[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty
        scale_id = scores.view(self.cfg.scale_num, -1).max(dim=1)[0].argmax()

        score = scores[scale_id].squeeze(0).cpu().numpy()
        score = cv2.resize(
            score, (self.final_score_sz, self.final_score_sz),
            interpolation=cv2.INTER_CUBIC)
        score -= score.min()
        score /= max(1e-12, score.sum())
        score = (1 - self.cfg.window_influence) * score + \
            self.cfg.window_influence * self.penalty

        return score, scale_id

    def _locate_target(self, center, score, final_score_sz,
                       total_stride, search_sz, response_up, x_sz):
        pos = np.unravel_index(score.argmax(), score.shape)[::-1]
        half = (final_score_sz - 1) / 2

        disp_in_area = np.asarray(pos) - half
        disp_in_xcrop = disp_in_area * total_stride / response_up
        disp_in_frame = disp_in_xcrop * x_sz / search_sz

        center = center + disp_in_frame

        return center
