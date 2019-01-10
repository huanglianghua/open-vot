from __future__ import absolute_import, division

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import torchvision.transforms.functional as TF
import numbers
from torch.optim.lr_scheduler import StepLR
from PIL import Image

from ..utils import dict2tuple
from ..models import SiameseNet, AlexNetV1, AlexNetV2
from ..utils.ioutil import load_siamfc_from_matconvnet
from ..utils.warp import crop
from ..utils.viz import show_frame


class BCEWeightedLoss(nn.Module):

    def __init__(self):
        super(BCEWeightedLoss, self).__init__()

    def forward(self, input, target, weight=None):
        return F.binary_cross_entropy_with_logits(
            input, target, weight, size_average=True)


class TrackerSiamFC(object):

    def __init__(self, branch='alexv2', net_path=None, **kargs):
        self.name = 'SiamFC'
        self.parse_args(**kargs)
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.setup_model(branch, net_path)
        self.setup_optimizer()

    def parse_args(self, **kargs):
        # default branch is AlexNetV2
        self.cfg = {
            'exemplar_sz': 127,
            'search_sz': 255,
            'response_up': 8,
            'context': 0.5,
            'window_influence': 0.25,
            'z_lr': 0.01,
            'scale_num': 3,
            'scale_step': 1.0816,
            'scale_penalty': 0.97,
            'scale_lr': 0.59,
            'r_pos': 8,
            'r_neg': 0,
            'initial_lr': 1e-2,
            'final_lr': 1e-5,
            'step_size': 50,
            'epoch_num': 1000,
            'lr_mult_conv_weight': 1,
            'lr_mult_conv_bias': 2,
            'lr_mult_bn_weight': 2,
            'lr_mult_bn_bias': 1,
            'lr_mult_linear_weight': 0,
            'lr_mult_linear_bias': 1,
            'weight_decay': 5e-4,
            'batch_size': 32}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = dict2tuple(self.cfg)

    def setup_model(self, branch='alexv2', net_path=None):
        assert branch in ['alexv1', 'alexv2']
        if branch == 'alexv1':
            self.model = SiameseNet(AlexNetV1(), norm='linear')
        elif branch == 'alexv2':
            self.model = SiameseNet(AlexNetV2(), norm='bn')

        if net_path is not None:
            ext = os.path.splitext(net_path)[1]
            if ext == '.mat':
                load_siamfc_from_matconvnet(net_path, self.model)
            elif ext == '.pt':
                state_dict = torch.load(
                    net_path, map_location=lambda storage, loc: storage)
                self.model.load_state_dict(state_dict)
            else:
                raise Exception('unsupport file extension')

        self.branch = nn.DataParallel(self.model.branch).to(self.device)
        self.model = nn.DataParallel(self.model).to(self.device)

    def setup_optimizer(self):
        params = []
        for name, param in self.model.named_parameters():
            lr = self.cfg.initial_lr
            weight_decay = self.cfg.weight_decay
            if '.0' in name:  # conv
                if 'weight' in name:
                    lr *= self.cfg.lr_mult_conv_weight
                    weight_decay *= 1
                elif 'bias' in name:
                    lr *= self.cfg.lr_mult_conv_bias
                    weight_decay *= 0
            elif '.1' in name or 'bn' in name:  # bn
                if 'weight' in name:
                    lr *= self.cfg.lr_mult_bn_weight
                    weight_decay *= 0
                elif 'bias' in name:
                    lr *= self.cfg.lr_mult_bn_bias
                    weight_decay *= 0
            elif 'linear' in name:
                if 'weight' in name:
                    lr *= self.cfg.lr_mult_linear_weight
                    weight_decay *= 1
                elif 'bias' in name:
                    lr *= self.cfg.lr_mult_linear_bias
                    weight_decay *= 0
            params.append({
                'params': param,
                'initial_lr': lr,
                'weight_decay': weight_decay})

        self.optimizer = optim.SGD(
            params, lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay)
        gamma = (self.cfg.final_lr / self.cfg.initial_lr) ** \
            (1 / (self.cfg.epoch_num // self.cfg.step_size))
        self.scheduler = StepLR(self.optimizer, self.cfg.step_size, gamma=gamma)
        self.criterion = BCEWeightedLoss().to(self.device)

    def init(self, image, init_rect):
        # initialize parameters
        self.center = init_rect[:2] + init_rect[2:] / 2
        self.target_sz = init_rect[2:]
        context = self.cfg.context * self.target_sz.sum()
        self.z_sz = np.sqrt((self.target_sz + context).prod())
        self.x_sz = self.z_sz * self.cfg.search_sz / self.cfg.exemplar_sz

        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)
        self.score_sz, self.total_stride = self._deduce_network_params(
            self.cfg.exemplar_sz, self.cfg.search_sz)
        self.final_score_sz = self.cfg.response_up * (self.score_sz - 1) + 1

        hann_1d = np.expand_dims(np.hanning(
            self.final_score_sz), axis=0)
        self.penalty = np.transpose(hann_1d) * hann_1d
        self.penalty = self.penalty / self.penalty.sum()

        # extract template features
        crop_z = crop(image, self.center, self.z_sz,
                      out_size=self.cfg.exemplar_sz)
        self.z = self._extract_feature(crop_z)

    def update(self, image):
        # update scaled sizes
        scaled_exemplar = self.scale_factors * self.z_sz
        scaled_search_area = self.scale_factors * self.x_sz
        scaled_target = self.scale_factors[:, np.newaxis] * self.target_sz

        # locate target
        crops_x = self._crop(image, self.center, scaled_search_area,
                             out_size=self.cfg.search_sz)
        x = self._extract_feature(crops_x)
        score, scale_id = self._calc_score(self.z, x)

        self.x_sz = (1 - self.cfg.scale_lr) * self.x_sz + \
            self.cfg.scale_lr * scaled_search_area[scale_id]
        self.center = self._locate_target(self.center, score, self.final_score_sz,
                                          self.total_stride, self.cfg.search_sz,
                                          self.cfg.response_up, self.x_sz)
        self.target_sz = (1 - self.cfg.scale_lr) * self.target_sz + \
            self.cfg.scale_lr * scaled_target[scale_id]

        # update the template
        # self.z_sz = (1 - self.cfg.scale_lr) * self.z_sz + \
        #     self.cfg.scale_lr * scaled_exemplar[scale_id]
        if self.cfg.z_lr > 0:
            crop_z = crop(image, self.center, self.z_sz,
                          out_size=self.cfg.exemplar_sz)
            new_z = self._extract_feature(crop_z)
            self.z = (1 - self.cfg.z_lr) * self.z + \
                self.cfg.z_lr * new_z
        self.z_sz = (1 - self.cfg.scale_lr) * self.z_sz + \
            self.cfg.scale_lr * scaled_exemplar[scale_id]

        return np.concatenate([
            self.center - self.target_sz / 2, self.target_sz])

    def track(self, img_files, init_rect, visualize=False):
        frame_num = len(img_files)
        bndboxes = np.zeros((frame_num, 4))
        bndboxes[0, :] = init_rect
        speed_fps = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            if image.mode == 'L':
                image = image.convert('RGB')

            start_time = time.time()
            if f == 0:
                self.init(image, init_rect)
            else:
                bndboxes[f, :] = self.update(image)
            elapsed_time = time.time() - start_time
            speed_fps[f] = elapsed_time

            if visualize:
                show_frame(image, bndboxes[f, :], fig_n=1)

        return bndboxes, speed_fps

    def step(self, batch, backward=True):
        if backward:
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

    def _crop(self, image, center, sizes, padding='avg', out_size=None):
        sizes = np.array(sizes)
        if sizes.ndim == 1:
            sizes = np.tile(sizes, (2, 1)).T

        max_size = np.max(sizes, axis=0)
        anchor_patch = crop(image, center, max_size, padding=padding)

        patches = []
        for i, size in enumerate(sizes):
            if np.all(size == max_size):
                patch = anchor_patch
            else:
                offset = (max_size - size) / 2
                patch = anchor_patch.crop((
                    int(offset[0]),
                    int(offset[1]),
                    int(offset[0] + round(size[0])),
                    int(offset[1] + round(size[1]))))
            if out_size is not None:
                patch = patch.resize((out_size, out_size), Image.BILINEAR)
            patches.append(patch)

        if len(sizes) == 1:
            patches = patches[0]

        return patches

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

    def _extract_feature(self, image):
        if isinstance(image, Image.Image):
            image = (255.0 * TF.to_tensor(image)).unsqueeze(0)
        elif isinstance(image, (list, tuple)):
            image = 255.0 * torch.stack([TF.to_tensor(c) for c in image])
        else:
            raise Exception('Incorrect input type: {}'.format(type(image)))

        with torch.set_grad_enabled(False):
            self.branch.eval()
            return self.branch(image.to(self.device))

    def _calc_score(self, z, x):
        scores = F.conv2d(x, z)
        with torch.set_grad_enabled(False):
            self.model.module.norm.eval()
            scores = self.model.module.norm(scores, z, x).squeeze(1)

        scores = np.stack(
            [self._resize(s.cpu().numpy(), self.final_score_sz) for s in scores])
        scores[:self.cfg.scale_num // 2, :, :] *= self.cfg.scale_penalty
        scores[self.cfg.scale_num // 2 + 1:, :, :] *= self.cfg.scale_penalty

        scale_id = np.argmax(np.amax(scores, axis=(1, 2)))
        score = scores[scale_id, :, :]
        score = score - np.min(score)
        score = score / (np.sum(score) + 1e-12)
        score = (1 - self.cfg.window_influence) * score + \
            self.cfg.window_influence * self.penalty

        return score, scale_id

    def _resize(self, array2d, size):
        if isinstance(size, numbers.Number):
            size = (size, size)
        image = Image.fromarray(array2d)
        image = image.resize(size, Image.BICUBIC)

        return np.array(image)

    def _locate_target(self, center, score, final_score_sz,
                       total_stride, search_sz, response_up, x_sz):
        pos = np.asarray(np.unravel_index(score.argmax(), score.shape))
        half = (final_score_sz - 1) / 2

        disp_in_area = pos - half
        disp_in_xcrop = disp_in_area * total_stride / response_up
        disp_in_frame = disp_in_xcrop * x_sz / search_sz

        center = center + disp_in_frame[::-1]

        return center
