from __future__ import absolute_import, print_function

import os
import platform
import json
import numpy as np
import torch
from tensorboardX import SummaryWriter
from datetime import datetime


class Logger(SummaryWriter):

    def __init__(self, log_dir=None, comment='', verbose=True):
        super(Logger, self).__init__(log_dir, comment)
        self.log_dir = log_dir
        self.verbose = verbose
        self.add_meta()

    def log(self, text_string):
        self.add_text('common_text', text_string)

    def add_meta(self, meta={}):
        filename = os.path.join(self.log_dir, 'meta/meta.json')
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        meta.update({
            'datetime': str(datetime.now()),
            'language': 'Python %s' % platform.python_version(),
            'platform': platform.platform(),
            'computer': platform.node()})
        json.dump(meta, open(filename, 'w'), indent=4)

    def add_text(self, tag, text_string, global_step=None):
        super(Logger, self).add_text(tag, text_string, global_step)
        if self.verbose:
            print('{}: {}'.format(tag, text_string))

    def add_array(self, tag, array, global_step=None):
        if global_step is None:
            filename = os.path.join(
                self.log_dir, 'arrays', tag + '.txt')
        else:
            filename = os.path.join(
                self.log_dir, 'arrays', 'step_%d' % (global_step + 1))

        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        np.savetxt(filename, array, fmt='%.3f', delimiter=',')

    def add_checkpoint(self, tag, state_dict, global_step=None):
        if global_step is None:
            filename = os.path.join(
                self.log_dir, 'checkpoints/%s.pt' % tag)
        else:
            filename = os.path.join(
                self.log_dir, 'checkpoints/%s_%d.pt' %
                (tag, global_step + 1))

        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        torch.save(state_dict, filename)
