from __future__ import absolute_import, print_function, division

import time
import sys
import os
import zipfile
import torch
import scipy.io
from urllib.request import urlretrieve


def download(url, filename):
    return urlretrieve(url, filename, _reporthook)


def _reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def extract(filename, extract_dir):
    if os.path.splitext(filename)[1] == '.zip':
        print('Extracting zip file...')
        if not os.path.isdir(extract_dir):
            os.makedirs(extract_dir)
        with zipfile.ZipFile(filename) as z:
            z.extractall(extract_dir)
    else:
        raise Exception('Unsupport extension {} of the compressed file {}.'.format(
            os.path.splitext(filename)[1]), filename)


def load_siamfc_from_matconvnet(filename, model):
    params_names_list, params_values_list = load_matconvnet(filename)
    params_values_list = [torch.from_numpy(p) for p in params_values_list]
    for l, p in enumerate(params_values_list):
        param_name = params_names_list[l]
        if 'conv' in param_name and param_name[-1] == 'f':
            p = p.permute(3, 2, 0, 1)
        p = torch.squeeze(p)
        params_values_list[l] = p

    net = (
        model.branch.conv1,
        model.branch.conv2,
        model.branch.conv3,
        model.branch.conv4,
        model.branch.conv5)

    for l, layer in enumerate(net):
        layer[0].weight.data[:] = params_values_list[
            params_names_list.index('br_conv%df' % (l + 1))]
        layer[0].bias.data[:] = params_values_list[
            params_names_list.index('br_conv%db' % (l + 1))]

        if l < len(net) - 1:
            layer[1].weight.data[:] = params_values_list[
                params_names_list.index('br_bn%dm' % (l + 1))]
            layer[1].bias.data[:] = params_values_list[
                params_names_list.index('br_bn%db' % (l + 1))]

            bn_moments = params_values_list[
                params_names_list.index('br_bn%dx' % (l + 1))]
            layer[1].running_mean[:] = bn_moments[:, 0]
            layer[1].running_var[:] = bn_moments[:, 1] ** 2
        elif model.norm.norm == 'bn':
            model.norm.bn.weight.data[:] = params_values_list[
                params_names_list.index('fin_adjust_bnm')]
            model.norm.bn.bias.data[:] = params_values_list[
                params_names_list.index('fin_adjust_bnb')]

            bn_moments = params_values_list[
                params_names_list.index('fin_adjust_bnx')]
            model.norm.bn.running_mean[:] = bn_moments[0]
            model.norm.bn.running_var[:] = bn_moments[1] ** 2

    return model


def load_matconvnet(filename):
    mat = scipy.io.loadmat(filename)
    net_dot_mat = mat.get('net')
    params = net_dot_mat['params']
    params = params[0][0]
    params_names = params['name'][0]
    params_names_list = [params_names[p][0] for p in range(params_names.size)]
    params_values = params['value'][0]
    params_values_list = [params_values[p] for p in range(params_values.size)]

    return params_names_list, params_values_list
