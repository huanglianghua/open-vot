from __future__ import absolute_import, print_function, division

import os
import glob
import numpy as np
import six

from . import VideoObjectDataset
from ..utils.ioutil import download, extract


class VOT(VideoObjectDataset):

    __valid_versions = range(2013, 2017 + 1)

    def __init__(self, root_dir, version=2017,
                 anno_type='rect', download=True):
        assert version in self.__valid_versions
        assert anno_type in ['rect', 'corner']
        super(VOT, self).__init__('vot{}'.format(version))

        self.root_dir = root_dir
        self.version = version
        self.anno_type = anno_type
        if download:
            self._download(root_dir, version)
        self._check_integrity(root_dir, version)

        list_file = os.path.join(root_dir, 'list.txt')
        with open(list_file, 'r') as f:
            self.seq_names = f.read().strip().split('\n')
        self.seq_dirs = [os.path.join(root_dir, s) for s in self.seq_names]
        self.anno_files = [os.path.join(s, 'groundtruth.txt')
                           for s in self.seq_dirs]

    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], '*.jpg')))
        anno = np.loadtxt(self.anno_files[index], delimiter=',')
        assert len(img_files) == len(anno)
        assert anno.shape[1] in [4, 8]
        if self.anno_type == 'rect' and anno.shape[1] == 8:
            anno = self._corner2rect(anno)

        return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _download(self, root_dir, version):
        assert version in self.__valid_versions

        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        elif os.path.isfile(os.path.join(root_dir, 'list.txt')):
            with open(os.path.join(root_dir, 'list.txt')) as f:
                seq_names = f.read().strip().split('\n')
            if all([os.path.isdir(os.path.join(root_dir, s)) for s in seq_names]):
                print('Files already downloaded.')
                return

        version_str = 'vot%d' % version
        url = 'http://data.votchallenge.net/%s/%s.zip' % (
            version_str, version_str)
        zip_file = os.path.join(root_dir, version_str + '.zip')

        print('Downloading to %s...' % zip_file)
        download(url, zip_file)
        print('\nExtracting to %s...' % root_dir)
        extract(zip_file, root_dir)

        return root_dir

    def _check_integrity(self, root_dir, version):
        assert version in self.__valid_versions
        list_file = os.path.join(root_dir, 'list.txt')

        if os.path.isfile(list_file):
            with open(list_file, 'r') as f:
                seq_names = f.read().strip().split('\n')

            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exist.' % seq_name)
        else:
            # dataset not exist
            raise Exception('Dataset not found or corrupted. ' +
                            'You can use download=True to download it.')

    def _corner2rect(self, corners, center=False):
        cx = np.mean(corners[:, 0::2], axis=1)
        cy = np.mean(corners[:, 1::2], axis=1)

        x1 = np.min(corners[:, 0::2], axis=1)
        x2 = np.max(corners[:, 0::2], axis=1)
        y1 = np.min(corners[:, 1::2], axis=1)
        y2 = np.max(corners[:, 1::2], axis=1)

        area1 = np.linalg.norm(corners[:, 0:2] - corners[:, 2:4], axis=1) * \
            np.linalg.norm(corners[:, 2:4] - corners[:, 4:6], axis=1)
        area2 = (x2 - x1) * (y2 - y1)
        scale = np.sqrt(area1 / area2)
        w = scale * (x2 - x1) + 1
        h = scale * (y2 - y1) + 1

        if center:
            return np.array([cx, cy, w, h]).T
        else:
            return np.array([cx - w / 2, cy - h / 2, w, h]).T
