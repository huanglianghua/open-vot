from __future__ import absolute_import, print_function, division

import os
import glob
import numpy as np
import six

from ..utils.ioutil import download, extract


class VOT(object):

    def __init__(self, root_dir, return_rect=False,
                 download=False, version=2017):
        super(VOT, self).__init__()
        self.root_dir = root_dir
        self.return_rect = return_rect
        if download:
            self._download(self.root_dir, version)

        if not self._check_integrity():
            raise Exception('Dataset not found or corrupted. ' +
                            'You can use download=True to download it.')

        self.anno_files = sorted(glob.glob(
            os.path.join(root_dir, '*/groundtruth.txt')))
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(s) for s in self.seq_dirs]

    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], '*.jpg')))
        anno = np.loadtxt(self.anno_files[index], delimiter=',')
        if self.return_rect and anno.shape[1] == 8:
            anno = self._corner2rect(anno)

        return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir=None):
        if not root_dir:
            root_dir = self.root_dir
        return os.path.isdir(root_dir) and \
            len(os.listdir(root_dir)) > 0

    def _download(self, root_dir, version):
        if self._check_integrity(root_dir):
            print('Files already downloaded.')
            return
        assert version in range(2013, 2017 + 1), 'Incorrect VOT version.'

        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        version = 'vot%d' % version
        url = 'http://data.votchallenge.net/%s/%s.zip' % (version, version)
        zip_file = os.path.join(root_dir, version + '.zip')

        download(url, zip_file)
        extract(zip_file, root_dir)

        return root_dir

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
            return np.array([cx-w/2, cy-h/2, w, h]).T
