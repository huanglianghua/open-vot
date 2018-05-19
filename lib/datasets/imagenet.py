from __future__ import absolute_import, division

import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import six
import random
from torch.utils.data import Dataset
from PIL import Image


class ImageNetVID(object):

    def __init__(self, root_dir, return_rect=False,
                 subset='train', rand_choice=True, download=False):
        r'''TODO: make the track_id sampling deterministic
        '''
        super(ImageNetVID, self).__init__()
        self.root_dir = root_dir
        self.return_rect = return_rect
        self.rand_choice = rand_choice
        if download:
            self._download(self.root_dir)

        if not self._check_integrity():
            raise Exception('Dataset not found or corrupted. ' +
                            'You can use download=True to download it.')

        if subset == 'val':
            self.seq_dirs = sorted(glob.glob(os.path.join(
                self.root_dir, 'Data/VID/val/ILSVRC2015_val_*')))
            self.seq_names = [os.path.basename(s) for s in self.seq_dirs]
            self.anno_dirs = [os.path.join(
                self.root_dir, 'Annotations/VID/val', s) for s in self.seq_names]
        elif subset == 'train':
            self.seq_dirs = sorted(glob.glob(os.path.join(
                self.root_dir, 'Data/VID/train/ILSVRC*/ILSVRC*')))
            self.seq_names = [os.path.basename(s) for s in self.seq_dirs]
            self.anno_dirs = [os.path.join(
                self.root_dir, 'Annotations/VID/train',
                *s.split('/')[-2:]) for s in self.seq_dirs]
        else:
            raise Exception('Unknown subset.')

    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)
        elif self.rand_choice:
            index = np.random.randint(len(self.seq_names))

        anno_files = sorted(glob.glob(
            os.path.join(self.anno_dirs[index], '*.xml')))
        objects = [ET.ElementTree(file=f).findall('object')
                   for f in anno_files]

        # choose the track id randomly
        track_ids, counts = np.unique([obj.find(
            'trackid').text for group in objects for obj in group], return_counts=True)
        track_id = random.choice(track_ids[counts >= 2])

        frames = []
        anno = []
        for f, group in enumerate(objects):
            for obj in group:
                if not obj.find('trackid').text == track_id:
                    continue
                frames.append(f)
                anno.append([
                    int(obj.find('bndbox/xmin').text),
                    int(obj.find('bndbox/ymin').text),
                    int(obj.find('bndbox/xmax').text),
                    int(obj.find('bndbox/ymax').text)])

        img_files = [os.path.join(
            self.seq_dirs[index], '%06d.JPEG' % f) for f in frames]
        anno = np.array(anno)
        if self.return_rect:
            anno[:, 2:] = anno[:, 2:] - anno[:, :2] + 1

        return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self):
        return os.path.isdir(self.root_dir) and \
            len(os.listdir(self.root_dir)) > 0

    def _download(self, root_dir):
        raise NotImplementedError()


class ImageNetObject(Dataset):

    def __init__(self, root_dir, return_rect=False,
                 subset='train', download=False, transform=None):
        super(ImageNetObject, self).__init__()
        self.root_dir = root_dir
        self.return_rect = return_rect
        if download:
            self._download(self.root_dir)
        self.transform = transform

        if not self._check_integrity():
            raise Exception('Dataset not found or corrupted. ' +
                            'You can use download=True to download it.')

        if subset == 'val':
            self.img_dirs = [os.path.join(self.root_dir, 'ILSVRC2012_img_val')]
            self.anno_dirs = [os.path.join(
                self.root_dir, 'ILSVRC2012_bbox_val/val')]
        elif subset == 'train':
            self.img_dirs = sorted(glob.glob(os.path.join(
                self.root_dir, 'ILSVRC2012_img_train/n*')))
            self.anno_dirs = [os.path.join(
                self.root_dir, 'ILSVRC2012_bbox_train',
                os.path.basename(s)) for s in self.img_dirs]
        else:
            raise Exception('Unknown subset.')

        self.img_nums = [len(glob.glob(os.path.join(d, '*.xml')))
                         for d in self.anno_dirs]
        self.acc_nums = [sum(self.img_nums[:i + 1])
                         for i in range(len(self.img_nums))]
        self.size = sum(self.img_nums)

    def __getitem__(self, index):
        # locate the annotation file
        dir_id = np.argmax(np.array(self.acc_nums) > index)
        anno_files = sorted(
            glob.glob(os.path.join(self.anno_dirs[dir_id], '*.xml')))

        if dir_id == 0:
            anno_id = index
        else:
            anno_id = index - self.acc_nums[dir_id - 1]
        anno_file = anno_files[anno_id]
        img_file = os.path.join(
            self.img_dirs[dir_id],
            os.path.splitext(os.path.basename(anno_file))[0] + '.JPEG')

        # read annotations
        objects = ET.ElementTree(file=anno_file).findall('object')
        rand_object = random.choice(objects)

        bndbox = np.array([
            int(rand_object.find('bndbox/xmin').text),
            int(rand_object.find('bndbox/ymin').text),
            int(rand_object.find('bndbox/xmax').text),
            int(rand_object.find('bndbox/ymax').text)])
        if self.return_rect:
            bndbox[2:] = bndbox[2:] - bndbox[:2] + 1

        img = Image.open(img_file)
        if img.mode == 'L':
            img = img.convert('RGB')

        if self.transform:
            return self.transform(img, bndbox)
        else:
            return img_file, bndbox

    def __len__(self):
        return self.size

    def _check_integrity(self):
        return os.path.isdir(self.root_dir) and \
            len(os.listdir(self.root_dir)) > 0

    def _download(self):
        raise NotImplementedError()
