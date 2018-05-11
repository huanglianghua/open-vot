from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import io
import six

from ..utils.ioutil import download, extract


class OTB(object):

    __otb50_seqs = ['BasketBall', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2',
                    'BlurFace', 'BlurOwl', 'Bolt', 'Box', 'Car1', 'Car4', 'CarDark',
                    'CarScale', 'ClifBar', 'Couple', 'Crowds', 'David', 'Deer',
                    'Diving', 'DragonBaby', 'Dudek', 'Football', 'Freeman4',
                    'Girl', 'Human3', 'Human4-2', 'Human6', 'Human9', 'Ironman',
                    'Jump', 'Jumping', 'Liquor', 'Matrix', 'MotorRolling', 'Panda',
                    'RedTeam', 'Shaking', 'Singer2', 'Skating1', 'Skating2-1',
                    'Skating2-2', 'Skiing', 'Soccer', 'Surfer', 'Sylvester',
                    'Tiger2', 'Trellis', 'Walking', 'Walking2', 'Woman']

    __otb100_seqs = ['Basketball', 'Biker', 'Bird1', 'Bird2', 'BlurBody',
                     'BlurCar1', 'BlurCar2', 'BlurCar3', 'BlurCar4', 'BlurFace',
                     'BlurOwl', 'Board', 'Bolt', 'Bolt2', 'Box', 'Boy', 'Car1',
                     'Car2', 'Car24', 'Car4', 'CarDark', 'CarScale', 'ClifBar',
                     'Coke', 'Couple', 'Coupon', 'Crossing', 'Crowds', 'Dancer',
                     'Dancer2', 'David', 'David2', 'David3', 'Deer', 'Diving',
                     'Dog', 'Dog1', 'Doll', 'DragonBaby', 'Dudek', 'FaceOcc1',
                     'FaceOcc2', 'Fish', 'FleetFace', 'Football', 'Football1',
                     'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Girl2', 'Gym',
                     'Human2', 'Human3', 'Human4.2', 'Human5', 'Human6', 'Human7',
                     'Human8', 'Human9', 'Ironman', 'Jogging.1', 'Jogging.2',
                     'Jump', 'Jumping', 'KiteSurf', 'Lemming', 'Liquor', 'Man',
                     'Matrix', 'Mhyang', 'MotorRolling', 'MountainBike', 'Panda',
                     'RedTeam', 'Rubik', 'Shaking', 'Singer1', 'Singer2',
                     'Skater', 'Skater2', 'Skating1', 'Skating2.1', 'Skating2.2',
                     'Skiing', 'Soccer', 'Subway', 'Surfer', 'Suv', 'Sylvester',
                     'Tiger1', 'Tiger2', 'Toy', 'Trans', 'Trellis', 'Twinnings',
                     'Vase', 'Walking', 'Walking2', 'Woman']

    def __init__(self, root_dir, download=False, version=2015):
        super(OTB, self).__init__()
        self.root_dir = root_dir
        if download:
            self._download(self.root_dir, version)

        if not self._check_integrity():
            raise Exception('Dataset not found or corrupted. ' +
                            'You can use download=True to download it.')

        self.anno_files = sorted(glob.glob(
            os.path.join(root_dir, '*/groundtruth_rect.txt')))
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(s) for s in self.seq_dirs]

    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], 'img/*.jpg')))

        # special sequences
        seq_name = self.seq_names[index]
        if seq_name.lower() == 'david':
            img_files = img_files[300-1:770]
        elif seq_name.lower() == 'football1':
            img_files = img_files[:74]
        elif seq_name.lower() == 'freeman3':
            img_files = img_files[:460]
        elif seq_name.lower() == 'freeman4':
            img_files = img_files[:283]
        elif seq_name.lower() == 'diving':
            img_files = img_files[:215]

        # to deal with multiple delimeters
        with open(self.anno_files[index], 'r') as f:
            anno = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))

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
        assert version in [2013, 2015]

        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        url_fmt = 'http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/%s.zip'
        seqs = self.__otb50_seqs if version == 2013 else self.__otb100_seqs
        for seq in seqs:
            url = url_fmt % seq
            zip_file = os.path.join(root_dir, seq + '.zip')
            download(url, zip_file)
            extract(zip_file, root_dir)

        return root_dir
