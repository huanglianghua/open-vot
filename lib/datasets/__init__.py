from __future__ import absolute_import


class VideoObjectDataset(object):

    def __init__(self, name):
        self.name = name

    def __getitem__(self, index):
        # typically returns paths and annotations of frames
        # i.e., `img_files, anno`
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


from .vot import VOT
from .otb import OTB
from .imagenet import ImageNetVID, ImageNetObject
from .pairwise import Pairwise
