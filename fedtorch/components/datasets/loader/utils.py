# -*- coding: utf-8 -*-
#TODO: to be removed.
import os
import glob
import sys

import lmdb
# import cv2
import numpy as np
from PIL import Image

import torch.utils.data as data


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def be_ncwh_pt(x):
    return x.permute(0, 3, 1, 2)  # pytorch is (n,c,w,h)


def uint8_to_float(x):
    x = x.permute(0, 3, 1, 2)  # pytorch is (n,c,w,h)
    return x.float()/128. - 1.


class IMDBPT(data.Dataset):
    """
    Args:
        root (string): Either root directory for the database files,
            or a absolute path pointing to the file.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_train'].
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        target_transform (callable, optional):
            A function/transform that takes in the target and transforms it.
    """

    def __init__(self, root, transform=None, target_transform=None, is_image=True, pattern=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.pattern = pattern
        self.lmdb_files = self._get_valid_lmdb_files()

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for lmdb_file in self.lmdb_files:
            self.dbs.append(LMDBPTClass(
                root=lmdb_file, transform=transform,
                target_transform=target_transform, is_image=is_image))

        # build up indices.
        self.indices = np.cumsum([len(db) for db in self.dbs])
        self.length = self.indices[-1]
        self._get_index_zones = self._build_indices()

    def _get_valid_lmdb_files(self):
        """get valid lmdb based on given root."""
        if self.pattern:
            file_list = sorted(glob.glob(self.root))
            for f in file_list:
                print(f)
                yield f
        elif not self.root.endswith('.lmdb'):
            for l in os.listdir(self.root):
                if '_' in l and '-lock' not in l and '_cache_' not in l:
                    yield os.path.join(self.root, l)
        else:
            yield self.root

    def _build_indices(self):
        indices = self.indices
        indices = np.insert(indices,0,0)
        from_to_indices = list(zip(indices[: -1], indices[1:]))

        def f(x):
            if len(from_to_indices) <= 1:
                return 0, x

            for ind, (from_index, to_index) in enumerate(from_to_indices):
                if from_index <= x and x < to_index:
                    return ind, x - from_index
        return f

    def _get_matched_index(self, index):
        return self._get_index_zones(index)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target)
        """
        block_index, item_index = self._get_matched_index(index)
        image, target = self.dbs[block_index][item_index]
        image = image.copy()
        target = target.copy()
        return image, target

    def __len__(self):
        return self.length

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace(
                '\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class LMDBPTClass(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, is_image=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.is_image = is_image

        # open lmdb env.
        self.env = self._open_lmdb()

        # get file stats.
        self._get_length()

        # prepare cache_file
        self._prepare_cache()

    def _open_lmdb(self):
        return lmdb.open(
            self.root,
            subdir=os.path.isdir(self.root),
            readonly=True, lock=False, readahead=False,
            map_size=1099511627776 * 2,
            max_readers=1, meminit=False)

    def _get_length(self):
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

            if txn.get(b'__keys__') is not None:
                self.length -= 1

    def _prepare_cache(self):
        cache_file = self.root + '_cache_'
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key
                             for key, _ in txn.cursor() if key != b'__keys__']
            pickle.dump(self.keys, open(cache_file, "wb"))

    def _image_decode(self, x):
        # image = cv2.imdecode(x, cv2.IMREAD_COLOR).astype('uint8')
        # return Image.fromarray(image, 'RGB')
        return Image.fromarray(x, 'RGB')

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            bin_file = txn.get(self.keys[index])

        image, target = serialize.loads(bin_file)

        if self.is_image:
            image = self._image_decode(image)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'
