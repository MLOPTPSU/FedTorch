# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import warnings
import urllib.request
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from fedtorch.components.datasets.loader.utils import TqdmUpTo
_DATASET_MAP = {
    'epsilon_train': 'https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2',
    'epsilon_test': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2',
    'rcv1_train': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2',
    'rcv1_test': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2',
    'url': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined.bz2',
    'higgs_train': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.bz2',
    'higgs_test': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.bz2',
    'MSD_train': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2',
    'MSD_test': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2'
}

class LibSVMDataset(object):
    @property
    def features(self):
        warnings.warn("features has been renamed data")
        return self.data
    @property
    def labels(self):
        warnings.warn("labels has been renamed targets")
        return self.targets
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data
    def __init__(self, root, name, split):
        self.root = root
        self.name = name
        self.split = split
        # get file url and file path.
        if name == 'url':
            data_url = _DATASET_MAP['url']
        else:
            data_url = _DATASET_MAP['{}_{}'.format(self.name, self.split)]
        raw_file_path = os.path.join(root, data_url.split('/')[-1])
        file_path = os.path.join(root, '{}_{}.pt'.format(name,split))

        # # download dataset or not.
        self.download(root, data_url, raw_file_path, file_path)

        # load dataset.
        self.data, self.targets = torch.load(file_path)

    def _get_images_and_labels(self, data):
        features, labels = data

        features = self._get_dense_tensor(features)
        labels = self._get_dense_tensor(labels)
        labels = self._correct_binary_labels(labels)
        return features, labels

    def _get_dense_tensor(self,tensor):
        if 'sparse' in str(type(tensor)):
            return tensor.todense()
        elif 'numpy' in str(type(tensor)):
            return tensor


    def _correct_binary_labels(self, labels, is_01_classes=True):
        classes = set(labels)
        if -1 in classes and is_01_classes:
            labels[labels == -1] = 0
        return labels

    def __len__(self):
        return self.features.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        for k in idxs:
            yield [self.features[k], self.labels[k]]
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def get_data(self):
        return self.__iter__()

    def size(self):
        return self.__len__()

    def _check_exists(self,path):
        return os.path.exists(path)

    def download(self,root, data_url, raw_file_path, file_path):
        """Download the LibSVM data if it doesn't exist in processed_folder already."""
        if self._check_exists(file_path):
            return
        if not self._check_exists(raw_file_path):
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=raw_file_path) as t:
                urllib.request.urlretrieve(data_url, raw_file_path, reporthook=t.update_to, data=None)
        dataset = load_svmlight_file(raw_file_path)
        features, labels = self._get_images_and_labels(dataset)
        if self.name == "MSD":
            features, labels = self.normalize(features, labels, self.name, root, self.split)
        
        dataset = (features,labels)
        with open(file_path, 'wb') as f:
            torch.save(dataset,f)


    def normalize(self, features, labels, name, root, split):
        scaler = StandardScaler()
        if split == 'test':
            train_dataset = LibSVMDataset(root,name,'train')
            train_features = train_dataset.data
            train_labels = train_dataset.labels
            scaler.fit(train_features)
            label_min = np.min(train_labels)
            label_max = np.max(train_labels)
            del train_dataset
        else:
            scaler.fit(features)
            label_min = np.min(labels)
            label_max = np.max(labels)
        features = scaler.transform(features)
        labels = (labels - label_min)/(label_max - label_min)
        return features, labels