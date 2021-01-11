# -*- coding: utf-8 -*-
import os
from os.path import join
import pandas as pd
import urllib.request
import numpy as np
from numpy import loadtxt
import urllib
import torch
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset



def maybe_download_and_extract(root, data_url, file_path):
    if not os.path.exists(root):
        os.makedirs(root)

    file_name = data_url.split('/')[-1]

    if len([x for x in os.listdir(root) if x == file_name]) == 0:
        e = os.system('wget -t inf {} -O {}'.format(data_url, file_path))
        if e==1:
            urllib.request.urlretrieve(data_url,file_path)

class AdultDataset(object):
    def __init__(self, root):
        # get file url and file path.
        data_url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        data_url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
        file_path_train = os.path.join(root, data_url_train.split('/')[-1])
        file_path_test = os.path.join(root, data_url_test.split('/')[-1])

        # download dataset or not.
        maybe_download_and_extract(root, data_url_train, file_path_train)
        maybe_download_and_extract(root, data_url_test, file_path_test)

        # load dataset.
        dataset = self.load_data(root)
        self.features, self.labels, self.features_test,self.labels_test = dataset
        

    def __len__(self):
        return self.features.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        for k in idxs:
            yield [self.features[k], self.labels[k]]

    def get_data(self):
        return self.__iter__()

    def size(self):
        return self.__len__()

    def reset_state(self):
        """
        Reset state of the dataflow.
        It **has to** be called once and only once before producing datapoints.
        Note:
            1. If the dataflow is forked, each process will call this method
               before producing datapoints.
            2. The caller thread of this method must remain alive to keep this dataflow alive.
        For example, RNG **has to** be reset if used in the DataFlow,
        otherwise it won't work well with prefetching, because different
        processes will have the same RNG state.
        """
        pass

    def load_data(self, data_dir, scale = True, target_scale='biased'):
        data = pd.read_csv(
        os.path.join(data_dir, "adult.data"),
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
            )
        len_train = len(data.values[:, -1])
        data_test = pd.read_csv(
            os.path.join(data_dir, "adult.test"),
            names=[
                "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
                "hours per week", "native-country", "income"]
        )
        data_test.index += len_train -1
        data = pd.concat([data, data_test])
        # Considering the relative low portion of missing data, we discard rows with missing data
#         domanda = data["workclass"][4].values[1]
#         data = data[data["workclass"] != domanda]
#         data = data[data["occupation"] != domanda]
#         data = data[data["native-country"] != domanda]
        # Here we apply discretisation on column marital_status
        data.replace(['Divorced', 'Married-AF-spouse',
                    'Married-civ-spouse', 'Married-spouse-absent',
                    'Never-married', 'Separated', 'Widowed'],
                    ['not married', 'married', 'married', 'married',
                    'not married', 'not married', 'not married'], inplace=True)
        # categorical fields
        category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                        'relationship', 'gender', 'native-country']
        self.categories = {}
        data=data.dropna()
        for col in category_col:
            b, c = np.unique(data[col], return_inverse=True)
            data[col] = c
            self.categories[col] = dict(zip(b,list(set(c))))
        
        # ind0 = data.index[data['income'].isin([' <=50K.',' <=50K'])].tolist()
        ind1 = data.index[data['income'].isin([' >50K', ' >50K.'])].tolist()
        target = np.zeros(len(data))
        target[ind1] = 1.0
        self.categories['income'] = {' <=50K':0.0,' >50K':1.0, ' <=50K.':0.0,' >50K.':1.0 }

        datamat = data.values
        datamat = datamat[:, :-1]
        if scale:
            scaler = StandardScaler()
            scaler.fit(datamat)
            datamat = scaler.transform(datamat)
            for col in category_col:
                ind = data.columns.tolist().index(col)
                for cat in self.categories[col].keys():
                    test_array = np.zeros((1,datamat.shape[1]))
                    test_array[0,ind] = self.categories[col][cat]
                    trans_test = scaler.transform(test_array)
                    self.categories[col][cat] = trans_test[0,ind]
        # if target_scale == 'biased':
        #     target = (target + 1 )/2
        return datamat[:len_train], target[:len_train], datamat[len_train:], target[len_train:]

class AdultDatasetTorch(Dataset):
    def __init__(self, dataset, split):
        self.split = split
        if self.split == 'train':
            self.train_data = torch.tensor(dataset.features).float()
            self.train_labels = torch.tensor(dataset.labels).long()
        else:
            self.test_data = torch.tensor(dataset.features_test).float()
            self.test_labels = torch.tensor(dataset.labels_test).long()
        self.categories = dataset.categories
        self.features_name = [
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country"]

    def __len__(self):
        if self.split == 'train':
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]

    def __getitem__(self, idx):
        if self.split == 'train':
            return self.train_data[idx], self.train_labels[idx]
        else:
            return self.test_data[idx], self.test_labels[idx]