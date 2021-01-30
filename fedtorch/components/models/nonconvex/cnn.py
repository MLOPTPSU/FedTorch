# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F


__all__ = ['cnn']

class CNN(nn.Module):
    def __init__(self,dataset):
        super(CNN, self).__init__()
        self.dataset = dataset
        self.num_classes = self._decide_num_classes()
        self.num_channels = self._decide_num_channels()
        self.conv1 = nn.Conv2d(self.num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.rep_out_dim = self._decide_output_representation_size()
        self.fc1 = nn.Linear(self.rep_out_dim, 512)
        self.fc2 = nn.Linear(512, self.num_classes)



    def _decide_num_channels(self):
        if self.dataset in ['cifar10', 'cifar100']:
            return 3
        elif self.dataset in ['mnist','fashion_mnist','emnist', 'emnist_full']:
            return 1

    def _decide_num_classes(self):
        if self.dataset in ['cifar10', 'mnist', 'fashion_mnist','emnist']:
            return 10
        elif self.dataset == 'cifar100':
            return 100
        elif self.dataset == 'emnist_full':
            return 62

    def _decide_input_feature_size(self):
        if 'mnist' in self.dataset:
            return 28 * 28
        elif 'cifar' in self.dataset:
            return  32 * 32 * 3
        else:
            raise NotImplementedError
    
    def _decide_output_representation_size(self):
        if 'mnist' in self.dataset:
            return 4 * 4 * 50
        elif 'cifar' in self.dataset:
            return  5 * 5 * 50
        else:
            raise NotImplementedError


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.rep_out_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def cnn(args):
    return CNN(args.data)