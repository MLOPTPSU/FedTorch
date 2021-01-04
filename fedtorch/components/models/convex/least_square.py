# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


__all__ = ['least_square']


class Least_square(nn.Module):

    def __init__(self, dataset):
        super(Least_square, self).__init__()
        self.dataset = dataset

        # get input and output dim.
        self._determine_problem_dims()

        # define layers.
        self.fc = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_classes, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x

    def _determine_problem_dims(self):
        if self.dataset == 'epsilon':
            self.num_features = 2000
            self.num_classes = 1
        elif self.dataset == 'url':
            self.num_features = 3231961
            self.num_classes = 1
        elif self.dataset == 'rcv1':
            self.num_features = 47236
            self.num_classes = 1
        elif self.dataset == 'MSD':
            self.num_features = 90
            self.num_classes = 1
        else:
            raise ValueError('convex methods only support epsilon, url, YearPredictionMSD and rcv1 for the moment')

class LinearMAFL(nn.Module):
    def __init__(self, in_features,  middle_features, out_features=1):
        super(LinearMAFL, self).__init__()
        self.middle_features = middle_features
        self.in_features = in_features
        self.out_features = out_features
        # modules
        if self.in_features == 0:
            self.in_features = self.middle_features
        self.Z = nn.Linear(self.in_features, self.middle_features, bias=False)
        self.W = nn.Linear(self.middle_features, self.out_features, bias=True) 
        
        self.core_params = self.W.parameters()
        self.extra_params = self.Z.parameters()
    
    @property
    def weight(self):
        return torch.matmul(self.W.weight, self.Z.weight)
    @property
    def bias(self):
        return self.W.bias
    

    def forward(self, x):
        return self.W(self.Z(x))

def least_square(args):
    if args.federated_type != 'mafl':
        return Least_square(dataset=args.data)
    else:
        return LinearMAFL(in_features=args.input_dim, middle_features=args.mafl_server_dim)
