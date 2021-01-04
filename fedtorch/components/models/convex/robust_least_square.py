# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


__all__ = ['robust_least_square']


class Robust_Least_Square(torch.nn.Module):

    def __init__(self, dataset):
        super(Robust_Least_Square, self).__init__()
        self.dataset = dataset

        # get input and output dim.
        self._determine_problem_dims()

        self.noise = torch.nn.Parameter(torch.randn(self.num_features)*0.001, requires_grad=True)
        # define layers.
        self.fc = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_classes, bias=True)

    def forward(self, x):
        x = self.fc(x + self.noise)
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
        elif self.dataset == 'synthetic':
            self.num_features = 60
            self.num_classes = 1
        else:
            raise ValueError('convex methods only support epsilon, url, YearPredictionMSD and rcv1 for the moment')


def robust_least_square(args):
    return Robust_Least_Square(dataset=args.data)
