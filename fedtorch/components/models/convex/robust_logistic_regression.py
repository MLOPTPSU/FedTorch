# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


__all__ = ['robust_logistic_regression']


class RobustLogisticRegression(torch.nn.Module):

    def __init__(self, dataset):
        super(RobustLogisticRegression, self).__init__()
        self.dataset = dataset

        # get input and output dim.
        self._determine_problem_dims()

        self.noise = torch.nn.Parameter(torch.randn(self.num_features)*0.001, requires_grad=True)
        # define layers.
        self.fc = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_classes, bias=True)

        self._weight_initialization()

    def forward(self, x):
        # We don't need the softmax layer here since CrossEntropyLoss already
        # uses it internally.
        if self.dataset in ['mnist', 'cifar10', 'cifar100', 'fashion_mnist','emnist', 'emnist_full']:
            x = x.view(-1, self.num_features)

        x = self.fc(x + self.noise)
        return x

    def _determine_problem_dims(self):
        if self.dataset == 'epsilon':
            self.num_features = 2000
            self.num_classes = 2
        elif self.dataset == 'url':
            self.num_features = 3231961
            self.num_classes = 2
        elif self.dataset == 'rcv1':
            self.num_features = 47236
            self.num_classes = 2
        elif self.dataset == 'higgs':
            self.num_features = 28
            self.num_classes = 2
        elif self.dataset == 'mnist':
            self.num_features = 784
            self.num_classes = 10
        elif self.dataset == 'emnist':
            self.num_features = 784
            self.num_classes = 10
        elif self.dataset == 'emnist_full':
            self.num_features = 784
            self.num_classes = 62
        elif self.dataset == 'cifar10':
            self.num_features = 3072
            self.num_classes = 10
        elif self.dataset == 'cifar100':
            self.num_features = 3072
            self.num_classes = 100
        elif self.dataset == 'fashion_mnist':
            self.num_features = 784
            self.num_classes = 10
        elif self.dataset == 'synthetic':
            self.num_features = 60
            self.num_classes = 10
        elif self.dataset == 'adult':
            self.num_features = 12
            self.num_classes = 2
        else:
            raise ValueError('convex methods only support epsilon, url, rcv1, higgs, synthetic, mnist, fashion_mnist, cifar, and adult for the moment')

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(mean=0, std=0.01)
                m.weight.data.zero_()
                m.bias.data.zero_()


def robust_logistic_regression(args):
    return RobustLogisticRegression(dataset=args.data)
