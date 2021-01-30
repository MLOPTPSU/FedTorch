# -*- coding: utf-8 -*-
import torch.nn as nn


__all__ = ['mlp']


class MLP(nn.Module):
    def __init__(self, dataset, num_layers, hidden_size, drop_rate):
        super(MLP, self).__init__()
        self.dataset = dataset

        # init
        self.num_layers = num_layers
        self.num_classes = self._decide_num_classes()
        input_size = self._decide_input_feature_size()

        # define layers.
        for i in range(1, self.num_layers + 1):
            in_features = input_size if i == 1 else hidden_size
            out_features = hidden_size

            layer = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features,track_running_stats=False),
                nn.ReLU(),
                nn.Dropout(p=drop_rate))
            setattr(self, 'layer{}'.format(i), layer)

        self.fc = nn.Linear(hidden_size, self.num_classes, bias=False)

    def _decide_num_classes(self):
        if self.dataset in ['cifar10', 'mnist','fashion_mnist','emnist']:
            return 10
        elif self.dataset == 'cifar100':
            return 100
        elif self.dataset == 'emnist_full':
            return 62
        elif self.dataset == 'adult':
            return 2

    def _decide_input_feature_size(self):
        if 'cifar' in self.dataset:
            return 32 * 32 * 3
        elif 'mnist' in  self.dataset:
            return 28 * 28
        elif self.dataset=='adult':
            return 14
        else:
            raise NotImplementedError

    def forward(self, x):
        out = x.view(x.size(0), -1)

        for i in range(1, self.num_layers + 1):
            out = getattr(self, 'layer{}'.format(i))(out)
        out = self.fc(out)
        return out


def mlp(args):
    return MLP(
        dataset=args.data, num_layers=args.mlp_num_layers,
        hidden_size=args.mlp_hidden_size, drop_rate=args.drop_rate)
