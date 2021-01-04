# -*- coding: utf-8 -*-

import torch.nn as nn


def define_criterion(args):
    if 'least_square' in args.arch:
        criterion = nn.MSELoss(reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    return criterion
