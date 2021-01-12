# -*- coding: utf-8 -*-
import torch

from fedtorch.comms.algorithms.distributed import global_average
from fedtorch.logs.meter import AverageMeter


def define_metrics(args, model):
    if 'least_square' not in args.arch:
        if args.arch not in ['rnn']:
            if model.num_classes >= 5:
                return (1, 5)
            else:
                return (1,)
        else:
            return (1,)
    else:
        return ()


class TopKAccuracy(object):
    def __init__(self, topk=1):
        self.topk = topk
        self.reset()

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        batch_size = target.size(0)

        _, pred = output.topk(self.topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topk].view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size)

    def reset(self):
        self.top = AverageMeter()

    def update(self, prec, size):
        self.top.update(prec, size)

    def average(self):
        return global_average(self.top.sum, self.top.count)

    @property
    def name(self):
        return "Prec@{}".format(self.topk)


def accuracy(output, target, topk=(1,), rnn=False):
    """Computes the precision@k for the specified values of k"""
    res = []

    if not rnn:
        if len(topk) > 0:
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
        else:
            res += [0]
    else:
        batch_size = target.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).float().mean()
        res.append(correct.mul_(100.0).item())
    return res



def accuracy_per_class(output,target,classes):
    # batch_size = target.size(0)
    acc = torch.zeros_like(classes).float()
    count = torch.zeros_like(classes).float()
    _ , pred = torch.max(output,1)
    target = torch.squeeze(target)
    correct = pred.eq(target)
    for i,c in enumerate(classes):
        c_inds = target == c
        count[i] = c_inds.float().sum()
        if count[i] == 0:
            acc[i] = 0.0
        else:
            acc[i] = (c_inds & correct).float().sum().mul_(100.0/count[i])
    return acc, count

def recall_per_group(output):
    pass

