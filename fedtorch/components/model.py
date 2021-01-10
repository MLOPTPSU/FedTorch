# -*- coding: utf-8 -*-
import torch.distributed as dist

import fedtorch.components.models as models


def define_model(args):
    if args.graph.rank % 100 == 0:
        print("=> creating model '{}' for rank {}/{}".format(args.arch,args.graph.rank,args.graph.n_nodes))
    if 'wideresnet' in args.arch:
        model = models.__dict__['wideresnet'](args)
    elif 'resnet' in args.arch:
        model = models.__dict__['resnet'](args)
    elif 'densenet' in args.arch:
        model = models.__dict__['densenet'](args)
    else:
        model = models.__dict__[args.arch](args)

    if args.is_distributed:
        consistent_model(args, model)
    if args.debug:
        get_model_stat(args, model)
    return model


def get_model_stat(args, model):
    print('Total params for process {}: {}M'.format(
        args.graph.rank,
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        ))


def consistent_model(args, model):
    """it might because of MPI, the model for each process is not the same.

    This function is proposed to fix this issue,
    i.e., use the  model (rank=0) as the global model.
    """
    print('consistent model for process (rank {})'.format(args.graph.rank))
    cur_rank = args.graph.rank
    for param in model.parameters():
        param.data = param.data if cur_rank == 0 else param.data - param.data
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)