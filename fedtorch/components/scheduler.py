# -*- coding: utf-8 -*-
from fedtorch.components.optimizers.learning import get_lr_scheduler


def define_scheduler(args):
    return define_lr_scheduler(args)


def adjust_learning_rate(args, optimizer, lr_scheduler, lr_external=None):
    """Sets the learning rate to the initial LR decayed by # of accessed sample
        We should decay the learning rate based on the number of samples that
        we have accessed.
    """
    # adjust and assign learning rate.
    if lr_external is None:
        lr = lr_scheduler(args.epoch_)

        if lr is None:
            lr = args.old_learning_rate

        if args.old_learning_rate != lr:
            args.old_learning_rate = lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_external
        lr = lr_external
    return lr


def define_lr_scheduler(args):
    # get the learning rate per sample.
    # TODO: Resolve confilict: base_batch_size confused with base_batch_size for growing_batch_size
    args.learning_rate_per_samples = args.lr / args.batch_size

    # get a valid learning rate.
    args.init_warmup_lr = args.lr

    if args.lr_scaleup:
        if args.lr_scaleup_type == 'linear':
            _lr = args.learning_rate_per_samples * args.batch_size
            _scale = args.graph.n_nodes
        elif args.lr_scaleup_type == 'sqrt':
            _lr = args.lr
            _scale = (
                1. * args.graph.n_nodes * args.batch_size /
                args.base_batch_size) ** 0.5
        else:
            raise NotImplementedError
        args.learning_rate = _lr * _scale
    else:
        _lr = args.learning_rate_per_samples * args.batch_size
        _scale = 1
    args.learning_rate = _lr * _scale

    # just backup the current learning rate.
    args.old_learning_rate = args.learning_rate

    # define the learning rate scheduler.
    lr_scheduler = get_lr_scheduler(args)
    return lr_scheduler
