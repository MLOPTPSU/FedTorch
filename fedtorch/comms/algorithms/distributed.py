# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
import time

import torch
import torch.distributed as dist

from fedtorch.utils.auxiliary import deepcopy_model
from fedtorch.comms.utils.flow_utils import (quantize_tensor, 
                                             dequantize_tensor,
                                             size_tensor, 
                                             compress_tensor, 
                                             decompress_tensor)

"""the frequency of communication"""
def configure_sync_scheme(args):
    args.local_steps = define_sync_freq(
        num_epochs=args.num_epochs,
        local_step=args.local_step,
        local_step_warmup_type=args.local_step_warmup_type,
        local_step_warmup_period=args.local_step_warmup_period,
        turn_on_local_step_from=args.turn_on_local_step_from,
        turn_off_local_step_from=args.turn_off_local_step_from,
        warmup_per_intervals=args.local_step_warmup_per_interval,
        lr_change_epochs=args.lr_change_epochs)

def define_sync_freq(
        num_epochs, local_step, local_step_warmup_type,
        local_step_warmup_period,
        turn_on_local_step_from, turn_off_local_step_from,
        warmup_per_intervals, lr_change_epochs):
    # TODO: should figure out a better sync scheme.
    # directly return a list of local steps.
    num_epochs = num_epochs + 2
    if local_step_warmup_period is None:
        local_step_warmup_period = local_step

    # we need local step warmup.
    # determine the local step warmup scheme.
    if local_step_warmup_type is None:
        tmp_steps = [local_step] * local_step_warmup_period
    elif 'exp' in local_step_warmup_type:
        log_local_step = int(np.log2(local_step_warmup_period))
        tmp_steps = [
            2 ** int(ind * log_local_step / local_step_warmup_period)
            for ind in range(1, 1 + local_step_warmup_period)
        ]
    elif 'linear' in local_step_warmup_type:
        tmp_steps = [
            max(1, int(ind * local_step / local_step_warmup_period))
            for ind in range(1, 1 + local_step_warmup_period)
        ]
    elif 'constant' in local_step_warmup_type:
        tmp_steps = [1] * local_step_warmup_period
    else:
        raise NotImplementedError

    if len(tmp_steps) > num_epochs:
        tmp_steps = tmp_steps[: num_epochs]

    # get lr_change_epochs.
    if lr_change_epochs is not None:
        lr_change_epochs = [int(x) for x in lr_change_epochs.split(',')]
        lr_change_epochs = [0] + lr_change_epochs + [num_epochs]
        lr_change_fromto_epochs = list(
            zip(lr_change_epochs[: -1], lr_change_epochs[1:])
        )

    # determine if we want to repeat the local step warmup or not.
    if not warmup_per_intervals:
        steps = []

        # add some specific operators.
        if lr_change_epochs is None:
            # allowed to use local step warmup
            steps = tmp_steps + [local_step] * (num_epochs - len(tmp_steps))
        else:
            # allowed to use local step warmup
            if turn_on_local_step_from is None and turn_off_local_step_from is None:
                return tmp_steps + [local_step] * (num_epochs - len(tmp_steps))

            # not allowed to use local step warmup.
            for from_ind, to_ind in lr_change_fromto_epochs:
                if turn_on_local_step_from is None and turn_off_local_step_from is not None:
                    if from_ind >= turn_off_local_step_from:
                        steps += [1] * (to_ind - from_ind)
                    else:
                        t = [local_step] * (to_ind - from_ind)
                        steps += t
                elif turn_on_local_step_from is not None and turn_off_local_step_from is None:
                    if from_ind >= turn_on_local_step_from:
                        t = [local_step] * (to_ind - from_ind)
                        steps += t
                    else:
                        steps += [1] * (to_ind - from_ind)
                elif turn_on_local_step_from is not None and turn_off_local_step_from is not None:
                    raise NotImplementedError
    elif warmup_per_intervals:
        steps = []
        for from_ind, to_ind in lr_change_fromto_epochs:
            t = [local_step] * (to_ind - from_ind - len(tmp_steps))
            steps += tmp_steps + t
    else:
        raise NotImplementedError
    return steps

def aggregate_gradients(args, old_model, model, optimizer, is_sync):
    """Aggregate gradients.

    Each local model first gets the difference between current model and
    previous synchronized model, and then all-reduce these difference by SUM.

    The previous synchronized model could be either from block/global sync,
    and the all-reduce range (group), can also be determined by sync status.

    We have a flag, i.e., args.avg_model, to determine if we want to average
    these gradients/difference or simply sum them up.
    """
    for old_param, param in zip(old_model.parameters(), model.parameters()):
        # get model difference.
        param.grad.data = old_param.data - param.data
        # recover to old model.
        param.data = old_param.data
        # all reduce.
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        # if or not averge the model.
        if args.avg_model:
            param.grad.data /= float(args.graph.n_nodes)

    # apply gradient again.
    # Note that when use local SGD, is_global is True for each aggregation.
    optimizer.step(
        apply_lr=False,
        scale=args.lr_scale_at_sync,
        apply_in_momentum=False,
        apply_out_momentum=args.out_momentum,
    )

    # reassign model to old_model.
    old_model = deepcopy_model(args, model)
    return old_model


"""functions."""


def global_average(sum, count, group=None):
    def helper(array, group=None):
        array = torch.FloatTensor(array)
        if group is None:
            dist.all_reduce(array, op=dist.ReduceOp.SUM)
        else:
            dist.all_reduce(array, op=dist.ReduceOp.SUM, group=group)
        all_sum, all_count = array
        if all_count == 0:
            return 0
        else:
            return all_sum / all_count
    avg = helper([sum, count], group)
    return avg


def elementwise_min(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return tensor