# -*- coding: utf-8 -*-
import time

import torch
import torch.distributed as dist

from fedtorch.utils.auxiliary import deepcopy_model

def afl_aggregation(args, model_server, model_client, lambda_weight, loss, group, online_clients, optimizer):
    
    if (0 not in online_clients) and (args.graph.rank == 0):
        rank_weight = 0
    else:
        rank_weight =  lambda_weight
    num_online_clients = len(online_clients) if 0 in online_clients else len(online_clients) + 1
    for i, (server_param, client_param) in enumerate(zip(model_server.parameters(), model_client.parameters())):
        # get model difference.
        client_param.grad.data = (server_param.data - client_param.data) * rank_weight
        # recover to old model.
        client_param.data = server_param.data

        # all reduce.This schema is not used in real federated setting
        # dist.all_reduce(client_param.grad.data,  op=dist.ReduceOp.SUM, group=group)

        gather_list = [torch.ones_like(client_param.grad.data) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
        st = time.time()
        dist.gather(client_param.grad.data, gather_list=gather_list, dst=0, group=group)
        args.comm_time[-1] += time.time() - st
        if args.graph.rank == 0:
            gather_list = gather_list if 0 in online_clients else gather_list[1:]
            d = torch.sum(torch.stack(gather_list,1), dim=1)
        else:
            d = torch.ones_like(client_param.grad.data)
        st = time.time()
        dist.broadcast(d, src=0, group=group)
        args.comm_time[-1] += time.time() - st
        client_param.grad.data = d

    # Gathering loss values
    gather_list_loss = [torch.tensor(0.0) for _ in range(num_online_clients)]
    if args.graph.rank == 0:
        st = time.time()
        dist.gather(loss, gather_list=gather_list_loss, dst=0, group=group)
        args.comm_time[-1] += time.time() - st
    else:
        dist.gather(loss, dst=0, group=group)
    loss_tensor_online = torch.stack(gather_list_loss)


    # apply gradient to each client's model
    optimizer.step(
        apply_lr=False,
        scale=args.lr_scale_at_sync,
        apply_in_momentum=False,
        apply_out_momentum=args.out_momentum,
    )

    # Reassign model_client to model_server
    model_server = deepcopy_model(args, model_client)

    return model_server, loss_tensor_online