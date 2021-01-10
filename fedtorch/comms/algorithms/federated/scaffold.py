# -*- coding: utf-8 -*-
import time
from copy import deepcopy

import torch
import torch.distributed as dist

from fedtorch.utils.auxiliary import deepcopy_model

def scaffold_aggregation(args, model_server, model_client, model_server_control, model_client_control,
                         group, online_clients, optimizer, lr, local_steps, lambda_weight=None):
    """Aggregate gradients for federated learning using SCAFFOLD.
    https://arxiv.org/abs/1910.06378
    """
    model_client_control_copy = deepcopy(model_client_control)
    num_online_clients = len(online_clients) if 0 in online_clients else len(online_clients) + 1
    if (0 not in online_clients) and (args.graph.rank == 0):
        rank_weight = 0
    else:
        if lambda_weight is None:
            rank_weight =  1.0 / num_online_clients
        else:
            #TODO: This is experimental. Test it.
            rank_weight = lambda_weight * args.graph.n_nodes / num_online_clients
        # Update local control variates for online clients only
        for cccp, ccp, scp, cp, sp in zip(model_client_control_copy.parameters(), model_client_control.parameters(), model_server_control.parameters(), model_client.parameters(), model_server.parameters()):
            cccp.data = ccp.data - scp.data + (sp.data - cp.data)/(local_steps * lr)

    
    for cccp, ccp, scp, cp, sp in zip(model_client_control_copy.parameters(), model_client_control.parameters(), model_server_control.parameters(), model_client.parameters(), model_server.parameters()):
        # get model difference.
        cp.grad.data = (sp.data - cp.data) * rank_weight
        # recover to old model.
        cp.data = sp.data
        # Control variate change
        ccp.data = (ccp.data - cccp.data) * rank_weight

        t = torch.stack([cp.grad.data, ccp.data])

        # all reduce. This schema is not used in real federated setting
        # dist.all_reduce(t ,  op=dist.ReduceOp.SUM, group=group)

        #### Federated communication #####
        gather_list = [torch.ones_like(t) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
        
        st = time.time()
        dist.gather(t, gather_list=gather_list, dst=0, group=group)
        args.comm_time[-1] += time.time() - st
        if args.graph.rank == 0:
            gather_list = gather_list if 0 in online_clients else gather_list[1:]
            d = torch.sum(torch.stack(gather_list,1), dim=1)
        else:
            d = torch.ones_like(t)
        st = time.time()
        dist.broadcast(d, src=0, group=group)
        args.comm_time[-1] += time.time() - st
        #####################################

        cp.grad.data = d[0]
        # Update server control variate
        scp.data -= d[1] * (len(online_clients) / args.num_workers)
        cp.grad.data = t[0]
        # Update server control variate
        scp.data -= t[1] * (len(online_clients) / args.num_workers)
    
    # apply gradient to each client's model
    optimizer.step(
        apply_lr=False,
        scale=args.lr_scale_at_sync,
        apply_in_momentum=False,
        apply_out_momentum=args.out_momentum,
    )

    # Reassign model_client to model_server
    model_server = deepcopy_model(args, model_client)

    # Reassing control variates
    model_client_control = deepcopy_model(args, model_client_control_copy)
    return model_server, model_client_control, model_server_control


def distribute_model_server_control(model_server, model_server_control, group, src=0):
    """
    Distributing the model on server from source node to the group of process
    #TODO: merge with distribute_model_server method
    """
    for server_param, server_control_param in zip(model_server.parameters(),model_server_control.parameters()):
        t = torch.stack([server_param.data, server_control_param.data])
        dist.broadcast(t, src=src, group=group)
        server_param.data = t[0]
        server_control_param.data = t[1]


    return model_server, model_server_control