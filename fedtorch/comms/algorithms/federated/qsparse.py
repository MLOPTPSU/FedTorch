# -*- coding: utf-8 -*-
import time

import torch
import torch.distributed as dist

from fedtorch.utils.auxiliary import deepcopy_model
from fedtorch.comms.utils.flow_utils import compress_tensor, decompress_tensor


def qsparse_aggregation(args, model_server, model_client, group, online_clients, optimizer, model_memory, lambda_weight=None):
    """Aggregate gradients for federated learning.

    Each local model first gets the difference between current model and
    previous synchronized model, and then all-reduce these difference by SUM.

    """
    num_online_clients = len(online_clients) if 0 in online_clients else len(online_clients) + 1
    if (0 not in online_clients) and (args.graph.rank == 0):
        rank_weight = 0
    else:
        if lambda_weight is None:
            rank_weight =  args.num_samples_per_epoch / args.train_dataset_size
            # rank_weight =  1.0 / num_online_clients
        else:
            #TODO: This is experimental. Test it.
            rank_weight = lambda_weight * args.graph.n_nodes / num_online_clients

    
    for server_param, client_param, memory_param in zip(model_server.parameters(), model_client.parameters(), model_memory.parameters()):
        # get model difference.
        client_param.grad.data = (server_param.data - client_param.data) * rank_weight
        # recover to old model.
        client_param.data = server_param.data

        g = client_param.grad.data + memory_param.data * rank_weight
        grad_v, grad_i, grad_s = compress_tensor(g, r=args.compressed_ratio, comp_type='topk')
        gather_list_value = [torch.ones_like(grad_v) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
        gather_list_index = [torch.ones_like(grad_i) for _ in range(num_online_clients)] if args.graph.rank == 0 else None
        
        # Gathering all the compressed gradients
        st = time.time()
        dist.gather(grad_i, gather_list=gather_list_index, dst=0, group=group)
        dist.gather(grad_v, gather_list=gather_list_value, dst=0, group=group)
        args.comm_time[-1] += time.time() - st
        if args.graph.rank == 0:
            gather_list_value = gather_list_value if 0 in online_clients else gather_list_value[1:]
            gather_list_index = gather_list_index if 0 in online_clients else gather_list_index[1:]
            gather_list_dec = [decompress_tensor(v,i, grad_s) for v,i in zip(gather_list_value,gather_list_index)]
            d = torch.sum(torch.stack(gather_list_dec,1), dim=1)
        else:
            d = torch.ones_like(g)
        
        st = time.time()
        dist.broadcast(d, src=0, group=group)
        args.comm_time[-1] += time.time() - st
        memory_param.data += client_param.grad.data/rank_weight - d 
        client_param.grad.data = d


    # apply gradient to each client's model
    optimizer.step(
        apply_lr=False,
        scale=args.lr_scale_at_sync,
        apply_in_momentum=False,
        apply_out_momentum=args.out_momentum,
    )

    # Reassign model_client to model_server
    model_server = deepcopy_model(args, model_client)
    return model_server