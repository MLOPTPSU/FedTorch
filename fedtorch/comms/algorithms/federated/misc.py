# -*- coding: utf-8 -*-
import time
import numpy as np

import torch
import torch.distributed as dist

from fedtorch.utils.auxiliary import deepcopy_model

def set_online_clients(args):
    # Define online clients for the current round of communication for Federated Learning setting
    useable_ranks = args.graph.ranks
    ranks_shuffled = np.random.permutation(useable_ranks)
    online_clients = ranks_shuffled[:int(args.online_client_rate * len(useable_ranks))]

    online_clients = torch.IntTensor(online_clients)
    group = dist.new_group(args.graph.ranks)
    dist.broadcast(online_clients, src=0, group=group)
    return list(online_clients.numpy())

def distribute_model_server(model_server, group, src=0):
    """
    Distributing the model on server from source node to the group of process
    """
    for server_param in model_server.parameters():
        dist.broadcast(server_param.data, src=src, group=group)

    return model_server

def set_online_clients_drfa(args, lambda_vector):
    # Define online clients for the current round of communication for Federated Learning setting
    online_clients = np.random.choice(np.arange(args.graph.n_nodes), size=int(args.online_client_rate * args.graph.n_nodes), replace=False, p=lambda_vector)

    online_clients = torch.IntTensor(online_clients)
    group = dist.new_group(args.graph.ranks)
    dist.broadcast(online_clients, src=0, group=group)
    return list(online_clients.numpy())

def aggregate_models_virtual(args, model, group, online_clients):
    virtual_model = deepcopy_model(args, model)
    # rank_weight =  args.num_samples_per_epoch / args.train_dataset_size
    if (0 not in online_clients) and (args.graph.rank == 0):
        rank_weight = 0
    else:
        rank_weight =   1 / len(online_clients)
    for param in virtual_model.parameters():
        param.data *= rank_weight
        # all reduce.
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=group)
        # if or not averge the model.
        
    return virtual_model

def loss_gather(args, loss, group, online_clients):
    num_online_clients = len(online_clients) if 0 in online_clients else len(online_clients) + 1
    gather_list = [torch.tensor(0.0) for _ in range(num_online_clients)]
    if args.graph.rank == 0:
        st = time.time()
        dist.gather(loss, gather_list=gather_list, dst=0, group=group)
        args.comm_time[-1] += time.time() - st
    else:
        dist.gather(loss, dst=0, group=group)
    return torch.stack(gather_list)