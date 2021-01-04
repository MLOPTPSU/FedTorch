# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.distributed as dist

def set_online_clients(args):
    # Define online clients for the current round of communication for Federated Learning setting
    useable_ranks = args.graph.ranks
    if args.fed_meta:
        # For this case, we want to reserve number of clients offline to join the training after it is finished for others
        useable_ranks = useable_ranks[:-3]
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