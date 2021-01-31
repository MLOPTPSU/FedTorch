# -*- coding: utf-8 -*-
import numpy as np

from fedtorch.comms.utils.flow_utils import zero_copy

def aggregate_kth_model_centered(OnlineClients, Server, online_clients):
    # This function is defined for DRFA algorithm.
    num_online_clients = len(online_clients)
    rank_weight =   1 / len(online_clients)
    Server.kth_model  = zero_copy(Server.model)
    for o in online_clients:
        for s_param,c_param in zip(Server.kth_model.parameters(),OnlineClients[o].kth_model.parameters()):
            c_param.data *= rank_weight
            # all reduce.
            s_param.data.add_(c_param.data)
    return

def set_online_clients_centered(args):
    # Define online clients for the current round of communication for Federated Learning setting
    ranks_shuffled = np.random.permutation(args.graph.ranks)
    online_clients = ranks_shuffled[:int(args.online_client_rate * len(args.graph.ranks))]
    return list(online_clients)
