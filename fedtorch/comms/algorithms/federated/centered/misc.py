# -*- coding: utf-8 -*-
import numpy as np
import torch

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

def set_online_clients_centered(cfg):
    # Define online clients for the current round of communication for Federated Learning setting
    ranks_shuffled = np.random.permutation(cfg.graph.ranks)
    online_clients = ranks_shuffled[:int(cfg.federated.online_client_rate * len(cfg.graph.ranks))]
    return list(online_clients)

def robust_noise_average(OnlineClients, Server, online_clients, lambda_weight=None):
    num_online_clients = len(online_clients)
    if lambda_weight is None:
        # rank_weight =  OnlineClient.cfg.num_samples_per_epoch / OnlineClient.cfg.train_dataset_size
        rank_weight =  1.0 / num_online_clients
    else:
        #TODO: This is experimental. Test it.
        rank_weight = lambda_weight * Server.cfg.graph.n_nodes / num_online_clients
    
    Server.avg_noise_optimizer.zero_grad(set_to_none=False)

    for o in online_clients:
        for server_param, client_param in zip(Server.avg_noise_model.parameters(), OnlineClients[o].model.parameters()):
            # get model difference.
            server_param.grad.data  += (server_param.data - client_param.data) * rank_weight
            
    Server.avg_noise_optimizer.step(
        apply_lr=False,
        scale=Server.cfg.lr.lr_scale_at_sync,
        apply_in_momentum=False,
        apply_out_momentum=Server.cfg.training.out_momentum,
    )

    return


def calc_clients_coefficient_centered(Clients, Server):
    """
    This function calculates the coefficient of correlation between the gradients of 
    clients in a federated learning setting.

    Arguments:
    Clients (dict): A dictionary of clients, where each key represents a client and its
                    value is an object that contains the train_loader.
    Server (object): An object that contains the model and the optimizer.

    Returns:
    corrs (float): The average coefficient of correlation between the gradients of clients.
    """
    Server.model.train()
    grads = {k:[] for k,_ in Server.model.named_parameters()}
    corrs = 0
    for c in Clients.keys():
        Server.optimizer.zero_grad(set_to_none=False)
        loss = torch.tensor([0.0])
        if Server.cfg.graph.on_cuda:
            loss=loss.cuda()
        for _input, _target in Clients[c].train_loader:
            if _input.size(0)==1:
                continue
            if Server.cfg.graph.on_cuda:
                _input, _target = _input.cuda(), _target.cuda()
            output = Server.model(_input)
            loss += Server.criterion(output, _target)
        loss.backward()
        for np, server_param in  Server.model.named_parameters():
            grads[np].append(server_param.grad.data.flatten().clone())
    count = 0
    for np, server_param in  Server.model.named_parameters():
        grads[np] = torch.stack(grads[np])
        g_corr = corr(grads[np])
        if g_corr[0,0] < 0.99:
            continue
        corrs += g_corr
        count += 1
    return corrs / count
    


def corr(X, eps=1e-08):
    D = X.shape[-1]
    std = torch.std(X, dim=-1).unsqueeze(-1)
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = (X - mean) / (std + eps)
    return 1/(D-1) * X @ X.transpose(-1, -2)


def fedshuffle_aggregation_centered(OnlineClients, Server, online_clients):
    """Aggregate gradients for federated learning using FedAvg algorithm.

    Each local model first gets the difference between current model and
    previous synchronized model, and then all-reduce these difference by SUM.

    """
    if not (Server.shuffle_list == torch.arange(Server.cfg.graph.n_nodes)).all():
        for o in online_clients:
            model_loc = Server.shuffle_list[o].item()
            weight = Server.alphas[o,model_loc]
            OnlineClients[o].optimizer_personal.zero_grad()
            if weight < 0:
                continue
            for i, (model_param, personal_param) in enumerate(zip(OnlineClients[model_loc].model.parameters(), OnlineClients[o].model_personal.parameters())):
                # get model difference.
                # personal_param.data = model_param.data.clone()
                param_diff = (personal_param.data - model_param.data) * weight

                personal_param.grad.data.add_(param_diff)

            OnlineClients[o].optimizer_personal.step(
                apply_lr=False,
                scale=Server.cfg.lr.lr_scale_at_sync,
                apply_in_momentum=False,
                apply_out_momentum=Server.cfg.training.out_momentum,
            )
    Server.shuffle_list = Server.shuffle_list.roll(1)
    return 