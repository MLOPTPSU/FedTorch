# -*- coding: utf-8 -*-
from fedtorch.comms.utils.flow_utils import (quantize_tensor, 
                                             dequantize_tensor,
                                             compress_tensor,
                                             decompress_tensor)

def fedgate_aggregation_centered(OnlineClients, Server, online_clients, local_steps, lr, lambda_weight=None):
    """Aggregate gradients for variance reduced federated learning with decoupled learning rates.

    Each local model first gets the difference between current model and
    previous synchronized model, and then all-reduce these difference by SUM.
    """
    Server.optimizer.zero_grad()
    num_online_clients = len(online_clients)
    if lambda_weight is None:
        # rank_weight =  OnlineClient.args.num_samples_per_epoch / OnlineClient.args.train_dataset_size
        rank_weight =  1.0 / num_online_clients
    else:
        #TODO: This is experimental. Test it.
        rank_weight = lambda_weight * Server.args.graph.n_nodes / num_online_clients

    
    for o in online_clients:
        for server_param, client_param, memory_param in zip(Server.model.parameters(), OnlineClients[o].model.parameters(), OnlineClients[o].model_memory.parameters()):
            # get model difference.
            param_diff = (server_param.data - client_param.data) * rank_weight
            if Server.args.quantized:
                grad_q, q_info = quantize_tensor(param_diff, num_bits= Server.args.quantized_bits, adaptive=True)
                grad_deq = dequantize_tensor(grad_q, q_info)
                server_param.grad.data.add_(grad_deq)
            elif Server.args.compressed:
                g = param_diff + memory_param.data * rank_weight
                grad_v, grad_i, grad_s = compress_tensor(g, r=Server.args.compressed_ratio, comp_type='topk')
                grad_decomp = decompress_tensor(grad_v, grad_i, grad_s)
                server_param.grad.data.add_(grad_decomp)
            else:
                server_param.grad.data.add_(param_diff)
                

    for o in online_clients:
        for server_param, client_param, delta_param, memory_param in zip(Server.model.parameters(), 
                                                                         OnlineClients[o].model.parameters(),
                                                                         OnlineClients[o].model_delta.parameters(), 
                                                                         OnlineClients[o].model_memory.parameters()
                                                                         ):

            delta_param.data += (server_param.data - server_param.grad.data - client_param.data)/(lr*local_steps)
            if Server.args.compressed:
                memory_param.data += server_param.data - client_param.data - server_param.grad.data


    Server.optimizer.step(
        apply_lr=False,
        scale=Server.args.lr_scale_at_sync,
        apply_in_momentum=False,
        apply_out_momentum=Server.args.out_momentum,
    )
    return 