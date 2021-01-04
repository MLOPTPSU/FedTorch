# -*- coding: utf-8 -*-
import numpy as np

def qffl_aggregation_centered(OnlineClients, Server, online_clients, lr):
    """Aggregate gradients for federated learning.

    Each local model first gets the difference between current model and
    previous synchronized model, and then all-reduce these difference by SUM.

    """
    Server.optimizer.zero_grad()
    num_online_clients = len(online_clients)
    h = 0.0
    for o in online_clients:
        for i, (server_param, client_param) in enumerate(zip(Server.model.parameters(), OnlineClients[o].model.parameters())):
            # get model difference.
            param_diff = (server_param.data - client_param.data) * \
                            ( np.float_power(OnlineClients[o].full_loss + 1e-10, Server.args.qffl_q) / lr )
            server_param.grad.data.add_(param_diff)
            h += Server.args.qffl_q * np.float_power(OnlineClients[o].full_loss + 1e-10, Server.args.qffl_q-1.0) * \
                param_diff.norm().pow(2).item()
        h += np.float_power(OnlineClients[o].full_loss + 1e-10, Server.args.qffl_q) / lr

    for server_param in Server.model.parameters():
        server_param.grad.data.div_(h+1e-10)

    Server.optimizer.step(
        apply_lr=False,
        scale=Server.args.lr_scale_at_sync,
        apply_in_momentum=False,
        apply_out_momentum=Server.args.out_momentum,
    ) 
    return 