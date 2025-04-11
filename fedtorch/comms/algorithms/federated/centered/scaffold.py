# -*- coding: utf-8 -*-

def scaffold_aggregation_centered(OnlineClients, Server, online_clients, local_steps, lr, lambda_weight=None):
    """Aggregate gradients for federated learning using SCAFFOLD algorithm.
    https://arxiv.org/abs/1910.06378
    """
    Server.optimizer.zero_grad(set_to_none=False)
    num_online_clients = len(online_clients)
    if lambda_weight is None:
        # rank_weight =  OnlineClient.cfg.data.num_samples_per_epoch / OnlineClient.cfg.data.train_dataset_size
        rank_weight =  1.0 / num_online_clients
    else:
        #TODO: This is experimental. Test it.
        rank_weight = lambda_weight * Server.cfg.graph.n_nodes / num_online_clients

    for o in online_clients:
        for ccp, scp, cp, sp in zip(OnlineClients[o].model_client_control.parameters(), \
                                    Server.model_server_control.parameters(), \
                                    OnlineClients[o].model.parameters(), \
                                    Server.model.parameters()):
            client_control_update = ccp.data - scp.data + (sp.data - cp.data)/(local_steps * lr)


            param_diff = (sp.data - cp.data) * rank_weight

            # Control variate change
            control_diff =  client_control_update - ccp.data
            
            # Aggregate gradients
            sp.grad.data.add_(param_diff)

            # Update server control variate
            scp.data.add_(control_diff / Server.cfg.graph.n_nodes)

            # Update client control parameter
            ccp.data = client_control_update
            # if o == 0:
            #     print(torch.norm(scp.data - ccp.data))


    Server.optimizer.step(
        apply_lr=False,
        scale=Server.cfg.lr.lr_scale_at_sync,
        apply_in_momentum=False,
        apply_out_momentum=Server.cfg.training.out_momentum,
    )


    return