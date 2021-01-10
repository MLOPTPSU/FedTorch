# -*- coding: utf-8 -*-
import time
from copy import deepcopy
import numpy as np

import torch
import torch.distributed as dist

from components.scheduler import adjust_learning_rate
from components.dataset import define_dataset, load_data_batch
from logs.checkpoint import save_to_checkpoint
from comms.utils.flow_utils import (get_current_epoch, 
                                    get_current_local_step,
                                    zero_copy, 
                                    is_sync_fed)
from comms.utils.eval import inference, do_validate
from comms.algorithms.federated import (fedavg_aggregation,
                                        fedgate_aggregation,
                                        scaffold_aggregation,
                                        qsparse_aggregation, 
                                        distribute_model_server_control,
                                        set_online_clients,
                                        distribute_model_server)
from logs.logging import (log,
                          logging_computing,
                          logging_sync_time,
                          logging_display_training,
                          logging_load_time,
                          logging_globally)
from logs.meter import define_local_training_tracker



def train_and_validate_federated(client):
    """The training scheme of Federated Learning systems.
        The basic model is FedAvg https://arxiv.org/abs/1602.05629
        TODO: Merge different models under this method
    """
    log('start training and validation with Federated setting.', client.args.debug)


    if client.args.evaluate and client.args.graph.rank==0:
        # Do the training on the server and return
        do_validate(client.args, client.model, client.optimizer,  client.criterion, client.metrics,
                         client.test_loader, client.all_clients_group, data_mode='test')
        return

    # init global variable.

    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    log('enter the training.', client.args.debug)

    # Number of communication rounds in federated setting should be defined
    for n_c in range(client.args.num_comms):
        client.args.rounds_comm += 1
        client.args.comm_time.append(0.0)
        # Configuring the devices for this round of communication
        log("Starting round {} of training".format(n_c+1), client.args.debug)
        online_clients = set_online_clients(client.args)
        if (n_c == 0) and  (0 not in online_clients):
            online_clients += [0]
        online_clients_server = online_clients if 0 in online_clients else online_clients + [0]
        online_clients_group = dist.new_group(online_clients_server)
        
        if client.args.graph.rank in online_clients_server:
            if  client.args.federated_type == 'scaffold':
                st = time.time()
                client.model_server, client.model_server_control = distribute_model_server_control(client.model_server, 
                                                                                                   client.model_server_control, 
                                                                                                   online_clients_group, 
                                                                                                   src=0)
                client.args.comm_time[-1] += time.time() - st
            else:
                st = time.time()
                client.model_server = distribute_model_server(client.model_server, online_clients_group, src=0)
                client.args.comm_time[-1] += time.time() - st
            client.model.load_state_dict(client.model_server.state_dict())
            local_steps = 0
            if client.args.graph.rank in online_clients:
                is_sync = False
                while not is_sync:
                    for _input, _target in client.train_loader:
                        local_steps += 1
                        client.model.train()

                        # update local step.
                        logging_load_time(tracker)

                        # update local index and get local step
                        client.args.local_index += 1
                        client.args.local_data_seen += len(_target)
                        get_current_epoch(client.args)
                        local_step = get_current_local_step(client.args)

                        # adjust learning rate (based on the # of accessed samples)
                        lr = adjust_learning_rate(client.args, client.optimizer, client.scheduler)

                        # load data
                        _input, _target = load_data_batch(client.args, _input, _target, tracker)

                        # Skip batches with one sample because of BatchNorm issue in some models!
                        if _input.size(0)==1:
                            is_sync = is_sync_fed(client.args)
                            break

                        # inference and get current performance.
                        client.optimizer.zero_grad()
                       
                        loss, performance = inference(client.model, client.criterion, client.metrics, _input, _target)

                        # compute gradient and do local SGD step.
                        loss.backward()

                        if client.args.federated_type == 'fedgate':
                            # Update gradients with control variates
                            for client_param, delta_param  in zip(client.model.parameters(), client.model_delta.parameters()):
                                client_param.grad.data -= delta_param.data 
                        elif client.args.federated_type == 'scaffold':
                            for cp, ccp, scp  in zip(client.model.parameters(), client.model_client_control.parameters(), client.model_server_control.parameters()):
                                cp.grad.data += scp.data - ccp.data
                        elif client.args.federated_type == 'fedprox':
                            # Adding proximal gradients and loss for fedprox
                            for client_param, server_param in zip(client.model.parameters(), client.model_server.parameters()):
                                if client.args.graph.rank == 0:
                                    print("distance norm for prox is:{}".format(torch.norm(client_param.data - server_param.data )))
                                loss += client.args.fedprox_mu /2 * torch.norm(client_param.data - server_param.data)
                                client_param.grad.data += client.args.fedprox_mu * (client_param.data - server_param.data)
                        
                        if 'robust' in client.args.arch:
                            client.model.noise.grad.data *= -1

                        client.optimizer.step(
                            apply_lr=True,
                            apply_in_momentum=client.args.in_momentum, apply_out_momentum=False
                        )

                        if 'robust' in client.args.arch:
                            if torch.norm(client.model.noise.data) > 1:
                                client.model.noise.data /= torch.norm(client.model.noise.data)
                        
                        # logging locally.
                        # logging_computing(tracker, loss_v, performance_v, _input, lr)
                        
                        # display the logging info.
                        # logging_display_training(args, tracker)


                        # reset load time for the tracker.
                        tracker['start_load_time'] = time.time()
                        # model_local = deepcopy(model_client)
                        is_sync = is_sync_fed(client.args)
                        if is_sync:
                            break

            else:
                log("Offline in this round. Waiting on others to finish!", client.args.debug)

            # Validate the local models befor sync
            do_validate(client.args, client.model, client.optimizer, client.criterion, client.metrics, 
                        client.train_loader, online_clients_group, data_mode='train', local=True)
            if client.args.fed_personal:
                do_validate(client.args, client.model, client.optimizer, client.criterion, client.metrics, 
                            client.val_loader, online_clients_group, data_mode='validation', local=True)
            # Sync the model server based on client models
            log('Enter synching', client.args.debug)
            tracker['start_sync_time'] = time.time()
            client.args.global_index += 1

            if client.args.federated_type == 'fedgate':
                client.model_server, client.model_delta = fedgate_aggregation(client.args, client.model_server, client.model, 
                                                                              client.model_delta, client.model_memory, 
                                                                              online_clients_group, online_clients, 
                                                                              client.optimizer, lr, local_steps)
            elif client.args.federated_type == 'scaffold':
                client.model_server, client.model_client_control, client.model_server_control = scaffold_aggregation(client.args, client.model_server, 
                                                                                                                     client.model, client.model_server_control, 
                                                                                                                     client.model_client_control, online_clients_group, 
                                                                                                                     online_clients, client.optimizer, lr, local_steps)
            elif client.args.federated_type == 'qsparse':
                client.model_server = qsparse_aggregation(client.args, client.model_server, client.model, 
                                                          online_clients_group, online_clients, 
                                                          client.optimizer, client.model_memory)
            else:
                client.model_server = fedavg_aggregation(client.args, client.model_server, client.model, 
                                                         online_clients_group, online_clients, client.optimizer)
             # evaluate the sync time
            logging_sync_time(tracker)
            
            # Do the validation on the server model
            do_validate(client.args, client.model_server, client.optimizer, client.criterion, client.metrics, 
                        client.train_loader, online_clients_group, data_mode='train')
            if client.args.fed_personal:
                do_validate(client.args, client.model_server, client.optimizer, client.criterion, client.metrics, 
                            client.val_loader, online_clients_group, data_mode='validation')

            # logging.
            logging_globally(tracker, start_global_time)
            
            # reset start round time.
            start_global_time = time.time()

            # validate the model at the server
            if client.args.graph.rank == 0:
                do_validate(client.args, client.model_server, client.optimizer, client.criterion, client.metrics, 
                            client.test_loader, online_clients_group, data_mode='test')
            log('This round communication time is: {}'.format(client.args.comm_time[-1]), client.args.debug)
        else:
            log("Offline in this round. Waiting on others to finish!", client.args.debug)
        dist.barrier(group=client.all_clients_group)

    return