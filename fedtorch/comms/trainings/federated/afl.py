# -*- coding: utf-8 -*-
import time
from copy import deepcopy
import numpy as np

import torch
import torch.distributed as dist

from fedtorch.components.scheduler import adjust_learning_rate
from fedtorch.components.dataset import define_dataset, load_data_batch
from fedtorch.logs.checkpoint import save_to_checkpoint
from fedtorch.comms.utils.flow_utils import (get_current_epoch, 
                                             get_current_local_step,
                                             zero_copy, 
                                             is_sync_fed,
                                             euclidean_proj_simplex)
from fedtorch.comms.algorithms.distributed import global_average
from fedtorch.comms.utils.eval import inference, do_validate, inference_personal
from fedtorch.comms.algorithms.federated import (afl_aggregation,
                                                 set_online_clients,
                                                 distribute_model_server)
from fedtorch.logs.logging import (log,
                                   logging_computing,
                                   logging_sync_time,
                                   logging_display_training,
                                   logging_load_time,
                                   logging_globally)
from fedtorch.logs.meter import define_local_training_tracker
from fedtorch.components.optimizer import define_optimizer


def train_and_validate_federated_afl(client):
    """The training scheme of Federated Learning systems.
        This the implementation of Agnostic Federated Learning
        https://arxiv.org/abs/1902.00146
    """
    log('start training and validation with Federated setting.', client.args.debug)

    if client.args.evaluate and client.args.graph.rank==0:
        # Do the testing on the server and return
        do_validate(client.args, client.model, client.optimizer,  client.criterion, client.metrics,
                         client.test_loader, client.all_clients_group, data_mode='test')
        return

    # Initialize lambda variable proportianate to their sample size
    if client.args.graph.rank == 0:
        gather_list_size = [torch.tensor(0.0) for _ in range(client.args.graph.n_nodes)]
        dist.gather(torch.tensor(client.args.num_samples_per_epoch, dtype=torch.float32), gather_list=gather_list_size, dst=0)
        client.lambda_vector = torch.stack(gather_list_size) / client.args.train_dataset_size
    else:
        dist.gather(torch.tensor(client.args.num_samples_per_epoch,  dtype=torch.float32), dst=0)
        client.lambda_vector = torch.tensor([1/client.args.graph.n_nodes]*client.args.graph.n_nodes)
    
    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    log('enter the training.', client.args.debug)

    # Number of communication rounds in federated setting should be defined
    for n_c in range(client.args.num_comms):
        client.args.rounds_comm += 1
        client.args.comm_time.append(0.0)
        # Configuring the devices for this round of communication
        # TODO: not make the server rank hard coded
        log("Starting round {} of training".format(n_c+1), client.args.debug)
        online_clients = set_online_clients(client.args)
        if n_c == 0:
            # The first round server should be in the communication to initilize its own training
            online_clients = online_clients if 0 in online_clients else online_clients + [0]
        online_clients_server = online_clients if 0 in online_clients else online_clients + [0]
        online_clients_group = dist.new_group(online_clients_server)
        
        if client.args.graph.rank in online_clients_server:
            st = time.time()
            client.model_server = distribute_model_server(client.model_server, online_clients_group, src=0)
            dist.broadcast(client.lambda_vector, src=0, group=online_clients_group)
            client.args.comm_time[-1] += time.time() - st
            client.model.load_state_dict(client.model_server.state_dict())

            # This loss tensor is for those clients not participating in the first round
            loss = torch.tensor(0.0)
            # Start running updates on local machines
            if client.args.graph.rank in online_clients:
                is_sync = False
                while not is_sync:
                    for _input, _target in client.train_loader:
                        
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
                        client.optimizer.step(
                            apply_lr=True,
                            apply_in_momentum=client.args.in_momentum, apply_out_momentum=False
                        )
                        
                        # logging locally.
                        # logging_computing(tracker, loss, performance, _input, lr)
                        
                        # display the logging info.
                        # logging_display_training(args, tracker)

                        # reset load time for the tracker.
                        tracker['start_load_time'] = time.time()
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
            
            client.model_server, loss_tensor_online = afl_aggregation(client.args, client.model_server, client.model, 
                                                                      client.lambda_vector[client.args.graph.rank].item(), 
                                                                      torch.tensor(loss.item()), online_clients_group, 
                                                                      online_clients, client.optimizer)

            # evaluate the sync time
            logging_sync_time(tracker)
            # Do the validation on the server model
            do_validate(client.args, client.model_server, client.optimizer, client.criterion, client.metrics, 
                        client.train_loader, online_clients_group, data_mode='train')
            if client.args.fed_personal:
                do_validate(client.args, client.model_server, client.optimizer, client.criterion, client.metrics, 
                            client.val_loader, online_clients_group, data_mode='validation')
            
            # Updating lambda variable
            if client.args.graph.rank == 0:
                num_online_clients = len(online_clients) if 0 in online_clients else len(online_clients) + 1
                loss_tensor = torch.zeros(client.args.graph.n_nodes)
                loss_tensor[sorted(online_clients_server)] = loss_tensor_online
                # Dual update
                client.lambda_vector += client.args.drfa_gamma * loss_tensor
                # Projection into a simplex
                client.lambda_vector = euclidean_proj_simplex(client.lambda_vector)
                # Avoid zero probability
                lambda_zeros = client.lambda_vector <= 1e-3
                if lambda_zeros.sum() > 0:
                    client.lambda_vector[lambda_zeros] = 1e-3
                    client.lambda_vector /= client.lambda_vector.sum()
            
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