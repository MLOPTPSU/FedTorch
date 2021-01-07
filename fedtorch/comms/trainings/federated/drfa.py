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
                                    is_sync_fed,
                                    projection_simplex_sort)
from comms.algorithms.distributed import global_average
from comms.utils.eval import inference, do_validate, inference_personal
from comms.algorithms.federated import (fedavg_aggregation,
                                        fedgate_aggregation,
                                        scaffold_aggregation,
                                        qsparse_aggregation, 
                                        distribute_model_server_control,
                                        set_online_clients,
                                        distribute_model_server,
                                        aggregate_models_virtual,
                                        loss_gather)
from logs.logging import (log,
                          logging_computing,
                          logging_sync_time,
                          logging_display_training,
                          logging_load_time,
                          logging_globally)
from logs.meter import define_local_training_tracker
from components.optimizer import define_optimizer


def train_and_validate_federated_drfa(args, model_client, criterion, scheduler, optimizer, metrics):
    """The training scheme of Distributionally Robust Federated Learning DRFA.
        paper: https://papers.nips.cc/paper/2020/hash/ac450d10e166657ec8f93a1b65ca1b14-Abstract.html
    """
    log('start training and validation with Federated setting.', args.debug)

    # get data loader.
    if args.fed_personal:
        train_loader, test_loader, val_loader = define_dataset(args, shuffle=True)
    else:
        train_loader, test_loader = define_dataset(args, shuffle=True)
    # add all clients to a group
    all_clients_group = dist.new_group(args.graph.ranks)
    if args.evaluate:
        do_validate(args, model_client, optimizer,  criterion, metrics,
                         test_loader, all_clients_group, data_mode='test')
        return

    # init global variable.
    if args.data in ['mnist','fashion_mnist','cifar10']:
        args.classes = torch.arange(10)
    elif args.data in ['synthetic']:
        args.classes = torch.arange(5)
    elif args.data in ['adult']:
        args.classes = torch.arange(2)
    
    args.finish_one_epoch = False
    model_server = deepcopy(model_client)

    if args.federated_type == 'fedgate':
        model_delta = zero_copy(model_client)
        model_memory = zero_copy(model_client)
    elif args.federated_type == 'scaffold':
        model_client_control = deepcopy(model_client)
        model_server_control = deepcopy(model_client)
    kth_model = deepcopy(model_client)
    
    # Initialize lambda variable proportianate to their sample size
    if args.graph.rank == 0:
        gather_list_size = [torch.tensor(0.0) for _ in range(args.graph.n_nodes)]
        dist.gather(torch.tensor(args.num_samples_per_epoch, dtype=torch.float32), gather_list=gather_list_size, dst=0)
        lambda_vector = torch.stack(gather_list_size) / args.train_dataset_size
    else:
        dist.gather(torch.tensor(args.num_samples_per_epoch,  dtype=torch.float32), dst=0)
        lambda_vector = torch.tensor([1/args.graph.n_nodes]*args.graph.n_nodes)

    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    log('enter the training.', args.debug)

    if args.federated_type == 'fedadam':
        # Initialize the parameter for FedAdam https://arxiv.org/abs/2003.00295
        args.fedadam_v =  [args.fedadam_tau ** 2] * len(list(model_server.parameters()))

    # Number of communication rounds in federated setting should be defined
    for n_c in range(args.num_comms):
        args.rounds_comm += 1
        args.comm_time.append(0.0)
        # Configuring the devices for this round of communication
        # TODO: not make the server rank hard coded
        log("Starting round {} of training".format(n_c), args.debug)
        online_clients = set_online_clients(args)
        if n_c == 0:
            # The first round server should be in the communication to initilize its own training
            online_clients = online_clients if 0 in online_clients else online_clients + [0]
        online_clients_server = online_clients if 0 in online_clients else online_clients + [0]
        online_clients_group = dist.new_group(online_clients_server)
        args.drfa_gamma *= 0.9
        if args.graph.rank in online_clients_server:
            if  args.federated_type == 'scaffold':
                st = time.time()
                model_server, model_server_control = distribute_model_server_control(model_server, model_server_control, online_clients_group, src=0)
                args.comm_time[-1] += time.time() - st
            else:
                st = time.time()
                model_server = distribute_model_server(model_server, online_clients_group, src=0)
                args.comm_time[-1] += time.time() - st
            model_client.load_state_dict(model_server.state_dict())

            # Send related variables to drfa algorithm
            st = time.time()
            dist.broadcast(lambda_vector, src=0, group=online_clients_group)
            # Sending the random number k to all nodes:
            # Does not fully support the epoch mode now
            k = torch.randint(low=1,high=args.local_step,size=(1,))
            dist.broadcast(k, src=0, group=online_clients_group)
            args.comm_time[-1] += time.time() - st
            
            k = k.tolist()[0]
            local_steps = 0
            # Start running updates on local machines
            if args.graph.rank in online_clients:
                is_sync = False
                while not is_sync:
                    for _input, _target in train_loader:
                        local_steps += 1
                        # Getting the k-th model for dual variable update
                        if k == local_steps:
                            kth_model.load_state_dict(model_client.state_dict())
                        model_client.train()

                        # update local step.
                        logging_load_time(tracker)

                        # update local index and get local step
                        args.local_index += 1
                        args.local_data_seen += len(_target)
                        get_current_epoch(args)
                        local_step = get_current_local_step(args)

                        # adjust learning rate (based on the # of accessed samples)
                        lr = adjust_learning_rate(args, optimizer, scheduler)

                        # load data
                        _input, _target = load_data_batch(args, _input, _target, tracker)
                        # inference and get current performance.
                        optimizer.zero_grad()
                        loss, performance = inference(model_client, criterion, metrics, _input, _target)

                        # compute gradient and do local SGD step.
                        loss.backward()

                        if args.federated_type == 'fedgate':
                            for client_param, delta_param  in zip(model_client.parameters(), model_delta.parameters()):
                                client_param.grad.data -= delta_param.data 
                        elif args.federated_type == 'scaffold':
                            for cp, ccp, scp  in zip(model_client.parameters(), model_client_control.parameters(), model_server_control.parameters()):
                                cp.grad.data += scp.data - ccp.data


                        optimizer.step(
                            apply_lr=True,
                            apply_in_momentum=args.in_momentum, apply_out_momentum=False
                        )
                        
                        # logging locally.
                        # logging_computing(tracker, loss, performance, _input, lr)
                        
                        # display the logging info.
                        # logging_display_training(args, tracker)

                        # reset load time for the tracker.
                        tracker['start_load_time'] = time.time()
                        is_sync = is_sync_fed(args)
                        if is_sync:
                            break
            else:
                log("Offline in this round. Waiting on others to finish!", args.debug)

            do_validate(args, model_client, optimizer, criterion, metrics, train_loader, online_clients_group, data_mode='train', local=True)
            if args.fed_personal:
                do_validate(args, model_client, optimizer, criterion, metrics, val_loader, online_clients_group, data_mode='validation', local=True)
            # Sync the model server based on model_clients
            log('Enter synching', args.debug)
            tracker['start_sync_time'] = time.time()
            args.global_index += 1

            if args.federated_type == 'fedgate':
                model_server, model_delta = fedgate_aggregation(args, model_server, model_client, model_delta, model_memory,
                                                                online_clients_group, online_clients, optimizer, lr, local_steps,
                                                                lambda_weight=lambda_vector[args.graph.rank].item())
            elif args.federated_type == 'scaffold':
                model_server, model_client_control, model_server_control = scaffold_aggregation(args, model_server, model_client, model_server_control, 
                                                                                                model_client_control, online_clients_group, online_clients, 
                                                                                                optimizer, lr, local_steps, lambda_weight=lambda_vector[args.graph.rank].item())
            else:
                model_server = fedavg_aggregation(args, model_server, model_client, online_clients_group, online_clients, optimizer, lambda_weight=lambda_vector[args.graph.rank].item())
             
            # Average the kth_model
            kth_model = aggregate_models_virtual(args, kth_model, online_clients_group, online_clients)
             # evaluate the sync time
            logging_sync_time(tracker)

            do_validate(args, model_client, optimizer, criterion, metrics, train_loader, online_clients_group, data_mode='train', local=True)
            if args.fed_personal:
                do_validate(args, model_client, optimizer, criterion, metrics, val_loader, online_clients_group, data_mode='validation', local=True)
            
            # validate the model at the server
            if args.graph.rank == 0:
                do_validate(args, model_server, optimizer, criterion, metrics, test_loader, online_clients_group, data_mode='test')

        else:
            log("Offline in this round. Waiting on others to finish!", args.debug)
        
        
        # Update lambda parameters
        online_clients_lambda = set_online_clients(args)
        online_clients_server_lambda = online_clients_lambda if 0 in online_clients_lambda else [0] + online_clients_lambda 
        online_clients_group_lambda = dist.new_group(online_clients_server_lambda)
        
        if args.graph.rank in online_clients_server_lambda:
            st = time.time()
            kth_model = distribute_model_server(kth_model, online_clients_group_lambda, src=0)
            args.comm_time[-1] += time.time() - st
            loss = torch.tensor(0.0)

            if args.graph.rank in online_clients_lambda:
                for _input, _target in train_loader:
                    _input, _target = load_data_batch(args, _input, _target, tracker)
                    loss, _ = inference(kth_model, criterion, metrics, _input, _target)
                    break
            loss_tensor_online = loss_gather(args, torch.tensor(loss.item()), group=online_clients_group_lambda, online_clients=online_clients_lambda)
            if args.graph.rank == 0:
                num_online_clients = len(online_clients_lambda) if 0 in online_clients_lambda else len(online_clients_lambda) + 1
                loss_tensor = torch.zeros(args.graph.n_nodes)
                loss_tensor[sorted(online_clients_server_lambda)] = loss_tensor_online * (args.graph.n_nodes / num_online_clients)
                # Dual update
                lambda_vector += args.drfa_gamma * args.local_step * loss_tensor
                lambda_vector = projection_simplex_sort(lambda_vector.numpy())

                # Avoid zero probability
                lambda_zeros = np.argwhere(lambda_vector <= 1e-3)
                if len(lambda_zeros)>0:
                    lambda_vector[lambda_zeros[0]] = 1e-3
                    lambda_vector /= np.sum(lambda_vector)
                lambda_vector = torch.tensor(lambda_vector)
        
        # logging.
        logging_globally(tracker, start_global_time)
        
        # reset start round time.
        start_global_time = time.time()
        log('This round communication time is: {}'.format(args.comm_time[-1]), args.debug)
        dist.barrier(group=all_clients_group)
    return

