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
from comms.algorithms.federated import (afl_aggregation,
                                        set_online_clients,
                                        distribute_model_server)
from logs.logging import (log,
                          logging_computing,
                          logging_sync_time,
                          logging_display_training,
                          logging_load_time,
                          logging_globally)
from logs.meter import define_local_training_tracker
from components.optimizer import define_optimizer


def train_and_validate_federated_afl(args, model_client, criterion, scheduler, optimizer, metrics):
    """The training scheme of Federated Learning systems.
        This the implementation of Agnostic Federated Learning
        https://arxiv.org/abs/1902.00146
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

    # Number of communication rounds in federated setting should be defined
    for n_c in range(args.num_comms):
        args.rounds_comm += 1
        args.comm_time.append(0.0)
        # Configuring the devices for this round of communication
        # TODO: not make the server rank hard coded
        log("Starting round {} of training".format(n_c+1), args.debug)
        online_clients = set_online_clients(args)
        if n_c == 0:
            # The first round server should be in the communication to initilize its own training
            online_clients = online_clients if 0 in online_clients else online_clients + [0]
        online_clients_server = online_clients if 0 in online_clients else online_clients + [0]
        online_clients_group = dist.new_group(online_clients_server)
        
        if args.graph.rank in online_clients_server:
            st = time.time()
            model_server = distribute_model_server(model_server, online_clients_group, src=0)
            dist.broadcast(lambda_vector, src=0, group=online_clients_group)
            args.comm_time[-1] += time.time() - st
            model_client.load_state_dict(model_server.state_dict())

            # This loss tensor is for those clients not participating in the first round
            loss = torch.tensor(0.0)
            # Start running updates on local machines
            if args.graph.rank in online_clients:
                is_sync = False
                while not is_sync:
                    for _input, _target in train_loader:
                        
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
            
            model_server, loss_tensor_online = afl_aggregation(args, model_server, model_client, lambda_vector[args.graph.rank].item(), 
                                                                    torch.tensor(loss.item()), online_clients_group, online_clients, optimizer)

            # evaluate the sync time
            logging_sync_time(tracker)
            do_validate(args, model_server, optimizer, criterion, metrics, train_loader, online_clients_group, data_mode='train')
            if args.fed_personal:
                do_validate(args, model_server, optimizer, criterion, metrics, val_loader, online_clients_group, data_mode='validation')
            
            # Updating lambda variable
            if args.graph.rank == 0:
                num_online_clients = len(online_clients) if 0 in online_clients else len(online_clients) + 1
                loss_tensor = torch.zeros(args.graph.n_nodes)
                loss_tensor[sorted(online_clients_server)] = loss_tensor_online
                # Dual update
                lambda_vector += args.drfa_gamma * loss_tensor
                # Projection into a simplex
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
            # validate the model at the server
            if args.graph.rank == 0:
                # TODO: run it at the end of each epoch not communication
                do_validate(args, model_server, optimizer, criterion, metrics, test_loader, online_clients_group, data_mode='test')
            log('This round communication time is: {}'.format(args.comm_time[-1]), args.debug)
        else:
            log("Offline in this round. Waiting on others to finish!", args.debug)
        dist.barrier(group=all_clients_group)


    return