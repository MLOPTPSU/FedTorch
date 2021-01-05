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
                                    alpha_update)
from comms.algorithms.distributed import global_average
from comms.utils.eval import inference, do_validate, inference_personal
from comms.algorithms.federated import (fedavg_aggregation,
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



def train_and_validate_federated_apfl(args, model_client, criterion, scheduler, optimizer, metrics):
    """The training scheme of Personalized Federated Learning.
        Official implementation for https://arxiv.org/abs/2003.13461
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
    args.finish_one_epoch = False
    model_personal = deepcopy(model_client)
    model_server = deepcopy(model_client)
    optimizer_personal = define_optimizer(args, model_personal)
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
        log("Starting round {} of training".format(n_c), args.debug)
        online_clients = set_online_clients(args)
        if (n_c == 0) and  (0 not in online_clients):
            online_clients += [0]
        online_clients_server = online_clients if 0 in online_clients else online_clients + [0]
        online_clients_group = dist.new_group(online_clients_server)
        
        if args.graph.rank in online_clients_server: 
            model_server = distribute_model_server(model_server, online_clients_group, src=0)
            model_client.load_state_dict(model_server.state_dict())
            if args.graph.rank in online_clients:
                is_sync = False
                ep = -1 # counting number of epochs
                while not is_sync:
                    ep += 1
                    for i, (_input, _target) in enumerate(train_loader):
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
                        
                        optimizer.zero_grad()
                        optimizer_personal.zero_grad()
                        loss_personal, performance_personal = inference_personal(model_personal, model_client, args.fed_personal_alpha, criterion, metrics, _input, _target)

                        # compute gradient and do local SGD step.
                        loss_personal.backward()
                        optimizer_personal.step(
                            apply_lr=True,
                            apply_in_momentum=args.in_momentum, apply_out_momentum=False
                        )

                        # update alpha
                        if args.fed_adaptive_alpha and i==0 and ep==0:
                            args.fed_personal_alpha = alpha_update(model_client, model_personal,args.fed_personal_alpha, lr) #0.1/np.sqrt(1+args.local_index))
                            average_alpha = args.fed_personal_alpha
                            average_alpha = global_average(average_alpha,args.graph.n_nodes, group=online_clients_group)
                            log("New alpha is:{}".format(average_alpha.item()), args.debug)
                        
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

            do_validate(args, model_client, optimizer_personal, criterion, metrics, train_loader, online_clients_group, data_mode='train',
                 personal=True, model_personal=model_personal,alpha=args.fed_personal_alpha)
            if args.fed_personal:
                do_validate(args, model_client, optimizer_personal, criterion, metrics, val_loader, online_clients_group, data_mode='validation',
                 personal=True, model_personal=model_personal,alpha=args.fed_personal_alpha)

            # Sync the model server based on model_clients
            log('Enter synching', args.debug)
            tracker['start_sync_time'] = time.time()
            args.global_index += 1
            model_server = fedavg_aggregation(args, model_server, model_client, online_clients_group, online_clients, optimizer)
            # evaluate the sync time
            logging_sync_time(tracker)

            do_validate(args, model_server, optimizer, criterion, metrics, train_loader, online_clients_group, data_mode='train')
            if args.fed_personal:
                do_validate(args, model_server, optimizer, criterion, metrics, val_loader, online_clients_group, data_mode='validation')

           
            # logging.
            logging_globally(tracker, start_global_time)
            
            # reset start round time.
            start_global_time = time.time()

            # validate the models at the test data
            if args.fed_personal_test:
                do_validate(args, model_client, optimizer_personal, criterion, metrics, test_loader, online_clients_group, data_mode='test',
                                    personal=True, model_personal=model_personal, alpha=args.fed_personal_alpha)
            elif args.graph.rank == 0:
                do_validate(args, model_server, optimizer, criterion, metrics, test_loader, online_clients_group, data_mode='test')
        else:
            log("Offline in this round. Waiting on others to finish!", args.debug)
        dist.barrier(group=all_clients_group)