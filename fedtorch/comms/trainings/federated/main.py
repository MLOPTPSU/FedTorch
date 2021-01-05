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



def train_and_validate_federated(args, model_client, criterion, scheduler, optimizer, metrics):
    """The training scheme of Federated Learning systems.
        The basic model is FedAvg https://arxiv.org/abs/1602.05629
        TODO: Merge different models under this method
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
    elif args.federated_type == 'qsparse':
        model_memory = zero_copy(model_client)
    elif args.federated_type == 'scaffold':
        model_client_control = deepcopy(model_client)
        model_server_control = deepcopy(model_client)
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
        args.quant_error = 0.0
        # Configuring the devices for this round of communication
        # TODO: not make the server rank hard coded
        log("Starting round {} of training".format(n_c+1), args.debug)
        online_clients = set_online_clients(args)
        if (n_c == 0) and  (0 not in online_clients):
            online_clients += [0]
        online_clients_server = online_clients if 0 in online_clients else online_clients + [0]
        online_clients_group = dist.new_group(online_clients_server)
        
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
            local_steps = 0
            if args.graph.rank in online_clients:
                is_sync = False
                while not is_sync:
                    for _input, _target in train_loader:
                        local_steps += 1
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

                        # Skip batches with one sample because of BatchNorm issue in some models!
                        if _input.size(0)==1:
                            is_sync = is_sync_fed(args)
                            break

                        # inference and get current performance.
                        optimizer.zero_grad()
                       
                        loss, performance = inference(model_client, criterion, metrics, _input, _target)
                        # print("loss in rank {} is {}".format(args.graph.rank,loss))

                        # compute gradient and do local SGD step.
                        loss.backward()

                        if args.federated_type == 'fedgate':
                            # Update gradients with control variates
                            for client_param, delta_param  in zip(model_client.parameters(), model_delta.parameters()):
                                client_param.grad.data -= delta_param.data 
                        elif args.federated_type == 'scaffold':
                            for cp, ccp, scp  in zip(model_client.parameters(), model_client_control.parameters(), model_server_control.parameters()):
                                cp.grad.data += scp.data - ccp.data
                        elif args.federated_type == 'fedprox':
                            # Adding proximal gradients and loss for fedprox
                            for client_param, server_param in zip(model_client.parameters(),model_server.parameters()):
                                if args.graph.rank == 0:
                                    print("distance norm for prox is:{}".format(torch.norm(client_param.data - server_param.data )))
                                loss += args.fedprox_mu /2 * torch.norm(client_param.data - server_param.data)
                                client_param.grad.data += args.fedprox_mu * (client_param.data - server_param.data)
                        
                        if 'robust' in args.arch:
                            model_client.noise.grad.data *= -1

                        optimizer.step(
                            apply_lr=True,
                            apply_in_momentum=args.in_momentum, apply_out_momentum=False
                        )

                        if 'robust' in args.arch:
                            if torch.norm(model_client.noise.data) > 1:
                                model_client.noise.data /= torch.norm(model_client.noise.data)
                        
                        # logging locally.
                        # logging_computing(tracker, loss_v, performance_v, _input, lr)
                        
                        # display the logging info.
                        # logging_display_training(args, tracker)


                        # reset load time for the tracker.
                        tracker['start_load_time'] = time.time()
                        # model_local = deepcopy(model_client)
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
                model_server, model_delta = fedgate_aggregation(args, model_server, model_client, model_delta, model_memory, online_clients_group, online_clients, optimizer, lr, local_steps)
            elif args.federated_type == 'scaffold':
                model_server, model_client_control, model_server_control = scaffold_aggregation(args, model_server, model_client, model_server_control, model_client_control,
                                                                                            online_clients_group, online_clients, optimizer, lr, local_steps)
            elif args.federated_type == 'qsparse':
                model_server = qsparse_aggregation(args, model_server, model_client, online_clients_group, online_clients, optimizer, model_memory)
            else:
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

            # validate the model at the server
            if args.graph.rank == 0:
                do_validate(args, model_server, optimizer, criterion, metrics, test_loader, online_clients_group, data_mode='test')
            log('This round communication time is: {}'.format(args.comm_time[-1]), args.debug)
        else:
            log("Offline in this round. Waiting on others to finish!", args.debug)
        dist.barrier(group=all_clients_group)

    return