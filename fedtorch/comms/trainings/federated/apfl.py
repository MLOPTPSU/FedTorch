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
                                             alpha_update)
from fedtorch.comms.algorithms.distributed import global_average
from fedtorch.comms.utils.eval import inference, do_validate, inference_personal
from fedtorch.comms.algorithms.federated import (fedavg_aggregation,
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



def train_and_validate_federated_apfl(client):
    """The training scheme of Personalized Federated Learning.
        Official implementation for https://arxiv.org/abs/2003.13461
    """
    log('start training and validation with Federated setting.', client.args.debug)

    if client.args.evaluate and client.args.graph.rank==0:
        # Do the testing on the server and return
        do_validate(client.args, client.model, client.optimizer,  client.criterion, client.metrics,
                         client.test_loader, client.all_clients_group, data_mode='test')
        return

    
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
        log("Starting round {} of training".format(n_c), client.args.debug)
        online_clients = set_online_clients(client.args)
        if (n_c == 0) and  (0 not in online_clients):
            online_clients += [0]
        online_clients_server = online_clients if 0 in online_clients else online_clients + [0]
        online_clients_group = dist.new_group(online_clients_server)
        
        if client.args.graph.rank in online_clients_server: 
            client.model_server = distribute_model_server(client.model_server, online_clients_group, src=0)
            client.model.load_state_dict(client.model_server.state_dict())
            if client.args.graph.rank in online_clients:
                is_sync = False
                ep = -1 # counting number of epochs
                while not is_sync:
                    ep += 1
                    for i, (_input, _target) in enumerate(client.train_loader):
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
                        
                        client.optimizer.zero_grad()
                        client.optimizer_personal.zero_grad()
                        loss_personal, performance_personal = inference_personal(client.model_personal, client.model, 
                                                                                 client.args.fed_personal_alpha, client.criterion, 
                                                                                 client.metrics, _input, _target)

                        # compute gradient and do local SGD step.
                        loss_personal.backward()
                        client.optimizer_personal.step(
                            apply_lr=True,
                            apply_in_momentum=client.args.in_momentum, apply_out_momentum=False
                        )

                        # update alpha
                        if client.args.fed_adaptive_alpha and i==0 and ep==0:
                            client.args.fed_personal_alpha = alpha_update(client.model, client.model_personal, client.args.fed_personal_alpha, lr) #0.1/np.sqrt(1+args.local_index))
                            average_alpha = client.args.fed_personal_alpha
                            average_alpha = global_average(average_alpha, client.args.graph.n_nodes, group=online_clients_group)
                            log("New alpha is:{}".format(average_alpha.item()), client.args.debug)
                        
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

            do_validate(client.args, client.model, client.optimizer_personal, client.criterion, client.metrics, 
                        client.train_loader, online_clients_group, data_mode='train', personal=True, 
                        model_personal=client.model_personal, alpha=client.args.fed_personal_alpha)
            if client.args.fed_personal:
                do_validate(client.args, client.model, client.optimizer_personal, client.criterion, client.metrics, 
                            client.val_loader, online_clients_group, data_mode='validation', personal=True, 
                            model_personal=client.model_personal, alpha=client.args.fed_personal_alpha)

            # Sync the model server based on model_clients
            log('Enter synching', client.args.debug)
            tracker['start_sync_time'] = time.time()
            client.args.global_index += 1
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

            # validate the models at the test data
            if client.args.fed_personal_test:
                do_validate(client.args, client.model_client, client.optimizer_personal, client.criterion, 
                            client.metrics, client.test_loader, online_clients_group, data_mode='test', personal=True,
                            model_personal=client.model_personal, alpha=client.args.fed_personal_alpha)
            elif client.args.graph.rank == 0:
                do_validate(client.args, client.model_server, client.optimizer, client.criterion, 
                            client.metrics, client.test_loader, online_clients_group, data_mode='test')
        else:
            log("Offline in this round. Waiting on others to finish!", client.args.debug)
        dist.barrier(group=client.all_clients_group)