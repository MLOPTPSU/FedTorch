# -*- coding: utf-8 -*-
import time
from copy import deepcopy
import numpy as np

import torch

from fedtorch.components.scheduler import adjust_learning_rate
from fedtorch.components.dataset import load_data_batch
from fedtorch.comms.utils.flow_utils import (get_current_epoch, 
                                             get_current_local_step, 
                                             is_sync_fed)
from fedtorch.comms.utils.eval import inference
from fedtorch.comms.utils.eval_centered import (do_validate_centered, 
                                                log_validation_centered,
                                                log_test_centered)
from fedtorch.comms.algorithms.federated import fedavg_aggregation_centered, set_online_clients_centered
from fedtorch.logs.logging import (log, 
                                   logging_sync_time,
                                   logging_load_time,
                                   logging_globally)
from fedtorch.logs.meter import define_local_training_tracker


def train_and_validate_perfedme_centered(Clients, Server):
    log('start training and validation with Federated setting in a centered way.')

    # For Sparsified SGD
    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    log('enter the training.')

    # Number of communication rounds in federated setting should be defined
    for n_c in range(Server.args.num_comms):
        Server.args.rounds_comm += 1
        Server.args.local_index += 1
        Server.args.quant_error = 0.0
        
        # Preset variables for this round of communication
        Server.zero_grad()
        Server.reset_tracker(Server.local_val_tracker)
        Server.reset_tracker(Server.global_val_tracker)
        if Server.args.fed_personal:
            Server.reset_tracker(Server.local_personal_val_tracker)
            Server.reset_tracker(Server.global_personal_val_tracker) 

        # Configuring the devices for this round of communication
        log("Starting round {} of training".format(n_c+1))
        online_clients = set_online_clients_centered(Server.args)
        
        for oc in online_clients:
            Clients[oc].model.load_state_dict(Server.model.state_dict())
            Clients[oc].args.rounds_comm = Server.args.rounds_comm
            local_steps = 0
            is_sync = False

            do_validate_centered(Clients[oc].args, Server.model, Server.criterion, Server.metrics, Server.optimizer,
                 Clients[oc].train_loader, Server.global_val_tracker, val=False, local=False)
            if Server.args.fed_personal:
                do_validate_centered(Clients[oc].args, Server.model, Server.criterion, Server.metrics, Server.optimizer,
                    Clients[oc].val_loader, Server.global_personal_val_tracker, val=True, local=False)

            while not is_sync:
                for _input, _target in Clients[oc].train_loader:
                    local_steps += 1
                    Clients[oc].model.train()
                    Clients[oc].model_personal.train()

                    # update local step.
                    logging_load_time(tracker)
                    
                    # update local index and get local step
                    Clients[oc].args.local_index += 1
                    Clients[oc].args.local_data_seen += len(_target)
                    get_current_epoch(Clients[oc].args)
                    local_step = get_current_local_step(Clients[oc].args)

                    # adjust learning rate (based on the # of accessed samples)
                    lr = adjust_learning_rate(Clients[oc].args, Clients[oc].optimizer_personal, Clients[oc].scheduler)

                    # load data
                    _input, _target = load_data_batch(Clients[oc].args, _input, _target, tracker)
        
                    # Skip batches with one sample because of BatchNorm issue in some models!
                    if _input.size(0)==1:
                        is_sync = is_sync_fed(Clients[oc].args)
                        break

                    # inference and get current performance.
                    Clients[oc].optimizer_personal.zero_grad()
                    
                    loss, performance = inference(Clients[oc].model_personal, Clients[oc].criterion, Clients[oc].metrics, _input, _target)
                    # print("loss in rank {} is {}".format(oc,loss))
        
                    # compute gradient and do local SGD step.
                    loss.backward()

                    for client_param, personal_param in zip(Clients[oc].model.parameters(),Clients[oc].model_personal.parameters()):
                        # loss += Clients[oc].args.perfedme_lambda * torch.norm(personal_param.data - client_param.data)**2
                        personal_param.grad.data += Clients[oc].args.perfedme_lambda * (personal_param.data - client_param.data)


                    Clients[oc].optimizer_personal.step(
                        apply_lr=True,
                        apply_in_momentum=Clients[oc].args.in_momentum, apply_out_momentum=False
                    )

                    if Clients[oc].args.local_index == 1:
                        Clients[oc].optimizer.zero_grad()
                        loss, performance = inference(Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, _input, _target)
                        loss.backward()
                    
                    is_sync = is_sync_fed(Clients[oc].args)
                    if Clients[oc].args.local_index % 5 == 0 or is_sync:
                        log('Updating the local version of the global model',Clients[oc].args.debug)
                        lr = adjust_learning_rate(Clients[oc].args, Clients[oc].optimizer, Clients[oc].scheduler)
                        Clients[oc].optimizer.zero_grad()
                        for client_param, personal_param in zip(Clients[oc].model.parameters(),Clients[oc].model_personal.parameters()):
                            client_param.grad.data = Clients[oc].args.perfedme_lambda * (client_param.data - personal_param.data)
                        Clients[oc].optimizer.step(
                            apply_lr=True,
                            apply_in_momentum=Clients[oc].args.in_momentum, apply_out_momentum=False
                            )

                    # reset load time for the tracker.
                    tracker['start_load_time'] = time.time()
                    # model_local = deepcopy(model_client)
                    
                    if is_sync:
                        break

            do_validate_centered(Clients[oc].args, Clients[oc].model_personal, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer_personal,
                 Clients[oc].train_loader, Server.local_val_tracker, val=False, local=True)
            if Server.args.fed_personal:
                do_validate_centered(Clients[oc].args, Clients[oc].model_personal, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer_personal,
                 Clients[oc].val_loader, Server.local_personal_val_tracker, val=True, local=True)
            # Sync the model server based on model_clients
            tracker['start_sync_time'] = time.time()
            Server.args.global_index += 1

            logging_sync_time(tracker)
        
        fedavg_aggregation_centered(Clients, Server, online_clients)
        # Log local performance
        log_validation_centered(Server.args, Server.local_val_tracker, val=False, local=True)
        if Server.args.fed_personal:
            log_validation_centered(Server.args, Server.local_personal_val_tracker, val=True, local=True)
            

        # Log server performance
        log_validation_centered(Server.args, Server.global_val_tracker, val=False, local=False)
        if Server.args.fed_personal:
            log_validation_centered(Server.args, Server.global_personal_val_tracker, val=True, local=False)


        # logging.
        logging_globally(tracker, start_global_time)
        
        # reset start round time.
        start_global_time = time.time()

        # validate the model at the server
        # if args.graph.rank == 0:
        #     do_test(args, model_server, optimizer, criterion, metrics, test_loader)
        # do_validate_test(args, model_server, optimizer, criterion, metrics, test_loader)
    return