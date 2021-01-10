# -*- coding: utf-8 -*-
import gc
import time
from copy import deepcopy

import torch
import torch.distributed as dist

from fedtorch.components.scheduler import adjust_learning_rate
from fedtorch.components.metrics import accuracy
from fedtorch.components.dataset import define_dataset, load_data_batch, \
    _load_data_batch
from fedtorch.logs.checkpoint import save_to_checkpoint
from fedtorch.comms.utils.eval import inference, do_validate
from fedtorch.comms.utils.flow_utils import is_stop, get_current_epoch, get_current_local_step
from fedtorch.comms.algorithms.distributed import aggregate_gradients
from fedtorch.logs.logging import log, logging_computing, logging_sync_time, \
    logging_display_training, logging_display_val, logging_load_time, \
    logging_globally, update_performancec_tracker
from fedtorch.logs.meter import define_local_training_tracker,\
    define_val_tracker, evaluate_gloabl_performance

def train_and_validate(client):
    """The training scheme of Distributed Local SGD."""
    log('start training and validation.', client.args.debug)

    if client.args.evaluate and client.args.graph.rank==0:
        # Do the training on the server and return
        do_validate(client.args, client.model, client.optimizer,  client.criterion, client.metrics,
                         client.test_loader, client.all_clients_group, data_mode='test')
        return

    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    log('enter the training.', client.args.debug)

    client.args.comm_time.append(0.0)
    # break until finish expected full epoch training.
    while True:
        # configure local step.
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
            logging_computing(tracker, loss, performance, _input, lr)

            # evaluate the status.
            is_sync = client.args.local_index % local_step == 0
            if client.args.epoch_ % 1 == 0:
                client.args.finish_one_epoch = True

            # sync
            if is_sync:
                log('Enter synching', client.args.debug)
                client.args.global_index += 1

                # broadcast gradients to other nodes by using reduce_sum.
                client.model_server = aggregate_gradients(client.args, client.model_server,
                                                          client.model, client.optimizer, is_sync)
                # evaluate the sync time
                logging_sync_time(tracker)

                # logging.
                logging_globally(tracker, start_global_time)
                
                # reset start round time.
                start_global_time = time.time()
            
            # finish one epoch training and to decide if we want to val our model.
            if client.args.finish_one_epoch:
                if client.args.epoch % client.args.eval_freq ==0 and client.args.graph.rank == 0:
                        do_validate(client.args, client.model, client.optimizer,  client.criterion, client.metrics,
                            client.test_loader, client.all_clients_group, data_mode='test')
                dist.barrier(group=client.all_clients_group)
                # refresh the logging cache at the begining of each epoch.
                client.args.finish_one_epoch = False
                tracker = define_local_training_tracker()
            
            # determine if the training is finished.
            if is_stop(client.args):
                #Last Sync
                log('Enter synching', client.args.debug)
                client.args.global_index += 1

                # broadcast gradients to other nodes by using reduce_sum.
                client.model_server = aggregate_gradients(client.args, client.model_server,
                                                          client.model, client.optimizer, is_sync)

                print("Total number of samples seen on device {} is {}".format(client.args.graph.rank, client.args.local_data_seen))
                if client.args.graph.rank == 0:
                    do_validate(client.args, client.model_server, client.optimizer,  client.criterion, client.metrics,
                        client.test_loader, client.all_clients_group, data_mode='test')
                return

            # display the logging info.
            logging_display_training(client.args, tracker)

            # reset load time for the tracker.
            tracker['start_load_time'] = time.time()

        # reshuffle the data.
        if client.args.reshuffle_per_epoch:
            log('reshuffle the dataset.', client.args.debug)
            del client.train_loader, client.test_loader
            gc.collect()
            log('reshuffle the dataset.', client.args.debug)
            client.load_local_dataset()