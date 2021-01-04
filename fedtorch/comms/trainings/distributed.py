# -*- coding: utf-8 -*-
import gc
import time
from copy import deepcopy

import torch
import torch.distributed as dist

from components.scheduler import adjust_learning_rate
from components.metrics import accuracy
from components.dataset import define_dataset, load_data_batch, \
    _load_data_batch
from logs.checkpoint import save_to_checkpoint
from comms.utils.eval import inference
from comms.utils.flow_utils import is_stop, get_current_epoch, get_current_local_step
from comms.algorithms.distributed import aggregate_gradients
from logs.logging import log, logging_computing, logging_sync_time, \
    logging_display_training, logging_display_val, logging_load_time, \
    logging_globally, update_performancec_tracker
from logs.meter import define_local_training_tracker,\
    define_val_tracker, evaluate_gloabl_performance


def train_and_validate(args, model, criterion, scheduler, optimizer, metrics):
    """The training scheme of Hierarchical Local SGD."""
    log('start training and validation.', args.debug)

    # get data loader.
    train_loader, val_loader = define_dataset(args, shuffle=True)

    if args.evaluate:
        validate(args, model, criterion, metrics, val_loader)
        return

    # init global variable.
    args.finish_one_epoch = False
    old_model = deepcopy(model)
    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    log('enter the training.', args.debug)

    args.comm_time.append(0.0)
    # break until finish expected full epoch training.
    while True:
        # configure local step.
        for _input, _target in train_loader:
            model.train()

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
            loss, performance = inference(model, criterion, metrics, _input, _target)

            # compute gradient and do local SGD step.
            loss.backward()
            optimizer.step(
                apply_lr=True,
                apply_in_momentum=args.in_momentum, apply_out_momentum=False
            )

            # logging locally.
            logging_computing(tracker, loss, performance, _input, lr)

            # evaluate the status.
            is_sync = args.local_index % local_step == 0
            if args.epoch_ % 1 == 0:
                args.finish_one_epoch = True

            # sync
            if is_sync:
                log('Enter synching', args.debug)
                args.global_index += 1

                # broadcast gradients to other nodes by using reduce_sum.
                old_model = aggregate_gradients(
                    args, old_model, model, optimizer, is_sync)
                # online_clients = args.graph.ranks
                # group = dist.new_group(online_clients)
                # old_model = fedavg_aggregation(args, old_model, model, group, online_clients, optimizer)

                # evaluate the sync time
                logging_sync_time(tracker)

                # logging.
                logging_globally(tracker, start_global_time)
                
                # reset start round time.
                start_global_time = time.time()

                # finish one epoch training and to decide if we want to val our model.
            if args.finish_one_epoch:
                # each worker finish one epoch training.
                if args.evaluate:
                    if args.epoch % args.eval_freq ==0:
                        do_validate(args, model, optimizer, criterion, metrics, val_loader)

                # refresh the logging cache at the begining of each epoch.
                args.finish_one_epoch = False
                tracker = define_local_training_tracker()

            # determine if the training is finished.
            if is_stop(args):
                #Last Synch
                log('Enter synching', args.debug)
                args.global_index += 1

                # broadcast gradients to other nodes by using reduce_sum.
                old_model = aggregate_gradients(
                    args, old_model, model, optimizer, True)

                print("Total number of samples seen on device {} is {}".format(args.graph.rank, args.local_data_seen))
                do_validate(args, model, optimizer, criterion, metrics, val_loader)
                return


            # display the logging info.
            logging_display_training(args, tracker)

            # reset load time for the tracker.
            tracker['start_load_time'] = time.time()

        # reshuffle the data.
        if args.reshuffle_per_epoch:
            log('reshuffle the dataset.', args.debug)
            del train_loader, val_loader
            gc.collect()
            log('reshuffle the dataset.', args.debug)
            train_loader, val_loader = define_dataset(args, shuffle=True)


def do_validate(args, model, optimizer, criterion, metrics, val_loader):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # wait until the whole group enters this function.
    dist.barrier()
    # evaluate the model.
    performance = validate(args, model, criterion, metrics, val_loader)

    # remember best prec@1 and save checkpoint.
    args.cur_prec1 = performance[0]
    is_best = args.cur_prec1 > args.best_prec1
    if is_best:
        args.best_prec1 = performance[0]
        args.best_epoch += [args.epoch_]

    # logging and display val info.
    logging_display_val(args)

    # save to the checkpoint.
    if args.graph.rank == 0:
        save_to_checkpoint({
            'arguments': args,
            'current_epoch': args.epoch,
            'local_index': args.local_index,
            'global_index': args.global_index,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': args.best_prec1,
            },
            is_best, dirname=args.checkpoint_root,
            filename='checkpoint.pth.tar',
            save_all=args.save_all_models)
    log('finished validation.', args.debug)


def validate(args, model, criterion, metrics, val_loader):
    """A function for model evaluation."""
    # define stat.
    tracker = define_val_tracker()

    # switch to evaluation mode
    model.eval()

    log('Do validation.', args.debug)
    for _input, _target in val_loader:
        # load data and check performance.
        _input, _target = _load_data_batch(args, _input, _target)

        with torch.no_grad():
            loss, performance = inference(
                model, criterion, metrics, _input, _target)
            tracker = update_performancec_tracker(
                tracker, loss, performance, _input.size(0))

    log('Aggregate val accuracy from different partitions.', args.debug)
    performance = [
        evaluate_gloabl_performance(tracker[x]) for x in ['top1', 'top5']
    ]

    log('Test at batch: {}. Epoch: {}. Process: {}. Prec@1: {:.3f} Prec@5: {:.3f}'.format(
        args.local_index, args.epoch, args.graph.rank, performance[0], performance[1]),
        debug=args.debug)
    return performance
