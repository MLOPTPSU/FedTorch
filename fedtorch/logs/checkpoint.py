# -*- coding: utf-8 -*-
import gc
import shutil
import time
from os.path import join, isfile

import torch

from fedtorch.utils.op_paths import build_dirs, remove_folder


def get_checkpoint_folder_name(args):
    # return datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    time_id = str(int(time.time()))
    if args.growing_batch_size:
        mode = 'growing_batch_size' 
    elif args.federated:
        mode = 'federated'
    else:
        mode = 'distributed'
    
    if args.federated:
        time_id += '_l2-{}_lr-{}_num_comms-{}_num_epochs-{}_batchsize-{}_blocksize-{}_localstep-{}_mode-{}_{}_clients_rate-{}'.format(
            args.weight_decay,
            args.lr,
            args.num_comms,
            args.num_epochs_per_comm,
            args.batch_size,
            args.blocks,
            args.local_step,
            mode,
            args.federated_type,
            args.online_client_rate
        )
    else:
        time_id += '_l2-{}_lr-{}_epochs-{}_batchsize-{}_blocksize-{}_localstep-{}_mode-{}'.format(
            args.weight_decay,
            args.lr,
            args.num_epochs,
            args.batch_size,
            args.blocks,
            args.local_step,
            mode
        )
    return time_id


def init_checkpoint(args):
    # init checkpoint dir.
    args.checkpoint_root = join(
        args.checkpoint, args.data, args.arch,
        args.experiment if args.experiment is not None else '',
        args.timestamp)
    args.checkpoint_dir = join(args.checkpoint_root, str(args.graph.rank))
    args.save_some_models = args.save_some_models.split(',')

    # if the directory does not exists, create them.
    if args.debug:
        build_dirs(args.checkpoint_dir)


def _save_to_checkpoint(state, dirname, filename):
    checkpoint_path = join(dirname, filename)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def save_to_checkpoint(state, is_best, dirname, filename, save_all=False):
    # save full state.
    args = state['arguments']
    checkpoint_path = _save_to_checkpoint(state, dirname, filename)
    best_model_path = join(dirname, 'model_best.pth.tar')
    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)
    if save_all:
        shutil.copyfile(checkpoint_path, join(
            dirname,
            'checkpoint_epoch_%s.pth.tar' % state['current_epoch']))
    elif str(state['current_epoch']) in args.save_some_models:
        shutil.copyfile(checkpoint_path, join(
            dirname,
            'checkpoint_epoch_%s.pth.tar' % state['current_epoch']))


def check_resume_status(args, old_args):
    signal = (args.data == old_args.data) and \
        (args.batch_size == old_args.batch_size) and \
        (args.num_epochs >= old_args.num_epochs)
    print('the status of previous resume: {}'.format(signal))
    return signal


def maybe_resume_from_checkpoint(args, model, optimizer):
    if args.resume:
        if args.checkpoint_index is not None:
            # reload model from a specific checkpoint index.
            checkpoint_index = '_epoch_' + args.checkpoint_index
        else:
            # reload model from the latest checkpoint.
            checkpoint_index = ''
        checkpoint_path = join(
            args.resume, 'checkpoint{}.pth.tar'.format(checkpoint_index))
        print('try to load previous model from the path:{}'.format(
              checkpoint_path))

        if isfile(checkpoint_path):
            print("=> loading checkpoint {} for {}".format(
                args.resume, args.graph.rank))

            # get checkpoint.
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if not check_resume_status(args, checkpoint['arguments']):
                print('=> the checkpoint is not correct. skip.')
            else:
                # restore some run-time info.
                args.local_index = checkpoint['local_index']
                args.best_prec1 = checkpoint['best_prec1']
                args.best_epoch = checkpoint['arguments'].best_epoch

                # reset path for log.
                # remove_folder(args.checkpoint_root)
                args.checkpoint_root = args.resume
                args.checkpoint_dir = join(args.resume, str(args.graph.rank))
                # restore model.
                model.load_state_dict(checkpoint['state_dict'])
                # restore optimizer.
                optimizer.load_state_dict(checkpoint['optimizer'])
                # logging.
                print("=> loaded model from path '{}' checkpointed at (epoch {})"
                      .format(args.resume, checkpoint['current_epoch']))

                # try to solve memory issue.
                del checkpoint
                torch.cuda.empty_cache()
                gc.collect()
                return
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
