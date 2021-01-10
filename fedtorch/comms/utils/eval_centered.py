# -*- coding: utf-8 -*-
import numpy as np

import torch

from fedtorch.components.dataset import _load_data_batch
from fedtorch.comms.utils.eval import inference, inference_personal
from fedtorch.logs.logging import log, update_performancec_tracker


def do_validate_centered(args, 
                        model,
                        criterion, 
                        metrics, 
                        optimizer,
                        val_loader, 
                        val_tracker,
                        personal=False,
                        val=True,
                        model_personal=None,
                        alpha=0.0,
                        local=False,
                        ):
    """Evaluate the model on the validation dataset."""
    
    # Finding the robust loss using gradient ascent
    if 'robust' in args.arch:
        tmp_noise = torch.clone(model.noise.data)
        model.noise.data = torch.zeros_like(tmp_noise)
        for _input, _target in val_loader:
            _input, _target = _load_data_batch(args, _input, _target)
            optimizer.zero_grad()
            loss, performance = inference(model, criterion, metrics, _input, _target)
            if model.noise.grad is None:
                loss.backward()
                optimizer.zero_grad()
                loss, performance = inference(model, criterion, metrics, _input, _target)
            model.noise.grad.data = torch.autograd.grad(loss, model.noise)[0]
            optimizer.step(
                apply_lr=False,
                scale= -0.01,
                apply_in_momentum=False,
                apply_out_momentum=args.out_momentum)
            if torch.norm(model.noise.data) > 1:
                model.noise.data /= torch.norm(model.noise.data)
        
    # switch to evaluation mode
    model.eval()    
    
    if personal:
        if model_personal is None:
            raise ValueError("model_personal should not be None for personalized mode!")
        model_personal.eval()
        # log('Do validation on the personal models.', args.debug)
    # else:
        # log('Do validation on the client models.', args.debug)
    for _input, _target in val_loader:
        # load data and check performance.
        _input, _target = _load_data_batch(args, _input, _target)

        if _input.size(0)==1:
            break

        with torch.no_grad():
            if personal:
                loss, performance = inference_personal(
                    model_personal, model, alpha, criterion, metrics, _input, _target)
            else:
                loss, performance = inference(
                    model, criterion, metrics, _input, _target, rnn= args.arch in ['rnn'])
            val_tracker = update_performancec_tracker(
                val_tracker, loss, performance, _input.size(0))
    # if personal and not val:
    #     print("acc in rank {} is {}".format(args.graph.rank,val_tracker['top1'].avg))
    if 'robust' in args.arch:
        model.noise.data = tmp_noise
    return


def log_validation_centered(args, val_tracker, personal=False, val=True, local=False):
    # log('Aggregate val performance from different clients.', args.debug)
    performance = [
        val_tracker[x].avg for x in ['top1', 'top5','losses']
    ]
    pretext = []
    pretext.append('Personal' if personal or local else 'Global')
    pretext.append('validation' if val else 'train')

    log('{} performance for {} at batch: {}. Epoch: {}. Process: {}. Prec@1: {:.3f} Prec@5: {:.3f} Loss: {:.3f} Comm: {}'.format(
                pretext[0], pretext[1], args.local_index, args.epoch, args.graph.rank, performance[0], performance[1], performance[2], args.rounds_comm),
                debug=args.debug)
    return

def log_validation_per_client_centered(args, OnlineClients, online_clients, val=True, local=False):
    # log('Aggregate val performance from different clients.', args.debug)
    acc = []
    for oc in online_clients:
        if local:
            if val:
                acc.append(OnlineClients[oc].local_personal_val_tracker['top1'].avg)
            else:
                acc.append(OnlineClients[oc].local_val_tracker['top1'].avg)
        else:
            if val:
                acc.append(OnlineClients[oc].global_personal_val_tracker['top1'].avg)
            else:
                acc.append(OnlineClients[oc].global_val_tracker['top1'].avg)

    log('{} per client stat for {} at batch: {}. Epoch: {}. Process: {}. Worst: {:.3f} Best: {:.3f} Var: {:.3f} Comm: {}'.format(
        'Personal' if local else 'Global', 'validation' if val else 'train',
        args.local_index, args.epoch, args.graph.rank, np.min(acc), np.max(acc), np.std(acc), args.rounds_comm),
        debug=args.debug)
    return

def log_test_centered(args, val_tracker):
    performance = [
        val_tracker[x].avg for x in ['top1', 'top5','losses']
    ]
    log('Test at batch: {}. Epoch: {}. Process: {}. Prec@1: {:.3f} Prec@5: {:.3f} Loss: {:.3f} Comm: {}'.format(
        args.local_index, args.epoch, args.graph.rank, performance[0], performance[1], performance[2], args.rounds_comm),
        debug=args.debug)