# -*- coding: utf-8 -*-
import torch

from fedtorch.components.metrics import accuracy, accuracy_per_class
from fedtorch.logs.logging import (log, 
                                   logging_display_val, 
                                   logging_display_test_summary,
                                   update_performancec_tracker, 
                                   update_performance_per_class)
from fedtorch.logs.meter import (define_val_tracker, 
                                 evaluate_gloabl_performance, 
                                 evaluate_local_performance, 
                                 define_per_class_acc_tracker)
from fedtorch.components.dataset import _load_data_batch
from fedtorch.logs.checkpoint import save_to_checkpoint

def inference(model, criterion, metrics, _input, _target, classes=None, rnn=False):
    """Inference on the given model and get loss and accuracy."""
    # if rnn:
    #     print("input size is:{}".format(_input.size(0)))
    #     model.init_hidden(_input.size(0))
    output = model(_input)
    loss = criterion(output, _target)
    performance = accuracy(output.data, _target, topk=metrics,rnn=rnn)

    if classes is not None:
        acc_per_class, count_per_class = accuracy_per_class(output.data, _target, classes)
        return loss, performance, (acc_per_class, count_per_class)
    return loss, performance

def inference_personal(model1, model2, alpha, criterion, metrics, _input, _target):
    """Inference on the given model and get loss and accuracy."""
    # TODO: merge with inference
    output1 = model1(_input)
    output2 = model2(_input)
    output = alpha * output1 + (1-alpha) * output2
    loss = criterion(output, _target)
    performance = accuracy(output.data, _target, topk=metrics)
    return loss, performance

def do_validate(args, 
                model, 
                optimizer, 
                criterion, 
                metrics, 
                data_loader, 
                group,
                data_mode='validation', 
                personal=False,
                model_personal=None,
                alpha=0.0,
                local=False,
                ):
    """Evaluate the model on the validation dataset."""

    model_mode = 'personal' if personal or local else 'global'

    tracker = define_val_tracker()
    if 'robust' in args.arch:
        tmp_noise = torch.clone(model.noise.data)
        # model.noise.data = torch.randn(tmp_noise.shape) * 0.1
        for _input, _target in data_loader:
            _input, _target = _load_data_batch(args, _input, _target)
            loss, performance = inference(model, criterion, metrics, _input, _target)
            grad = torch.autograd.grad(loss, model.noise)[0]
            model.noise.data.add_(grad,alpha=0.01)
            if torch.norm(model.noise.data) > 1:
                model.noise.data /= torch.norm(model.noise.data)
    # switch to evaluation mode
    model.eval()

    
    if personal:
        if model_personal is None:
            raise ValueError("model_personal should not be None for personalized mode for APFL!")
        model_personal.eval()
        # log('Do validation on the personal models.', args.debug)
    # else:
    #     if local:
    #         log('Do validation on the client models.', args.debug)
    #     else:
    #         log('Do validation on the global model.', args.debug)
    for _input, _target in data_loader:
        # load data and check performance.
        _input, _target = _load_data_batch(args, _input, _target)

        # Skip batches with one sample because of BatchNorm issue in some models!
        if _input.size(0)==1:
            break

        with torch.no_grad():
            if personal:
                loss, performance = inference_personal(
                    model_personal, model, alpha, criterion, metrics, _input, _target)
            else:
                loss, performance = inference(
                    model, criterion, metrics, _input, _target)
            tracker = update_performancec_tracker(
                tracker, loss, performance, _input.size(0))
    # if local and not val:
    #     print("loss in rank {} is {}".format(args.graph.rank,tracker['losses'].avg))
    # log('Aggregate val performance from different clients.', args.debug)
    if len(metrics) == 1:
        tracker['top5'].count = 1.0
        tracker['top5'].sum = 0.0
        tracker['top5'].avg = 0.0
    if data_mode == 'test' and model_mode =='global':
        # Only the server performs the test, do not need for aggregation
        performance = [
            evaluate_local_performance(tracker[x]) for x in ['top1', 'top5','losses']
        ]
    else:
        performance = [
            evaluate_gloabl_performance(tracker[x], group) for x in ['top1', 'top5','losses']
        ]


    logging_display_val(args,performance, mode=data_mode, personal=model_mode=='personal')

    if data_mode == 'test' and not personal:
        # remember best prec@1 and save checkpoint.
        args.cur_prec1 = performance[0]
        is_best = args.cur_prec1 > args.best_prec1
        if is_best:
            args.best_prec1 = performance[0]
            args.best_epoch += [args.epoch_]

        # logging and display val info.
        logging_display_test_summary(args, debug=args.debug)
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
    


    if 'robust' in args.arch:
        model.noise.data = tmp_noise
    return performance
