# -*- coding: utf-8 -*-
from fedtorch.components.optimizer import define_optimizer
from fedtorch.components.criterion import define_criterion
from fedtorch.components.metrics import define_metrics
from fedtorch.components.model import define_model
from fedtorch.components.scheduler import define_scheduler
from fedtorch.logs.checkpoint import maybe_resume_from_checkpoint


def create_components(args):
    """Create model, criterion and optimizer.
    If args.use_cuda is True, use ps_id as GPU_id.
    """
    model = define_model(args)

    # define the criterion and metrics.
    criterion = define_criterion(args)
    metrics = define_metrics(args, model)

    # define the lr scheduler.
    scheduler = define_scheduler(args)

    # define the optimizer.
    optimizer = define_optimizer(args, model)

    # place model and criterion.
    if args.graph.on_cuda:
        model.cuda()
        criterion = criterion.cuda()

    # (optional) reload checkpoint
    maybe_resume_from_checkpoint(args, model, optimizer)
    return model, criterion, scheduler, optimizer, metrics
