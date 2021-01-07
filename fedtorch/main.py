# -*- coding: utf-8 -*-
import platform

import torch.distributed as dist

from parameters import get_args
from components.comps import create_components
from utils.init_config import init_config
from comms.trainings.distributed import train_and_validate
from comms.trainings.federated import (train_and_validate_federated,
                                       train_and_validate_federated_apfl,
                                       train_and_validate_federated_drfa,
                                       train_and_validate_federated_afl)
from logs.logging import log, configure_log, log_args


def main(args):
    """distributed training via mpi backend."""
    dist.init_process_group('mpi')

    # init the config.
    init_config(args)
    print("Config is initialized")
    # create model and deploy the model.
    model, criterion, scheduler, optimizer, metrics = create_components(args)
    # config and report.
    configure_log(args)
    log_args(args, debug=args.debug)
    log(
        'Rank {} with block {} on {} {}-{}'.format(
            args.graph.rank,
            args.graph.ranks_with_blocks[args.graph.rank],
            platform.node(),
            'GPU' if args.graph.on_cuda else 'CPU',
            args.graph.device
            ),
        debug=args.debug)
    # train and evaluate model.
    if args.federated:
        if args.federated_drfa:
            train_and_validate_federated_drfa(args, model, criterion, scheduler, optimizer, metrics)
        else:
            if args.federated_type == 'apfl':
                train_and_validate_federated_apfl(args, model, criterion, scheduler, optimizer, metrics)
            elif args.federated_type =='afl':
                train_and_validate_federated_afl(args, model, criterion, scheduler, optimizer, metrics)
                # elif args.federated_type == 'perfedavg':
                #     train_and_validate_federated_perfedavg(args, model, criterion, scheduler, optimizer, metrics)
                # else:
            elif args.federated_type in ['fedavg','scaffold','fedgate','qsparse','fedprox']:
                train_and_validate_federated(args, model, criterion, scheduler, optimizer, metrics)
            else:
                raise NotImplementedError
    else:
        # train_and_validate(args, model, criterion, scheduler, optimizer, metrics)
        train_and_validate(args, model, criterion, scheduler, optimizer, metrics)

    return

if __name__ == '__main__':
    args = get_args()
    main(args)
