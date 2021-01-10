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
from nodes import Client
from logs.logging import log, configure_log, log_args


def main(args):
    """distributed training via mpi backend."""
    dist.init_process_group('mpi')

    client = Client(args, dist.get_rank())
    # train and evaluate model.
    if args.federated:
        if args.federated_drfa:
            train_and_validate_federated_drfa(client)
        else:
            if args.federated_type == 'apfl':
                train_and_validate_federated_apfl(client)
            elif args.federated_type =='afl':
                train_and_validate_federated_afl(client)
            elif args.federated_type in ['fedavg','scaffold','fedgate','qsparse','fedprox']:
                train_and_validate_federated(client)
            else:
                raise NotImplementedError
    else:
        train_and_validate(client)
        pass

    return

if __name__ == '__main__':
    args = get_args()
    main(args)
