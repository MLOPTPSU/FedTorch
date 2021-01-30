# -*- coding: utf-8 -*-
import torch
import platform
from copy import deepcopy,copy

import torch.distributed as dist

from fedtorch.parameters import get_args
from fedtorch.comms.trainings.federated import (train_and_validate_federated_centered,
                                                train_and_validate_apfl_centered,
                                                train_and_validate_drfa_centered,
                                                train_and_validate_afl_centered,
                                                train_and_validate_perfedme_centered)
from fedtorch.nodes import ClientCentered, ServerCentered


def main(args):
    """Non-distributed training."""
    # Create Clients and the Server
    ClientNodes ={}
    for i in range(args.num_workers):
        if args.data in ['emnist', 'emnist_full','synthetic'] or i==0:
            ClientNodes[i] = ClientCentered(args,i)
        else:
            ClientNodes[i] = ClientCentered(args,i, Partitioner=ClientNodes[0].Partitioner)
        

    ServerNode = ServerCentered(ClientNodes[0].args,ClientNodes[0].model) 
    ServerNode.enable_grad(ClientNodes[0].train_loader)
    # train and evaluate model.
    if ServerNode.args.federated_drfa:
        train_and_validate_drfa_centered(ClientNodes, ServerNode)
    else:
        if ServerNode.args.federated_type == 'apfl':
            train_and_validate_apfl_centered(ClientNodes, ServerNode)
        elif ServerNode.args.federated_type == 'perfedme':
            train_and_validate_perfedme_centered(ClientNodes, ServerNode)
        elif ServerNode.args.federated_type == 'afl':
            train_and_validate_afl_centered(ClientNodes, ServerNode)
        elif ServerNode.args.federated_type in ['fedavg','scaffold','fedgate','qsparse','fedprox','qffl','perfedavg']:
            train_and_validate_federated_centered(ClientNodes, ServerNode)
        else:
            raise NotImplementedError
    return



if __name__ == '__main__':
    args = get_args()
    main(args)
