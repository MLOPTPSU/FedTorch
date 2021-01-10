# -*- coding: utf-8 -*-
import platform
from copy import deepcopy,copy

import torch
import torch.distributed as dist

from components.comps import create_components
from components.optimizer import define_optimizer
from utils.init_config import init_config
from components.dataset import define_dataset, _load_data_batch
from comms.utils.flow_utils import zero_copy
from logs.logging import log, configure_log, log_args
from logs.meter import define_val_tracker
from comms.communication import configure_sync_scheme

class Node():
    def __init__(self, rank):
        self.rank = rank

    def initialize(self):
        pass

    def reset_tracker(self, tracker):
        for k in tracker.keys():
            tracker[k].reset()

class Client(Node):
    def __init__(self, args, rank):
        super(Client, self).__init__(rank)
        self.args = copy(args)

        # Initialize the node
        self.initialize()
        # Load the dataset
        self.load_local_dataset()
        # Generate auxiliary models
        self.gen_aux_models()


    def initialize(self):
        init_config(self.args)
        self.model, self.criterion, self.scheduler, self.optimizer, self.metrics = create_components(self.args)
        self.args.finish_one_epoch = False
        # Create a model server on each client to keep a copy of the server model at each communication round.
        self.model_server = deepcopy(self.model)

        configure_log(self.args)
        log_args(self.args, debug=self.args.debug)
        log(
        'Rank {} with block {} on {} {}-{}'.format(
            self.args.graph.rank,
            self.args.graph.ranks_with_blocks[self.args.graph.rank],
            platform.node(),
            'GPU' if self.args.graph.on_cuda else 'CPU',
            self.args.graph.device
            ),
        debug=self.args.debug)

        self.all_clients_group = dist.new_group(self.args.graph.ranks)

    def load_local_dataset(self):
        load_test = True if self.args.graph.rank ==0 else False
        if self.args.fed_personal:
            self.train_loader, self.test_loader, self.val_loader= define_dataset(self.args, shuffle=True, test=load_test)
        else:
            self.train_loader, self.test_loader = define_dataset(self.args, shuffle=True, test=load_test)
        
        if self.args.data in ['mnist','fashion_mnist','cifar10', 'cifar100']:
            self.args.classes = torch.arange(10)
        elif self.args.data in ['synthetic']:
            self.args.classes = torch.arange(5)
        elif self.args.data in ['adult']:
            self.args.classes = torch.arange(2)

    def gen_aux_models(self):
        if self.args.federated:
            if self.args.federated_type == 'fedgate':
                self.model_delta = zero_copy(self.model)
                self.model_memory = zero_copy(self.model)
            elif self.args.federated_type == 'qsparse':
                self.model_memory = zero_copy(self.model)
            elif self.args.federated_type == 'scaffold':
                self.model_client_control = zero_copy(self.model)
                self.model_server_control = zero_copy(self.model)
            elif self.args.federated_type == 'fedadam':
                # Initialize the parameter for FedAdam https://arxiv.org/abs/2003.00295
                self.args.fedadam_v =  [self.args.fedadam_tau ** 2] * len(list(self.model.parameters()))
            elif self.args.federated_type == 'apfl':
                self.model_personal = deepcopy(self.model)
                self.optimizer_personal = define_optimizer(self.args, self.model_personal)
            elif self.args.federated_type == 'afl':
                self.lambda_vector = torch.zeros(self.args.graph.n_nodes)
            elif self.args.federated_type == 'perfedme':
                self.model_personal = deepcopy(self.model)
                self.optimizer_personal = define_optimizer(self.args, self.model_personal)
            elif self.args.federated_type == 'qffl':
                self.full_loss = 0.0
            if self.args.federated_drfa:
                self.kth_model  = zero_copy(self.model)
                self.lambda_vector = torch.zeros(self.args.graph.n_nodes)
    
    def zero_avg(self):
        self.model_avg = zero_copy(self.model)
        self.model_avg_tmp = zero_copy(self.model)