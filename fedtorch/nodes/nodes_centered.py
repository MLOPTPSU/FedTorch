# -*- coding: utf-8 -*-
from copy import deepcopy,copy

import torch

from fedtorch.components.comps import create_components
from fedtorch.components.optimizer import define_optimizer
from fedtorch.utils.init_config import init_config_centered
from fedtorch.components.dataset import define_dataset, _load_data_batch
from fedtorch.comms.utils.flow_utils import zero_copy
from fedtorch.logs.logging import log, configure_log, log_args
from fedtorch.logs.meter import define_val_tracker
from fedtorch.comms.algorithms.distributed import configure_sync_scheme

class Node():
    def __init__(self, rank):
        self.rank = rank

    def initialize(self):
        pass

    def reset_tracker(self, tracker):
        for k in tracker.keys():
            tracker[k].reset()


class ClientCentered(Node):
    def __init__(self, args, rank, Partitioner=None):
        super(ClientCentered, self).__init__(rank)
        self.args = copy(args)
        self.Partitioner = None

        # Initialize the node
        self.initialize()
        # Load the dataset
        self.load_local_dataset(Partitioner)
        # Generate auxiliary models
        self.gen_aux_models()

        # Create trackers
        self.local_val_tracker = define_val_tracker()
        self.global_val_tracker = define_val_tracker()
        if self.args.fed_personal:
            self.local_personal_val_tracker = define_val_tracker()
            self.global_personal_val_tracker = define_val_tracker()

        if self.args.federated_sync_type == 'epoch':
            self.args.local_step = self.args.num_epochs_per_comm * len(self.train_loader)
            # Rebuild the sync scheme
            configure_sync_scheme(self.args)

    def initialize(self):
        init_config_centered(self.args, self.rank)
        self.model, self.criterion, self.scheduler, self.optimizer, self.metrics = create_components(self.args)
        self.args.finish_one_epoch = False

        if self.rank==0:
            configure_log(self.args)
            log_args(self.args, debug=self.args.debug)

    def make_model_consistent(self, ref_model):
        """make all initial models consistent with the server model (rank 0)
        """
        print('consistent model for process (rank {})'.format(self.args.graph.rank))
        for param, ref_param in zip(self.model.parameters(), ref_model.parameters()):
            param.data = ref_param.data
    
    def load_local_dataset(self,Partitioner):
        if self.args.data in ['emnist', 'emnist_full', 'synthetic','shakespeare']:
            if self.args.fed_personal:
                if self.args.federated_type == 'perfedavg':
                    self.train_loader, self.test_loader, self.val_loader,  self.val_loader1 = define_dataset(self.args, shuffle=True, test=False)
                    if len(self.val_loader1.dataset) == 1 and self.args.batch_size > 1:
                        raise ValueError('Size of the validation dataset is too low!')
                    # self.val_iterator = iter(self.val_loader1)
                else:
                    self.train_loader, self.test_loader, self.val_loader = define_dataset(self.args, shuffle=True, test=False)
                if len(self.val_loader.dataset) == 1 and self.args.batch_size > 1:
                    raise ValueError('Size of the validation dataset is too low!')
            else:
                self.train_loader, self.test_loader = define_dataset(self.args, shuffle=True,test=False)
            
        else:
            if self.rank == 0:
                if self.args.fed_personal:
                    if self.args.federated_type == 'perfedavg':
                        (self.train_loader, self.test_loader, self.val_loader, self.val_loader1), self.Partitioner = define_dataset(self.args,
                                                                                                                                    shuffle=True,
                                                                                                                                    test=False, 
                                                                                                                                    return_partitioner=True)
                        # self.val_iterator = iter(self.val_loader1)
                    else:
                        (self.train_loader, self.test_loader, self.val_loader), self.Partitioner = define_dataset(self.args, shuffle=True,
                                                                                                    test=False, return_partitioner=True)
                else:
                    (self.train_loader, self.test_loader), self.Partitioner = define_dataset(self.args, shuffle=True,test=False,
                                                                                            return_partitioner=True)
            else:
                if self.args.fed_personal:
                    if self.args.federated_type == 'perfedavg':
                        self.train_loader, self.test_loader, self.val_loader, self.val_loader1 = define_dataset(self.args, shuffle=True,
                                                                                                test=False, Partitioner=Partitioner)
                        self.val_iterator = iter(self.val_loader1)
                    else:
                        self.train_loader, self.test_loader, self.val_loader= define_dataset(self.args, shuffle=True,
                                                                                                test=False, Partitioner=Partitioner)
                else:
                    self.train_loader, self.test_loader = define_dataset(self.args, shuffle=True, test=False, Partitioner=Partitioner)
        
        if self.args.data in ['mnist','fashion_mnist','cifar10']:
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
            elif self.args.federated_type == 'fedadam':
                # Initialize the parameter for FedAdam https://arxiv.org/abs/2003.00295
                self.args.fedadam_v =  [self.args.fedadam_tau ** 2] * len(list(self.model.parameters()))
            elif self.args.federated_type == 'apfl':
                self.model_personal = deepcopy(self.model)
                self.optimizer_personal = define_optimizer(self.args, self.model_personal)
            elif self.args.federated_type == 'perfedme':
                self.model_personal = deepcopy(self.model)
                self.optimizer_personal = define_optimizer(self.args, self.model_personal)
            elif self.args.federated_type == 'qffl':
                self.full_loss = 0.0
            if self.args.federated_drfa:
                self.kth_model  = zero_copy(self.model)
    
    def zero_avg(self):
        self.model_avg = zero_copy(self.model)
        self.model_avg_tmp = zero_copy(self.model)
    


class ServerCentered(Node):
    def __init__(self, args, model_server, rank=0):
        super(ServerCentered, self).__init__(0)
        self.args = copy(args)
        self.args.epoch=1
        self.rnn = self.args.arch in ['rnn']
        
        # Initialize the node
        self.initialize()
        self.gen_aux_models()
        # Create trackers
        self.local_val_tracker = define_val_tracker()
        self.global_val_tracker = define_val_tracker()
        if self.args.fed_personal:
            self.local_personal_val_tracker = define_val_tracker()
            self.global_personal_val_tracker = define_val_tracker()
        self.global_test_tracker = define_val_tracker()
        
        # Load test dataset for server
        self.load_test_dataset()

    def initialize(self):
        self.model, self.criterion, self.scheduler, self.optimizer, self.metrics = create_components(self.args)

    def zero_grad(self):
        self.grad = zero_copy(self.model,self.rnn)
    
    def zero_avg(self):
        self.model_avg = zero_copy(self.model,self.rnn)

    def update_model(self):
        # with torch.no_grad():
        for p,g in zip(self.model.parameters(),self.grad.parameters()):
            p.data -= self.args.lr_scale_at_sync * g.data
    
    def enable_grad(self,dataloader):
        # Initialize the grad on model params
        dataiter = iter(dataloader)
        _input, _target = next(dataiter)
        _input, _target = _load_data_batch(self.args, _input, _target)
        self.optimizer.zero_grad()
        output = self.model(_input)
        loss = self.criterion(output, _target)
        loss.backward()
        self.optimizer.zero_grad()
        return

    def gen_aux_models(self):
        if self.args.federated:
            if self.args.federated_type == 'scaffold':
                self.model_server_control = zero_copy(self.model)
            elif self.args.federated_type == 'sgdap':
                self.avg_noise_model = deepcopy(self.model)
                self.avg_noise_optimizer = define_optimizer(self.args, self.avg_noise_model)
            elif self.args.federated_type == 'afl':
                self.lambda_vector = torch.zeros(self.args.graph.n_nodes)
            if self.args.federated_drfa:
                self.kth_model  = zero_copy(self.model)
                self.lambda_vector = torch.zeros(self.args.graph.n_nodes)
    
    def load_test_dataset(self):
        if self.args.fed_personal:
            if self.args.federated_type == 'perfedavg':
                _, self.test_loader,_,  _ = define_dataset(self.args, shuffle=True, test=True)
            else:
                _, self.test_loader, _ = define_dataset(self.args, shuffle=True, test=True)
        else:
            _, self.test_loader = define_dataset(self.args, shuffle=True,test=True)