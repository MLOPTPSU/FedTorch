# -*- coding: utf-8 -*-
from copy import deepcopy,copy

import torch
from torch.utils.tensorboard import SummaryWriter

from fedtorch.components.comps import create_components
from fedtorch.components.optimizer_builder import build_optimizer_from_config
from fedtorch.utils.init_config import init_config_centered
from fedtorch.components.dataset_builder import build_dataset, _load_data_batch
from fedtorch.comms.utils.flow_utils import zero_copy
from fedtorch.logs.logging import log, configure_log, log_cfgs
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
    def __init__(self, cfg, rank, Partitioner=None):
        super(ClientCentered, self).__init__(rank)
        self.cfg = deepcopy(cfg)
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
        if hasattr(self.cfg,'federated') and self.cfg.federated.personal:
            self.local_personal_val_tracker = define_val_tracker()
            self.global_personal_val_tracker = define_val_tracker()

        if hasattr(self.cfg,'federated') and self.cfg.federated.sync_type == 'epoch':
            self.cfg.training.local_step = self.cfg.training.num_epochs_per_comm * len(self.train_loader)
            # Rebuild the sync scheme
            configure_sync_scheme(self.cfg)

    def initialize(self):
        init_config_centered(self.cfg, self.rank)
        self.model, self.criterion, self.scheduler, self.optimizer, self.metrics = create_components(self.cfg)
        self.cfg.finish_one_epoch = False

        if self.rank==0:
            configure_log(self.cfg)
            log_cfgs(self.cfg, debug=self.cfg.graph.debug)

    def make_model_consistent(self, ref_model):
        """make all initial models consistent with the server model (rank 0)
        """
        print('consistent model for process (rank {})'.format(self.cfg.graph.rank))
        for param, ref_param in zip(self.model.parameters(), ref_model.parameters()):
            param.data = ref_param.data
    
    def load_local_dataset(self,Partitioner):
        if self.cfg.data.dataset.type in ['emnist', 'emnist_full', 'synthetic',  'synthetic_polar','shakespeare']:
            if hasattr(self.cfg,'federated') and self.cfg.federated.personal:
                if self.cfg.federated.type == 'perfedavg':
                    self.train_loader, self.test_loader, self.val_loader,  self.val_loader1 = build_dataset(self.cfg, test=True)
                    if len(self.val_loader1.dataset) == 1 and self.cfg.training.batch_size > 1:
                        raise ValueError('Size of the validation dataset is too low!')
                    # self.val_iterator = iter(self.val_loader1)
                else:
                    self.train_loader, self.test_loader, self.val_loader = build_dataset(self.cfg, test=False)
                if len(self.val_loader.dataset) == 1 and self.cfg.training.batch_size > 1:
                    raise ValueError('Size of the validation dataset is too low!')
            else:
                self.train_loader, self.test_loader = build_dataset(self.cfg, test=False)
            
        else:
            if self.rank == 0:
                if self.cfg.federated.personal:
                    if self.cfg.federated.type == 'perfedavg':
                        (self.train_loader, self.test_loader, self.val_loader, self.val_loader1), self.Partitioner = build_dataset(self.cfg,
                                                                                                                                    test=False, 
                                                                                                                                    return_partitioner=True)
                        # self.val_iterator = iter(self.val_loader1)
                    else:
                        (self.train_loader, self.test_loader, self.val_loader), self.Partitioner = build_dataset(self.cfg, 
                                                                                                    test=False, return_partitioner=True)
                else:
                    (self.train_loader, self.test_loader), self.Partitioner = build_dataset(self.cfg, test=False,
                                                                                            return_partitioner=True)
            else:
                if self.cfg.federated.personal:
                    if self.cfg.federated.type == 'perfedavg':
                        self.train_loader, self.test_loader, self.val_loader, self.val_loader1 = build_dataset(self.cfg,
                                                                                                test=False, Partitioner=Partitioner)
                        self.val_iterator = iter(self.val_loader1)
                    else:
                        self.train_loader, self.test_loader, self.val_loader= build_dataset(self.cfg, test=False, Partitioner=Partitioner)
                else:
                    self.train_loader, self.test_loader = build_dataset(self.cfg, test=False, Partitioner=Partitioner)
        
        # if self.cfg.data in ['mnist','fashion_mnist','cifar10']:
        #     self.cfg.classes = torch.arange(10)
        # elif self.cfg.data in ['synthetic']:
        #     self.cfg.classes = torch.arange(5)
        # elif self.cfg.data in ['adult']:
        #     self.cfg.classes = torch.arange(2)

    def gen_aux_models(self):
        if self.cfg.training.type == 'federated':
            if self.cfg.federated.type == 'fedgate':
                self.model_delta = zero_copy(self.model)
                self.model_memory = zero_copy(self.model)
            elif self.cfg.federated.type == 'qsparse':
                self.model_memory = zero_copy(self.model)
            elif self.cfg.federated.type == 'scaffold':
                self.model_client_control = zero_copy(self.model)
            elif self.cfg.federated.type == 'fedadam':
                # Initialize the parameter for FedAdam https://arxiv.org/abs/2003.00295
                self.cfg.federated.fedadam_v =  [self.cfg.federated.fedadam_tau ** 2] * len(list(self.model.parameters()))
            elif self.cfg.federated.type in ['apfl', 'perfedme', 'fedshuffle', 'fedshuffleone']:
                self.model_personal = deepcopy(self.model)
                self.optimizer_personal = build_optimizer_from_config(self.cfg.optimizer, self.model_personal, self.cfg.graph.n_nodes, self.cfg.lr.lr)
            elif self.cfg.federated.type == 'qffl':
                self.full_loss = 0.0
            if self.cfg.training.drfa:
                self.kth_model  = zero_copy(self.model)
    
    def zero_avg(self):
        self.model_avg = zero_copy(self.model)
        self.model_avg_tmp = zero_copy(self.model)
    


class ServerCentered(Node):
    def __init__(self, cfg, model_server, rank=0):
        super(ServerCentered, self).__init__(0)
        self.cfg = copy(cfg)
        self.cfg.epoch=1
        self.rnn = self.cfg.model.type in ['rnn']
        
        # Initialize the node
        self.initialize()
        self.gen_aux_models()
        # Create trackers
        self.local_val_tracker = define_val_tracker()
        self.global_val_tracker = define_val_tracker()
        if hasattr(self.cfg, 'federated')  and self.cfg.federated.personal:
            self.local_personal_val_tracker = define_val_tracker()
            self.global_personal_val_tracker = define_val_tracker()
        self.global_test_tracker = define_val_tracker()
        
        # Load test dataset for server
        self.load_test_dataset()

    def initialize(self):
        self.cfg.tb_writer = SummaryWriter(log_dir=self.cfg.work_dir)
        self.model, self.criterion, self.scheduler, self.optimizer, self.metrics = create_components(self.cfg)

    def zero_grad(self):
        self.grad = zero_copy(self.model,self.rnn)
    
    def zero_avg(self):
        self.model_avg = zero_copy(self.model,self.rnn)

    def update_model(self):
        # with torch.no_grad():
        for p,g in zip(self.model.parameters(),self.grad.parameters()):
            p.data -= self.cfg.lr.lr_scale_at_sync * g.data
    
    def enable_grad(self,dataloader):
        # Initialize the grad on model params
        self.model.train()
        dataiter = iter(dataloader)
        _input, _target = next(dataiter)
        _input, _target = _load_data_batch(self.cfg, _input, _target)
        self.optimizer.zero_grad()
        output = self.model(_input)
        loss = self.criterion(output, _target)
        loss.backward()
        self.optimizer.zero_grad(set_to_none=False)
        assert next(self.model.parameters()).grad is not None
        return
    

    def gen_aux_models(self):
        if self.cfg.training.type == 'federated':
            if self.cfg.federated.type == 'scaffold':
                self.model_server_control = zero_copy(self.model)
            elif self.cfg.federated.type == 'sgdap':
                self.avg_noise_model = deepcopy(self.model)
                self.avg_noise_optimizer = build_optimizer_from_config(self.cfg.optimizer, self.avg_noise_model, self.cfg.graph.n_nodes, self.cfg.lr.lr)
            elif self.cfg.federated.type == 'afl':
                self.lambda_vector = torch.zeros(self.cfg.graph.n_nodes)
            elif self.cfg.federated.type == 'fedshuffle':
                self.shuffle_list = torch.arange(self.cfg.graph.n_nodes)[torch.randperm(self.cfg.graph.n_nodes)]
            elif self.cfg.federated.type == 'fedshuffleone':
                self.alphas = torch.ones(self.cfg.graph.n_nodes,self.cfg.graph.n_nodes) / self.cfg.graph.n_nodes
                self.shuffle_list = torch.arange(self.cfg.graph.n_nodes)[torch.randperm(self.cfg.graph.n_nodes)]
                self.personal_models = [deepcopy(self.model) for _ in range(self.cfg.graph.n_nodes)] 
                self.cfg.federated.personal=True
            if getattr(self.cfg.training,'drfa', False):
                self.kth_model  = zero_copy(self.model)
                self.lambda_vector = torch.zeros(self.cfg.graph.n_nodes)
    
    def load_test_dataset(self):
        if hasattr(self.cfg, 'federated') and self.cfg.federated.personal:
            if self.cfg.federated.type == 'perfedavg':
                _, self.test_loader,_,  _ = build_dataset(self.cfg, test=True)
            else:
                _, self.test_loader, _ = build_dataset(self.cfg, test=True)
        else:
            _, self.test_loader = build_dataset(self.cfg, test=True)