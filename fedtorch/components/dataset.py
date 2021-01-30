# -*- coding: utf-8 -*-
import time

import torch
import numpy as np

from fedtorch.logs.logging import log
from fedtorch.components.datasets.partition import DataPartitioner, GrowingBatchPartitioner, FederatedPartitioner
from fedtorch.components.datasets.prepare_data import get_dataset


def _load_data_batch(args, _input, _target):
    if 'least_square' in args.arch:
        _input = _input.float()
        _target = _target.unsqueeze_(1).float()
    else:
        if 'epsilon' in args.data or 'url' in args.data or 'rcv1' in args.data or 'higgs' in args.data:
            _input, _target = _input.float(), _target.long()

    if args.graph.on_cuda:
        _input, _target = _input.cuda(), _target.cuda()
    return _input, _target


def load_data_batch(args, _input, _target, tracker):
    """Load a mini-batch and record the loading time."""
    # get variables.
    start_data_time = time.time()

    _input, _target = _load_data_batch(args, _input, _target)

    # measure the data loading time
    end_data_time = time.time()
    tracker['data_time'].update(end_data_time - start_data_time)
    tracker['end_data_time'] = end_data_time
    return _input, _target


def define_dataset(args, shuffle, test=True, Partitioner=None, return_partitioner=False):
    log('create {} dataset for rank {}'.format(args.data, args.graph.rank), args.debug)

    train_loader = partition_dataset(args, shuffle, dataset_type='train', 
                            Partitioner=Partitioner, return_partitioner=return_partitioner)
    if return_partitioner:
        train_loader, Partitioner = train_loader
    if args.fed_personal:
        if args.federated_type == 'perfedavg':
            train_loader, val_loader, val_loader1 = train_loader
        else:
            train_loader, val_loader = train_loader
    if test:
        test_loader = partition_dataset(args, shuffle, dataset_type='test')
    else:
        test_loader=None

    get_data_stat(args, train_loader, test_loader)
    if args.fed_personal:
        if args.federated_type == 'perfedavg':
            out = [train_loader, test_loader, val_loader, val_loader1]
        else:
            out = [train_loader, test_loader, val_loader]
    else:
        out = [train_loader, test_loader]
    if return_partitioner:
        out = (out, Partitioner)
    return out

def partitioner(args, dataset, shuffle, world_size,
                partition_type='normal', return_partitioner=False):
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    if partition_type == 'normal':
        partition = DataPartitioner(args, dataset, shuffle, partition_sizes)
    elif partition_type == 'growing':
        partition = GrowingBatchPartitioner(args, dataset, partition_sizes)
    elif partition_type == 'noniid':
        partition = FederatedPartitioner(args,dataset, shuffle)
    if return_partitioner:
        return partition.use(args.graph.rank), partition
    else:
        return partition.use(args.graph.rank)


def partition_dataset(args, shuffle, dataset_type, Partitioner=None, return_partitioner=False):
    """ Given a dataset, partition it. """
    if Partitioner is None:
        dataset = get_dataset(args, args.data, args.data_dir, split=dataset_type)
    else:
        dataset = Partitioner.data
    # Federated Dataset with Validation
    # if args.data in ['emnist'] and args.fed_personal and dataset_type=='train':
    #     dataset, dataset_val = dataset
    batch_size = args.batch_size
    world_size = args.graph.n_nodes

    # partition data.
    if args.partition_data and dataset_type == 'train':
        if args.iid_data:
            if args.data in ['emnist', 'emnist_full','synthetic','shakespeare']:
                raise ValueError('The dataset {} does not have a structure for iid distribution of data'.format(args.data))
            if args.growing_batch_size:
                pt = 'growing'
            else:
                pt = 'normal'
        else:
            if args.data not in ['mnist','fashion_mnist','emnist', 'emnist_full','cifar10','cifar100','adult','synthetic','shakespeare']:
                    raise NotImplementedError("""Non-iid distribution of data for dataset {} is not implemented.
                        Set the distribution to iid.""".format(args.data))
            if args.growing_batch_size:
                raise ValueError('Growing Minibatch Size is not designed for non-iid data distribution')
            else:
                pt = 'noniid'
        
        if Partitioner is None:
            if return_partitioner:
                data_to_load, Partitioner = partitioner(args, dataset, shuffle,world_size, 
                                                    partition_type=pt, return_partitioner=True)
            else:
                data_to_load = partitioner(args, dataset, shuffle,world_size, partition_type=pt)
            log('Make {} data partitions and use the subdata.'.format(pt), args.debug)
        else:
            data_to_load = Partitioner.use(args.graph.rank)
            log('use the loaded partitioner to load the data.', args.debug)
    else:
        if Partitioner is not None:
            raise ValueError('Partitioner is provided but data partition method is not defined!')
        # If test dataset needs to be partitioned this should be changed
        data_to_load = dataset
        log('used whole data.', args.debug)

    # Log stats about the dataset to laod
    if dataset_type == 'train':
        args.train_dataset_size = len(dataset)
        log('  We have {} samples for {}, \
            load {} data for process (rank {}).'.format(
            len(dataset), dataset_type, len(data_to_load), args.graph.rank), args.debug)
    else:
        args.val_dataset_size = len(dataset)
        log('  We have {} samples for {}, \
            load {} val data for process (rank {}).'.format(
            len(dataset), dataset_type, len(data_to_load), args.graph.rank), args.debug)

    # Batching
    if (args.growing_batch_size) and (dataset_type == 'train'):
        batch_sampler = GrowingMinibatchSampler(data_source=data_to_load,
                                                num_epochs=args.num_epochs,
                                                num_iterations=args.num_iterations,
                                                base_batch_size=args.base_batch_size,
                                                max_batch_size=args.max_batch_size
                                                )
        args.num_epochs = batch_sampler.num_epochs
        args.num_iterations = batch_sampler.num_iterations
        args.total_data_size = len(data_to_load)
        args.num_samples_per_epoch = len(data_to_load) / args.num_epochs
        data_loader = torch.utils.data.DataLoader(
            data_to_load, batch_sampler=batch_sampler,
            num_workers=args.num_workers, pin_memory=args.pin_memory)
        log('we have {} batches for {} for rank {}.'.format(
                len(data_loader), dataset_type, args.graph.rank), args.debug)
    elif dataset_type == 'train':
        # Adjust stopping criteria
        if args.stop_criteria == 'epoch':
            args.num_iterations = int(len(data_to_load) * args.num_epochs / batch_size)
        else:
            args.num_epochs = int(args.num_iterations * batch_size / len(data_to_load))
        args.total_data_size = len(data_to_load) * args.num_epochs
        args.num_samples_per_epoch = len(data_to_load)

        # Generate validation data part
        if args.fed_personal:
            if args.data in ['emnist', 'emnist_full', 'shakespeare']:
                data_to_load_train = data_to_load
                data_to_load_val = get_dataset(args, args.data, args.data_dir, split='val')
            
            if args.federated_type == "perfedavg":
                if args.data in ['emnist', 'emnist_full', 'shakespeare']:
                    #TODO: make this size a paramter
                    val_size = int(0.1*len(data_to_load))
                    data_to_load_train, data_to_load_val1 = torch.utils.data.random_split(data_to_load,[len(data_to_load) - val_size, val_size])
                else:
                    val_size = int(0.1*len(data_to_load))
                    data_to_load_train, data_to_load_val, data_to_load_val1 = torch.utils.data.random_split(data_to_load,[len(data_to_load) - 3 * val_size, 2 * val_size, val_size])
                data_loader_val1 = torch.utils.data.DataLoader(
                                    data_to_load_val1, batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=5, pin_memory=args.pin_memory,
                                    drop_last=False)
            else:
                if args.data not in ['emnist', 'emnist_full', 'shakespeare']:
                    val_size = int(0.2*len(data_to_load))
                    data_to_load_train, data_to_load_val = torch.utils.data.random_split(data_to_load,[len(data_to_load) - val_size, val_size])
            # Generate data loaders
            data_loader_val = torch.utils.data.DataLoader(
                                    data_to_load_val, batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=5, pin_memory=args.pin_memory,
                                    drop_last=False)
        
            data_loader_train = torch.utils.data.DataLoader(
                                        data_to_load_train, batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=5, pin_memory=args.pin_memory,
                                        drop_last=False)
            if args.federated_type == 'perfedavg':
                data_loader = [data_loader_train, data_loader_val, data_loader_val1]
            else:
                data_loader = [data_loader_train, data_loader_val]
                
            log('we have {} batches for {} for rank {}.'.format(
                len(data_loader[0]), 'train', args.graph.rank), args.debug)
            log('we have {} batches for {} for rank {}.'.format(
                len(data_loader[1]), 'val', args.graph.rank), args.debug)
        else:
            data_loader = torch.utils.data.DataLoader(
                                        data_to_load, batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=5, pin_memory=args.pin_memory,
                                        drop_last=False)
            log('we have {} batches for {} for rank {}.'.format(
            len(data_loader), 'train', args.graph.rank), args.debug)
    else:
        data_loader = torch.utils.data.DataLoader(
            data_to_load, batch_size=batch_size,
            shuffle=False,
            num_workers=5, pin_memory=args.pin_memory,
            drop_last=False)
        log('we have {} batches for {} for rank {}.'.format(
            len(data_loader), dataset_type, args.graph.rank), args.debug)
    if return_partitioner:
        return data_loader, Partitioner
    else:
        return data_loader


def get_data_stat(args, train_loader, test_loader=None):
    # get the data statictics (on behalf of each worker) for train.
    # args.num_batches_train_per_device_per_epoch = \
    #     len(train_loader) // args.graph.n_nodes \
    #     if not args.partition_data else len(train_loader)
    args.num_batches_train_per_device_per_epoch = len(train_loader)
    args.num_whole_train_batches_per_worker = \
        args.num_batches_train_per_device_per_epoch * args.num_epochs
    args.num_warmup_train_batches_per_worker = \
        args.num_batches_train_per_device_per_epoch * args.lr_warmup_epochs
    args.num_iterations_per_worker = args.num_iterations #// args.graph.n_nodes

    # get the data statictics (on behalf of each worker) for val.
    if test_loader is not None:
        args.num_batches_val_per_device_per_epoch = len(test_loader)
    else:
        args.num_batches_val_per_device_per_epoch=0


    # define some parameters for training.
    log('we have {} epochs, \
        {} mini-batches per device for training. \
        {} mini-batches per device for test. \
        The batch size: {}.'.format(
            args.num_epochs,
            args.num_batches_train_per_device_per_epoch,
            args.num_batches_val_per_device_per_epoch,
            args.batch_size), args.debug)


class GrowingMinibatchSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, 
                data_source,
                num_epochs=None, 
                num_iterations=None, 
                base_batch_size=2,
                rho=1.01, 
                max_batch_size=0):
        self.data_source = data_source
        self.base_batch_size = base_batch_size
        self.rho = rho
        self.num_samples_per_epoch = len(data_source)
        self.idx_pool = []
        self.max_batch_size = max_batch_size
        if num_epochs is None:
            if num_iterations is None:
                raise ValueError('One of the number of epochs or number of iterations should be provided.')
            self.num_iterations = num_iterations
            self.num_epochs = int(self.base_batch_size * (rho**self.num_iterations - 1) / ((rho - 1) * self.num_samples_per_epoch)) + 1
        else:
            self.num_epochs = num_epochs
            self.num_iterations = int(np.log(self.num_samples_per_epoch*self.num_epochs*(self.rho-1)/self.base_batch_size + 1) / np.log(self.rho))+1
        for _ in range(self.num_epochs):
            self.idx_pool.extend(np.random.permutation(self.num_samples_per_epoch).tolist())
    
        self.batch_size=[int(self.base_batch_size * self.rho**i)+1 for i in range(self.num_iterations)]
        if max_batch_size:
            b = np.array(self.batch_size)
            idx = np.squeeze(np.argwhere(b > max_batch_size))
            if len(idx) >= 1:
                self.batch_size = self.batch_size[:idx[0]] + [max_batch_size] * (np.sum(b[idx]) // max_batch_size)
                if np.sum(b[idx]) // max_batch_size:
                    self.batch_size += [np.sum(b[idx]) % max_batch_size]
                self.num_iterations = len(self.batch_size)
        self.total_num_data = np.sum(self.batch_size)

    def __iter__(self):
        for bs in self.batch_size:
            batch = self.idx_pool[:bs]
            self.idx_pool = self.idx_pool[bs:]
            yield batch

    def __len__(self):
        return self.num_iterations
