# -*- coding: utf-8 -*-
import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from components.datasets.loader.imagenet_folder import define_imagenet_folder
from components.datasets.loader.svhn_folder import define_svhn_folder
from components.datasets.loader.epsilon_or_rcv1_folder import define_epsilon_or_rcv1_or_MSD_folder
from components.datasets.loader.synthetic_folder import define_synthetic_folder
from components.datasets.loader.adult_loader import AdultDataset, AdultDatasetTorch
from components.datasets.loader.federated_datasets import EMNIST, Synthetic, Shakespeare
from components.datasets.loader.libsvm_datasets import LibSVMDataset


def _get_cifar(name, root, split, transform, target_transform, download):
    is_train = (split == 'train')

    # decide normalize parameter.
    if name == 'cifar10':
        dataset_loader = datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif name == 'cifar100':
        dataset_loader = datasets.CIFAR100
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    # decide data type.
    if is_train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), 4),
            transforms.ToTensor(),
            normalize])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    return dataset_loader(root=root,
                          train=is_train,
                          transform=transform,
                          target_transform=target_transform,
                          download=download)


def _get_mnist(root, split, transform, target_transform, download):
    is_train = (split == 'train')

    if is_train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    return datasets.MNIST(root=root,
                          train=is_train,
                          transform=transform,
                          target_transform=target_transform,
                          download=download)

def _get_emnist(root, split, client_id, download, val=False):
    # if split == 'train':
    #     train_dataset = EMNIST(root,'train',client_id=client_id, download=download)
    #     if val:
    #         val_dataset = EMNIST(root,'val',client_id=client_id, download=download)
    #         dataset = (train_dataset, val_dataset)
    #     else:
    #         dataset = train_dataset
    # else:
    #     dataset = EMNIST(root,'test', download=download)
    
    # return dataset
    if split == 'train':
        return EMNIST(root,'train',client_id=client_id, download=download)
    elif split == 'val':
        return EMNIST(root,'val',client_id=client_id, download=download)
    elif split == 'test':
        return EMNIST(root,'test', download=download)
    else:
        raise ValueError('The split {} does not exist! It should be from train, val, or test.'.format(split))

def _get_shakespeare(root, split, client_id, download, val=False, batch_size=2, seq_len=50):
    if split == 'train':
        return Shakespeare(root,'train',client_id=client_id, download=download, batch_size=batch_size, seq_len=seq_len)
    elif split == 'val':
        return Shakespeare(root,'val',client_id=client_id, download=download, batch_size=batch_size, seq_len=seq_len)
    elif split == 'test':
        return Shakespeare(root,'test', download=download)
    else:
        raise ValueError('The split {} does not exist! It should be from train, val, or test.'.format(split))


def _get_fashion_mnist(root, split, transform, target_transform, download):
    is_train = (split == 'train')
    return  datasets.FashionMNIST(
                    root = root,
                    train = is_train,
                    download = download,
                    transform = transforms.Compose([
                        transforms.ToTensor()                                 
                    ])
                )


def _get_stl10(root, split, transform, target_transform, download):
    return datasets.STL10(root=root,
                          split=split,
                          transform=transform,
                          target_transform=target_transform,
                          download=download)


def _get_svhn(root, split, transform, target_transform, download):
    is_train = (split == 'train')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return define_svhn_folder(root=root,
                              is_train=is_train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)


def _get_imagenet(args, name, datasets_path, split):
    is_train = (split == 'train')
    root = os.path.join(
        datasets_path,
        'lmdb' if 'downsampled' not in name else 'lmdb_32x32'
        ) if args.use_lmdb_data else datasets_path

    if is_train:
        root = os.path.join(root, 'train{}'.format(
            '' if not args.use_lmdb_data else '.lmdb')
        )
    else:
        root = os.path.join(root, 'val{}'.format(
            '' if not args.use_lmdb_data else '.lmdb')
        )
    return define_imagenet_folder(args,
        name=name, root=root, flag=args.use_lmdb_data,
        cuda=args.graph.on_cuda)


def _get_epsilon_or_rcv1_or_MSD(args, root, name, split):
    # root = os.path.join(root, '{}_{}.lmdb'.format(name, split))
    # return define_epsilon_or_rcv1_or_MSD_folder(args, root)
    return LibSVMDataset(root,name,split)


def _get_synthetic(args, root, name, split):
    reg = 'least_square' in args.arch
    # pattern = split == 'train'
    # if pattern:
    #     root = os.path.join(root, '{}_{}_{}.lmdb'.format(name, args.graph.rank, split))
    # else:
    #     root = os.path.join(root, '{}_{}.lmdb'.format(name, split))
    # return define_synthetic_folder(args, root, pattern=pattern)
    if args.federated_type == 'mafl':
        dim = 0 #The dimension will be randomly decided
    else:
        dim = 60
    return Synthetic(root, split=split, client_id=args.graph.rank,
                    num_tasks=args.graph.n_nodes, alpha=args.synthetic_alpha, 
                    beta=args.synthetic_beta, regression=reg,num_dim=dim)



def _get_adult(args, root, name, split):
    dataset = AdultDataset(root)
    # dataset = AdultDataset1(root)
    return AdultDatasetTorch(dataset, split)


def get_dataset(
        args, name, datasets_path, split='train', transform=None,
        target_transform=None, download=True):
    # create data folder if it does not exist.
    # if args.data =='synthetic':
    #     root = os.path.join(datasets_path, name + '{}-{}'.format(args.synthetic_alpha, args.synthetic_beta))
    # else:
    root = os.path.join(datasets_path, name)

    if name == 'cifar10' or name == 'cifar100':
        return _get_cifar(
            name, root, split, transform, target_transform, download)
    elif name == 'svhn':
        return _get_svhn(root, split, transform, target_transform, download)
    elif name == 'mnist':
        return _get_mnist(root, split, transform, target_transform, download)
    elif name == 'emnist':
        return _get_emnist(root, split,args.graph.rank, download, args.fed_personal)
    elif name == 'fashion_mnist':
        return _get_fashion_mnist(root, split, transform, target_transform, download)
    elif name == 'stl10':
        return _get_stl10(root, split, transform, target_transform, download)
    elif 'imagenet' in name:
        return _get_imagenet(args, name, datasets_path, split)
    elif name == 'epsilon':
        return _get_epsilon_or_rcv1_or_MSD(args, root, name, split)
    elif name == 'rcv1':
        return _get_epsilon_or_rcv1_or_MSD(args, root, name, split)
    elif name == 'higgs':
        return _get_epsilon_or_rcv1_or_MSD(args, root, name, split)
    elif name == 'MSD':
        return _get_epsilon_or_rcv1_or_MSD(args, root, name, split)
    elif name == 'synthetic':
        return _get_synthetic(args, root, name, split)
    elif name == 'adult':
        return _get_adult(args, root, name, split)
    elif name == 'shakespeare':
        return _get_shakespeare(root, split,args.graph.rank, download, 
                args.fed_personal, args.batch_size, args.rnn_seq_len)
    else:
        raise NotImplementedError
