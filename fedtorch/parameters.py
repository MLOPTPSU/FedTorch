# -*- coding: utf-8 -*-
"""
Define parameters needed for training a model using distributed or federated schema
"""
import argparse
from os.path import join

import fedtorch.components.models as models
from fedtorch.logs.checkpoint import get_checkpoint_folder_name


def get_args():
    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__"))

    # feed them to the parser.
    parser = argparse.ArgumentParser(
        description='Parameters for running training on FedTorch package.')

    # add arguments.
    # dataset.
    parser.add_argument('-d', '--data', default='cifar10',
                        choices=['cifar10','cifar100','mnist','fashion_mnist',
                            'emnist','emnist_full', 'synthetic', 'shakespeare','adult',
                            'epsilon','MSD', 'higgs', 'rcv1', 'stl10'],
                        help='Dataset name.')
    parser.add_argument('-p', '--data_dir', default='./data/',
                        help='path to dataset')
    parser.add_argument('--partition_data', default=True, type=str2bool,
                        help='decide if each worker will access to all data.')
    parser.add_argument('--pin_memory', default=True, type=str2bool)
    parser.add_argument('--synthetic_alpha', default=0.0, type=float,
                        help='Setting alpha variable for a Synthetic dataset')
    parser.add_argument('--synthetic_beta', default=0.0, type=float,
                        help='Setting beta variable for a Synthetic dataset')
    parser.add_argument('--sensitive_feature', default=9, type=int,
                        help='Setting sensitive feature index for dividing the dataset')
    
    # Federated setting parameters
    parser.add_argument('-f', '--federated', default=False, type=str2bool,
                        help='Setup Federate Learning environment')
    parser.add_argument('--num_class_per_client', default=1, type=int,
                        help="Number of classes to attribute the data for each client \
                               for non-iid distribution in the Federated setting")
    parser.add_argument('--num_comms', default=100, type=int,
                        help="Number of communication rounds in Federated setting.")
    parser.add_argument('--online_client_rate', default=0.1, type=float,
                        help="The rate of clients to be online in each round of communication.")
    parser.add_argument('--federated_sync_type', default='epoch', type=str,
                        choices=['epoch','local_step']) # Not implemented for all federated types such as APFL
    parser.add_argument('--num_epochs_per_comm', default=1, type=int,
                        help="Number of epochs for each device on each round of communication") 
    parser.add_argument('--iid_data', default=True, type=str2bool,
                        help="Whether the data will distributed iid or non-iid among clients.")  
    parser.add_argument('--federated_type', default='fedavg', type=str,
                        choices=['fedavg','scaffold','fedprox','fedgate',
                            'fedadam','apfl','afl','perfedavg','qsparse',
                            'perfedme', 'qffl'],
                        help="Types of federated learning algorithm and/or training procedure.")
    parser.add_argument('--unbalanced', default=False, type=str2bool,
                        help="If set, the data will be distributed with unbalanced number of samples randomly.")
    parser.add_argument('--dirichlet', default=False, type=str2bool,
                        help="To distribute data among clients using a Dirichlet distribution.\
                               See paper: https://arxiv.org/pdf/2003.13461.pdf")
    parser.add_argument('--fed_personal', default=False, type=str2bool,
                        help="If set, the personalizied model will be evaluated during training.")    
    parser.add_argument('--fed_personal_alpha', default=0.5, type=float,
                        help="The alpha variable for the personalized training in APFL algorithm") 
    parser.add_argument('--fed_adaptive_alpha', default=False, type=str2bool,
                        help="If set, the alpha variable for APFL training will be optimized during training.")
    parser.add_argument('--fed_personal_test', default=False, type=str2bool,
                        help="If set, the personalized model will be evaluated using test dataset.")
    parser.add_argument('--fedadam_beta', default=0.9, type=float,
                        help="The beta vaiabale for FedAdam training. \
                            See paper: https://arxiv.org/pdf/2003.00295.pdf")  
    parser.add_argument('--fedadam_tau', default=0.1, type=float,
                        help="The tau vaiabale for FedAdam training. \
                            See paper: https://arxiv.org/pdf/2003.00295.pdf")
    parser.add_argument('--quantized', default=False, type=str2bool,
                            help="Quantized gradient for federated learning") 
    parser.add_argument('--quantized_bits', default=8, type=int,
                        help="The bit precision for quantization.")
    parser.add_argument('--compressed', default=False, type=str2bool,
                            help="Compressed gradient for federated learning")
    parser.add_argument('--compressed_ratio', default=1.0, type=float,
                        help="The ratio of keeping data after compression, where 1.0 means no compression.") 
    parser.add_argument('--federated_drfa', default=False, type=str2bool,
                        help="Indicator for using DRFA algorithm for training. \
                              The federated aggregation should be set using --federated_type. \
                              Paper: https://papers.nips.cc/paper/2020/hash/ac450d10e166657ec8f93a1b65ca1b14-Abstract.html")     
    parser.add_argument('--drfa_gamma', default=0.1, type=float,
                        help="Setting the gamma value for DRFA algorithm. \
                            See paper: https://papers.nips.cc/paper/2020/hash/ac450d10e166657ec8f93a1b65ca1b14-Abstract.html")
    parser.add_argument('--per_class_acc', default=False, type=str2bool,
                        help="If set, the validation will be reported per each class. Will be deprecated!")
    parser.add_argument('--perfedavg_beta', default=0.001, type=float,
                        help="The beta parameter in PerFedAvg algorithm. \
                              See paper: https://arxiv.org/pdf/2002.07948.pdf")
    parser.add_argument('--fedprox_mu', default=0.002, type=float,
                        help="The Mu parameter in the FedProx algorithm. \
                              See paper: https://arxiv.org/pdf/1812.06127.pdf")
    parser.add_argument('--perfedme_lambda', default=15, type=float,
                        help="The Lambda parameter for PerFedMe algorithm. \
                              See paper: https://arxiv.org/pdf/2006.08848.pdf")
    parser.add_argument('--qffl_q', default=0.0, type=float,
                        help="The q parameter in qffl algorithm. \
                              See paper: https://arxiv.org/pdf/1905.10497.pdf")


    # model
    parser.add_argument('-a', '--arch', default='mlp',
                        help='model architecture: ' +
                             ' | '.join(model_names) + ' (default: mlp)')

    # training and learning scheme
    parser.add_argument('--stop_criteria', type=str, default='epoch')
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--num_iterations', type=int, default=None)

    parser.add_argument('--local_step', type=int, default=1)
    parser.add_argument(
        '--local_step_warmup_per_interval', default=False, type=str2bool)
    parser.add_argument('--local_step_warmup_type', default=None, type=str)
    parser.add_argument('--local_step_warmup_period', default=None, type=int)
    parser.add_argument('--turn_on_local_step_from', default=None, type=int)
    parser.add_argument('--turn_off_local_step_from', default=None, type=int)

    parser.add_argument('--avg_model', type=str2bool, default=False)
    parser.add_argument('--reshuffle_per_epoch', default=False, type=str2bool)
    parser.add_argument('-b', '--batch_size', default=50, type=int,
                        help='mini-batch size (default: 50)')
    parser.add_argument('--growing_batch_size', default=False, type=str2bool,
                        help="If set, the batch size is growing during the training.")
    parser.add_argument('--base_batch_size', default=None, type=int,
                        help="The minimum batch size in the growing batch size mode.")
    parser.add_argument('--max_batch_size', default=0, type=int,
                        help="The maximum batch size in the growing batch size mode.")

    # learning rate scheme
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_schedule_scheme', type=str, default=None)

    parser.add_argument('--lr_change_epochs', type=str, default=None)
    parser.add_argument('--lr_fields', type=str, default=None)
    parser.add_argument('--lr_scale_indicators', type=str, default=None)

    parser.add_argument('--lr_scaleup', type=str2bool, default=False)
    parser.add_argument('--lr_scaleup_type', type=str, default='linear')
    parser.add_argument('--lr_scale_at_sync', type=float, default=1.0)
    parser.add_argument('--lr_warmup', type=str2bool, default=False)
    parser.add_argument('--lr_warmup_epochs', type=int, default=5)
    parser.add_argument('--lr_decay', type=float, default=10)

    parser.add_argument('--lr_onecycle_low', type=float, default=0.15)
    parser.add_argument('--lr_onecycle_high', type=float, default=3)
    parser.add_argument('--lr_onecycle_extra_low', type=float, default=0.0015)
    parser.add_argument('--lr_onecycle_num_epoch', type=int, default=46)

    parser.add_argument('--lr_gamma', type=float, default=None)
    parser.add_argument('--lr_mu', type=float, default=None)
    parser.add_argument('--lr_alpha', type=float, default=None)

    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd')

    # momentum scheme
    parser.add_argument('--in_momentum', type=str2bool, default=False)
    parser.add_argument('--in_momentum_factor', default=0.9, type=float)
    parser.add_argument('--out_momentum', type=str2bool, default=False)
    parser.add_argument('--out_momentum_factor', default=None, type=float)
    parser.add_argument('--use_nesterov', default=False, type=str2bool)

    # regularization
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--correct_wd', type=str2bool, default=False)
    parser.add_argument('--drop_rate', default=0.0, type=float)

    # different models' parameters.
    parser.add_argument('--densenet_growth_rate', default=12, type=int)
    parser.add_argument('--densenet_bc_mode', default=False, type=str2bool)
    parser.add_argument('--densenet_compression', default=0.5, type=float)

    parser.add_argument('--wideresnet_widen_factor', default=4, type=int)

    parser.add_argument('--mlp_num_layers', default=2, type=int)
    parser.add_argument('--mlp_hidden_size', default=500, type=int)

    parser.add_argument('--rnn_seq_len', default=50, type=int)
    parser.add_argument('--rnn_hidden_size', default=50, type=int)
    parser.add_argument('--vocab_size', default=86, type=int)
    

    # miscs
    parser.add_argument('--manual_seed', type=int,
                        default=6, help='manual seed')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        type=str2bool, default=False,
                        help='evaluate model on validation set')
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--summary_freq', default=10, type=int)
    parser.add_argument('--timestamp', default=None, type=str)

    # checkpoint
    parser.add_argument('--debug', type=str2bool, default=False,
                        help="Showing the training and evaluation results.\
                              By default, the server's debug is True, but all other nodes are False")
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--check_model_at_sync', default=False, type=str2bool)
    parser.add_argument('--track_model_aggregation', default=False, type=str2bool)
    parser.add_argument('--checkpoint', '-c', default='./checkpoint/',
                        type=str,
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--checkpoint_index', type=str, default=None)
    parser.add_argument('--save_all_models', type=str2bool, default=False)
    parser.add_argument('--save_some_models', type=str, default='1,29,59')
    parser.add_argument('--log_dir', default='./logdir/')
    parser.add_argument('--plot_dir', default=None,
                        type=str, help='path to plot the result')
    parser.add_argument('--pretrained', dest='pretrained', type=str2bool,
                        default=False, help='use pre-trained model')

    # device
    parser.add_argument('--is_distributed', default=True, type=str2bool)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--hostfile', type=str, default='hostfile')
    parser.add_argument('-j', '--num_workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--dist_backend', default='mpi', type=str,
                        help='distributed backend')

    parser.add_argument('--blocks', default='2,2', type=str,
                        help='number of blocks (divide processes to blocks)')
    parser.add_argument('--on_cuda', type=str2bool, default=True)
    parser.add_argument('--world', default=None, type=str)

    # parse args.
    args = parser.parse_args()
    if args.timestamp is None:
        args.timestamp = get_checkpoint_folder_name(args)
    if args.growing_batch_size:
        if args.base_batch_size is None:
            args.base_batch_size = 1
    if args.federated:
        if args.reshuffle_per_epoch:
            raise ValueError("In the Federated Learning mode, we cannot shuffle data in the middle of training! set --reshuffle_per_epoch False")
        args.num_epochs = int(args.num_epochs_per_comm * args.num_comms * args.online_client_rate)
        if args.federated_type == 'afl':
            args.federated_sync_type = 'local_step'
            args.local_step = 1
        if args.federated_type == 'qsparse':
            args.compressed == True
        if args.quantized and args.compressed:
            raise ValueError("Quantization is mutually exclusive with compression! Choose only one of them.")
    # args.data_dir += '/data'
        if args.federated_type in ['apfl','perfedme','perfedavg']:
            # These are personalized models and need to have vaildation data on local devices
            args.fed_personal = True
    return args


def str2bool(v):
    """Convert different forms of bool string to boolean value

    Args:
        v (str): String bool input

    Raises:
        argparse.ArgumentTypeError: The string should be one of the mentioned values.

    Returns:
        bool: Boolean value corresponding to the input.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_args(args):
    print('parameters: ')
    for arg in vars(args):
        print(arg, getattr(args, arg))


if __name__ == '__main__':
    args = get_args()
