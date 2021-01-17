import os
import argparse


def main(args):
    model={'epsilon':'logistic_regression', 
           'MSD':'robust_least_square', 
           'cifar10':'logistic_regression', 
           'emnist':'mlp', 
           'mnist':'mlp',
           'synthetic':'logistic_regression',
           'fashion_mnist':'mlp',
           'adult':'logistic_regression'}
    
    mlp_size = {'mnist':200,'fashion_mnist':200,'cifar10':200,'cifar100':500,'adult':50,'MSD':50,'emnist':200}
    NUM_NODES=1
    NUM_WORKER_PER_NODE =  int(args.num_clients / NUM_NODES)
    NUM_WORKERS_NODE = [NUM_WORKER_PER_NODE] * NUM_NODES 

    BLOCKS=(',').join([str(i) for i in NUM_WORKERS_NODE])
    WORLD = ",".join([ ",".join([str(x) for x in range(i)]) for i in NUM_WORKERS_NODE])


    training_params = {
        '--avg_model': True,
        '--debug': True,
        '--eval_freq': 1,
        '--partition_data': True,
        '--reshuffle_per_epoch': False,
        '--stop_criteria': "epoch",
        '--num_epochs': args.num_epochs_per_comm * args.num_comms,
        '--on_cuda': False,
        '--num_workers': args.num_clients, 
        '--blocks': BLOCKS,
        '--world': WORLD,
        '--weight_decay': args.weight_decay,
        '--use_nesterov': False,
        '--in_momentum': False,
        '--in_momentum_factor': 0.9, 
        '--out_momentum': False, 
        '--out_momentum_factor': 0.9,
        '--local_step': args.local_steps,
        '--turn_on_local_step_from': 0,
        '--checkpoint': args.data_path,
        '--drop_rate': 0.25,
    }

    model_params = {
        '--arch':  model[args.dataset],
        '--mlp_num_layers': 2,
        '--mlp_hidden_size':mlp_size[args.dataset],
    }
    data_params = {
        '--data': args.dataset,
        '--data_dir': args.data_path,
        '--synthetic_alpha':args.synthetic_params[0],
        '--synthetic_beta':args.synthetic_params[1],
        '--batch_size':args.batch_size,
    }
    federated_params = {
        '--federated': args.federated,
        '--federated_type':args.federated_type,
        '--federated_sync_type':args.federated_sync_type,
        '--num_comms':args.num_comms,
        '--online_client_rate':args.online_client_rate,
        '--num_epochs_per_comm': args.num_epochs_per_comm,
        '--num_class_per_client':args.num_class_per_client,
        '--iid_data':args.iid,
        '--fed_personal': args.fed_personal,
        '--quantized':args.quantized,
        '--quantized_bits':args.quantized_bits,
        '--compressed':args.compressed,
        '--compressed_ratio':args.compressed_ratio,
        '--federated_drfa': args.federated_drfa,
        '--drfa_gamma': args.drfa_gamma,
        '--fed_adaptive_alpha': args.fed_adaptive_alpha,
        '--fed_personal_alpha': args.fed_personal_alpha,
        '--fedprox_mu': args.fedprox_mu,
        '--perfedavg_beta': 0.03,
        '--sensitive_feature':args.sensitive_feature,
        '--unbalanced':args.unbalanced,
    }
    learning_rate = {
        '--lr_schedule_scheme': 'custom_multistep',
        '--lr_change_epochs': ','.join([str(x) for x in range(1,args.num_epochs_per_comm * args.num_comms)]),
        '--lr_warmup': False,
        '--lr': args.lr_gamma,
        '--lr_scale_at_sync': args.lr_sync ,
        '--lr_warmup_epochs': 3,
        '--lr_decay':1.01,
    }

    # learning_rate = {
    #     '--lr_schedule_scheme':'custom_convex_decay',
    #     '--lr_gamma': args.lr_gamma,
    #     '--lr_mu': args.lr_mu,
    #     '--lr_scale_at_sync': args.lr_sync ,
    #     '--lr_alpha': 1,
    # }

    training_params.update(model_params)
    training_params.update(data_params)
    training_params.update(federated_params)
    training_params.update(learning_rate)

    if os.environ['TMPDIR'] == '':
        os.environ['TMPDIR'] = args.tmp_dir
    
    prefix_cmd = 'mpirun -np {} --allow-run-as-root  --oversubscribe --mca btl_tcp_if_exclude docker0,lo --mca orte_base_help_aggregate 0 \
                 --mca orte_tmpdir_base {} --mca opal_warn_on_missing_libcuda 0 '.format(args.num_clients, os.environ['TMPDIR'])

    cmd = 'python main.py '
    for k, v in training_params.items():
        if v is not None:
            cmd += ' {} {} '.format(k, v)

    cmd = prefix_cmd + cmd
    # run the cmd.
    print('\nRunnig the following command:\n' + cmd)
    os.system(cmd)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
        description='Running FedTorch using MPI backend!')
  
  parser.add_argument('-e',  '--num_epochs_per_comm', default=1, type=int)
  parser.add_argument('-n',  '--num_clients', default=20, type=int)
  parser.add_argument('-d',  '--dataset', default='mnist', type=str)
  parser.add_argument('-p',  '--data_path', default='./data', type=str)
  parser.add_argument('-b',  '--batch_size', default=50, type=int)
  parser.add_argument('-c',  '--num_comms', default=100, type=int)
  parser.add_argument('-lg', '--lr_gamma', default=1.0, type=float)
  parser.add_argument('-lm', '--lr_mu', default=1, type=float)
  parser.add_argument('-ls', '--lr_sync', default=1.0, type=float)
  parser.add_argument('-w',  '--weight_decay', default=1e-4, type=float)
  parser.add_argument('-i',  '--iid', action='store_true')
  parser.add_argument('-l',  '--local_steps', default=1, type=int)
  parser.add_argument('-td', '--tmp_dir', default='/tmp', type=str)
  # Federated Params
  parser.add_argument('-f',  '--federated', action='store_true')
  parser.add_argument('-ft', '--federated_type', default='fedavg', type=str)
  parser.add_argument('-fd', '--federated_drfa', action='store_true')
  parser.add_argument('-dg', '--drfa_gamma', default=0.1, type=float)
  parser.add_argument('-fs', '--federated_sync_type', default='epoch', type=str, choices=['epoch', 'local_step'])
  parser.add_argument('-k',  '--online_client_rate', default=1.0, type=float)
  parser.add_argument('-r',  '--num_class_per_client', default=2, type=int)
  parser.add_argument('-sp', '--synthetic_params', nargs='+', type=float, default=[0.0,0.0])
  parser.add_argument('-q',  '--quantized', action='store_true')
  parser.add_argument('-cp', '--compressed', action='store_true')
  parser.add_argument('-cr', '--compressed_ratio', default=1.0, type=float)
  parser.add_argument('-u',  '--unbalanced', action='store_true')
  parser.add_argument('-fp', '--fed_personal', action='store_true')
  parser.add_argument('-pa', '--fed_personal_alpha', default=0.0, type=float)
  parser.add_argument('-pd', '--fed_adaptive_alpha', action='store_true')
  parser.add_argument('-sf', '--sensitive_feature', default=9, type=int)
  parser.add_argument('-B',  '--quantized_bits', default=8, type=int)
  parser.add_argument('-pm', '--fedprox_mu', default=0.002, type=float)
  
  args = parser.parse_args()

  main(args)
  
