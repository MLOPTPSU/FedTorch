import os
import argparse


def main(args):
    NUM_WORKERS=args.num_clients
    LOCAL_STEP=args.local_steps
    DATASET=args.dataset

    MODEL={'epsilon':'logist_regression', 'MSD':'robust_least_square', 'cifar10':'mlp', 'emnist':'mlp', 'mnist':'mlp','synthetic':'logist_regression','fashion_mnist':'mlp','adult':'logist_regression'}
    
    mlp_size = {'mnist':200,'fashion_mnist':200,'cifar10':500,'cifar100':500,'adult':50,'MSD':50,'emnist':200}
    NUM_NODES=1
    NUM_WORKER_PER_NODE =  int(NUM_WORKERS / NUM_NODES)
    NUM_WORKERS_NODE = [NUM_WORKER_PER_NODE] * NUM_NODES 

    BLOCKS=(',').join([str(i) for i in NUM_WORKERS_NODE])
    WORLD = ",".join([ ",".join([str(x) for x in range(i)]) for i in NUM_WORKERS_NODE])


    script_params = {
        '--arch':  MODEL[DATASET],
        '--mlp_num_layers': 2,
        '--mlp_hidden_size':mlp_size[DATASET],
        '--avg_model': True,
        '--experiment': 'demo',
        '--debug': True,
        '--data': DATASET,
        '--synthetic_alpha':args.synthetic_params[0],
        '--synthetic_beta':args.synthetic_params[1],
        '--pin_memory': True,
        '--federated': True,
        '--federated_type':args.federated_type,
        '--federated_sync_type':args.federated_sync_type,
        '--num_comms':args.num_comms,
        '--online_client_rate':args.online_client_rate,
        '--num_epochs_per_comm': args.num_epochs_per_comm,
        '--num_class_per_client':args.num_class_per_client,
        '--iid_data':args.iid,
        '--fed_personal': True,
        '--quantized':args.quantized,
        '--quantized_bits':args.quantized_bits,
        '--compressed':args.compressed,
        '--compressed_ratio':args.compressed_ratio,
        '--federated_drfa': args.federated_drfa,
        '--drfa_gamma': args.drfa_gamma,
        '--per_class_acc':args.per_class_acc,
        '--comm_delay_coef':args.comm_delay_coef,
        '--fed_adaptive_alpha': args.fed_adaptive_alpha,
        '--fed_personal_alpha': args.fed_personal_alpha,
        '--fedprox_mu': args.fedprox_mu,
        '--perfedavg_beta': 0.03,
        '--sensitive_feature':args.sensitive_feature,
        '--unbalanced':args.unbalanced,
        '--batch_size':args.batch_size, 
        '--eval_freq': 1,
        '--partition_data': True,
        '--reshuffle_per_epoch': False,
        '--stop_criteria': "epoch",
        '--num_epochs': args.num_epochs_per_comm * args.num_comms,
        '--on_cuda': False,
        '--num_workers': NUM_WORKERS, 
        '--blocks': BLOCKS,
        '--world': WORLD,
        '--weight_decay': args.weight_decay,
        '--use_nesterov': False,
        '--in_momentum': False,
        '--in_momentum_factor': 0.9, 
        '--out_momentum': False, 
        '--out_momentum_factor': 0.9,
        '--is_kube': True, 
        '--hostfile': 'local_hostfile',
        '--python_path': "/usr/bin/python",
        '--mpi_path': "/usr/",
        '--local_step': LOCAL_STEP,
        '--turn_on_local_step_from': 0,
        '--data_dir': args.data_path,
        '--checkpoint': args.data_path,
        '--drop_rate': 0.25,
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

    script_params.update(learning_rate)

    prefix_cmd = 'mpirun -np {} --allow-run-as-root  --oversubscribe --mca btl_tcp_if_exclude docker0,lo --mca orte_base_help_aggregate 0 --mca orte_tmpdir_base {} --mca opal_warn_on_missing_libcuda 0 '.format(args.num_clients,os.environ['TMPDIR'])

    cmd = 'python main.py '
    for k, v in script_params.items():
        if v is not None:
            cmd += ' {} {} '.format(k, v)

    cmd = prefix_cmd + cmd
    # run the cmd.
    print('\nRun the following cmd:\n' + cmd)
    os.system(cmd)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
        description='Running distributed optimization in CPU')
  
  parser.add_argument('-e', '--num_epochs_per_comm', default=1, type=int)
  parser.add_argument('-n', '--num_clients', default=20, type=int)
  parser.add_argument('-d', '--dataset', default='mnist', type=str)
  parser.add_argument('-y', '--lr_gamma', default=1.0, type=float)
  parser.add_argument('-l', '--lr_sync', default=1.0, type=float)
  parser.add_argument('-m', '--lr_mu', default=1, type=float)
  parser.add_argument('-b', '--batch_size', default=50, type=int)
  parser.add_argument('-c', '--num_comms', default=100, type=int)
  parser.add_argument('-k', '--online_client_rate', default=1.0, type=float)
  parser.add_argument('-p', '--data_path', default='./data', type=str)
  parser.add_argument('-t', '--federated_type', default='fedavg', type=str)
  parser.add_argument('-i', '--iid', action='store_true')
  parser.add_argument('-s', '--local_steps', default=1, type=int)
  parser.add_argument('-f', '--federated_sync_type', default='epoch', type=str, choices=['epoch', 'local_step'])
  parser.add_argument('-r', '--num_class_per_client', default=2, type=int)
  parser.add_argument('-a', '--synthetic_params', nargs='+', type=float, default=[0.0,0.0])
  parser.add_argument('-w', '--weight_decay', default=1e-4, type=float)
  parser.add_argument('-q', '--quantized', action='store_true')
  parser.add_argument('-S', '--compressed', action='store_true')
  parser.add_argument('-R', '--compressed_ratio', default=1.0, type=float)
  parser.add_argument('-D', '--federated_drfa', action='store_true')
  parser.add_argument('-G', '--drfa_gamma', default=0.1, type=float)
  parser.add_argument('-P', '--per_class_acc', action='store_true')
  parser.add_argument('-u', '--unbalanced', action='store_true')
  parser.add_argument('-C', '--comm_delay_coef', default=0.0, type=float)
  parser.add_argument('-L', '--fed_personal_alpha', default=0.0, type=float)
  parser.add_argument('-A', '--fed_adaptive_alpha', action='store_true')
  parser.add_argument('-F', '--sensitive_feature', default=9, type=int)
  parser.add_argument('-B', '--quantized_bits', default=8, type=int)
  parser.add_argument('-M', '--fedprox_mu', default=0.002, type=float)
  
  args = parser.parse_args()

  main(args)
  
