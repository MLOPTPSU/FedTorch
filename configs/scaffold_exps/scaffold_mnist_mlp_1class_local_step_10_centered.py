_base_ = [
    '../_base_/model/mlp.py', # Model config
    '../_base_/dataset/mnist.py', # Data Config
    '../_base_/optimizer/sgd.py', # Optimizer Config
    '../_base_/scheduler/convex_decay.py', # Scheduler Config
    '../_base_/checkpoint.py', # Checkpoint Config
    '../_base_/device.py', # Device Config
    '../_base_/partitioner/federated_partitioner.py', # Partitioner Config
    '../_base_/training/federated.py', # Training Config
    '../_base_/federated/scaffold.py',  # Federated Learning Config
]

training=dict(
    centered=True,
    local_step=10,
)

data = dict(
    dataset=dict(
        download=True,
    )
)

federated = dict(
    sync_type='local_step',
)

device=dict(
    on_cuda=False,
)