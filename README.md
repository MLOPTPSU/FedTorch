![FedTorch Logo](./logo.png)
--------------------------------------------------------------------------------

FedTorch is an open-source Python package for distributed and federated training of machine learning models using [PyTorch distributed API](https://pytorch.org/docs/stable/distributed.html). Various algorithms for federated learning and local SGD are implemented for benchmarking and research, including our own proposed methods:
* [Redundancy Infused SGD (RI-SGD)](http://proceedings.mlr.press/v97/haddadpour19a.html) ![official](https://img.shields.io/badge/code-Official-green)
* [Local SGD with Adaptive Synchoronization (LUPA-SGD)](https://papers.nips.cc/paper/2019/hash/c17028c9b6e0c5deaad29665d582284a-Abstract.html)  ![official](https://img.shields.io/badge/code-Official-green)
* [Adaptive Personalized Federated Learning (APFL)](https://arxiv.org/abs/2003.13461) ![official](https://img.shields.io/badge/code-Official-green)
* [Distributionally Robust Federated Learning (DRFA)](https://papers.nips.cc/paper/2020/file/ac450d10e166657ec8f93a1b65ca1b14-Paper.pdf) ![official](https://img.shields.io/badge/code-Official-green)
* [Federated Learning with Gradient Tracking and Compression (FedGATE and FedCOMGATE)](https://arxiv.org/abs/2007.01154) ![official](https://img.shields.io/badge/code-Official-green)

And other common algorithms such as:
* [FedAvg](http://proceedings.mlr.press/v54/mcmahan17a.html)
* [SCAFFOLD](http://proceedings.mlr.press/v119/karimireddy20a.html)
* [Qsparse Local SGD](https://ieeexplore.ieee.org/abstract/document/9057579)
* [AFL](https://arxiv.org/abs/1902.00146)
* [FedProx](https://arxiv.org/abs/1812.06127)
* and more ...

We are actively trying to expand the library to include more training algorithms as well.

## NEWS
Recent updates to the package:
* **Paper Accepted** (01/22/2021): Our paper titled [`Federated Learning with Compression: Unified Analysis and Sharp Guarantees`](https://arxiv.org/abs/2007.01154) accepted to [AISTAT 2021](https://aistats.org/aistats2021/)🎉
* **Public Release** (01/17/2021): We are releasing the package to the public with a docker image

## Installation
First you need to clone the repo into your computer:
```cli
git clone https://github.com/MLOPTPSU/FedTorch.git
```
The PyPi package will be added soon.

This package is built based on PyTorch Distributed API. Hence, it could be run with any supported distributed backend of GLOO, MPI, and NCCL. Among these three, MPI backend since it can be used for both CPU and CUDA runnings, is the main backend we use for developement. Unfortunately installing the built version of PyTorch does not support MPI backend for distributed training and needed to be built from source with a version of MPI installed that supports CUDA as well. However, do not worry since we got you covered. We provide a docker file that can create an image with all dependencies needed for FedTorch. The Dockerfile can be found [here](docker/README.md), where you can edit based on your need to create your customized image. In addition, since building this docker image might take a lot of time, we provide different versions that we built before along with this repository in the [packages](https://github.com/orgs/MLOPTPSU/packages?repo_name=FedTorch) section. 

For instance, you can pull one of the images that is built with CUDA 10.2 and OpenMPI 4.0.1 with CUDA support and PyTorch 1.6.0, using the following command:
```cli
docker pull docker.pkg.github.com/mloptpsu/fedtorch/fedtorch:cuda10.2-mpi
```
The docker images can be used for cloud services such as [Azure Machine Learning API](https://azure.microsoft.com/en-us/services/machine-learning/) for running FedTorch. The instructions for running on cloud services will be added in near future.


## Get Started
Running different trainings with different settings is easy in FedTorch. The only thing you need to take care of is to set the correct parameters for the experiment. The list of all parameters used in this package is in [`parameters.py`](fedtorch/parameters.py#L23) file. For different algorithms we will provide examples, so the relevant parameters can be set correctly. YAML support will be added in future for each distinct training. The parameters can be parsed from the input of the commandline using the following method:
```python
from fedtorch.parameters import get_args

args = get_args()
```

When the parameters are set, we need to setup the nodes using those parameters and start the training.
First, we need to setup the distributed backend. For instance, we can use MPI backend as:
```python
import torch.distributed as dist

dist.init_process_group('mpi')
```

This will initialize the backend for distributed training. Next, we need to setup each node based on the parameters and create the graph of nodes. Then we need to initialize nodes and load their data. For this, we can use the node object in the FedTorch:
```python
from fedtorch.nodes import Client

client = Client(args, dist.get_rank())
# Initialize the node
client.initialize()
# Initialize the dataset if not downloaded
client.initialize_dataset()
# Load the dataset
client.load_local_dataset()
# Generate auxiliary models and params for training
client.gen_aux_models()
```

Then, we need to call the appropriate training method and run the training. For instance, if the parameters are set for a FedAvg training, we can run:
```python
from fedtorch.comms.trainings.federated import train_and_validate_federated

train_and_validate_federated(client)
```
Different distributed and federated algorithms in this package can be run using this procedure, and for simplicity, we provide [`main.py`](main.py) file, where can be used for running those algorithms following the same procedure. To run this file, we should run it using mpi and define number of clients (processes) that will run the same file for training using:
```cli
mpirun -np {NUM_CLIENTS} python main.py {args:values}
```
where `{args:values}` should be filled with appropriate parameters needed for the training. Next we provide a file to automatically build this command for mpi running for various situations.

## Running Examples
To make the process easier, we have provided a [`run_mpi.py`](run_mpi.py) file that covers most of parameters needed for running different training algrithms. We first get into the details of different parameters and then provide some examples for running.

### Dataset Parameters
For setting up the dataset there are some parameters involved. The main parameters are:
* `--data` : Defines the dataset name for training.
* `--batch_size` : Defines the size of the batch in the training.
* `--partition_data` : Defines whether the data should be partitioned or each client access to the whole dataset.
* `--reshuffle_per_epoch` : This can be set `True` for distributed training to have iid data accross clients and faster convergence. This is not inline with Federated Learning settings.
* `--iid_data` : If set `True`, the data is randomly distributed accross clients. If is set to `False`, either the dataset itself is non-iid by default (like the `EMNIST` dataset) or it can be manullay distributed to be non-iid (like the `MNIST` dataset using parameter `num_class_per_client`). The default is `True` in the package, but in the `run_mpi.py` file the default is `False`.
* `--num_class_per_client`: If the parameter `iid` is set to `False`, we can distribute the data heterogeneously by attributing certain number of classes to each client. For instance if setting `--num_class_per_client 2`, then each client will only has access to two randomly selected classes' data in the entire training process.

### Federated Learning Parameters
To run the training using federated learning setups some main parameters are:
* `--federated` : If set to `True` the training will be in one of federated learning setups. If not, it will be in a distributed mode using local SGD and with periodic averaging (that could be set using `--local_step`) and possibly reshuffling after each epoch. The default is `False`.
* `--federated_type` : defines the type of fderated learning algorithm we want to use. The default is `fedavg`.
* `--federated_sync_type` : It could be either `epoch` or `local_step` and it will be used to determine when to synchronize the models. If set to `epoch`, then the parameter `--num_epochs_per_comm` should be set as well. If set to `local_step`, then the parameter `--local_steps` should be set. The default is `epoch`.
* `--num_comms` : Defines the number of communication rounds needed for trainings. This is only for federated learning, while in normal distirbuted mode the number of total iterations should be set either by `--num_epochs` or `--num_iterations`, and hence the `--stop_criteria` should be either `epoch` or `iteration`.
* `--online_client_rate` : Defines the ratio of clients that are online and active during each round of communication. This is only for federated learning. The default value is `1.0`, which means all clients will be active.

### Learning Rate Schedule
Different learning rate schedules can be set using their corresponding parameters. The main parameter is `--lr_schedule_scheme`, which defines the scheme for learning rate scheduling. For more information about different learning rate schedulers, please see [`learning.py`](fedtorch/components/optimizers/learning.py) file.

### Examples
Now we provide some simple examples for running some of the training algorithms on a single node with multiple processes using mpi. To do so, we first need to run the docker container with installed dependencies.
```cli
docker run --rm -it --mount type=bind,source="{path/to/FedTorch}",target=/FedTorch docker.pkg.github.com/mloptpsu/fedtorch/fedtorch:cuda10.2-mpi
cd /FedTorch
```
This will run the container and will mount the FedTorch repo to it. The `{path/to/FedTorch}` should be replaced with your local path to the FedTorch repo directory. Now we can run the training on it.

#### FedAvg and FedGATE
Now, we can run the FedAvg algorithm for training an MLP model using MNIST data by the following command.
```cli
python run_mpi.py -f -ft fedavg -n 10 -d mnist -lg 0.1 -b 50 -c 20 -k 1.0 -fs local_step -l 10 -r 2
```
This will run the training on 10 nodes with initial learning rate of 0.1, the batch size of 50, for 20 communication rounds each with 10 local steps of SGD. The dataset is distributed hetergeneously with each client has access to only 2 classes of data from the MNIST dataset.

Changing `-ft fedavg` to `-ft fedgate` will run the same training using the FedGATE algorithm. To run the FedCOMGATE algorithm we need to add `-q` to the parameter to enable quantization as well. Hence the command will be:
```cli
python run_mpi.py -f -ft fedgate -n 10 -d mnist -lg 0.1 -b 50 -c 20 -k 1.0 -fs local_step -l 10 -r 2 -q
```

#### APFL
To run APFL algorithm a simple command will be:
```cli
python run_mpi.py -f -ft apfl -n 10 -d mnist -lg 0.1 -b 50 -c 20 -k 1.0 -fs local_step -l 10 -r 2 -pa 0.5 -fp
```
where `-pa 0.5` sets the alpha parameter of the APFL algorithm. The last parameter `-fp` will turn on the `fed_personal` parameter, which evaluate the personalized or the localized model using a local validation dataset. This will be mostly used for personalization algorithms such as APFL.

#### DRFA
To run a DRFA training we can use the following command:
```cli
python run_mpi.py -f -fd -ft fedavg -n 10 -d mnist -lg 0.1 -b 50 -c 20 -k 1.0 -fs local_step -l 10 -r 2 -dg 0.1 
```
where `-dg 0.1` sets the gamma parameter in the DRFA algorithm. Note that DRFA is a framework that can be run using any federated learning aggergator such as FedAvg or FedGATE. Hence the parameter `-fd` will enable DRFA training and `-ft` will define the federated type to be used for aggregation.


## References 
For this repository there are several different references used for each training procedure. If you use this repository in your research, please cite the following paper:
```ref
@article{haddadpour2020federated,
  title={Federated learning with compression: Unified analysis and sharp guarantees},
  author={Haddadpour, Farzin and Kamani, Mohammad Mahdi and Mokhtari, Aryan and Mahdavi, Mehrdad},
  journal={arXiv preprint arXiv:2007.01154},
  year={2020}
}
```
Our other papers developed using this repository should be cited using the following bibitems:
```ref
@inproceedings{haddadpour2019local,
  title={Local sgd with periodic averaging: Tighter analysis and adaptive synchronization},
  author={Haddadpour, Farzin and Kamani, Mohammad Mahdi and Mahdavi, Mehrdad and Cadambe, Viveck},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11082--11094},
  year={2019}
}
@article{deng2020distributionally,
  title={Distributionally Robust Federated Averaging},
  author={Deng, Yuyang and Kamani, Mohammad Mahdi and Mahdavi, Mehrdad},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
@article{deng2020adaptive,
  title={Adaptive Personalized Federated Learning},
  author={Deng, Yuyang and Kamani, Mohammad Mahdi and Mahdavi, Mehrdad},
  journal={arXiv preprint arXiv:2003.13461},
  year={2020}
}
```

### Acknowledgement and Disclaimer
This repository is developed, mainly by [MM. Kamani](https://github.com/mmkamani7), based on our group's research on distributed and federated learning algorithms. We also developed the works of other groups' proposed methods using FedTorch for a better comparison. However, this repo is not the official code for those methods other than our group's. Some parts of the initial stages of this repository were based on a forked repo of Local SGD code from Tao Lin, which is not public now.
