FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV PYTORCH_VERSION=1.6.0
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.2
ENV NCCL_VERSION=2.5.6-1+cuda10.2

ARG python=3.8
ENV PYTHON_VERSION=${python}

RUN apt update && apt install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
		build-essential \
		cmake \
		git \
		curl \
		vim \
		wget \
		ca-certificates \
		libcudnn7=${CUDNN_VERSION} \
		libnccl2=${NCCL_VERSION} \
		libnccl-dev=${NCCL_VERSION} \
		python${PYTHON_VERSION} \
		python${PYTHON_VERSION}-dev

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN apt install -y --allow-downgrades --allow-change-held-packages --no-install-recommends  python3-distutils

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
	rm get-pip.py

# setup openmpi with CUDA and multi-threading support
WORKDIR "/workspace"

RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz && \
    gunzip -c openmpi-4.0.1.tar.gz | tar xf - && cd openmpi-4.0.1 && \
	mkdir build && cd build/ && \
	../configure --prefix=/usr --with-cuda --enable-mpi-thread-multiple && \
	make -j $(nproc) all && \
	make install && \
	ldconfig

# setup pytorch build dependencies
RUN pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
ENV TORCH_CUDA_ARCH_LIST="3.5 3.7 5.2 6.0 6.1 7.0+PTX"
RUN git clone --branch v${PYTORCH_VERSION} --recursive https://github.com/pytorch/pytorch  && \
		cd pytorch && \
		BUILD_TEST=0 BUILD_BINARY=0 python setup.py install

#    setup OpenSSH for MPI (should be deleted when FROM PHILLY CONTAINER)
RUN apt install -y --no-install-recommends openssh-client openssh-server && \
		mkdir -p /var/run/sshd

# Higher versions of torchvision will uninstall the built PyTorch and reinstall the latest version without MPI support
RUN pip install torchvision==0.2.0

RUN pip install lmdb tensorboard_logger pyarrow msgpack msgpack_numpy mpi4py cvxopt tensorpack opencv-python pandas scikit-learn tensorflow_federated tqdm