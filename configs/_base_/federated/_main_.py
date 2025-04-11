federated = dict(
    personal=False,
    num_comms=100,
    online_client_rate=1.0,
    sync_type='epoch', # 'epoch' or 'local_step'
    num_epochs_per_comm=1,
    unbalanced=False, # If set, the data will be distributed with unbalanced number of samples randomly.
    dirichlet=False, # To distribute data among clients using a Dirichlet distribution.See paper: https://arxiv.org/pdf/2003.13461.pdf",
    quantized=False, # Quantized gradient for federated learning
    quantized_bits=8, # The bit precision for quantization.
    compressed=False, # Compressed gradient for federated learning
    compressed_ratio=1.0, # The ratio of keeping data after compression, where 1.0 means no compression.
    per_class_acc=False, # If set, the validation will be reported per each class. Will be deprecated!
    num_early_stop=0, # Number of clients to stop early
)