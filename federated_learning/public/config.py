# Overall settings
k_folds = 2 # number of folds for cross-validation, if 1, no cross-validation
strategy = 'fedavg' # ['fedavg', 'ssfl']
random_seed = 42
gpu = -2 # set the GPU to use, if -1 use CPU, -2 for multigpus
n_clients = 10
n_samples_clients = -1 # if -1, use all samples

# Dynamic dataset for ANDA (heterogeneous data)
dataset_name = "MNIST"
n_clusters = 2 # number of data distribution

# Training model settings
model_name = "LeNet5"   # ["LeNet5", "ResNet9"]
batch_size = 64
test_batch_size = 64
client_eval_ratio = 0.2
n_rounds = 10
local_epochs = 2
lr = 0.005
momentum = 0.9

# FL settings - Communications
port = '8018'
ip = '0.0.0.0' # Local Host=0.0.0.0, or IP address of the server
