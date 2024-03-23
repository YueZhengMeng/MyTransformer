import torch

seed = 42

max_length = 16
pad = 0
bos = 8
eos = 9

d_model = 512
vocab_size = 10
nhead = 8
num_layers = 3
dropout = 0.1

batch_size = 4
learning_rate = 4e-4
epoch = 3000
log_step = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
