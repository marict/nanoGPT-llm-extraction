# Default training configuration used when no config file is provided.
import torch

out_dir = 'out'
eval_interval = 250
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'

dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

dag_depth = 0
dag_hidden_dim = 16
dag_num_ops = 5

learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
