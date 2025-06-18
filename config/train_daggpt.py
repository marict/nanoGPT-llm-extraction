# Training config for DAGGPT model on OpenWebText.
# Mirrors the hyperparameters used in the ChatGPT-2 setup.

wandb_project = "dag-gpt"
wandb_run_name = "daggpt-124M"

# total batch size of ~0.5M tokens
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# train for 300B tokens
max_iters = 600000
lr_decay_iters = 600000

# evaluation settings
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# DAG-specific settings
dag_depth = 4  # Number of DAG steps to perform

# Model architecture (124M parameters)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1
bias = True

# Learning rate settings
learning_rate = 6e-4
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 2000
min_lr = 6e-5

# Dataset
dataset = "shakespeare"

# Training infrastructure
backend = "nccl"  # Use NCCL for multi-GPU training
device = "cuda"
dtype = "bfloat16"  # Use bfloat16 for better performance
compile = True  # Use torch.compile() for faster training

# Checkpointing
out_dir = "out"
always_save_checkpoint = False
init_from = "scratch"
