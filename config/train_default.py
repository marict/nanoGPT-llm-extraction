"""Minimal CPU configuration used by default when running ``train.py``."""

# evaluate every step and run a single iteration by default
out_dir = "out"
eval_interval = 1
log_interval = 1
eval_iters = 1
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

wandb_project = "dag-gpt"

# tiny dataset and network for quick local testing
dataset = "shakespeare"
gradient_accumulation_steps = 1
batch_size = 2
block_size = 32

n_layer = 1
n_head = 1
n_embd = 32
dropout = 0.0
bias = False

dag_depth = 0

learning_rate = 6e-4
max_iters = 10
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

backend = "gloo"
device = "cpu"
dtype = "float32"
compile = False
