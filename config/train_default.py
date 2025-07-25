"""Minimal CPU configuration used by default when running ``train.py``."""

# evaluate every step and run a single iteration by default
eval_interval = 3
log_interval = 1
eval_iters = 1
eval_once = False
always_save_checkpoint = True
init_from = "scratch"

# Math evaluation settings for quick testing
math_eval_examples = 1

name = "daggpt-default"

# tiny dataset and network for quick local testing
dataset = "proofpile"
gradient_accumulation_steps = 1
batch_size = 4
block_size = 32
subset = 0.000001
clear_previous_checkpoints = True

n_layer = 1
n_head = 1
n_embd = 32
dropout = 0.0
bias = False

dag_depth = 4

learning_rate = 6e-4
max_iters = 6
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

math_eval_max_examples = 1
decay_lr = False
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5


backend = "gloo"
dtype = "bfloat16"  # keep this, fine on A100/H100
# dtype = "float32"
compile = False
