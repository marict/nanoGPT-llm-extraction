"""Ultra-fast CPU configuration used by default when running ``train.py``."""

from dataclasses import field

# evaluate every step and run minimal iterations by default
eval_interval = 1
log_interval = 1
eval_iters = 1
eval_once = False
always_save_checkpoint = False  # Skip checkpointing for speed
init_from = "scratch"

# Math evaluation settings for quick testing
math_eval_examples = 1

name = "daggpt-default"

# tiny dataset and network for quick local testing
dataset = "dagset"  # Fast synthetic data instead of file-based proofpile
gradient_accumulation_steps = 1
batch_size = 2  # Minimal batch size
block_size = 8  # Minimal block size
clear_previous_checkpoints = True


# DAG dataset parameters - minimal for speed
max_dag_depth = 2  # Minimal depth
max_digits = 2  # Minimal digits
max_decimal_places = 1  # Minimal decimal places


n_layer = 1
n_head = 1
n_embd = 8  # Minimal embedding
dropout = 0.0
bias = False

dag_depth = 2  # Match max_dag_depth for consistency

learning_rate = 1e-3
max_iters = 2  # Ultra-minimal iterations
weight_decay = 0.0  # Skip for speed
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

math_eval_max_examples = 1
decay_lr = False
warmup_iters = 0  # Skip warmup
lr_decay_iters = 2
min_lr = 1e-4

backend = "gloo"
dtype = "float32"  # Fastest on most systems
compile = False  # Skip compilation overhead
