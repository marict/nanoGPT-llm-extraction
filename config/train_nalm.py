"""Configuration for training on the NALM dataset."""

from pathlib import Path

# Model configuration (matching predictor training config)
n_layer = 2
n_head = 4
n_embd = n_head * 64  # 256
block_size = 32  # Matching predictor training
bias = True
dropout = 0.1

# Training configuration
learning_rate = 6e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 6e-5

# System configuration
device = "auto"
dtype = "bfloat16" if device == "cuda" else "float16"
compile = True

# Data configuration
dataset = "nalm"
data_dir = Path("data")
subset = 1.0
num_proc = 8
force = False

# NALM-specific configuration
nalm_config = {
    "train_range": (-10.0, 10.0),
    "val_range": (-10.0, 10.0),
    "extrapolation_range": (-100.0, 100.0),
    "operations": ["add", "sub", "mul", "div"],
    "batch_size": 32,
    "train_examples": 10000,
    "val_examples": 1000,
    "seed": 42,
}

# Evaluation configuration
eval_interval = 100
eval_iters = 200
log_interval = 10

# Checkpointing
out_dir = Path("out")
always_save_checkpoint = True

# Weights & Biases
wandb_log = True
wandb_project = "nalm-training"
wandb_run_name = "nalm-gpt-mini"

# DAG configuration (if using DAG-GPT)
use_dag = True  # Set to True to use DAG integration
dag_config = {
    "dag_depth": 4,  # Must match the checkpoint's dag_depth
    "max_digits": 4,  # Must match the checkpoint's max_digits
    "max_decimal_places": 4,  # Must match the checkpoint's max_decimal_places
}

# Checkpoint configuration
init_from = "/runpod-volume/checkpoints/rhfuok9btu6t8m-pretrained_sharp/ckpt_predictor_pretrain_8600_99.98acc.pt"
