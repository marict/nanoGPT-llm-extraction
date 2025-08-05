# Minimal CPU configuration used by default when running ``train_predictor.py``.
"""Minimal CPU configuration for quick local testing of DAG predictor training.

This is the predictor analogue of ``config/train_default.py`` – all settings are scaled
down for fast execution on a laptop/CI runner. Adjust as needed for real training.
"""

# Project settings
# ----------------
# Mirrors the naming convention used in ``train_default.py``.
name = "dag_predictor-default"

# Training intervals
# ------------------
# Ultra-minimal for speed.
max_iters = 2
eval_interval = 1
log_interval = 1
eval_iters = 1
eval_once = False
always_save_checkpoint = False  # Skip checkpointing for speed
# Remove any stale checkpoints when starting a new run.
clear_previous_checkpoints = True

# Model initialization
init_from = "scratch"  # or "resume"

# Dataset configuration
dataset = "dagset"  # Use DAG dataset for predictor training

# DAG dataset parameters - minimal for speed
max_dag_depth = 2  # Minimal depth
max_digits = 2  # Minimal digits
max_decimal_places = 1  # Minimal decimal places


# Training hyperparameters - minimal
gradient_accumulation_steps = 1
batch_size = 2  # Minimal batch size
block_size = 8  # Minimal block size

# Model architecture - tiny for speed
n_head = 1
n_layer = 1  # Minimal layers
n_embd = 8  # Minimal embedding
dropout = 0.0
bias = False  # Simpler for speed
dag_depth = 2  # Match max_dag_depth for consistency


# Optimization - minimal
learning_rate = 1e-3
weight_decay = 0.0  # Skip for speed
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule - minimal
warmup_iters = 0  # Skip warmup
lr_decay_iters = 2
min_lr = 1e-4

use_cyclical_lr = False  # Skip cyclical LR
cyclical_lr_period = 1
cyclical_lr_amplitude = 0.0

# System settings
backend = "gloo"
dtype = "float32"  # Fastest on most systems
compile = False  # Skip compilation overhead
keep_alive = False
check_nans = False

sharp_training = True
# Random seeds
seed = 42
