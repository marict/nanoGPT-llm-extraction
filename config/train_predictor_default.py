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
# Evaluate every step and keep logs chatty for debugging.
eval_interval = 1
log_interval = 1
eval_iters = 20
eval_only = False
always_save_checkpoint = True
# Remove any stale checkpoints when starting a new run.
clear_previous_checkpoints = True

# Model initialization
init_from = "latest"  # or "resume"

# Dataset configuration
dataset = "dagset"  # Use DAG dataset for predictor training

# DAG dataset parameters
max_dag_depth = 4
max_digits = 4  # Maximum number of integer digits for uniform digit distribution
max_decimal_places = (
    None  # Auto-derived from max_digits for uniform string distribution
)

train_examples_per_batch = 100
val_examples_per_batch = 20

# English conversion settings
english_conversion_rate = 0.3  # Probability of converting tokens to English (0.0 = disabled, 1.0 = always convert)

# Data generation settings

# Training hyperparameters
gradient_accumulation_steps = 1
batch_size = 4
sequence_length = 32

# Model architecture (should match target model)
n_head = 1
n_embd = 32
dropout = 0.0
bias = False
dag_depth = 4  # Target DAG depth

# Optimization
learning_rate = 6e-4
max_iters = 10
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
warmup_iters = 2
lr_decay_iters = 6
min_lr = 1e-5

use_cyclical_lr = True
cyclical_lr_period = 2
cyclical_lr_amplitude = 0.1

# System settings
backend = "gloo"
# Keep this for mixed-precision support on modern CPUs/GPUs; fine on most hardware.
dtype = "bfloat16"
compile = False
keep_alive = False
check_nans = False

# Loss weights (can tune these)
sign_loss_weight = 1.0
log_loss_weight = 1.0
op_loss_weight = 1.0

# Random seeds
seed = -1
