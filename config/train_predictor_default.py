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
max_iters = 10
eval_interval = 3
log_interval = 1
eval_iters = 1
eval_once = False
always_save_checkpoint = True
# Remove any stale checkpoints when starting a new run.
clear_previous_checkpoints = True

# Model initialization
init_from = "scratch"  # or "resume"

# Dataset configuration
dataset = "dagset"  # Use DAG dataset for predictor training

# DAG dataset parameters
max_dag_depth = 6  # Match the model dag_depth for consistency
# Choose 4 to match the NALU paper
max_digits = 4
max_decimal_places = 4
base = 10  # Dataset generation now supports configurable bases

# Expression generation settings
english_conversion_probability = 0.5
integer_no_decimal_probability = 0.5
expression_simplification_probability = 0.5
expression_expansion_probability = 0.5
printing_style_probs = {
    "sstr": 0.25,
    "pretty": 0.25,
    "ascii": 0.25,
    "latex": 0.25,
}
# Data generation settings

# Training hyperparameters
gradient_accumulation_steps = 1
batch_size = 4
block_size = 32

# Model architecture (should match target model)
n_head = 1
n_layer = 2
n_embd = 32
dropout = 0.0
bias = True  # Enable bias for ALU weight patching compatibility
dag_depth = 4  # Target DAG depth


# Optimization
learning_rate = 6e-4
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
compile = True
keep_alive = False
check_nans = False

# Random seeds
seed = 42
