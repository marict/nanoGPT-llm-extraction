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

# Loss weights (can tune these)
sign_loss_weight = 1.0
digit_loss_weight = 1.0
op_loss_weight = 1.0
value_loss_weight = 1.0  # MSE loss on initial values
exec_loss_weight = 0.5

# Curriculum Learning Parameters for Enhanced Value-Based Learning
# ================================================================

# Global curriculum learning toggle
enable_curriculum_learning = False  # Set to False to disable all curriculum learning

# Value Loss Curriculum (Initial Values)
value_curriculum_beta_start = 1.0  # Start lenient (larger Huber threshold)
value_curriculum_beta_end = 0.1  # End strict (smaller Huber threshold)
value_curriculum_steps = max_iters * 0.1  # Steps to transition over
sign_penalty_start = 0.05  # Start with low sign penalty weight
sign_penalty_end = 0.2  # End with higher sign penalty weight

# Exec Loss Curriculum (Final Execution Values)
exec_curriculum_beta_start = 1.0  # Start lenient for Huber loss
exec_curriculum_beta_end = 0.05  # End very strict
exec_curriculum_steps = max_iters * 0.16  # Longer transition for exec loss
exec_rel_weight_start = 0.005  # Start with low relative error weight
exec_rel_weight_end = 0.03  # End with higher relative error weight
exec_overflow_start = 30.0  # Start with lenient overflow threshold
exec_overflow_end = 25.0  # End with stricter overflow threshold

# Digit Loss Curriculum (Digit Prediction Confidence)
digit_entropy_weight_start = 0.0  # Start with no entropy penalty
digit_entropy_weight_end = 0.05  # End with entropy penalty for sharper predictions
digit_entropy_curriculum_steps = max_iters * 0.12  # Steps to ramp up entropy penalty

# Exec loss smoothing to prevent spikes
exec_loss_ema_decay = 0.95  # EMA decay factor (0.95 = keep 95% of history)
exec_loss_max_clip = 5.0  # Maximum allowed exec_loss value
exec_loss_warmup_steps = max_iters * 0.002  # Steps before EMA kicks in

# Random seeds
seed = 42
