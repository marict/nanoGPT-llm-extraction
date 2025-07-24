# Configuration for DAG predictor pretraining on RunPod
# This config is optimized for cloud training on RunPod instances

# Project settings
name = "predictor_pretrain"

# Training intervals
eval_interval = 50
log_interval = 10
eval_iters = 10
eval_once = False
clear_previous_checkpoints = False
reload_reset_iters = False

# Model initialization
# init_from = "/runpod-volume/checkpoints/932rfb4cs2izun-resume_add_digit_tau_2/ckpt_predictor_pretrain.pt"
init_from = "scratch"

# Dataset configuration
dataset = "dagset"  # Use DAG dataset for predictor training

# DAG dataset parameters
max_dag_depth = 6  # Match the model dag_depth for consistency
# The original NALU paper had values in range 9999
max_digits = 14
max_decimal_places = 14
base = 2

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

# Model configuration
gradient_accumulation_steps = 1
batch_size = 256
block_size = 128

# Model architecture (larger for RunPod training)
n_head = 12
n_layer = 6
n_embd = n_head * 64
dropout = 0.1
bias = True
dag_depth = max_dag_depth  # MUST match max_dag_depth above

# Optimization (tuned for longer training)
max_iters = 50000
# weight_decay = 1e-1
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
warmup_iters = max_iters * 0.02  # 1/50 of max_iters
lr_decay_iters = max_iters  # Updated to match max_iters
min_lr = 2e-4
learning_rate = 3e-4

# System settings (optimized for RunPod)
backend = "nccl"
dtype = "bfloat16"  # Use bfloat16 for efficiency if available
compile = True
keep_alive = False  # Auto-stop by default
check_nans = False  # Check for NaNs in cloud training

# Loss weights
sign_loss_weight = 1.0
digit_loss_weight = 1.0
op_loss_weight = 1.0
value_loss_weight = 1.0
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
