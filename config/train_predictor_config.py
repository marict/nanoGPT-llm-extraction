# Configuration for DAG predictor pretraining on RunPod
# This config is optimized for cloud training on RunPod instances

# Project settings
name = "predictor_pretrain"

# Training intervals
eval_interval = 100  # We don't need to do this very often because train is seeing unseen data as well.
log_interval = 1  # Log every iteration for better monitoring
eval_iters = 10  # Reduced from 5 - each eval iter is much more informative
eval_only = False
clear_previous_checkpoints = False
reload_reset_iters = False

# Model initialization
init_from = "/runpod-volume/checkpoints/vthtoes0nb3rd6-resume_add_digit_tau/ckpt_predictor_pretrain.pt"
# init_from = "scratch"

# Dataset configuration
dataset = "dagset"  # Use DAG dataset for predictor training

train_examples_per_batch = 4000
val_examples_per_batch = 800

# DAG dataset parameters
max_dag_depth = 6  # Match the model dag_depth for consistency
# Choose 4 to match the NALU paper
max_digits = 4
max_decimal_places = 4

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
compile = False  # Disable compilation for now to see if there are any gradient issues.
keep_alive = False  # Auto-stop by default
check_nans = False  # Check for NaNs in cloud training

# Loss weights (balanced for full training)
sign_loss_weight = 1.0
digit_loss_weight = 1.0
op_loss_weight = 1.0
value_loss_weight = 1.0
exec_loss_weight = 0.083

# Exec loss smoothing to prevent spikes
exec_loss_ema_decay = 0.95  # EMA decay factor (0.95 = keep 95% of history)
exec_loss_max_clip = 5.0  # Maximum allowed exec_loss value
exec_loss_warmup_steps = 100  # Steps before EMA kicks in

# Random seeds
seed = 42
