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
# init_from = "/runpod-volume/checkpoints/wujclmlklb90c6-fixdepthweight/ckpt_predictor_pretrain.pt"
init_from = "scratch"

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
gradient_accumulation_steps = 16
batch_size = 256
sequence_length = 128

# Model architecture (larger for RunPod training)
n_head = 12
n_embd = n_head * 64
dropout = 0.35  # Slight dropout for regularization
bias = False
dag_depth = max_dag_depth  # MUST match max_dag_depth above

# Optimization (tuned for longer training)
learning_rate = 5e-5  # Based on when performance started to degrade
max_iters = 50000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
warmup_iters = max_iters * 0.02  # 1/50 of max_iters
lr_decay_iters = max_iters  # Updated to match max_iters
min_lr = 1e-5

# System settings (optimized for RunPod)
backend = "nccl"
dtype = "bfloat16"  # Use bfloat16 for efficiency if available
compile = True  # Enable compilation for speed
keep_alive = False  # Auto-stop by default
check_nans = False  # Check for NaNs in cloud training

# Loss weights (balanced for full training)
sign_loss_weight = 1.0
digit_loss_weight = 1.0
op_loss_weight = 1.0
value_loss_weight = 1.0
exec_loss_weight = 1.0

# Random seeds
seed = 42
