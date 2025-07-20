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
reload_reset_iters = True

# Model initialization
# init_from = (
#     "/runpod-volume/checkpoints/6jtm4x7kn65w4m-new_losses/ckpt_predictor_pretrain.pt"
# )
init_from = "scratch"

# Dataset configuration
dataset = "dagset"  # Use DAG dataset for predictor training

# DAG dataset parameters
max_dag_depth = 6  # Match the model dag_depth for consistency
max_digits = 6  # Maximum number of integer digits for uniform digit distribution
max_decimal_places = 6  # Seems like a reasonable rounding for the numbers initially

# OPTIMIZED: Significantly increase data generation for better GPU utilization
train_examples_per_batch = 8000  # Increased from 4000 - match larger batch sizes
val_examples_per_batch = 1600  # Increased from 800 - match larger batch sizes

# English conversion settings
english_conversion_probability = 0.3
integer_no_decimal_probability = 0.5
expression_simplification_probability = 0.5
expression_expansion_probability = 0.5

# Expression rendering style probabilities - showcasing diverse styles for RunPod training
printing_style_probs = {
    "sstr": 0.4,
    "pretty": 0.2,
    "ascii": 0.1,
    "latex": 0.2,
    "str": 0.1,
}

# Model configuration
gradient_accumulation_steps = 16  # Updated from 8
batch_size = 512  # Updated from 128
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
