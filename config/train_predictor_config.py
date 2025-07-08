# Configuration for DAG predictor pretraining on RunPod
# This config is optimized for cloud training on RunPod instances

# Project settings
name = "predictor_pretrain"
note = "DAG predictor pretraining on structure prediction - RunPod"

# Training intervals
eval_interval = 50  # Reduced from 50 - each iter has 128x more data
log_interval = 1  # Log every iteration for better monitoring
eval_iters = 2  # Reduced from 5 - each eval iter is much more informative
eval_only = False
always_save_checkpoint = False
clear_previous_checkpoints = False  # Save space on RunPod

# Model initialization
init_from = "scratch"

# Dataset configuration
dataset = "dagset"  # Use DAG dataset for predictor training

# DAG dataset parameters
max_dag_depth = 6  # Match the model dag_depth for consistency
max_digits = 4  # Maximum number of integer digits for uniform digit distribution
max_decimal_places = 6  # Seems like a reasonable rounding for the numbers initially

# OPTIMIZED: Significantly increase data generation for better GPU utilization
train_examples_per_batch = 8000  # Increased from 4000 - match larger batch sizes
val_examples_per_batch = 1600  # Increased from 800 - match larger batch sizes

# English conversion settings
english_conversion_rate = 0.3  # Probability of converting tokens to English (0.0 = disabled, 1.0 = always convert)

# Model configuration
gradient_accumulation_steps = (
    16  # Increased from 8 - effective batch size = 1024 * 16 = 16,384
)
batch_size = 512
sequence_length = 128

# Model architecture (larger for RunPod training)
n_layer = 32
n_head = n_layer
n_embd = n_head * 64
dropout = 0.3  # Slight dropout for regularization
bias = False
dag_depth = max_dag_depth  # MUST match max_dag_depth above

# Optimization (tuned for longer training)
learning_rate = (
    1e-3  # Scaled up from 3e-4 for large batch size (conservative 3x scaling)
)
max_iters = 50_000  # Longer training for RunPod
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
warmup_iters = max_iters * 0.1  # Longer warmup for stability
lr_decay_iters = max_iters  # Match max_iters
min_lr = 1e-5

use_cyclical_lr = True
cyclical_lr_period = max_iters * 0.2
cyclical_lr_amplitude = 0.1

# System settings (optimized for RunPod)
backend = "nccl"
dtype = "bfloat16"  # Use bfloat16 for efficiency if available
compile = True  # Enable compilation for speed
keep_alive = False  # Auto-stop by default
check_nans = False  # Check for NaNs in cloud training

# Loss weights (balanced for full training)
sign_loss_weight = 1.0
log_loss_weight = 1.0
op_loss_weight = 1.0

# Random seeds
train_seed = 42
val_seed = 42
