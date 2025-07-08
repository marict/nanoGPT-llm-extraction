# Configuration for DAG predictor pretraining on RunPod
# This config is optimized for cloud training on RunPod instances

# Project settings
name = "predictor_pretrain"
note = "DAG predictor pretraining on structure prediction - RunPod"

# Training intervals
eval_interval = 100
log_interval = 50
eval_iters = 10
eval_only = False
always_save_checkpoint = False
clear_previous_checkpoints = True  # Save space on RunPod

# Model initialization
init_from = "scratch"  # or "resume"

# Dataset configuration
dataset = "dagset"  # Use DAG dataset for predictor training

# DAG dataset parameters
max_dag_depth = 6  # Match the model dag_depth for consistency
value_range = (
    -10000.0,
    10000.0,
)  # Allow negative values for meaningful sign prediction

train_examples_per_batch = 1000  # Larger batches for efficiency
val_examples_per_batch = 200  # More validation examples

# English conversion settings
english_conversion_rate = 0.3  # Probability of converting tokens to English (0.0 = disabled, 1.0 = always convert)

# Expression permutation settings
permutation_probability = (
    0.0  # Probability of applying permutation (0.0 = disabled, 1.0 = always permute)
)

# Training hyperparameters (optimized for RunPod)
gradient_accumulation_steps = 1
batch_size = 128
sequence_length = 128

# Model architecture (larger for RunPod training)
n_layer = 12  # Full size model
n_head = 12  # Full attention heads
n_embd = 768  # Full embedding size
dropout = 0.1  # Slight dropout for regularization
bias = False
dag_depth = max_dag_depth  # MUST match max_dag_depth above

# Optimization (tuned for longer training)
learning_rate = 3e-4  # Standard learning rate
max_iters = 50_000  # Longer training for RunPod
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = max_iters * 0.05  # Longer warmup for stability
lr_decay_iters = max_iters  # Match max_iters
min_lr = 3e-5

# System settings (optimized for RunPod)
backend = "nccl"
dtype = "bfloat16"  # Use bfloat16 for efficiency if available
compile = True  # Enable compilation for speed
keep_alive = False  # Auto-stop by default
check_nans = False  # Check for NaNs in cloud training

# Loss weights (balanced for full training)
sign_loss_weight = 0.3
log_loss_weight = 1.0
op_loss_weight = 1.0

# Random seeds
train_seed = 42
val_seed = 42
