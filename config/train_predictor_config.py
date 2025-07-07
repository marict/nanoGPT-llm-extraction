# Configuration for DAG predictor pretraining on RunPod
# This config is optimized for cloud training on RunPod instances

# Project settings
name = "predictor_pretrain"
note = "DAG predictor pretraining on structure prediction - RunPod"

# Training intervals
eval_interval = 200  # Less frequent evaluation to save time
log_interval = 20  # More frequent logging for monitoring
eval_iters = 100  # More evaluation iterations for better metrics
eval_only = False
always_save_checkpoint = True
clear_previous_checkpoints = True  # Save space on RunPod

# Model initialization
init_from = "scratch"  # or "resume"

# Dataset configuration
dataset = "dagset"  # Use DAG dataset for predictor training

# DAG dataset parameters
max_dag_depth = 8  # Deeper for more complex training

train_examples_per_batch = 1000  # Larger batches for efficiency
val_examples_per_batch = 200  # More validation examples

# Training hyperparameters (optimized for RunPod)
gradient_accumulation_steps = 4  # Larger for better gradient estimates
batch_size = 32  # Larger batch size for GPU efficiency
sequence_length = 512  # Full sequence length

# Model architecture (larger for RunPod training)
n_layer = 12  # Full size model
n_head = 12  # Full attention heads
n_embd = 768  # Full embedding size
dropout = 0.1  # Slight dropout for regularization
bias = False
dag_depth = 6  # Deeper DAG for complex structures

# Optimization (tuned for longer training)
learning_rate = 3e-5  # Standard learning rate
max_iters = 20_000  # Longer training for RunPod
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 1000  # Longer warmup for stability
lr_decay_iters = 20000  # Match max_iters
min_lr = 3e-5

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
