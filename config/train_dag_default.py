# Default configuration for DAG predictor pretraining
# This config is optimized for pretraining the DAG predictor on structure prediction

# Project settings
name = "dag_pretrain"
note = "DAG predictor pretraining on structure prediction"

# Training intervals
eval_interval = 100
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = True
clear_previous_checkpoints = False

# Model initialization
init_from = "scratch"  # or "resume"

# DAG dataset parameters
max_dag_depth = 6
min_dag_depth = 1
train_examples_per_batch = 500
val_examples_per_batch = 100

# Training hyperparameters
gradient_accumulation_steps = 2
batch_size = 16
sequence_length = 256

# Model architecture (should match target model)
n_layer = 6  # Smaller for faster pretraining
n_head = 6
n_embd = 384  # Smaller for faster pretraining
dropout = 0.1
bias = False
dag_depth = 4  # Target DAG depth

# Optimization
learning_rate = 5e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 5e-5

# System settings
backend = "nccl"
dtype = "float16"  # Use float16 for faster training
compile = True
keep_alive = False
check_nans = False

# Loss weights (can tune these)
sign_loss_weight = 1.0
log_loss_weight = 1.0
op_loss_weight = 2.0  # Emphasize operation prediction

# Random seeds
train_seed = 42
val_seed = 43
