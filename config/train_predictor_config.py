# Configuration for DAG predictor pretraining on RunPod
# This config is optimized for cloud training on RunPod instances

# Project settings
name = "predictor_pretrain"

# Training intervals
eval_interval = 20
log_interval = 1  # Logging interval for training metrics
eval_iters = 1
eval_once = False
clear_previous_checkpoints = False
reload_reset_iters = False

# Model initialization
# init_from = "/runpod-volume/checkpoints/b2gv3ae90i77nj-base10_weightbalance/ckpt_predictor_pretrain.pt"
init_from = "scratch"

# Dataset configuration
dataset = "dagset"

# DAG dataset parameters
max_dag_depth = 4  # Must match dag_depth in model config
# The original NALU paper had values in range 9999

max_digits = 4
max_decimal_places = 4
base = 10

# Model configuration
gradient_accumulation_steps = 1
batch_size = 512
block_size = 32

# Model architecture (larger for RunPod training)
n_layer = 2
n_head = 4
n_embd = n_head * 64
dropout = 0.1
bias = True
dag_depth = max_dag_depth  # MUST match max_dag_depth above

# Optimization (tuned for longer training)
max_iters = 10000

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
train_uncertainty_params = True

# Random seeds
seed = 42

loss_flags = {
    "sign": True,
    "digit": True,
    "op": True,
    "value": True,
    "exec": True,
    "stats": False,
}
