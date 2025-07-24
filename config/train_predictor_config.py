# Configuration for DAG predictor pretraining on RunPod
# This config is optimized for cloud training on RunPod instances

# Project settings
name = "predictor_pretrain"

# Training intervals
eval_interval = 10
log_interval = 1
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
max_dag_depth = 4  # Must match dag_depth in model config
# The original NALU paper had values in range 9999
max_digits = 3
max_decimal_places = 3
base = 22

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
n_head = 6
n_layer = 1
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
compile = False
keep_alive = False  # Auto-stop by default
check_nans = False  # Check for NaNs in cloud training

# Loss weights
sign_loss_weight = 1.0
digit_loss_weight = 1.0
op_loss_weight = 1.0
value_loss_weight = 0
exec_loss_weight = 0

# Random seeds
seed = 42
