name = "daggpt-frozen-gpt2"

# Use pretrained GPT-2 weights and freeze backbone
init_from = "gpt2"  # load OpenAI GPT-2 (small) weights
freeze_gpt = True  # training script will freeze non-DAG parameters

# Dataset / task
dataset = "proofpile"
subset = 1.0

# Model architecture
n_layer = 12
n_head = 12
n_embd = 768
bias = True

dag_depth = 8  # enable DAG augmentation

dropout = 0.1

# Training hyper-parameters (same as regular DAG config)
block_size = 512
batch_size = 16
gradient_accumulation_steps = 2

learning_rate = 3e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

max_iters = 100_000
warmup_iters = 10_000
lr_decay_iters = 100_000
min_lr = 3e-5
decay_lr = True

# Logging / evaluation
eval_interval = 1000
eval_iters = 20
log_interval = 50
always_save_checkpoint = True

backend = "nccl"
dtype = "bfloat16"
compile = False  # compile currently unstable for DAG models
