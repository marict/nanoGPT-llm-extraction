"""Training config for DAGGPT on RunPod."""

wandb_log = True
wandb_project = 'daggpt'
wandb_run_name = 'daggpt-runpod'

# OpenWebText dataset with GPT-2 size model
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# Model dimensions match GPT-2 (124M parameters)
n_layer = 12
n_head = 12
n_embd = 768

dag_depth = 4
dag_hidden_dim = 16

# 300B tokens
max_iters = 600000
lr_decay_iters = 600000

eval_interval = 1000
eval_iters = 200
log_interval = 10

weight_decay = 1e-1
