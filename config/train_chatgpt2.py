# Training config for ChatGPT-2 equivalent model on OpenWebText.
# Mirrors the hyperparameters used in the README instructions.

wandb_project = "owt"

# total batch size of ~0.5M tokens
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# train for 300B tokens
max_iters = 600000
lr_decay_iters = 600000

# evaluation settings
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
