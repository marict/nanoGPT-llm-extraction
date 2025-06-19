wandb_project = "dag-gpt"
wandb_run_name = "daggpt-h100-proofpile"

batch_size = 64
block_size = 4096
gradient_accumulation_steps = 8

max_iters = 1500000
lr_decay_iters = 1500000

eval_interval = 100
eval_iters = 50
log_interval = 10

weight_decay = 1e-1

dag_depth = 4

n_layer = 48
n_head = 48
n_embd = 6144
dropout = 0.1
bias = True

learning_rate = 3e-4
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 10000
min_lr = 3e-5

dataset = "proofpile"

backend = "nccl"
dtype = "bfloat16"
compile = True

out_dir = "out"
always_save_checkpoint = True
init_from = "scratch"
