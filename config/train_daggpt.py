name = "daggpt"

batch_size = 16
block_size = 512
gradient_accumulation_steps = 2

max_iters = 100_000
lr_decay_iters = 100_000

eval_interval = 1000
eval_iters = 20
log_interval = 50

weight_decay = 1e-1

dag_depth = 8

# n_embed must be divisible by n_head
n_layer = 12
n_head = 12
n_embd = 768
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
compile = False  # Currently broken for dag model

always_save_checkpoint = True
init_from = "scratch"
