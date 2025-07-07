name = "daggpt"

batch_size = 48
block_size = 512
gradient_accumulation_steps = 1

# Whatever is enough to show saturation
max_iters = 6_000
lr_decay_iters = 6_000
warmup_iters = int(max_iters * 0.05)


eval_interval = max_iters // 15
# No math eval by for now until we can get a better model
math_eval_examples = 0
eval_iters = 5
log_interval = 50

weight_decay = 0.05

dag_depth = 4

# n_embed must be divisible by n_head
n_layer = 12
n_head = 12
n_embd = n_head * 64
dropout = 0.01
bias = True

learning_rate = 3e-4
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
min_lr = 3e-5

dataset = "proofpile"

backend = "nccl"
dtype = "bfloat16"
compile = True

always_save_checkpoint = True
init_from = "scratch"
check_nans = False
