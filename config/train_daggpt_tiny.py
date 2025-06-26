name = "daggpt_tiny"

batch_size = 8  # smaller to reduce memory pressure
block_size = 512  # shorter context for faster training

gradient_accumulation_steps = 2  # reduces step latency

max_iters = 2000  # 1–2 GPU hours on A100 class cards
lr_decay_iters = 2000

eval_interval = 500
eval_iters = 10
log_interval = 20

weight_decay = 1e-1

dag_depth = 4  # reduce DAG cost

n_layer = 2  # very shallow
n_head = 4  # just enough heads
n_embd = 128  # smallest safe embedding

dropout = 0.1
bias = True

learning_rate = 3e-4
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
min_lr = 1e-5

dataset = "proofpile"

backend = "nccl"
dtype = "bfloat16"  # keep this, fine on A100/H100
compile = False  # set False to avoid extra compilation overhead

always_save_checkpoint = True
init_from = "scratch"
