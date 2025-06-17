"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
from pathlib import Path

from python_version_check import check_python_version

check_python_version()
import argparse
import math
import pickle
import runpy
import time
from ast import literal_eval
from contextlib import nullcontext
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import runpod_service
from dag_model import DAGGPT, DAGGPTConfig
from model import GPT, GPTConfig


@dataclass
class TrainConfig:
    out_dir: str = "out"
    eval_interval: int = 250
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = "scratch"

    wandb_project: str = "owt"
    wandb_run_name: str = "gpt2"

    dataset: str = "openwebtext"
    gradient_accumulation_steps: int = 5 * 8
    batch_size: int = 12
    block_size: int = 1024

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    dag_depth: int = 0
    dag_hidden_dim: int = 16

    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    backend: str = "nccl"
    device: str = "cuda"
    dtype: str = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    compile: bool = True


def load_config_file(path: str) -> dict:
    cfg_dict = runpy.run_path(path)
    return {k: v for k, v in cfg_dict.items() if not k.startswith("_")}


def update_config(cfg: TrainConfig, data: dict) -> None:
    for f in fields(cfg):
        if f.name in data:
            setattr(cfg, f.name, data[f.name])


def apply_overrides(cfg: TrainConfig, overrides: list[str]) -> None:
    for arg in overrides:
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(f"Invalid override: {arg}")
        key, val = arg[2:].split("=", 1)
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key: {key}")
        cur = getattr(cfg, key)
        try:
            lit = literal_eval(val)
        except Exception:
            lit = val
        if not isinstance(lit, type(cur)):
            raise ValueError(f"Invalid type for {key}")
        setattr(cfg, key, lit)


parser = argparse.ArgumentParser(description="nanoGPT Trainer")
parser.add_argument("config", nargs="?", default="config/train_default.py")
parser.add_argument("--use-runpod", action="store_true")
parser.add_argument("--dag-depth", type=int)
parser.add_argument("--gpu-type")
parser.add_argument("--wandb-api-key", help="Weights & Biases API key")
args, overrides = parser.parse_known_args()

cfg = TrainConfig()
update_config(cfg, load_config_file(args.config))
apply_overrides(cfg, overrides)
if args.dag_depth is not None:
    cfg.dag_depth = args.dag_depth

# use cfg directly instead of polluting globals
_use_runpod_flag = args.use_runpod
_dag_depth_override = args.dag_depth
_gpu_type_flag = args.gpu_type or runpod_service.DEFAULT_GPU_TYPE
config_path = args.config
config = vars(cfg)

# propagate API keys via environment variables
if args.wandb_api_key:
    os.environ["WANDB_API_KEY"] = args.wandb_api_key

# validate required keys
if not os.getenv("WANDB_API_KEY"):
    parser.error("WANDB_API_KEY is required for logging to Weights & Biases")

# local aliases for config values
out_dir = cfg.out_dir
eval_interval = cfg.eval_interval
log_interval = cfg.log_interval
eval_iters = cfg.eval_iters
eval_only = cfg.eval_only
always_save_checkpoint = cfg.always_save_checkpoint
init_from = cfg.init_from
wandb_project = os.getenv("RUNPOD_POD_NAME", cfg.wandb_project)
wandb_run_name = cfg.wandb_run_name
dataset = cfg.dataset
gradient_accumulation_steps = cfg.gradient_accumulation_steps
batch_size = cfg.batch_size
block_size = cfg.block_size
n_layer = cfg.n_layer
n_head = cfg.n_head
n_embd = cfg.n_embd
dropout = cfg.dropout
bias = cfg.bias
dag_depth = cfg.dag_depth
dag_hidden_dim = cfg.dag_hidden_dim
learning_rate = cfg.learning_rate
max_iters = cfg.max_iters
weight_decay = cfg.weight_decay
beta1 = cfg.beta1
beta2 = cfg.beta2
grad_clip = cfg.grad_clip
decay_lr = cfg.decay_lr
warmup_iters = cfg.warmup_iters
lr_decay_iters = cfg.lr_decay_iters
min_lr = cfg.min_lr
backend = cfg.backend
device = cfg.device
dtype = cfg.dtype
compile = cfg.compile

if _use_runpod_flag:
    remote_args = config_path
    for ov in overrides:
        remote_args += f" {ov}"
    if _dag_depth_override is not None:
        # use the public argument name with dash, not underscore
        remote_args += f" --dag-depth={_dag_depth_override}"
    runpod_service.start_cloud_training(
        remote_args, _gpu_type_flag, api_key=os.getenv("RUNPOD_API_KEY")
    )
    raise SystemExit
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# poor man's data loader
DATA_DIR = Path("data") / dataset

# run prepare script if train.bin doesn't exist
if not (DATA_DIR / "train.bin").exists():
    print(f"Preparing dataset {dataset}...")
    from data import prepare_dataset
    train_tokens, val_tokens = prepare_dataset(dataset, DATA_DIR)
    print(f"Dataset preparation complete. Train tokens: {train_tokens:,}, Val tokens: {val_tokens:,}")

# attempt to derive vocab_size from the dataset
META_PATH = DATA_DIR / "meta.pkl"
META_VOCAB_SIZE = None
META_DTYPE = np.uint16  # default dtype
if META_PATH.exists():
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    META_VOCAB_SIZE = meta["vocab_size"]
    META_DTYPE = np.uint8 if meta.get("byte_level", False) else np.uint16
    print(f"found vocab_size = {META_VOCAB_SIZE} (inside {META_PATH})")
    print(f"found meta_dtype = {META_DTYPE} (inside {META_PATH})")


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(
            DATA_DIR / "train.bin", dtype=META_DTYPE, mode="r"
        )
    else:
        data = np.memmap(DATA_DIR / "val.bin", dtype=META_DTYPE, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)  # start with model_args from command line
if dag_depth > 0:
    model_args.update(dag_depth=dag_depth, dag_hidden_dim=dag_hidden_dim)

ModelConfig = DAGGPTConfig if dag_depth > 0 else GPTConfig
ModelClass = DAGGPT if dag_depth > 0 else GPT
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if META_VOCAB_SIZE is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = META_VOCAB_SIZE if META_VOCAB_SIZE is not None else 50304
    gptconf = ModelConfig(**model_args)
    model = ModelClass(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = Path(out_dir) / "ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelConfig(**model_args)
    model = ModelClass(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    base_model = GPT.from_pretrained(init_from, override_args)
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(base_model.config, k)
    if dag_depth > 0:
        print(
            "DAG model cannot load gpt2 weights, starting from scratch with same dimensions"
        )
        gptconf = ModelConfig(**model_args)
        model = ModelClass(gptconf)
    else:
        model = base_model
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = (
        block_size  # so that the checkpoint will have the right value
    )
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
try:
    scaler = torch.amp.GradScaler(device_type=device_type, enabled=(dtype == "float16"))
except (AttributeError, TypeError):
    # fall back for older PyTorch versions
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss() -> dict[str, torch.Tensor]:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if master_process:
    import wandb

    run = wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    print(f"W&B run URL: {run.url}")

# training loop
X, Y = get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        wandb.log(
            {
                "iter": iter_num,
                "train/loss": losses["train"],
                "val/loss": losses["val"],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            }
        )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, Path(out_dir) / "ckpt.pt")
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = model(X, Y)
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
