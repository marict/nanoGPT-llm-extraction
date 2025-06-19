"""
Training script for nanoGPT and DAGGPT.

This version removes global runtime state in favor of explicit arguments,
making the code easier to test. All side effects are contained in `main()`
and `train()`.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import runpy
import subprocess
import time
from ast import literal_eval
from contextlib import nullcontext
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import runpod_service
from dag_model import DAGGPT, DAGGPTConfig
from model import GPT, GPTConfig
from python_version_check import check_python_version

TORCH_2_2_1 = torch.__version__ >= "2.2.1"
CUDA_AVAILABLE = torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# Configuration utilities
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Container for all training-related hyperparameters."""

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

    learning_rate: float = 6e-4
    max_iters: int = 600_000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 2_000
    lr_decay_iters: int = 600_000
    min_lr: float = 6e-5

    backend: str = "nccl"
    dtype: str = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    compile: bool = True


def load_config_file(path: str) -> Dict[str, object]:
    """Executes a python config file and returns its public symbols."""
    cfg_dict = runpy.run_path(path)
    return {k: v for k, v in cfg_dict.items() if not k.startswith("_")}


def update_config(cfg: TrainConfig, data: Dict[str, object]) -> None:
    """Overwrite fields in <cfg> with matching keys from <data>."""
    for f in fields(cfg):
        if f.name in data:
            setattr(cfg, f.name, data[f.name])


def apply_overrides(cfg: TrainConfig, overrides: List[str]) -> None:
    """Apply --key=value CLI overrides."""
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


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.ArgumentParser:
    """Return an argparse parser pre-populated with CLI flags."""
    parser = argparse.ArgumentParser(description="nanoGPT Trainer")
    parser.add_argument("config", nargs="?", default="config/train_default.py")
    parser.add_argument("--use-runpod", action="store_true")
    parser.add_argument("--dag-depth", type=int)
    parser.add_argument("--gpu-type")
    parser.add_argument("--wandb-api-key")
    return parser


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #
def get_lr(it: int, *, cfg: TrainConfig) -> float:
    """Cosine LR schedule with linear warmup."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def estimate_loss(
    model: torch.nn.Module,
    eval_iters: int,
    batch_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
    ctx: nullcontext | torch.autocast,
) -> Dict[str, torch.Tensor]:
    """Return mean loss over <eval_iters> batches for train/val splits."""
    out: Dict[str, torch.Tensor] = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batch_fn(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# --------------------------------------------------------------------------- #
# Core training routine
# --------------------------------------------------------------------------- #
def train(cfg: TrainConfig) -> None:
    """Run the training loop using hyperparameters in <cfg>."""
    # --------------------------------------------------------------------- #
    # DDP / environment setup
    # --------------------------------------------------------------------- #
    setup_start = time.time()
    print(f"[{time.time() - setup_start:.2f}s] Starting training")
    print(f"[{time.time() - setup_start:.2f}s] PyTorch version: {torch.__version__}")

    ddp_start = time.time()
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=cfg.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert cfg.gradient_accumulation_steps % ddp_world_size == 0
        cfg.gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    print(
        f"[{time.time() - setup_start:.2f}s] DDP setup completed in {time.time() - ddp_start:.2f}s"
    )

    tokens_per_iter = (
        cfg.gradient_accumulation_steps
        * ddp_world_size
        * cfg.batch_size
        * cfg.block_size
    )
    if master_process:
        print(f"[{time.time() - setup_start:.2f}s] Tokens / iter: {tokens_per_iter:,}")

    seed_start = time.time()
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.dtype]
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )
    print(
        f"[{time.time() - setup_start:.2f}s] Device setup completed in {time.time() - seed_start:.2f}s"
    )

    # --------------------------------------------------------------------- #
    # Data loading
    # --------------------------------------------------------------------- #
    data_start = time.time()
    print(f"[{time.time() - setup_start:.2f}s] Loading data")
    data_dir = Path("data") / cfg.dataset
    if not (data_dir / "train.bin").exists():
        if master_process:
            print(
                f"[{time.time() - setup_start:.2f}s] Preparing dataset {cfg.dataset}..."
            )
            from data import prepare_dataset  # local import

            train_tokens, val_tokens = prepare_dataset(cfg.dataset, data_dir)
            print(
                f"[{time.time() - setup_start:.2f}s] Dataset prepared. Train: {train_tokens:,}, Val: {val_tokens:,}"
            )

    meta_start = time.time()
    print(f"[{time.time() - setup_start:.2f}s] Loading meta")
    meta_path = data_dir / "meta.pkl"
    meta_dtype = np.uint16
    vocab_size = None
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
        meta_dtype = np.uint8 if meta.get("byte_level", False) else np.uint16
        if master_process:
            print(
                f"[{time.time() - setup_start:.2f}s] Found vocab_size {vocab_size} and dtype {meta_dtype}"
            )
    print(
        f"[{time.time() - setup_start:.2f}s] Data loading completed in {time.time() - data_start:.2f}s"
    )

    def get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one batch of tokens for <split>."""
        file = "train.bin" if split == "train" else "val.bin"
        data = np.memmap(data_dir / file, dtype=meta_dtype, mode="r")
        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy(data[i : i + cfg.block_size].astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(data[i + 1 : i + 1 + cfg.block_size].astype(np.int64))
                for i in ix
            ]
        )
        if device == "cuda":
            x, y = (
                x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # --------------------------------------------------------------------- #
    # Model creation
    # --------------------------------------------------------------------- #
    model_start = time.time()
    print(f"[{time.time() - setup_start:.2f}s] Initializing model")
    model_args: Dict[str, object] = dict(
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        bias=cfg.bias,
        vocab_size=vocab_size or 50_304,
        dropout=cfg.dropout,
    )
    if cfg.dag_depth > 0:
        model_args["dag_depth"] = cfg.dag_depth
        ModelConfig, ModelClass = DAGGPTConfig, DAGGPT
    else:
        ModelConfig, ModelClass = GPTConfig, GPT

    if cfg.init_from == "scratch":
        if master_process:
            print(f"[{time.time() - setup_start:.2f}s] Initializing model from scratch")
        gptconf = ModelConfig(**model_args)
        model = ModelClass(gptconf)
    elif cfg.init_from == "resume":
        ckpt_path = Path(cfg.out_dir) / "ckpt.pt"
        checkpoint = torch.load(ckpt_path, map_location=device)
        for k in ("n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"):
            model_args[k] = checkpoint["model_args"][k]
        gptconf = ModelConfig(**model_args)
        model = ModelClass(gptconf)
        state_dict = {
            k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()
        }
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    elif cfg.init_from.startswith("gpt2"):
        if master_process:
            print(
                f"[{time.time() - setup_start:.2f}s] Loading GPT-2 weights {cfg.init_from}"
            )
        base_model = GPT.from_pretrained(cfg.init_from, dict(dropout=cfg.dropout))
        for k in ("n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"):
            model_args[k] = getattr(base_model.config, k)
        if cfg.dag_depth > 0:
            gptconf = ModelConfig(**model_args)
            model = ModelClass(gptconf)
        else:
            model = base_model
        iter_num, best_val_loss = 0, 1e9
    else:
        raise ValueError(f"Unsupported init_from {cfg.init_from}")

    if cfg.block_size < model.config.block_size:
        model.crop_block_size(cfg.block_size)
        model_args["block_size"] = cfg.block_size

    print(f"[{time.time() - setup_start:.2f}s] Model type: {type(model).__name__}")
    model.to(device)
    print(
        f"[{time.time() - setup_start:.2f}s] Model creation completed in {time.time() - model_start:.2f}s"
    )

    # Different scaler for different torch versions
    # One for the local version and one for the runpod version
    scalar_enabled = cfg.dtype == "float16"
    scaler = (
        torch.cuda.amp.GradScaler(enabled=scalar_enabled)
        if TORCH_2_2_1
        else torch.amp.GradScaler("cuda", enabled=scalar_enabled)
    )

    optimizer_start = time.time()
    print(f"[{time.time() - setup_start:.2f}s] Initializing optimizer")
    optimizer = model.configure_optimizers(
        cfg.weight_decay,
        cfg.learning_rate,
        (cfg.beta1, cfg.beta2),
        device,
    )
    if cfg.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(
        f"[{time.time() - setup_start:.2f}s] Optimizer initialization completed in {time.time() - optimizer_start:.2f}s"
    )

    compile_start = time.time()
    if cfg.compile:
        print(f"[{time.time() - setup_start:.2f}s] Compiling model")
        model = torch.compile(model)
        print(
            f"[{time.time() - setup_start:.2f}s] Model compilation completed in {time.time() - compile_start:.2f}s"
        )
    else:
        print(f"[{time.time() - setup_start:.2f}s] Compilation disabled")

    if ddp:
        model = DDP(model, device_ids=[int(device.split(":")[-1])])

    # --------------------------------------------------------------------- #
    # W&B
    # --------------------------------------------------------------------- #
    wandb_start = time.time()
    print(f"[{time.time() - setup_start:.2f}s] Initializing wandb")
    if master_process:
        import wandb

        try:
            run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=cfg.__dict__,
                settings=wandb.Settings(
                    start_method="thread"
                ),  # Use thread-based initialization
            )
            print(f"[{time.time() - setup_start:.2f}s] W&B URL: {run.url}")
        except Exception as e:
            print(
                f"[{time.time() - setup_start:.2f}s] Warning: Failed to initialize wandb: {e}"
            )
            run = None
    print(
        f"[{time.time() - setup_start:.2f}s] W&B initialization completed in {time.time() - wandb_start:.2f}s"
    )

    print(f"[{time.time() - setup_start:.2f}s] Entering training loop")

    # --------------------------------------------------------------------- #
    # Training loop
    # --------------------------------------------------------------------- #
    train_start = time.time()
    try:
        X, Y = get_batch("train")
        t0 = time.time()
        iter_num = locals().get("iter_num", 0)
        best_val_loss = locals().get("best_val_loss", 1e9)
        running_mfu = -1.0
        raw_model = model.module if ddp else model

        extra_vals = {}
        while True:
            lr = get_lr(iter_num, cfg=cfg) if cfg.decay_lr else cfg.learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if iter_num % cfg.eval_interval == 0 and master_process:
                try:
                    losses = estimate_loss(model, cfg.eval_iters, get_batch, ctx)
                    print(
                        f"step {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}"
                    )

                    # Log everything to wandb if available
                    if run is not None:
                        try:
                            wandb.log(
                                {
                                    "iter": iter_num,
                                    "train/loss": losses["train"],
                                    "val/loss": losses["val"],
                                    "lr": lr,
                                    "mfu": running_mfu * 100,
                                    **extra_vals,  # Add any extra values from the model
                                }
                            )
                        except Exception as e:
                            print(f"Warning: Failed to log to wandb: {e}")

                    if losses["val"] < best_val_loss or cfg.always_save_checkpoint:
                        best_val_loss = losses["val"]
                        if iter_num > 0:
                            ckpt = {
                                "model": raw_model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "model_args": model_args,
                                "iter_num": iter_num,
                                "best_val_loss": best_val_loss,
                                "config": cfg.__dict__,
                            }
                            if master_process:
                                print(f"Saving checkpoint to {cfg.out_dir}")
                            torch.save(ckpt, Path(cfg.out_dir) / f"ckpt_{iter_num}.pt")
                except Exception as e:
                    print(f"Warning: Error during evaluation: {e}")

            if iter_num == 0 and cfg.eval_only:
                break

            try:
                for micro_step in range(cfg.gradient_accumulation_steps):
                    if ddp:
                        model.require_backward_grad_sync = (
                            micro_step == cfg.gradient_accumulation_steps - 1
                        )
                    with ctx:
                        logits, loss = model(X, Y)
                        loss = loss / cfg.gradient_accumulation_steps
                    X, Y = get_batch("train")
                    scaler.scale(loss).backward()
                    # We have to grab the extra vals after the backwards pass
                    # because some might be gradients.
                    extra_vals = model.extra_vals()

                if cfg.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                dt = time.time() - t0
                t0 = time.time()
                if iter_num % cfg.log_interval == 0 and master_process:
                    lossf = loss.item() * cfg.gradient_accumulation_steps
                    if iter_num >= 5:
                        mfu = raw_model.estimate_mfu(
                            cfg.batch_size * cfg.gradient_accumulation_steps, dt
                        )
                        running_mfu = (
                            0.9 * running_mfu + 0.1 * mfu if running_mfu >= 0 else mfu
                        )
                    print(
                        f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f} ms, "
                        f"mfu {running_mfu*100:.2f}%"
                    )

                iter_num += 1
                if iter_num > cfg.max_iters:
                    break
            except Exception as e:
                print(f"Warning: Error during training step: {e}")
                # Try to recover by getting a new batch
                try:
                    X, Y = get_batch("train")
                except Exception as e2:
                    print(f"Fatal: Could not recover from error: {e2}")
                    break

    except Exception as e:
        print(f"Fatal error in training loop: {e}")
    finally:
        print(
            f"[{time.time() - train_start:.2f}s] Training loop completed in {time.time() - train_start:.2f}s"
        )
        if ddp:
            destroy_process_group()
        if run is not None:
            try:
                run.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")

        # Stop RunPod instance if we're running on RunPod
        if os.getenv("RUNPOD_POD_ID"):
            runpod_service.stop_runpod()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    main_start = time.time()
    print(f"[{time.time() - main_start:.2f}s] Starting main function")

    check_start = time.time()
    check_python_version()
    print(
        f"[{time.time() - main_start:.2f}s] Python version check completed in {time.time() - check_start:.2f}s"
    )

    parser_start = time.time()
    parser = parse_args()
    args, overrides = parser.parse_known_args()
    print(
        f"[{time.time() - main_start:.2f}s] Argument parsing completed in {time.time() - parser_start:.2f}s"
    )

    config_start = time.time()
    cfg = TrainConfig()
    update_config(cfg, load_config_file(args.config))
    apply_overrides(cfg, overrides)
    if args.dag_depth is not None:
        cfg.dag_depth = args.dag_depth
    print(
        f"[{time.time() - main_start:.2f}s] Configuration setup completed in {time.time() - config_start:.2f}s"
    )

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    if not os.getenv("WANDB_API_KEY"):
        parser.error("WANDB_API_KEY is required")

    if args.use_runpod:
        remote_args = " ".join([args.config] + overrides)
        if args.dag_depth is not None:
            remote_args += f" --dag-depth={args.dag_depth}"
        runpod_service.start_cloud_training(
            remote_args,
            args.gpu_type or runpod_service.DEFAULT_GPU_TYPE,
            api_key=os.getenv("RUNPOD_API_KEY"),
        )
        return

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()
