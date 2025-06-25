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
import random
import runpy
import string
import time
from ast import literal_eval
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tiktoken
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import run_math_eval
import runpod_service
import wandb
from dag_logger import DAGLogger
from dag_model import GPT, GPTConfig
from data import prepare_dataset
from evaluation import estimate_loss, evaluate_math
from python_version_check import check_python_version

TORCH_2_2_1 = torch.__version__ >= "2.2.1"
CUDA_AVAILABLE = torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# Configuration utilities
# --------------------------------------------------------------------------- #
# Use RunPod volume if available, otherwise use local directory
CHECKPOINT_DIR = (
    "/runpod-volume/checkpoints" if os.path.exists("/runpod-volume") else "checkpoints"
)


def generate_run_name(cfg) -> str:
    """Generate a run name using RunPod identifier or local random string."""
    # Check if we're running on RunPod
    runpod_id = os.environ.get("RUNPOD_POD_ID")

    if runpod_id and runpod_id.strip():
        # Use RunPod identifier
        return runpod_id
    else:
        # Generate local identifier with random string
        random_str = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=12)
        )
        return f"local_{random_str}"


@dataclass
class TrainConfig:
    """Container for all training-related hyperparameters."""

    eval_interval: int = 250
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    clean_previous_runs: bool = False  # Remove previous checkpoints on startup
    init_from: str = "scratch"

    # Math evaluation settings
    eval_math: bool = True  # Whether to run math evaluation during training
    math_eval_tasks: List[str] = field(default_factory=lambda: ["gsm8k", "svamp"])
    math_eval_max_examples: int = 50  # Max examples per math task during training

    name: str = (
        "owt"  # Project/run name (used for both wandb project and runpod naming)
    )

    dataset: str = "openwebtext"
    subset: float = 1.0  # Fraction of dataset to use (0.0 < subset <= 1.0)
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
    parser.add_argument("--subset", type=float)
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


def get_checkpoint_filename(cfg: TrainConfig, iter_num: int) -> str:
    """Generate checkpoint filename with config name and iteration number."""
    # Sanitize the name for filesystem compatibility
    safe_name = "".join(c for c in cfg.name if c.isalnum() or c in ("-", "_"))
    return f"ckpt_{safe_name}_{iter_num}.pt"


def clean_previous_checkpoints(cfg: TrainConfig) -> None:
    """Remove all previous checkpoint files for this config name."""
    if not cfg.clean_previous_runs:
        print("Skipping checkpoint cleanup")
        return

    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return

    # Sanitize the name for filesystem compatibility
    safe_name = "".join(c for c in cfg.name if c.isalnum() or c in ("-", "_"))
    pattern = f"ckpt_{safe_name}_*.pt"

    removed_count = 0
    for ckpt_file in checkpoint_dir.glob(pattern):
        try:
            ckpt_file.unlink()
            removed_count += 1
        except Exception as e:
            print(f"Warning: Failed to remove checkpoint {ckpt_file}: {e}")

    if removed_count > 0:
        print(f"Cleaned {removed_count} previous checkpoint(s) for '{cfg.name}'")


def find_latest_checkpoint(cfg: TrainConfig) -> Path | None:
    """Find the latest checkpoint file for this config name."""
    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return None

    # Sanitize the name for filesystem compatibility
    safe_name = "".join(c for c in cfg.name if c.isalnum() or c in ("-", "_"))
    pattern = f"ckpt_{safe_name}_*.pt"

    checkpoint_files = list(checkpoint_dir.glob(pattern))
    if not checkpoint_files:
        return None

    # Extract iteration numbers and find the latest
    latest_file = None
    latest_iter = -1

    for ckpt_file in checkpoint_files:
        try:
            # Extract iteration number from filename: ckpt_{name}_{iter}.pt
            parts = ckpt_file.stem.split("_")
            if len(parts) >= 3:
                iter_num = int(parts[-1])
                if iter_num > latest_iter:
                    latest_iter = iter_num
                    latest_file = ckpt_file
        except (ValueError, IndexError):
            continue

    return latest_file


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

    # --------------------------------------------------------------------- #
    # W&B
    # --------------------------------------------------------------------- #
    wandb_start = time.time()
    print(f"[{time.time() - setup_start:.2f}s] Initializing wandb")
    if master_process:
        try:
            run = wandb.init(
                project=cfg.name,
                name=generate_run_name(cfg),
                config=cfg.__dict__,
                settings=wandb.Settings(
                    start_method="thread"
                ),  # Use thread-based initialization
            )
            print(f"[{time.time() - setup_start:.2f}s] W&B URL: {run.url}")
        except Exception as e:
            print(
                f"[{time.time() - setup_start:.2f}s] Error: Failed to initialize wandb: {e}"
            )
            raise  # Re-raise the exception to fail the job
    print(
        f"[{time.time() - setup_start:.2f}s] W&B initialization completed in {time.time() - wandb_start:.2f}s"
    )

    # Clean previous checkpoints if requested
    clean_previous_checkpoints(cfg)

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
                f"[{time.time() - setup_start:.2f}s] Preparing dataset {cfg.dataset}... with subset {cfg.subset}"
            )
            train_tokens, val_tokens = prepare_dataset(
                cfg.dataset, data_dir, cfg.subset
            )
            print(
                f"[{time.time() - setup_start:.2f}s] Dataset prepared. Train: {train_tokens:,}, Val: {val_tokens:,}"
            )

    print(f"[{time.time() - setup_start:.2f}s] Loading meta")
    meta_path = data_dir / "meta.pkl"
    meta_dtype = np.uint16
    vocab_size = None

    # Load tokenizer for text generation
    encode = decode = None
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
        meta_dtype = np.uint8 if meta.get("byte_level", False) else np.uint16
        if master_process:
            print(
                f"[{time.time() - setup_start:.2f}s] Found vocab_size {vocab_size} and dtype {meta_dtype}"
            )
        # Set up encoder/decoder for text generation
        if "stoi" in meta and "itos" in meta:
            stoi, itos = meta["stoi"], meta["itos"]
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: "".join([itos[i] for i in l])

    # Fallback to GPT-2 tokenizer if no meta available
    if encode is None or decode is None:
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        if master_process:
            print(f"[{time.time() - setup_start:.2f}s] Using GPT-2 tokenizer fallback")

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
    model_args["dag_depth"] = cfg.dag_depth
    ModelConfig, ModelClass = GPTConfig, GPT

    if cfg.init_from == "scratch":
        if master_process:
            print(f"[{time.time() - setup_start:.2f}s] Initializing model from scratch")
        gptconf = ModelConfig(**model_args)
        model = ModelClass(gptconf)
    elif cfg.init_from == "resume":
        # Find the latest checkpoint for this config name
        ckpt_path = find_latest_checkpoint(cfg)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found for config name '{cfg.name}' in {CHECKPOINT_DIR}"
            )

        print(f"Resuming from checkpoint: {ckpt_path}")
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
        # Always create a new model with the desired dag_depth
        gptconf = ModelConfig(**model_args)
        model = ModelClass(gptconf)
        # If dag_depth=0, copy weights from the pretrained model
        if cfg.dag_depth == 0:
            model.load_state_dict(base_model.state_dict())
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
    # Training loop
    # --------------------------------------------------------------------- #
    train_start = time.time()
    try:
        X, Y = get_batch("train")
        print(f"[{time.time() - setup_start:.2f}s] Got batch")

        t0 = time.time()
        iter_num = locals().get("iter_num", 0)
        best_val_loss = locals().get("best_val_loss", 1e9)
        running_mfu = -1.0
        raw_model = model.module if ddp else model
        extra_vals = {}

        print(f"[{time.time() - setup_start:.2f}s] Entering training loop")

        # Initialize DAG logger for models that support it
        dag_logger = DAGLogger()

        while True:
            lr = get_lr(iter_num, cfg=cfg) if cfg.decay_lr else cfg.learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if iter_num % cfg.eval_interval == 0 and iter_num > 0 and master_process:
                try:
                    losses = estimate_loss(model, cfg.eval_iters, get_batch, ctx)

                    # Generate a sample sentence to track generation quality
                    generated_sample = ""
                    if encode is not None and decode is not None:
                        try:
                            # Simple prompt for generation
                            sample_prompt = "Two plus 5 is equal to: "
                            encoded = encode(sample_prompt)
                            prompt_ids = torch.tensor(
                                encoded, dtype=torch.long, device=device
                            ).unsqueeze(0)

                            with torch.no_grad():
                                generated = raw_model.generate(
                                    prompt_ids,
                                    max_new_tokens=20,
                                    temperature=0.8,
                                    top_k=40,
                                )
                                generated_sample = decode(generated[0].cpu().tolist())
                                print(f"Generated sample: {generated_sample}")
                        except Exception as e:
                            print(f"Warning: Failed to generate sample text: {e}")
                            generated_sample = f"Error: {str(e)}"
                    else:
                        print("Skipping text generation (no tokenizer available)")
                        generated_sample = "No tokenizer available"

                    # Setup DAG logger and get extra values AFTER text generation
                    eval_extra = model.extra_vals()
                    if dag_logger is not None:
                        # For evaluation, we need to do a separate forward/backward pass to get gradients
                        # since estimate_loss() doesn't compute gradients
                        try:
                            model.train()  # Temporarily switch to train mode
                            dag_logger.setup_gradient_tracking(raw_model)

                            # Do a single forward/backward pass to capture gradients
                            eval_X, eval_Y = get_batch("train")
                            with ctx:
                                _, eval_loss = model(eval_X, eval_Y)
                            eval_loss.backward()

                            # Now get the gradient information
                            eval_extra.update(dag_logger.get_extra_vals(raw_model))
                            dag_logger.format_console_logging(raw_model)

                            # Clear gradients and switch back to eval mode
                            model.zero_grad()
                            model.eval()
                        except Exception as e:
                            print(
                                f"Warning: Failed to capture gradients during evaluation: {e}"
                            )
                            # Fallback to just getting non-gradient extra values
                            eval_extra.update(dag_logger.get_extra_vals(raw_model))
                            dag_logger.format_console_logging(raw_model)

                    # Run math evaluation if enabled
                    math_scores = {}
                    if cfg.eval_math:
                        math_scores = evaluate_math(
                            model,
                            device,
                            tasks=cfg.math_eval_tasks,
                            max_examples=cfg.math_eval_max_examples,
                        )

                    print(
                        f"step {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}"
                    )
                    if math_scores:
                        math_str = ", ".join(
                            [
                                f"{task}: {score:.4f}"
                                for task, score in math_scores.items()
                            ]
                        )
                        print(f"math eval: {math_str}")

                    # Log everything to wandb if available
                    if run is not None:
                        try:
                            base_log_dict = {
                                "iter": iter_num,
                                "train/loss": losses["train"],
                                "val/loss": losses["val"],
                                "lr": lr,
                                "mfu": running_mfu * 100,
                                **eval_extra,
                            }

                            # Add math evaluation scores
                            for task, score in math_scores.items():
                                base_log_dict[f"math_eval/{task}"] = score

                            # Get comprehensive logging dict
                            log_dict = (
                                dag_logger.get_wandb_logging_dict(
                                    raw_model, base_log_dict
                                )
                                if dag_logger
                                else base_log_dict
                            )

                            # Check if gradients are being logged
                            grad_keys = [
                                k for k in log_dict.keys() if k.startswith("op_grad/")
                            ]
                            if dag_logger and len(grad_keys) == 0:
                                print(
                                    f"Warning: No gradient keys found in wandb log_dict for evaluation at iter {iter_num}"
                                )
                                print(f"Available keys: {list(log_dict.keys())}")
                            elif dag_logger and len(grad_keys) > 0:
                                print(
                                    f"Logging {len(grad_keys)} gradient values to wandb (evaluation)"
                                )

                            wandb.log(log_dict, step=iter_num, commit=False)
                        except Exception as e:
                            print(
                                f"ERROR: Failed to log evaluation data to wandb at iter {iter_num}: {e}"
                            )
                            import traceback

                            traceback.print_exc()

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
                            checkpoint_filename = get_checkpoint_filename(cfg, iter_num)
                            checkpoint_path = Path(CHECKPOINT_DIR) / checkpoint_filename
                            if master_process:
                                print(f"Saving checkpoint: {checkpoint_path}")
                            torch.save(ckpt, checkpoint_path)
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
                        _, loss = model(X, Y)
                        loss = loss / cfg.gradient_accumulation_steps

                    # Set up gradient tracking AFTER forward pass but BEFORE backward pass
                    if dag_logger is not None and micro_step == 0:  # Only set up once
                        dag_logger.setup_gradient_tracking(raw_model)

                    X, Y = get_batch("train")
                    scaler.scale(loss).backward()

                # Get extra values after backward pass (for gradients)
                extra_vals = model.extra_vals()
                if dag_logger is not None:
                    extra_vals.update(dag_logger.get_extra_vals(raw_model))

                if cfg.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                dt = time.time() - t0
                t0 = time.time()
                if (
                    iter_num % cfg.log_interval == 0
                    and master_process
                    and run is not None
                ):
                    try:
                        # Prepare logging dict
                        base_dict = {"iter": iter_num, **extra_vals}
                        log_dict = (
                            dag_logger.get_wandb_logging_dict(raw_model, base_dict)
                            if dag_logger
                            else base_dict
                        )

                        # Check if gradients are being logged
                        grad_keys = [
                            k for k in log_dict.keys() if k.startswith("op_grad/")
                        ]
                        if dag_logger and len(grad_keys) == 0:
                            print(
                                f"Warning: No gradient keys found in wandb log_dict for training at iter {iter_num}"
                            )
                            print(f"Available keys: {list(log_dict.keys())}")
                        elif dag_logger and len(grad_keys) > 0:
                            print(
                                f"Logging {len(grad_keys)} gradient values to wandb (training)"
                            )

                        wandb.log(log_dict, step=iter_num, commit=True)
                    except Exception as e:
                        print(
                            f"ERROR: Failed to log training data to wandb at iter {iter_num}: {e}"
                        )
                        import traceback

                        traceback.print_exc()
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
    if args.subset is not None:
        cfg.subset = args.subset
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

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()
