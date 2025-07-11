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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tiktoken
import torch
from torch import Tensor
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import runpod_service
import wandb
from checkpoint_manager import CheckpointManager
from dag_logger import DAGLogger
from data import prepare_dataset
from evaluation import estimate_loss, evaluate_math
from models.dag_model import GPT, OP_NAMES, GPTConfig
from python_version_check import check_python_version
from training_utils import (CHECKPOINT_DIR, BaseConfig, _check_for_nonfinite,
                            apply_overrides, generate_run_name, get_lr,
                            load_config_file, parse_args, update_config)

TORCH_2_2_1 = torch.__version__ >= "2.2.1"
CUDA_AVAILABLE = torch.cuda.is_available()

# Optional safetensors for pure-tensor checkpoints
try:
    import safetensors.torch as _st  # type: ignore

    _HAVE_ST = True
except ModuleNotFoundError:
    _HAVE_ST = False


# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------

# Default sample prompt for evaluation - consistent across sample.py and train.py
DEFAULT_SAMPLE_PROMPT = "Two plus 5 = "


# --------------------------------------------------------------------------- #
# Configuration utilities
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig(BaseConfig):
    """Container for all training-related hyperparameters."""

    name: str = "owt"

    # Learning rate schedule
    use_cyclical_lr: bool = False
    cyclical_lr_period: int = 1000
    cyclical_lr_amplitude: float = 0.1

    # Math evaluation settings
    eval_math: bool = True  # Whether to run math evaluation during training
    math_eval_tasks: List[str] = field(default_factory=lambda: ["gsm8k", "svamp"])
    math_eval_examples: int = 50  # Max examples per math task during training

    dataset: str = "openwebtext"
    subset: float = 1.0  # Fraction of dataset to use (0.0 < subset <= 1.0)
    gradient_accumulation_steps: int = 5 * 8
    batch_size: int = 12
    sequence_length: int = 1024

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    dag_depth: int = 0

    # DAG streaming dataset parameters
    max_dag_depth: int = 8
    train_examples_per_batch: int = 1000
    val_examples_per_batch: int = 100
    seed: int = 42

    # Loss weights for DAG predictor training
    sign_loss_weight: float = 1.0
    log_loss_weight: float = 1.0
    op_loss_weight: float = 1.0

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

    # Fine-tuning options
    freeze_gpt: bool = False  # If True, freeze GPT backbone and train DAG layers only


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Core training routine
# --------------------------------------------------------------------------- #
def train(cfg: TrainConfig, wandb_run_id: str | None = None) -> None:
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
    # Initialize checkpoint manager and clean previous checkpoints if requested
    # --------------------------------------------------------------------- #
    checkpoint_manager = CheckpointManager("regular")
    if cfg.clear_previous_checkpoints:
        checkpoint_manager.clean_previous_checkpoints(cfg.name)

    # --------------------------------------------------------------------- #
    # W&B
    # --------------------------------------------------------------------- #
    wandb_start = time.time()
    print(f"[{time.time() - setup_start:.2f}s] Initializing wandb")
    if master_process:
        try:
            if wandb_run_id:
                # Resume existing wandb run
                print(
                    f"[{time.time() - setup_start:.2f}s] Resuming wandb run: {wandb_run_id}"
                )
                run = wandb.init(
                    project=cfg.name,
                    id=wandb_run_id,
                    resume="must",
                    config=cfg.__dict__,
                )
            else:
                # Create new wandb run
                run = wandb.init(
                    project=cfg.name,
                    name=generate_run_name(cfg),
                    config=cfg.__dict__,
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

    tokens_per_iter = (
        cfg.gradient_accumulation_steps
        * ddp_world_size
        * cfg.batch_size
        * cfg.sequence_length
    )
    if master_process:
        print(f"[{time.time() - setup_start:.2f}s] Tokens / iter: {tokens_per_iter:,}")

    seed_start = time.time()
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Determine the best available device
    if CUDA_AVAILABLE:
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Smart dtype handling with fallback for unsupported bfloat16
    original_dtype = cfg.dtype
    if cfg.dtype == "bfloat16":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            # Test if BFloat16 actually works with a simple operation
            try:
                test_tensor = torch.tensor([1.0], device=device, dtype=torch.bfloat16)
                test_result = test_tensor * 2.0  # Simple operation
                actual_dtype = "bfloat16"
            except Exception as e:
                # BFloat16 claimed to be supported but doesn't work
                actual_dtype = "float16"
                print(
                    f"⚠️  BFloat16 theoretically supported but failed in practice on this CUDA device. Falling back to Float16. Error: {e}"
                )
        elif device == "cuda":
            # CUDA available but BFloat16 not supported, fallback to float16
            actual_dtype = "float16"
            print(
                f"⚠️  BFloat16 requested but not supported on this CUDA device. Falling back to Float16."
            )
        else:
            # Non-CUDA device (CPU/MPS), fallback to float32 for stability
            actual_dtype = "float32"
            print(
                f"⚠️  BFloat16 requested but not supported on {device} device. Falling back to Float32."
            )
    else:
        actual_dtype = cfg.dtype

    # Update the config to use the actual dtype throughout the rest of the system
    cfg.dtype = actual_dtype

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[actual_dtype]

    if original_dtype != actual_dtype:
        print(
            f"[{time.time() - setup_start:.2f}s] Dtype fallback: {original_dtype} → {actual_dtype}"
        )

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
    # Look for data in dataset-specific subfolder (new structure)
    data_dir = Path("data") / cfg.dataset / cfg.dataset
    # If dataset-specific subfolder doesn't exist, try legacy location
    if not data_dir.exists():
        data_dir = Path("data") / cfg.dataset

    # -------------------------------------------------------------
    # Dataset preparation (streaming-aware)
    # -------------------------------------------------------------
    needs_prep = False
    if cfg.dataset == "dagset":
        # Purely streaming: we consider the dataset ready if its meta file exists.
        needs_prep = not (data_dir / "meta.pkl").exists()
    else:
        # File-based datasets still look for train.bin as before.
        needs_prep = not (data_dir / "train.bin").exists()

    if needs_prep:
        if master_process:
            print(
                f"[{time.time() - setup_start:.2f}s] Preparing dataset {cfg.dataset}... with subset {cfg.subset}"
            )
            train_tokens, val_tokens = prepare_dataset(
                cfg.dataset, Path("data"), cfg.subset
            )
            print(
                f"[{time.time() - setup_start:.2f}s] Dataset prepared. Train: {train_tokens:,}, Val: {val_tokens:,}"
            )
            # Update data_dir to point to the newly created dataset-specific folder
            data_dir = Path("data") / cfg.dataset

    print(f"[{time.time() - setup_start:.2f}s] Loading meta")
    meta_path = data_dir / "meta.pkl"
    meta_dtype = np.uint16
    vocab_size = None
    if master_process:
        print(
            f"[{time.time() - setup_start:.2f}s] Found vocab_size {vocab_size} and dtype {meta_dtype}"
        )

    # (Streaming dataset pathway removed.)

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
        # Use on-disk token dataset
        file = "train.bin" if split == "train" else "val.bin"
        data = np.memmap(data_dir / file, dtype=meta_dtype, mode="r")
        ix = torch.randint(len(data) - cfg.sequence_length, (cfg.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy(data[i : i + cfg.sequence_length].astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    data[i + 1 : i + 1 + cfg.sequence_length].astype(np.int64)
                )
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
        block_size=cfg.sequence_length,
        bias=cfg.bias,
        vocab_size=vocab_size or 50_304,
        dropout=cfg.dropout,
    )
    model_args["dag_depth"] = cfg.dag_depth
    ModelConfig, ModelClass = GPTConfig, GPT

    # Handle checkpoint loading using unified checkpoint manager
    expected_keys = ["model_args", "model", "iter_num", "best_val_loss"]
    checkpoint, iter_num, best_val_loss = checkpoint_manager.handle_checkpoint_loading(
        cfg, device, expected_keys=expected_keys
    )

    if cfg.init_from == "scratch":
        if master_process:
            print(f"[{time.time() - setup_start:.2f}s] Initializing model from scratch")
        gptconf = ModelConfig(**model_args)
        model = ModelClass(gptconf)
    elif cfg.init_from.startswith("gpt2"):
        if master_process:
            print(
                f"[{time.time() - setup_start:.2f}s] Loading GPT-2 weights {cfg.init_from}"
            )
        base_model = GPT.from_pretrained(cfg.init_from, dict(dropout=cfg.dropout))
        for k in ("n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"):
            model_args[k] = getattr(base_model.config, k)
        # Create new model with desired dag_depth (may be >0)
        gptconf = ModelConfig(**model_args)
        model = ModelClass(gptconf)

        # Load overlapping GPT weights regardless of dag_depth
        try:
            model.load_state_dict(base_model.state_dict(), strict=False)
        except Exception as e:
            print(f"Warning: partial weight loading failed: {e}")
    elif checkpoint is not None:
        # Loading from checkpoint (resume or direct path)
        print(f"[{time.time() - setup_start:.2f}s] Loading model from checkpoint")
        for k in ("n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"):
            model_args[k] = checkpoint["model_args"][k]
        gptconf = ModelConfig(**model_args)
        model = ModelClass(gptconf)
        state_dict = {
            k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()
        }
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported init_from {cfg.init_from}")

    if cfg.sequence_length < model.config.block_size:
        model.crop_block_size(cfg.sequence_length)
        model_args["block_size"] = cfg.sequence_length

    print(f"[{time.time() - setup_start:.2f}s] Model type: {type(model).__name__}")
    model.to(device)

    # Optionally freeze GPT backbone parameters (except DAG components)
    if getattr(cfg, "freeze_gpt", False):
        frozen_count = 0
        for name, param in model.named_parameters():
            # Identify parameters belonging to transformer backbone or lm_head
            if name.startswith("transformer") or name.startswith("lm_head"):
                if not name.startswith("dag") and not name.startswith(
                    "scalar_to_embed"
                ):
                    param.requires_grad = False
                    frozen_count += 1
        if master_process:
            print(
                f"Froze {frozen_count} GPT parameters; DAG-specific layers remain trainable."
            )

    print(
        f"[{time.time() - setup_start:.2f}s] Model creation completed in {time.time() - model_start:.2f}s"
    )

    # Different scaler for different torch versions
    # One for the local version and one for the runpod version
    scalar_enabled = actual_dtype == "float16"
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
    if checkpoint is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(
        f"[{time.time() - setup_start:.2f}s] Optimizer initialization completed in {time.time() - optimizer_start:.2f}s"
    )

    compile_start = time.time()
    if cfg.compile:
        print(f"[{time.time() - setup_start:.2f}s] Compiling model")
        try:
            # Disable CUDA graphs; they crash when tensors share storage in
            # complex ways ("complex memory overlap" segfault). This keeps the
            # bulk of Inductor speed-ups while avoiding the runtime fault.
            model = torch.compile(
                model,
                mode="reduce-overhead",
                disable="cudagraphs",
            )
        except TypeError:
            # Older PyTorch without the 'disable' kwarg – fall back to safe compile
            model = torch.compile(model, mode="reduce-overhead")
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

        # Track consecutive errors to prevent infinite retry loops
        consecutive_errors = 0
        max_consecutive_errors = 5

        while True:
            lr = get_lr(iter_num, cfg=cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            if iter_num % cfg.eval_interval == 0 and iter_num > 0 and master_process:
                try:
                    losses = estimate_loss(model, cfg.eval_iters, get_batch, ctx)

                    # Generate a sample sentence to track generation quality
                    if encode is None or decode is None:
                        raise ValueError("No tokenizer available")
                    try:
                        # Simple prompt for generation
                        sample_prompt = DEFAULT_SAMPLE_PROMPT
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
                            # Show only the continuation (exclude the prompt prefix)
                            continuation_ids = generated[0, len(encoded) :]
                            continuation_text = decode(continuation_ids.cpu().tolist())
                            print(
                                f"Generated sample: {sample_prompt}{{BEGIN}}{continuation_text}{{END}}"
                            )
                    except Exception as e:
                        print(f"Warning: Failed to generate sample text: {e}")
                        raise e

                    # Compute logging statistics and collect non-gradient extra values
                    eval_extra = {}
                    dag_logger.compute_log_statistics(raw_model)
                    eval_extra.update(dag_logger.get_extra_vals(raw_model))
                    dag_logger.format_console_logging(
                        raw_model, decode_fn=decode, input_ids=prompt_ids
                    )

                    # Run math evaluation if enabled
                    math_scores = {}
                    if cfg.eval_math:
                        math_scores = evaluate_math(
                            model,
                            device,
                            tasks=cfg.math_eval_tasks,
                            max_examples=cfg.math_eval_examples,
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
                    if run is None:
                        raise ValueError("Run is not available")
                    try:
                        base_log_dict = {
                            "iter": iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            # Log the *actual* learning rate in the optimizer to ensure we record
                            # the value that was truly applied this step.
                            "lr": optimizer.param_groups[0]["lr"],
                            "mfu": running_mfu * 100,
                            **eval_extra,
                        }

                        # Add math evaluation scores
                        for task, score in math_scores.items():
                            base_log_dict[f"math_eval/{task}"] = score

                        # Get comprehensive logging dict
                        log_dict = (
                            dag_logger.get_wandb_logging_dict(raw_model, base_log_dict)
                            if dag_logger
                            else base_log_dict
                        )

                        # Note: Gradient logging happens during training steps, not evaluation
                        # Evaluation logs non-gradient metrics only

                        wandb.log(log_dict, step=iter_num, commit=False)
                    except Exception as e:
                        print(
                            f"ERROR: Failed to log evaluation data to wandb at iter {iter_num}: {e}"
                        )
                        traceback.print_exc()
                        raise e

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

                            # Calculate validation accuracy for filename
                            val_acc = None
                            if math_scores:
                                # Use average of math evaluation scores as validation accuracy
                                valid_scores = [
                                    score
                                    for score in math_scores.values()
                                    if score >= 0
                                ]
                                if valid_scores:
                                    val_acc = sum(valid_scores) / len(valid_scores)

                            checkpoint_filename = (
                                checkpoint_manager.generate_checkpoint_filename(
                                    cfg.name, iter_num, val_acc=val_acc
                                )
                            )
                            if master_process:
                                print(f"Saving checkpoint: {checkpoint_filename}")
                                checkpoint_manager.save_checkpoint(
                                    ckpt, checkpoint_filename
                                )
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
                        # Immediately abort training if loss becomes NaN or Inf
                        if not torch.isfinite(loss):
                            # Log one last time
                            # Prepare logging dict
                            base_dict = {"iter": iter_num, **extra_vals}
                            log_dict = (
                                dag_logger.get_wandb_logging_dict(raw_model, base_dict)
                                if dag_logger
                                else base_dict
                            )
                            wandb.log(log_dict, step=iter_num, commit=True)

                            raise RuntimeError(
                                f"Non-finite loss detected (value={loss.item()}) at iteration {iter_num}. Aborting training."
                            )
                        loss = loss / cfg.gradient_accumulation_steps

                    # Set up gradient tracking AFTER forward pass but BEFORE backward pass
                    if micro_step == 0:  # Only set up once
                        dag_logger.setup_gradient_tracking(raw_model)

                    X, Y = get_batch("train")
                    scaler.scale(loss).backward()
                    for n, p in model.named_parameters():
                        if p.grad is not None and (
                            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                        ):
                            print(
                                f"[GRAD NAN] {n}  →  min={p.grad.min():.3e}  max={p.grad.max():.3e}"
                            )
                            break

                # Compute logging statistics (populates non-gradient metrics)
                dag_logger.compute_log_statistics(raw_model)

                # Get extra values after backward pass (includes gradients captured by hooks)
                extra_vals = dag_logger.get_extra_vals(raw_model)

                if cfg.grad_clip:
                    scaler.unscale_(optimizer)
                    # Gradient sanity check BEFORE clipping so we detect raw overflows
                    if cfg.check_nans:
                        _check_for_nonfinite(
                            (
                                (n, p.grad)
                                for n, p in model.named_parameters()
                                if p.grad is not None
                            ),
                            "GRAD",
                        )
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                else:
                    scaler.unscale_(optimizer)
                    if cfg.check_nans:
                        _check_for_nonfinite(
                            (
                                (n, p.grad)
                                for n, p in model.named_parameters()
                                if p.grad is not None
                            ),
                            "GRAD",
                        )
                scaler.step(optimizer)
                # Parameter sanity check immediately after the update but before scaler.update()
                if cfg.check_nans:
                    _check_for_nonfinite(
                        ((n, p) for n, p in model.named_parameters()), "PARAM"
                    )
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                dt = time.time() - t0
                t0 = time.time()
                if iter_num % cfg.log_interval == 0 and master_process:
                    try:
                        # Prepare logging dict
                        base_dict = {"iter": iter_num, **extra_vals}
                        log_dict = (
                            dag_logger.get_wandb_logging_dict(raw_model, base_dict)
                            if dag_logger
                            else base_dict
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

                # Reset error counter on successful training step
                consecutive_errors = 0

                iter_num += 1
                if iter_num > cfg.max_iters:
                    break
            except Exception as e:
                error_msg = str(e)
                consecutive_errors += 1
                print(
                    f"Warning: Error during training step ({consecutive_errors}/{max_consecutive_errors}): {e}"
                )

                # Check for critical errors that indicate fundamental incompatibility
                critical_errors = [
                    "Got unsupported ScalarType BFloat16",
                    "CUDA out of memory",
                    "device-side assert triggered",
                    "not implemented for",
                    "RuntimeError: Expected",
                    "Logging data not available",
                    "Non-finite loss",
                ]

                is_critical = any(
                    critical_pattern in error_msg
                    for critical_pattern in critical_errors
                )

                if is_critical:
                    print(f"CRITICAL ERROR: {error_msg}")
                    print(
                        "This error indicates a fundamental compatibility issue that cannot be recovered from."
                    )
                    print("Stopping training to prevent resource waste.")
                    break

                # Check if we've had too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    print(
                        f"FATAL: Too many consecutive errors ({consecutive_errors}). Stopping training."
                    )
                    break

                # For non-critical errors, try to recover by getting a new batch
                try:
                    X, Y = get_batch("train")
                    print("Recovered from non-critical error, continuing training...")
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
        run.finish()

        # Stop RunPod instance if we're running on RunPod and keep-alive is not enabled
        if os.getenv("RUNPOD_POD_ID") and not getattr(cfg, "keep_alive", False):
            runpod_service.stop_runpod()


# --------------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------------
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
    if args.keep_alive:
        cfg.keep_alive = args.keep_alive

    # Store note separately without altering project name
    if args.note:
        cfg.note = args.note
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
    train(cfg, wandb_run_id=args.wandb_run_id)


if __name__ == "__main__":
    main()
