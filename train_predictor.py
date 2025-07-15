"""DAG predictor pre-training entry-point (lean version)."""

from __future__ import annotations

import argparse
import math
import os
import random
import random as _eval_random
import runpy
import string
import time
from ast import literal_eval
from contextlib import nullcontext
# no local dataclass definitions needed – imported from predictor_config
from pathlib import Path
from typing import Dict, List

import torch
import torch as _torch
import torch.nn as nn
import torch.nn.functional as F
from tiktoken import get_encoding
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import runpod_service
import wandb
from checkpoint_manager import CheckpointManager
from data.dagset.streaming import create_dag_structure_dataloaders
from models.dag_model import GPT, LOG_LIM, OP_NAMES
from models.predictor_only_model import PredictorOnlyModel
from predictor_config import DAGTrainConfig
from predictor_utils import (compute_dag_structure_loss, evaluate_dag_model,
                             tokenize_texts)
from python_version_check import check_python_version
from training_utils import (CHECKPOINT_DIR, BaseConfig, apply_overrides,
                            generate_run_name, get_lr, load_config_file,
                            parse_args, update_config)

# Local helper utilities


def _empty_metrics() -> dict[str, float]:
    """Return a zero-filled metrics accumulator used during training."""
    return {
        "total_loss": 0.0,
        "sign_loss": 0.0,
        "log_loss": 0.0,
        "op_loss": 0.0,
        "op_accuracy": 0.0,
        "full_dag_op_match": 0.0,
        "sign_accuracy": 0.0,
        "log_magnitude_mape": 0.0,
    }


TORCH_2_2_1 = torch.__version__ >= "2.2.1"
CUDA_AVAILABLE = torch.cuda.is_available()

# Optional safetensors for pure-tensor checkpoints
try:
    import safetensors.torch as _st  # type: ignore

    _HAVE_ST = True
except ModuleNotFoundError:
    _HAVE_ST = False


# Configuration constants

# Checkpoint directory
CHECKPOINT_DIR = (
    "/runpod-volume/checkpoints" if os.path.exists("/runpod-volume") else "checkpoints"
)


# The original `tokenize_texts` implementation has been moved to
# `predictor_utils.tokenize_texts` to avoid code duplication.


# Safe checkpoint saving (logic unchanged)


# DAGTrainConfig is now defined in `predictor_config.py` – importing keeps the
# original public interface intact while significantly shrinking this file.


# Duplicate `get_lr` implementation removed – we use the one provided by
# `training_utils.get_lr`.


# The local implementations of ``compute_dag_structure_loss`` and
# ``evaluate_dag_model`` have been moved to *predictor_utils* (imported above).
# The auxiliary ``_get_dag_predictions`` helper was unused and has been dropped.


def parse_args() -> argparse.ArgumentParser:  # type: ignore[override]
    """Light wrapper that makes the *config* positional argument optional with a
    sensible default, while re-using the common parser from ``training_utils``.
    """
    import training_utils as _tu

    parser = _tu.parse_args()

    # Make ``config`` optional and set the default for predictor runs.
    for action in parser._actions:
        if action.dest == "config":
            action.nargs = "?"
            action.default = "config/train_predictor_default.py"
            break

    return parser


# Duplicate get_lr removed – we rely on the imported `training_utils.get_lr`.


# The local implementations of ``compute_dag_structure_loss`` and
# ``evaluate_dag_model`` have been moved to *predictor_utils* (imported above).
# The auxiliary ``_get_dag_predictions`` helper was unused and has been dropped.


def train_predictor(cfg: DAGTrainConfig, wandb_run_id: str | None = None) -> None:
    """Run DAG predictor training loop."""
    # Setup
    setup_start = time.time()
    # (prints removed to keep script minimal)
    # W&B URL debug print removed

    # DDP setup (simplified for DAG training)
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
    ddp_start = time.time()
    print(
        f"[{time.time() - setup_start:.2f}s] DDP setup completed in {time.time() - ddp_start:.2f}s"
    )

    # Determine model name and create appropriate config
    # Model configuration will be created (or reconstructed) later by the
    # checkpoint manager, so we only need the model name here.
    model_name = GPT.__name__ if cfg.full_backbone else PredictorOnlyModel.__name__

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager("dag")

    # Clean previous checkpoints
    if master_process and cfg.clear_previous_checkpoints:
        checkpoint_manager.clean_previous_checkpoints(cfg.name, model_name)

    # W&B initialization
    if master_process:
        try:
            if wandb_run_id:
                run = wandb.init(
                    project=cfg.name,
                    id=wandb_run_id,
                    resume="must",
                    config=cfg.__dict__,
                )
            else:
                run = wandb.init(
                    project=cfg.name,
                    name=generate_run_name(cfg),
                    config=cfg.__dict__,
                )
            # print(f"[{time.time() - setup_start:.2f}s] W&B URL: {run.url}")
        except Exception as e:
            print(
                f"[{time.time() - setup_start:.2f}s] Error: Failed to initialize wandb: {e}"
            )
            raise
    else:
        # Non-master processes don't initialize wandb
        run = None

    # Device and dtype setup
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if CUDA_AVAILABLE:
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Dtype handling
    actual_dtype = cfg.dtype
    if cfg.dtype == "bfloat16" and device != "cuda":
        actual_dtype = "float32"
    elif cfg.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        actual_dtype = "float16"

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[actual_dtype]

    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )

    # --------------------------------------------------------------------- #
    # Data loading
    # --------------------------------------------------------------------- #

    train_loader, val_loader = create_dag_structure_dataloaders(
        train_batch_size=cfg.batch_size,
        val_batch_size=cfg.batch_size,
        max_depth=cfg.dag_depth,  # All examples have exactly this depth to match the model
        seed=cfg.seed,
        english_conversion_rate=cfg.english_conversion_rate,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
    )

    # --------------------------------------------------------------------- #
    # Model creation and checkpoint loading
    # --------------------------------------------------------------------- #

    # Load checkpoint (if any) to resume training and obtain iteration/metrics
    expected_keys = [
        "model",
        "optimizer",
        "model_config",
        "iter_num",
        "best_val_loss",
    ]
    checkpoint, iter_num, best_val_loss = checkpoint_manager.handle_checkpoint_loading(
        cfg,
        device,
        model_name,
        expected_keys,
        prefer_best=cfg.save_best,
    )

    # Initialise or load the model via the checkpoint manager helper. This
    # centralises all checkpoint/model-config logic in one place.
    model, model_config = checkpoint_manager.initialize_dag_model(
        cfg, checkpoint, device, setup_start
    )

    model.to(device)

    if master_process:
        pass  # logging removed

    # Optimizer setup
    # Optimiser initialisation logged elsewhere

    # All parameters are trainable in this standalone model
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    if checkpoint is not None and "optimizer" in checkpoint:
        # Checkpoint state message removed
        optimizer.load_state_dict(checkpoint["optimizer"])
        # print(
        #     f"[{time.time() - setup_start:.2f}s] ✅ Optimizer state loaded (Adam momentum preserved)"
        # )
    else:
        # Fresh optimiser state message removed
        pass

    # Gradient scaler
    scalar_enabled = actual_dtype == "float16"
    scaler = (
        torch.cuda.amp.GradScaler(enabled=scalar_enabled)
        if TORCH_2_2_1
        else torch.amp.GradScaler("cuda", enabled=scalar_enabled)
    )

    # Model compilation
    if cfg.compile:
        # Compiling model message removed
        try:
            model = torch.compile(model, mode="reduce-overhead", disable="cudagraphs")
        except TypeError:
            model = torch.compile(model, mode="reduce-overhead")

    if ddp:
        model = DDP(model, device_ids=[int(device.split(":")[-1])])

    # --------------------------------------------------------------------- #
    # Training loop
    # --------------------------------------------------------------------- #

    t0 = time.time()
    raw_model = model.module if ddp else model
    train_start = time.time()

    # Initialize loss accumulator for logging
    loss_accum = _empty_metrics()

    # Store validation metrics for combined logging
    pending_val_metrics = None

    try:
        while iter_num <= cfg.max_iters:
            lr = get_lr(iter_num, cfg=cfg)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Evaluation
            if iter_num % cfg.eval_interval == 0 and master_process:
                # Recreate validation loader each time so that with a fixed
                # `seed` we always evaluate on the *same* sequence of
                # examples (starting from the very first batch). Without this
                # the underlying generator would advance each epoch, giving
                # different samples even though the RNG seed itself is fixed.
                if cfg.seed == -1:
                    seed = random.randint(0, 10000)
                else:
                    seed = cfg.seed

                _train_loader_unused, val_loader_eval = (
                    create_dag_structure_dataloaders(
                        train_batch_size=cfg.batch_size,
                        val_batch_size=cfg.batch_size,
                        max_depth=cfg.dag_depth,
                        seed=seed,
                        english_conversion_rate=cfg.english_conversion_rate,
                        max_digits=cfg.max_digits,
                        max_decimal_places=cfg.max_decimal_places,
                    )
                )

                model.eval()
                eval_losses = evaluate_dag_model(
                    raw_model, val_loader_eval, device, ctx, cfg, cfg.eval_iters, seed
                )
                # evaluation progress message removed

                # Log validation metrics to wandb
                if run is not None:
                    val_log_dict = {
                        "iter": iter_num,
                        "val/total_loss": eval_losses["total_loss"],
                        "val/sign_loss": eval_losses["sign_loss"],
                        "val/log_loss": eval_losses["log_loss"],
                        "val/op_loss": eval_losses["op_loss"],
                        "val/op_accuracy": eval_losses["op_accuracy"],
                        "val/full_dag_op_match": eval_losses["full_dag_op_match"],
                        "val/sign_accuracy": eval_losses["sign_accuracy"],
                        "val/log_magnitude_mape": eval_losses["log_magnitude_mape"],
                    }
                    # Store validation metrics for combined logging with training metrics
                    pending_val_metrics = val_log_dict

                is_new_best = eval_losses["total_loss"] < best_val_loss
                if is_new_best:
                    best_val_loss = eval_losses["total_loss"]

                if iter_num > 0:
                    checkpoint_data = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": model_config.__dict__,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                    }

                    if cfg.save_best and is_new_best:
                        # Save a single checkpoint for the best model
                        val_acc = eval_losses.get("op_accuracy", None)
                        checkpoint_filename = (
                            checkpoint_manager.generate_checkpoint_filename(
                                cfg.name,
                                iter_num,
                                raw_model.__class__.__name__,
                                val_acc=val_acc,
                                is_best=True,
                            )
                        )
                        # verbose checkpoint message removed
                        checkpoint_manager.save_checkpoint(
                            checkpoint_data, checkpoint_filename
                        )

                    if cfg.always_save_checkpoint or (
                        not cfg.save_best and is_new_best
                    ):
                        # Save a checkpoint with the iteration number
                        val_acc = eval_losses.get("op_accuracy", None)
                        checkpoint_filename = (
                            checkpoint_manager.generate_checkpoint_filename(
                                cfg.name,
                                iter_num,
                                raw_model.__class__.__name__,
                                val_acc=val_acc,
                            )
                        )
                        # verbose checkpoint message removed
                        checkpoint_manager.save_checkpoint(
                            checkpoint_data, checkpoint_filename
                        )

                model.train()

            # Early stopping
            if iter_num == 0 and cfg.eval_only:
                break

            # Forward and backward pass
            optimizer.zero_grad(set_to_none=True)

            # Get a batch
            texts, structures, _ = next(train_loader)

            # Move targets to device
            target_sgn = structures["initial_sgn"].to(device)
            target_log = structures["initial_log"].to(device)
            target_ops = structures["operation_probs"].to(device)

            loss_accum = _empty_metrics()

            for micro_step in range(cfg.gradient_accumulation_steps):
                if ddp:
                    model.require_backward_grad_sync = (
                        micro_step == cfg.gradient_accumulation_steps - 1
                    )

                with ctx:
                    # Tokenize texts properly using the mathematical expressions
                    input_tokens = tokenize_texts(texts, cfg.sequence_length, device)

                    # Forward pass through shallow attention DAG predictor model

                    if cfg.full_backbone and hasattr(raw_model, "dag"):
                        hidden = raw_model.forward_hidden(input_tokens)
                        pred_sgn, pred_log, pred_ops = raw_model.dag.plan_predictor(
                            hidden
                        )
                    else:
                        pred_sgn, pred_log, pred_ops = raw_model(input_tokens)

                    # Average over sequence dimension
                    pred_sgn_avg = pred_sgn.mean(dim=1)  # (B, num_nodes_pred)
                    pred_log_avg = pred_log.mean(dim=1)  # (B, num_nodes_pred)
                    pred_ops_avg = pred_ops.mean(dim=1)  # (B, depth_pred, n_ops)

                    # Ensure target and prediction tensors have compatible shapes
                    target_nodes = target_sgn.size(1)
                    pred_nodes = pred_sgn_avg.size(1)
                    target_depth = target_ops.size(1)
                    pred_depth = pred_ops_avg.size(1)

                    # Handle mismatched node dimensions
                    if pred_nodes != target_nodes:
                        if pred_nodes > target_nodes:
                            # Truncate predictions
                            pred_sgn_avg = pred_sgn_avg[:, :target_nodes]
                            pred_log_avg = pred_log_avg[:, :target_nodes]
                        else:
                            # Pad predictions with zeros
                            pad_nodes = target_nodes - pred_nodes
                            pred_sgn_avg = F.pad(pred_sgn_avg, (0, pad_nodes))
                            pred_log_avg = F.pad(pred_log_avg, (0, pad_nodes))

                    if pred_depth != target_depth:
                        if pred_depth > target_depth:
                            # Truncate predictions
                            pred_ops_avg = pred_ops_avg[:, :target_depth]
                        else:
                            # Pad predictions with zeros
                            pad_depth = target_depth - pred_depth
                            pred_ops_avg = F.pad(pred_ops_avg, (0, 0, 0, pad_depth))

                    # Add sequence dimension
                    pred_sgn_seq = pred_sgn_avg.unsqueeze(1)
                    pred_log_seq = pred_log_avg.unsqueeze(1)
                    pred_ops_seq = pred_ops_avg.unsqueeze(1)

                    # Add sequence dimension to targets
                    target_sgn_seq = target_sgn.unsqueeze(1)
                    target_log_seq = target_log.unsqueeze(1)
                    target_ops_seq = target_ops.unsqueeze(1)

                    # Compute loss
                    losses = compute_dag_structure_loss(
                        pred_sgn_seq,
                        pred_log_seq,
                        pred_ops_seq,
                        target_sgn_seq,
                        target_log_seq,
                        target_ops_seq,
                        cfg,
                    )

                    loss = losses["total_loss"] / cfg.gradient_accumulation_steps

                    # Accumulate losses for logging
                    for key, value in losses.items():
                        loss_accum[key] += (
                            value.item() / cfg.gradient_accumulation_steps
                        )

                    # ------------------------------------------------------------------ #
                    # Compute and accumulate training metrics (same as evaluation)
                    # ------------------------------------------------------------------ #
                    pred_ops_idx = pred_ops_avg.argmax(dim=-1)  # (B, depth)
                    target_ops_idx = target_ops.argmax(dim=-1)
                    op_correct = pred_ops_idx.eq(target_ops_idx)
                    loss_accum["op_accuracy"] += (
                        op_correct.float().mean().item()
                        / cfg.gradient_accumulation_steps
                    )
                    loss_accum["full_dag_op_match"] += (
                        op_correct.all(dim=-1).float().mean().item()
                        / cfg.gradient_accumulation_steps
                    )

                    sign_correct = torch.sign(pred_sgn_avg).eq(torch.sign(target_sgn))
                    loss_accum["sign_accuracy"] += (
                        sign_correct.float().mean().item()
                        / cfg.gradient_accumulation_steps
                    )

                    pred_mag = pred_log_avg.exp()
                    target_mag = target_log.exp()
                    log_mape = (
                        (pred_mag - target_mag).abs() / target_mag.clamp_min(1e-8)
                    ).mean()
                    loss_accum["log_magnitude_mape"] += (
                        log_mape.item() / cfg.gradient_accumulation_steps
                    )

                # Backward pass
                scaler.scale(loss).backward()

            # Optimization step
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # Timing and logging
            dt = time.time() - t0
            t0 = time.time()

            if iter_num % cfg.log_interval == 0 and master_process:
                # Build dynamic log message for main losses
                log_msg = (
                    f"iter {iter_num}: loss {loss_accum['total_loss']:.4f}, "
                    f"sign {loss_accum['sign_loss']:.4f}, "
                    f"log {loss_accum['log_loss']:.4f}, "
                    f"op {loss_accum['op_loss']:.4f}"
                    f", op_acc {loss_accum['op_accuracy']:.4f}, "
                    f"full_op_match {loss_accum['full_dag_op_match']:.4f}, "
                    f"sign_acc {loss_accum['sign_accuracy']:.4f}, "
                    f"log_mape {loss_accum['log_magnitude_mape']:.4f}"
                )

                # Add internal losses if present
                internal_loss_keys = [
                    k
                    for k in loss_accum.keys()
                    if k.endswith("_loss")
                    and k not in ["total_loss", "sign_loss", "log_loss", "op_loss"]
                ]
                if internal_loss_keys:
                    internal_msg = ", ".join(
                        [
                            f"{k.replace('_loss', '')} {loss_accum[k]:.6f}"
                            for k in internal_loss_keys
                        ]
                    )
                    log_msg += f", {internal_msg}"

                log_msg += f", time {dt*1000:.2f}ms"
                # realtime training log emitted via wandb

                # Log to wandb
                if run is not None:
                    # Capture the *actual* learning rate currently in the optimizer. This avoids any
                    # issues where the local `lr` variable drifts from the value that is ultimately
                    # used for the step (e.g. if the optimizer or scheduler modifies it).
                    current_lr = optimizer.param_groups[0]["lr"]
                    log_dict = {
                        "iter": iter_num,
                        "train/total_loss": loss_accum["total_loss"],
                        "train/sign_loss": loss_accum["sign_loss"],
                        "train/log_loss": loss_accum["log_loss"],
                        "train/op_loss": loss_accum["op_loss"],
                        "lr": current_lr,
                        "train/op_accuracy": loss_accum["op_accuracy"],
                        "train/full_dag_op_match": loss_accum["full_dag_op_match"],
                        "train/sign_accuracy": loss_accum["sign_accuracy"],
                        "train/log_magnitude_mape": loss_accum["log_magnitude_mape"],
                    }

                    # Add internal losses to wandb logging
                    for key, value in loss_accum.items():
                        if (
                            key.endswith("_loss")
                            and key not in log_dict
                            and key != "total_loss"
                        ):
                            log_dict[f"train/{key}"] = value

                    # Combine pending validation metrics if they exist
                    if pending_val_metrics is not None:
                        log_dict.update(pending_val_metrics)
                        pending_val_metrics = None  # Clear after using

                    wandb.log(log_dict, step=iter_num, commit=True)

            iter_num += 1

            # Log any pending validation metrics that weren't combined with training metrics
            if pending_val_metrics is not None and run is not None:
                wandb.log(
                    pending_val_metrics, step=pending_val_metrics["iter"], commit=True
                )
                pending_val_metrics = None

    except Exception as e:
        print("Fatal error in training loop – see traceback above")
        import traceback

        traceback.print_exc()
    finally:
        # Train loop duration print removed
        # Cleanup
        if ddp:
            destroy_process_group()
        if run is not None:
            run.finish()

        # Stop RunPod instance if we're running on RunPod and keep-alive is not enabled
        if os.getenv("RUNPOD_POD_ID") and not getattr(cfg, "keep_alive", False):
            runpod_service.stop_runpod()


def main() -> None:
    """Main entry point."""
    check_python_version()

    parser = parse_args()
    args, overrides = parser.parse_known_args()

    # Handle --use-runpod before any other processing
    if args.use_runpod:
        # Basic config setup for runpod args
        cfg = DAGTrainConfig()
        try:
            update_config(cfg, load_config_file(args.config))
        except Exception:
            # If config loading fails, continue with defaults for runpod launch
            pass

        apply_overrides(cfg, overrides)

        if args.dag_depth is not None:
            cfg.dag_depth = args.dag_depth
        if args.keep_alive:
            cfg.keep_alive = args.keep_alive
        if args.note:
            cfg.note = args.note

        remote_args = " ".join([args.config] + overrides)
        if args.dag_depth is not None:
            remote_args += f" --dag-depth={args.dag_depth}"
        runpod_service.start_cloud_training(
            remote_args,
            args.gpu_type or runpod_service.DEFAULT_GPU_TYPE,
            api_key=os.getenv("RUNPOD_API_KEY"),
            script_name="train_predictor.py",
            note=args.note,
            keep_alive=args.keep_alive,
        )
        return

    # For local training, wrap everything in error handling for proper runpod termination
    cfg = None
    try:
        cfg = DAGTrainConfig()
        update_config(cfg, load_config_file(args.config))
        apply_overrides(cfg, overrides)

        if args.dag_depth is not None:
            cfg.dag_depth = args.dag_depth
        if args.keep_alive:
            cfg.keep_alive = args.keep_alive
        if args.note:
            cfg.note = args.note

        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        if not os.getenv("WANDB_API_KEY"):
            parser.error("WANDB_API_KEY is required")

        Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        train_predictor(cfg, wandb_run_id=args.wandb_run_id)

    except Exception as e:
        print(f"Fatal error in predictor training: {e}")
        import traceback

        traceback.print_exc()

        # Stop RunPod instance on error if we're running on RunPod
        if os.getenv("RUNPOD_POD_ID") and (
            cfg is None or not getattr(cfg, "keep_alive", False)
        ):
            pass

        # Re-raise the exception to ensure proper exit code
        raise


if __name__ == "__main__":
    main()
