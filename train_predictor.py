"""DAG predictor pre-training entry-point (lean version)."""

from __future__ import annotations

import argparse
import os
import random
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import runpod_service
import training_utils as _tu
import wandb
from checkpoint_manager import CheckpointManager
from data.dagset.streaming import create_dag_structure_dataloaders
from evaluate import evaluate_dag_model
from predictor_config import DAGTrainConfig
from predictor_utils import (
    compute_dag_loss,
    tokenize_texts,
)
from python_version_check import check_python_version

# Centralised runtime constants & env tweaks
from runtime import CHECKPOINT_DIR, CUDA_AVAILABLE, TORCH_2_2_1
from training_utils import (
    apply_overrides,
    generate_run_name,
    get_lr,
    load_config_file,
    log_config_values,
    log_git_commit_info,
    parse_args,
    setup_distributed,
    update_config,
)


# Local helper utilities
def _empty_metrics() -> dict[str, float]:
    """Return a zero-filled metrics accumulator used during training."""
    return {
        "total_loss": 0.0,
        "digit_loss": 0.0,
        "V_mag_loss": 0.0,
        "V_sign_loss": 0.0,
        "O_loss": 0.0,
        "G_loss": 0.0,
        "exec_loss": 0.0,
    }


def parse_args() -> argparse.ArgumentParser:  # type: ignore[override]
    """Light wrapper that makes the *config* positional argument optional with a
    sensible default, while re-using the common parser from ``training_utils``.
    """

    parser = _tu.parse_args()

    # Make ``config`` optional and set the default for predictor runs.
    for action in parser._actions:
        if action.dest == "config":
            action.nargs = "?"
            action.default = "config/train_predictor_default.py"
            break

    return parser


def train_predictor(cfg: DAGTrainConfig, wandb_run_id: str | None = None) -> None:
    """Run DAG predictor training loop."""
    # Setup
    setup_start = time.time()
    # (prints removed to keep script minimal)
    # W&B URL debug print removed

    # DDP setup
    ddp_start = time.time()
    ddp, master_process, ddp_world_size, device = setup_distributed(cfg)
    print(
        f"[{time.time() - setup_start:.2f}s] DDP setup completed in {time.time() - ddp_start:.2f}s"
    )

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager("dag")
    # Will hold the sanitised W&B run name once available.
    safe_run_name: str = "default_run"

    # Clean previous checkpoints
    if master_process and cfg.clear_previous_checkpoints:
        checkpoint_manager.clean_previous_checkpoints(cfg.name)

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
            # Create a dedicated checkpoint sub-directory for this run.
            safe_run_name = "".join(
                c for c in run.name if c.isalnum() or c in ("-", "_")
            )
            (checkpoint_manager.checkpoint_dir / safe_run_name).mkdir(
                parents=True, exist_ok=True
            )

            # Log git commit info and config now that W&B is initialized
            log_git_commit_info()
            log_config_values(cfg)
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

    ctx = nullcontext()

    train_loader = None
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
        device=device,
        expected_keys=expected_keys,
        prefer_best=cfg.save_best,
    )

    # Initialise or load the model via the checkpoint manager helper. This
    # centralises all checkpoint/model-config logic in one place.
    model, model_config = checkpoint_manager.initialize_dag_model(
        cfg, checkpoint, setup_start_time=setup_start
    )

    model.to(device)

    if master_process:
        pass

    # Optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    if checkpoint is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
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

    raw_model = model.module if ddp else model
    train_loader, val_loader = create_dag_structure_dataloaders(
        train_batch_size=cfg.batch_size,
        val_batch_size=cfg.batch_size,
        max_depth=cfg.dag_depth,
        seed=cfg.seed,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
        block_size=cfg.block_size,
    )

    # --------------------------------------------------------------------- #
    # Training loop
    # --------------------------------------------------------------------- #

    t0 = time.time()
    raw_model = model.module if ddp else model

    # Initialize loss accumulator for logging
    loss_accum = _empty_metrics()

    # Store validation metrics for combined logging
    pending_val_metrics = None

    try:
        while iter_num <= cfg.max_iters:
            # Apply learning rate schedule (single parameter group for DAG model)
            lr = get_lr(iter_num, cfg=cfg)
            optimizer.param_groups[0]["lr"] = lr

            # Evaluation
            if iter_num % cfg.eval_interval == 0 and master_process:
                if cfg.seed == -1:
                    seed = random.randint(0, 10000)
                else:
                    seed = cfg.seed

                model.eval()
                eval_start_time = time.time()

                eval_losses = evaluate_dag_model(
                    raw_model, val_loader, device, ctx, cfg, cfg.eval_iters, seed
                )
                eval_time_ms = (time.time() - eval_start_time) * 1000 / cfg.eval_iters

                if master_process:
                    eval_msg = (
                        f"[val] iter {iter_num}: total_loss {eval_losses['total_loss']:.4f}, "
                        f"digit_loss {eval_losses.get('digit_loss', 0.0):.4f}, "
                        f"digit_acc {eval_losses.get('digit_accuracy', 0.0):.1%}, "
                        f"V_mag_loss {eval_losses.get('V_mag_loss', 0.0):.4f}, "
                        f"V_sign_loss {eval_losses.get('V_sign_loss', 0.0):.4f}, "
                        f"sign_acc {eval_losses.get('sign_accuracy', 0.0):.1%}, "
                        f"O_loss {eval_losses.get('O_loss', 0.0):.4f}, "
                        f"op_acc {eval_losses.get('op_accuracy', 0.0):.1%}, "
                        f"G_loss {eval_losses.get('G_loss', 0.0):.4f}, "
                        f"gate_acc {eval_losses.get('gate_accuracy', 0.0):.1%}, "
                        f"exec_loss {eval_losses.get('exec_loss', 0.0):.4f}, "
                        f"valid_rate {eval_losses.get('expression_valid_rate', 0.0):.1%}, "
                        f"time {eval_time_ms:.1f}ms"
                    )
                    print(eval_msg)

                # Log validation metrics to wandb
                if run is not None and eval_losses is not None:
                    val_log_dict = {
                        "iter": iter_num,
                        "val/total_loss": eval_losses["total_loss"],
                        "val/digit_loss": eval_losses.get("digit_loss", 0.0),
                        "val/digit_accuracy": eval_losses.get("digit_accuracy", 0.0),
                        "val/V_sign_loss": eval_losses.get("V_sign_loss", 0.0),
                        "val/sign_accuracy": eval_losses.get("sign_accuracy", 0.0),
                        "val/O_loss": eval_losses.get("O_loss", 0.0),
                        "val/op_accuracy": eval_losses.get("op_accuracy", 0.0),
                        "val/G_loss": eval_losses.get("G_loss", 0.0),
                        "val/gate_accuracy": eval_losses.get("gate_accuracy", 0.0),
                        "val/exec_loss": eval_losses.get("exec_loss", 0.0),
                        "val/expression_valid_rate": eval_losses.get(
                            "expression_valid_rate", 0.0
                        ),
                        "val/time_per_iter_ms": eval_time_ms,
                    }
                    # Store validation metrics for combined logging with training metrics
                    pending_val_metrics = val_log_dict
                else:
                    pending_val_metrics = None

                is_new_best = (
                    eval_losses is not None
                    and eval_losses["total_loss"] < best_val_loss
                )
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

                    # Default behavior: only save on new best validation loss
                    # Override: always save if always_save_checkpoint is enabled
                    if cfg.always_save_checkpoint or is_new_best:
                        val_acc = eval_losses.get("op_accuracy", None)

                        # Always use unique checkpoint names (no backwards compatibility)
                        rel_name = checkpoint_manager.generate_checkpoint_filename(
                            cfg.name,
                            iter_num,
                            val_acc=val_acc,
                            is_best=is_new_best,
                        )
                        checkpoint_filename = f"{safe_run_name}/{rel_name}"

                        if is_new_best:
                            print(
                                f"[{time.time() - setup_start:.2f}s] 🎯 BEST validation loss: {best_val_loss:.6f} "
                                f"(iter {iter_num}) - saving checkpoint: {checkpoint_filename}"
                            )

                        checkpoint_manager.save_checkpoint(
                            checkpoint_data, checkpoint_filename
                        )

                model.train()

            # Eval once
            if iter_num == 0 and cfg.eval_once:
                print(
                    "🎯 EVAL_ONCE: Evaluation completed. Exiting after single evaluation run."
                )
                break

            # Forward and backward pass
            optimizer.zero_grad(set_to_none=True)

            texts, target_tensors, valid_mask = next(train_loader)
            for seq_targets in target_tensors:
                for target_dict in seq_targets:
                    for key, tensor in target_dict.items():
                        if isinstance(tensor, torch.Tensor):
                            target_dict[key] = tensor.to(device)

            valid_mask = valid_mask.to(device)

            loss_accum = _empty_metrics()

            for micro_step in range(cfg.gradient_accumulation_steps):
                if ddp:
                    model.require_backward_grad_sync = (
                        micro_step == cfg.gradient_accumulation_steps - 1
                    )

                with ctx:
                    # Tokenize texts properly using the mathematical expressions
                    input_tokens = tokenize_texts(texts, cfg.block_size, device)

                    hidden_states = raw_model.forward_hidden(
                        input_tokens
                    )  # (B, T, n_embd)

                    assert (
                        raw_model.dag_predictor is not None
                    ), "DAG models should always have dag_predictor"
                    pred_digit_logits, pred_V_sign, pred_O, pred_G = (
                        raw_model.dag_predictor(hidden_states)
                    )

                    dag_executor = getattr(raw_model, "dag_executor", None)
                    losses = compute_dag_loss(
                        pred_digit_logits,
                        pred_V_sign,
                        pred_O,
                        pred_G,
                        target_tensors,
                        valid_mask,
                        dag_executor=dag_executor,
                        cfg=cfg,
                    )
                    loss = losses["total_loss"] / cfg.gradient_accumulation_steps

                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item() / cfg.gradient_accumulation_steps
                    loss_accum[key] = value

                # ------------------------------------------------------------------ #
                # Add per-token training metrics (OUTSIDE the loss accumulation loop)
                # ------------------------------------------------------------------ #
                # Calculate expression-level valid rate (excluding padding)
                batch_size, seq_len = valid_mask.shape
                expression_valid_tokens = 0
                expression_total_tokens = 0

                for b in range(batch_size):
                    # Find the last non-padding token (expression length)
                    sequence_mask = valid_mask[b]  # (seq_len,)

                    # Find expression length by looking for the transition to padding
                    # Padding is always False values at the end
                    expression_length = seq_len
                    for i in range(seq_len - 1, -1, -1):
                        if (
                            i == 0
                            or sequence_mask[i]
                            or (i < seq_len - 1 and sequence_mask[i + 1])
                        ):
                            expression_length = i + 1
                            break

                    # Count valid tokens within the expression (before padding)
                    expression_tokens = sequence_mask[:expression_length]
                    expression_valid_tokens += expression_tokens.sum().item()
                    expression_total_tokens += expression_length

                if expression_total_tokens > 0:
                    loss_accum["expression_valid_rate"] = loss_accum.get(
                        "expression_valid_rate", 0
                    ) + (
                        (expression_valid_tokens / expression_total_tokens)
                        / cfg.gradient_accumulation_steps
                    )

                # Backward pass
                # Skip backward if loss doesn't require gradients (e.g., when all losses disabled)
                if loss.requires_grad:
                    scaler.scale(loss).backward()
                else:
                    if master_process:
                        print(
                            f"[{time.time() - setup_start:.2f}s] Warning: Loss doesn't require gradients (total_loss={loss.item():.6f}), skipping backward pass"
                        )

                # Check for NaN/Inf gradients in training (conditional on config)
                if cfg.check_nans:
                    for n, p in model.named_parameters():
                        if p.grad is not None and (
                            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                        ):
                            print(
                                f"[GRAD NAN] {n}  →  min={p.grad.min():.3e}  max={p.grad.max():.3e}"
                            )
                            break

            # Optimization step
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Timing and logging
            dt = time.time() - t0
            t0 = time.time()

            if iter_num % cfg.log_interval == 0 and master_process:
                # Display losses for DAG model
                digit_loss_val = loss_accum.get("digit_loss", 0.0)
                digit_accuracy_val = loss_accum.get("digit_accuracy", 0.0)
                V_mag_loss_val = loss_accum.get("V_mag_loss", 0.0)
                V_sign_loss_val = loss_accum.get("V_sign_loss", 0.0)
                O_loss_val = loss_accum.get("O_loss", 0.0)
                G_loss_val = loss_accum.get("G_loss", 0.0)
                exec_loss_val = loss_accum.get("exec_loss", 0.0)
                expression_valid_rate_val = loss_accum.get("expression_valid_rate", 0.0)
                sign_accuracy_val = loss_accum.get("sign_accuracy", 0.0)
                op_accuracy_val = loss_accum.get("op_accuracy", 0.0)
                gate_accuracy_val = loss_accum.get("gate_accuracy", 0.0)

                log_msg = (
                    f"iter {iter_num}: loss {loss_accum['total_loss']:.4f}, "
                    f"digit {digit_loss_val:.4f}, "
                    f"digit_acc {digit_accuracy_val:.1%}, "
                    f"V_mag {V_mag_loss_val:.4f}, "
                    f"V_sign {V_sign_loss_val:.4f}, "
                    f"sign_acc {sign_accuracy_val:.1%}, "
                    f"O {O_loss_val:.4f}, "
                    f"op_acc {op_accuracy_val:.1%}, "
                    f"G {G_loss_val:.4f}, "
                    f"gate_acc {gate_accuracy_val:.1%}, "
                    f"exec {exec_loss_val:.4f}, "
                    f"valid_rate {expression_valid_rate_val:.1%}, "
                    f"time {dt*1000:.2f}ms"
                )

                # Realtime training log emitted via wandb
                if master_process:
                    print(log_msg)

                # Log to wandb
                if run is not None:
                    # Capture the current learning rate from optimizer
                    current_lr = optimizer.param_groups[0]["lr"]

                    log_dict = {
                        "iter": iter_num,
                        "train/total_loss": loss_accum["total_loss"],
                        "train/digit_loss": loss_accum.get("digit_loss", 0.0),
                        "train/digit_accuracy": loss_accum.get("digit_accuracy", 0.0),
                        "train/V_sign_loss": loss_accum.get("V_sign_loss", 0.0),
                        "train/sign_accuracy": loss_accum.get("sign_accuracy", 0.0),
                        "train/O_loss": loss_accum.get("O_loss", 0.0),
                        "train/op_accuracy": loss_accum.get("op_accuracy", 0.0),
                        "train/G_loss": loss_accum.get("G_loss", 0.0),
                        "train/gate_accuracy": loss_accum.get("gate_accuracy", 0.0),
                        "train/exec_loss": loss_accum.get("exec_loss", 0.0),
                        "train/expression_valid_rate": loss_accum.get(
                            "expression_valid_rate", 0.0
                        ),
                        "lr": current_lr,
                        "train/time_per_iter_ms": dt * 1000,
                    }

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
                        pending_val_metrics = None

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
        traceback.print_exc()
        raise

    finally:
        # Cleanup
        if ddp:
            destroy_process_group()
        if run is not None:
            run.finish()

        wandb.finish()
        if not getattr(cfg, "keep_alive", False):
            runpod_service.stop_runpod()
        wandb.finish()


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
        except Exception as e:
            print("Failed to load config file, continuing with defaults")
            raise  # Re-raise the caught exception

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

    # Wrap everything in error handling for proper runpod termination
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
        traceback.print_exc()

        wandb.finish()
        if not getattr(cfg, "keep_alive", False):
            runpod_service.stop_runpod()
        wandb.finish()

        raise e


if __name__ == "__main__":
    main()
