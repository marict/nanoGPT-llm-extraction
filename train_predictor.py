"""DAG predictor pre-training entry-point (lean version)."""

from __future__ import annotations

import argparse
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import runpod_service
import wandb
from checkpoint_manager import CheckpointManager
from data.dagset.streaming import create_dag_structure_dataloaders
from models.dag_model import GPT, OP_NAMES
from models.predictor_only_model import PredictorOnlyModel
from predictor_config import DAGTrainConfig
from predictor_utils import (
    compute_dag_structure_loss,
    compute_gradient_cosines,
    evaluate_dag_model,
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
        "sign_loss": 0.0,
        "digit_loss": 0.0,
        "op_loss": 0.0,
        "value_loss": 0.0,  # Robust loss on initial values
        "exec_loss": 0.0,  # Robust loss on final execution values
        "op_accuracy": 0.0,
        "full_dag_op_match": 0.0,
        "sign_accuracy": 0.0,
        # Gradient cosine metrics are added dynamically when computed
        "grad_cosine_sign_loss": 0.0,
        "grad_cosine_digit_loss": 0.0,
        "grad_cosine_op_loss": 0.0,
        "grad_cosine_value_loss": 0.0,
        "grad_cosine_exec_loss": 0.0,
    }


# (TORCH_2_2_1, CUDA_AVAILABLE, CHECKPOINT_DIR) come from runtime


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

    # Determine model name and create appropriate config
    # Model configuration will be created (or reconstructed) later by the
    # checkpoint manager, so we only need the model name here.
    model_name = GPT.__name__ if cfg.full_backbone else PredictorOnlyModel.__name__

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager("dag")
    # Will hold the sanitised W&B run name once available.
    safe_run_name: str = "default_run"

    # Clean previous checkpoints unless we're running with overwrite_previous
    if (
        master_process
        and cfg.clear_previous_checkpoints
        and not getattr(cfg, "overwrite_previous", False)
    ):
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

    # --------------------------------------------------------------------- #
    # Model-aware Data loading (will be created after model is initialized)
    # --------------------------------------------------------------------- #
    train_loader = val_loader = None  # placeholders; real loaders created later

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

    # Separate uncertainty_params for higher learning rate
    uncertainty_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "uncertainty_params" in name:
                uncertainty_params.append(param)
            else:
                other_params.append(param)

    # Create optimizer groups with different learning rates
    # uncertainty_params get 2 orders of magnitude higher learning rate for faster adaptation
    uncertainty_params_lr = cfg.learning_rate * 100  # 2 orders of magnitude jump

    optim_groups = [
        {"params": other_params, "lr": cfg.learning_rate},
        {"params": uncertainty_params, "lr": uncertainty_params_lr},
    ]

    print(f"Optimizer groups created:")
    print(f"  Main parameters: {len(other_params)} params with lr={cfg.learning_rate}")
    print(
        f"  uncertainty_params: {len(uncertainty_params)} params with lr={uncertainty_params_lr}"
    )

    optimizer = torch.optim.AdamW(
        optim_groups,
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

    # --------------------------------------------------------------------- #
    # Create DataLoaders that respect the model's available operations
    # --------------------------------------------------------------------- #

    raw_model = model.module if ddp else model
    # Restrict dataset to the subset of ops requested by the config (defaults to all ops)
    allowed_operations = getattr(cfg, "op_names", OP_NAMES)

    train_loader, _ = create_dag_structure_dataloaders(
        train_batch_size=cfg.batch_size,
        val_batch_size=cfg.batch_size,
        max_depth=cfg.dag_depth,
        seed=cfg.seed,
        english_conversion_probability=cfg.english_conversion_probability,
        integer_no_decimal_probability=cfg.integer_no_decimal_probability,
        expression_simplification_probability=cfg.expression_simplification_probability,
        expression_expansion_probability=cfg.expression_expansion_probability,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
        base=cfg.base,
        allowed_operations=allowed_operations,
        printing_style_probs=cfg.printing_style_probs,
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

                _, val_loader_eval = create_dag_structure_dataloaders(
                    train_batch_size=cfg.batch_size,
                    val_batch_size=cfg.batch_size,
                    max_depth=cfg.dag_depth,
                    # Large prime to avoid overlap with training and repeated values on each eval.
                    seed=seed + iter_num * 97919,
                    english_conversion_probability=cfg.english_conversion_probability,
                    integer_no_decimal_probability=cfg.integer_no_decimal_probability,
                    expression_simplification_probability=cfg.expression_simplification_probability,
                    expression_expansion_probability=cfg.expression_expansion_probability,
                    max_digits=cfg.max_digits,
                    max_decimal_places=cfg.max_decimal_places,
                    base=cfg.base,
                    allowed_operations=allowed_operations,
                    printing_style_probs=cfg.printing_style_probs,
                )

                model.eval()
                eval_start_time = time.time()
                eval_losses = evaluate_dag_model(
                    raw_model, val_loader_eval, device, ctx, cfg, cfg.eval_iters, seed
                )
                eval_time_ms = (time.time() - eval_start_time) * 1000 / cfg.eval_iters
                if master_process:
                    # Format validation weights (get current learned weights from model)
                    # Use consistent access pattern via dag_predictor property
                    uncertainty_weights = raw_model.dag_predictor.uncertainty_params
                    uncertainty_weights = torch.exp(-uncertainty_weights).detach()
                    uncertainty_weights_str = f"[{uncertainty_weights[0]:.3f},{uncertainty_weights[1]:.3f},{uncertainty_weights[2]:.3f},{uncertainty_weights[3]:.3f},{uncertainty_weights[4]:.3f},{uncertainty_weights[5]:.3f}]"

                    eval_msg = (
                        f"[val] iter {iter_num}: loss {eval_losses['total_loss']:.4f}, "
                        f"sign {eval_losses['sign_loss']:.4f}, "
                        f"digit {eval_losses['digit_loss']:.4f}, "
                        f"op {eval_losses['op_loss']:.4f}, "
                        f"value {eval_losses['value_loss']:.4f}, "
                        f"exec {eval_losses['exec_loss']:.4f}, "
                        f"stats {eval_losses['stats_loss']:.4f}, "
                        f"uncertainty_weights {uncertainty_weights_str}, "
                        f"op_acc {eval_losses['op_accuracy']:.4f}, "
                        f"full_op_match {eval_losses['full_dag_op_match']:.4f}, "
                        f"sign_acc {eval_losses['sign_accuracy']:.4f}, "
                        f"executed_mse {eval_losses['executed_mse']:.4f}"
                    )
                    print(eval_msg)

                # Log validation metrics to wandb
                if run is not None:
                    val_log_dict = {
                        "iter": iter_num,
                        "val/total_loss": eval_losses["total_loss"],
                        "val/sign_loss": eval_losses["sign_loss"],
                        "val/digit_loss": eval_losses["digit_loss"],
                        "val/op_loss": eval_losses["op_loss"],
                        "val/value_loss": eval_losses["value_loss"],
                        "val/exec_loss": eval_losses["exec_loss"],
                        "val/stats_loss": eval_losses["stats_loss"],
                        "val/op_accuracy": eval_losses["op_accuracy"],
                        "val/full_dag_op_match": eval_losses["full_dag_op_match"],
                        "val/sign_accuracy": eval_losses["sign_accuracy"],
                        "val/executed_mse": eval_losses["executed_mse"],
                        "val/initial_values_mse": eval_losses["initial_values_mse"],
                        "val/time_per_iter_ms": eval_time_ms,
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

                    # Default behavior: only save on new best validation loss
                    # Override: always save if always_save_checkpoint is enabled
                    if cfg.always_save_checkpoint or is_new_best:
                        val_acc = eval_losses.get("op_accuracy", None)

                        if getattr(cfg, "overwrite_previous", False):
                            checkpoint_filename = f"{safe_run_name}/ckpt_{cfg.name}.pt"
                        else:
                            rel_name = checkpoint_manager.generate_checkpoint_filename(
                                cfg.name,
                                iter_num,
                                val_acc=val_acc,
                                is_best=is_new_best,
                            )
                            checkpoint_filename = f"{safe_run_name}/{rel_name}"

                        if is_new_best:
                            print(
                                f"[{time.time() - setup_start:.2f}s] 🎯 NEW BEST validation loss: {best_val_loss:.6f} "
                                f"(iter {iter_num}) - saving checkpoint: {checkpoint_filename}"
                            )

                        checkpoint_manager.save_checkpoint(
                            checkpoint_data, checkpoint_filename
                        )

                model.train()

            # Early stopping
            if iter_num == 0 and cfg.eval_once:
                print(
                    "🎯 EVAL_ONCE: Evaluation completed. Exiting after single evaluation run."
                )
                break

            # Forward and backward pass
            optimizer.zero_grad(set_to_none=True)

            # Get a batch
            texts, structures, _ = next(train_loader)

            # Move targets to device
            target_sgn = structures["target_initial_sgn"].to(device)
            target_digits = structures["target_initial_digits"].to(device)
            target_ops = structures["target_operation_probs"].to(device)
            target_initial_values = structures["target_initial_values"].to(device)
            target_final_exec = structures["target_final_exec"].to(device)
            target_statistics = {
                "initial": structures["target_initial_stats"].to(device),
                "intermediate": structures["target_intermediate_stats"].to(device),
                "final": structures["target_final_stats"].to(device),
            }

            loss_accum = _empty_metrics()

            for micro_step in range(cfg.gradient_accumulation_steps):
                if ddp:
                    model.require_backward_grad_sync = (
                        micro_step == cfg.gradient_accumulation_steps - 1
                    )

                with ctx:
                    # Tokenize texts properly using the mathematical expressions
                    input_tokens = tokenize_texts(texts, cfg.block_size, device)

                    # Forward pass through shallow attention DAG predictor model

                    if cfg.full_backbone and hasattr(raw_model, "dag"):
                        hidden = raw_model.forward_hidden(input_tokens)
                        pred_sgn, _, pred_ops, pred_statistics = (
                            raw_model.dag.plan_predictor(hidden)
                        )
                    else:
                        pred_sgn, _, pred_ops, pred_statistics = raw_model(input_tokens)

                    # Average over sequence dimension
                    pred_sgn_avg = pred_sgn.mean(dim=1)  # (B, num_nodes_pred)
                    if hasattr(raw_model, "dag"):  # GPT backbone with DAG
                        last_digit_logits = (
                            raw_model.dag.plan_predictor.last_digit_logits
                            if hasattr(
                                raw_model.dag.plan_predictor, "last_digit_logits"
                            )
                            else None
                        )
                        last_operation_logits = (
                            raw_model.dag.plan_predictor.last_operation_logits
                            if hasattr(
                                raw_model.dag.plan_predictor, "last_operation_logits"
                            )
                            else None
                        )
                        last_sign_logits = (
                            raw_model.dag.plan_predictor.last_sign_logits
                            if hasattr(raw_model.dag.plan_predictor, "last_sign_logits")
                            else None
                        )
                    else:  # PredictorOnlyModel
                        last_digit_logits = (
                            raw_model.dag_predictor.last_digit_logits
                            if hasattr(raw_model.dag_predictor, "last_digit_logits")
                            else None
                        )
                        last_operation_logits = (
                            raw_model.dag_predictor.last_operation_logits
                            if hasattr(raw_model.dag_predictor, "last_operation_logits")
                            else None
                        )
                        last_sign_logits = (
                            raw_model.dag_predictor.last_sign_logits
                            if hasattr(raw_model.dag_predictor, "last_sign_logits")
                            else None
                        )
                    if last_digit_logits is None:
                        raise RuntimeError(
                            "last_digit_logits not set in plan_predictor"
                        )
                    if last_sign_logits is None:
                        raise RuntimeError("last_sign_logits not set in plan_predictor")
                    if last_operation_logits is None:
                        raise RuntimeError(
                            "last_operation_logits not set in plan_predictor"
                        )

                    digit_logits_avg = last_digit_logits.mean(dim=1)  # (B,N,D,10)
                    operation_logits_avg = last_operation_logits.mean(
                        dim=1
                    )  # (B,depth,n_ops)
                    sign_logits_avg = last_sign_logits.mean(dim=1)  # (B,N)
                    pred_ops_avg = pred_ops.mean(dim=1)  # (B, depth_pred, n_ops)

                    # Ensure target and prediction tensors have compatible shapes
                    target_nodes = target_sgn.size(1)
                    pred_nodes = pred_sgn_avg.size(1)
                    target_depth = target_ops.size(1)
                    pred_depth = pred_ops_avg.size(1)

                    # Verify dimension consistency - fail fast on mismatches
                    if pred_nodes != target_nodes:
                        raise ValueError(
                            f"Model node dimension mismatch: predicted {pred_nodes} nodes, "
                            f"but target has {target_nodes} nodes. This indicates a configuration "
                            f"inconsistency between model dag_depth and data max_dag_depth. "
                            f"Check that cfg.dag_depth == cfg.max_dag_depth."
                        )

                    if pred_depth != target_depth:
                        raise ValueError(
                            f"Model depth dimension mismatch: predicted depth {pred_depth}, "
                            f"but target has depth {target_depth}. This indicates a configuration "
                            f"inconsistency between model dag_depth and data max_dag_depth. "
                            f"Check that cfg.dag_depth == cfg.max_dag_depth."
                        )

                    # Add sequence dimension
                    sign_logits_seq = sign_logits_avg.unsqueeze(1)
                    digit_logits_seq = digit_logits_avg.unsqueeze(1)
                    operation_logits_seq = operation_logits_avg.unsqueeze(1)

                    # Add sequence dimension to targets
                    target_sgn_seq = target_sgn.unsqueeze(1)
                    target_digits_seq = target_digits.unsqueeze(1)
                    target_ops_seq = target_ops.unsqueeze(1)

                    # Add sequence dimension to new target values
                    target_initial_values_seq = target_initial_values.unsqueeze(1)
                    target_final_exec_seq = target_final_exec.unsqueeze(1)

                    # Get sequence length from predictions
                    T = pred_statistics["initial"].shape[1]

                    # Broadcast statistics targets to match per-token predictions (B, T, num_stats)
                    # Same statistics apply to all tokens since they relate to the same expression
                    target_statistics_seq = {
                        "initial": target_statistics["initial"]
                        .unsqueeze(1)
                        .expand(-1, T, -1),
                        "intermediate": target_statistics["intermediate"]
                        .unsqueeze(1)
                        .expand(-1, T, -1),
                        "final": target_statistics["final"]
                        .unsqueeze(1)
                        .expand(-1, T, -1),
                    }

                    # Use consistent access pattern via dag_predictor property
                    uncertainty_params = raw_model.dag_predictor.uncertainty_params

                    # Compute losses with learned uncertainty weighting
                    losses = compute_dag_structure_loss(
                        sign_logits_seq,
                        digit_logits_seq,
                        operation_logits_seq,
                        pred_statistics,
                        target_sgn_seq,
                        target_digits_seq,
                        target_ops_seq,
                        target_initial_values_seq,
                        target_final_exec_seq,
                        target_statistics_seq,
                        cfg,
                        uncertainty_params=uncertainty_params,
                    )
                    loss = losses["total_loss"] / cfg.gradient_accumulation_steps

                    # Compute gradient cosines when we're doing regular logging
                    should_compute_gradient_cosines = (
                        iter_num % cfg.log_interval == 0
                        and micro_step
                        == cfg.gradient_accumulation_steps
                        - 1  # Only on last micro-step
                        and master_process
                    )
                    if should_compute_gradient_cosines:
                        # Compute gradient cosines for analysis
                        model_params = list(raw_model.parameters())
                        # All losses use automatic balancing (no manual weights)
                        weighted_losses = {
                            "sign_loss": losses["sign_loss"],
                            "digit_loss": losses["digit_loss"],
                            "op_loss": losses["op_loss"],
                            "value_loss": losses["value_loss"],
                            "exec_loss": (
                                losses["exec_loss"]  # Already scaled automatically
                                if not cfg.check_nans
                                or torch.isfinite(losses["exec_loss"]).all()
                                else torch.tensor(
                                    0.0, device=losses["exec_loss"].device
                                )
                            ),
                            "stats_loss": losses[
                                "stats_loss"
                            ],  # Already scaled automatically
                        }
                        gradient_cosines = compute_gradient_cosines(
                            weighted_losses,
                            losses["total_loss"],
                            model_params,
                        )
                        losses.update(gradient_cosines)
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        if key == "uncertainty_weights":
                            # Store the full tensor for detailed logging
                            loss_accum[key] = value.detach().cpu()
                        else:
                            value = value.item() / cfg.gradient_accumulation_steps
                            loss_accum[key] = value
                    else:
                        loss_accum[key] = value

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

                # Backward pass
                scaler.scale(loss).backward()

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
                # Display losses (all use automatic balancing)
                sign_loss_val = loss_accum["sign_loss"]
                digit_loss_val = loss_accum["digit_loss"]
                op_loss_val = loss_accum["op_loss"]
                value_loss_val = loss_accum["value_loss"]
                exec_loss_val = loss_accum["exec_loss"]  # Automatically scaled
                stats_loss_val = loss_accum["stats_loss"]  # Automatically scaled

                # Format uncertainty weights for logging (exponential of log-variance)
                uncertainty_weights = loss_accum["uncertainty_weights"]
                uncertainty_weights_str = f"[{uncertainty_weights[0]:.3f},{uncertainty_weights[1]:.3f},{uncertainty_weights[2]:.3f},{uncertainty_weights[3]:.3f},{uncertainty_weights[4]:.3f},{uncertainty_weights[5]:.3f}]"

                log_msg = (
                    f"iter {iter_num}: loss {loss_accum['total_loss']:.4f}, "
                    f"sign {sign_loss_val:.4f}, "
                    f"digit {digit_loss_val:.4f}, "
                    f"op {op_loss_val:.4f}, "
                    f"value {value_loss_val:.4f}, "
                    f"exec {exec_loss_val:.4f}, "
                    f"stats {stats_loss_val:.4f}, "
                    f"uncertainty_weights {uncertainty_weights_str}, "
                    f"op_acc {loss_accum['op_accuracy']:.4f}, "
                    f"full_op_match {loss_accum['full_dag_op_match']:.4f}, "
                    f"sign_acc {loss_accum['sign_accuracy']:.4f}"
                    f", time {dt*1000:.2f}ms"
                )

                # realtime training log emitted via wandb
                if master_process:
                    print(log_msg)

                # Log to wandb
                if run is not None:
                    # Capture the *actual* learning rate currently in the optimizer. This avoids any
                    # issues where the local `lr` variable drifts from the value that is ultimately
                    # used for the step (e.g. if the optimizer or scheduler modifies it).
                    current_lr = optimizer.param_groups[0]["lr"]
                    current_uncertainty_params_lr = (
                        optimizer.param_groups[1]["lr"]
                        if len(optimizer.param_groups) > 1
                        else current_lr
                    )

                    # Get current uncertainty_params values for monitoring via consistent access pattern
                    current_uncertainty_params = (
                        raw_model.dag_predictor.uncertainty_params.detach().cpu()
                    )

                    log_dict = {
                        "iter": iter_num,
                        "train/total_loss": loss_accum["total_loss"],
                        "train/sign_loss": loss_accum["sign_loss"],
                        "train/digit_loss": loss_accum["digit_loss"],
                        "train/op_loss": loss_accum["op_loss"],
                        "train/value_loss": loss_accum["value_loss"],
                        "train/exec_loss": loss_accum["exec_loss"],
                        "train/stats_loss": loss_accum["stats_loss"],
                        "lr": current_lr,
                        "lr_uncertainty_params": current_uncertainty_params_lr,
                        "train/op_accuracy": loss_accum["op_accuracy"],
                        "train/full_dag_op_match": loss_accum["full_dag_op_match"],
                        "train/sign_accuracy": loss_accum["sign_accuracy"],
                        "train/time_per_iter_ms": dt * 1000,
                        # Uncertainty weights (exp(-uncertainty_params))
                        "uncertainty_weights/sign": uncertainty_weights[0].item(),
                        "uncertainty_weights/digit": uncertainty_weights[1].item(),
                        "uncertainty_weights/op": uncertainty_weights[2].item(),
                        "uncertainty_weights/value": uncertainty_weights[3].item(),
                        "uncertainty_weights/exec": uncertainty_weights[4].item(),
                        "uncertainty_weights/stats": uncertainty_weights[5].item(),
                        # Raw uncertainty_params values for monitoring adaptation
                        "uncertainty_params/sign": current_uncertainty_params[0].item(),
                        "uncertainty_params/digit": current_uncertainty_params[
                            1
                        ].item(),
                        "uncertainty_params/op": current_uncertainty_params[2].item(),
                        "uncertainty_params/value": current_uncertainty_params[
                            3
                        ].item(),
                        "uncertainty_params/exec": current_uncertainty_params[4].item(),
                        "uncertainty_params/stats": current_uncertainty_params[
                            5
                        ].item(),
                    }

                    # Add internal losses and gradient cosines to wandb logging
                    for key, value in loss_accum.items():
                        if key.startswith("grad_cosine_"):
                            log_dict[f"grad/{key}"] = value
                        elif (
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
