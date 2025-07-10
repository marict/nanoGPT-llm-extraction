"""
Training script for DAG predictor pretraining on structure prediction.

This script trains the DAG predictor components to predict DAG structures
from text descriptions, serving as pretraining before full model training.
"""

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
from dataclasses import dataclass
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
from data.dagset.streaming import create_dag_structure_dataloaders
from models.dag_model import GPT, OP_NAMES, GPTConfig
from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel
from python_version_check import check_python_version
from training_utils import (CHECKPOINT_DIR, BaseConfig, CheckpointLoadError,
                            _check_for_nonfinite, _safe_torch_save,
                            apply_overrides, clean_previous_checkpoints,
                            find_latest_checkpoint, generate_run_name,
                            get_checkpoint_filename, get_lr,
                            handle_checkpoint_loading, load_config_file,
                            parse_args, update_config)

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

# Checkpoint directory
CHECKPOINT_DIR = (
    "/runpod-volume/checkpoints" if os.path.exists("/runpod-volume") else "checkpoints"
)


# --------------------------------------------------------------------------- #
# Tokenization utility
# --------------------------------------------------------------------------- #


def tokenize_texts(texts: List[str], sequence_length: int, device: str) -> torch.Tensor:
    """Tokenize a list of mathematical expressions.

    Args:
        texts: List of mathematical expressions to tokenize
        sequence_length: Target sequence length
        device: Device to place tensor on

    Returns:
        Tensor of token IDs with shape (batch_size, sequence_length)
    """
    # Initialize GPT-2 tokenizer
    enc = get_encoding("gpt2")

    batch_size = len(texts)
    input_tokens = torch.zeros(
        (batch_size, sequence_length), dtype=torch.long, device=device
    )

    for i, text in enumerate(texts):
        # Tokenize the text
        tokens = enc.encode_ordinary(text)

        # Truncate or pad to sequence_length
        if len(tokens) >= sequence_length:
            # Truncate to sequence_length
            tokens = tokens[:sequence_length]
        else:
            # Pad with zeros (or use a special padding token if preferred)
            tokens = tokens + [0] * (sequence_length - len(tokens))

        input_tokens[i] = torch.tensor(tokens, dtype=torch.long)

    return input_tokens


# --------------------------------------------------------------------------- #
# Safe checkpoint saving with retry (copied from train.py)
# --------------------------------------------------------------------------- #


class CheckpointSaveError(Exception):
    """Raised when saving a checkpoint fails after retries."""


def _safe_torch_save(obj, path: Path, retries: int = 1) -> None:
    """Save <obj> to <path> with retry; raise CheckpointSaveError on failure."""
    tmp_path = path.with_suffix(".tmp")

    for attempt in range(retries + 1):
        try:
            if _HAVE_ST and _all_tensors(obj):
                # Use safetensors for pure tensor checkpoints
                final_path = path.with_suffix(".safetensors")
                _st.save_file(obj, str(tmp_path))
            else:
                # Fallback to torch.save for mixed-type or no-safetensors state
                final_path = path.with_suffix(".pt")
                torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)

            try:
                # Perform atomic rename
                tmp_path.rename(final_path)
                return
            except Exception as exc:
                if attempt >= retries:
                    raise CheckpointSaveError(
                        f"Failed to rename temporary checkpoint {tmp_path} to {final_path}: {exc}"
                    ) from exc
                print(
                    f"Retrying checkpoint rename ({attempt+1}/{retries}) due to error: {exc}"
                )
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries:
                raise CheckpointSaveError(
                    f"Failed to save checkpoint {path}: {exc}"
                ) from exc
            print(
                f"Retrying checkpoint save ({attempt+1}/{retries}) due to error: {exc}"
            )


def _all_tensors(state):
    """Return True if every leaf value in state dict is a torch.Tensor."""
    for v in state.values():
        if isinstance(v, dict):
            if not _all_tensors(v):
                return False
        elif not isinstance(v, torch.Tensor):
            return False
    return True


@dataclass
class DAGTrainConfig(BaseConfig):
    """Container for DAG predictor training hyperparameters."""

    name: str = "dag_pretrain"  # Project/run name

    # Learning rate schedule
    use_cyclical_lr: bool = False
    cyclical_lr_period: int = 1000
    cyclical_lr_amplitude: float = 0.1

    # DAG dataset parameters
    max_dag_depth: int = 8
    max_digits: int = (
        4  # Maximum number of integer digits for uniform digit distribution
    )
    max_decimal_places: int = (
        None  # Auto-derived from max_digits for uniform string distribution
    )
    train_examples_per_batch: int = 1000
    val_examples_per_batch: int = 100

    # English conversion settings
    english_conversion_rate: float = (
        0.3  # Probability of converting tokens to English (0.0 = disabled, 1.0 = always convert)
    )

    gradient_accumulation_steps: int = 4
    batch_size: int = 32
    sequence_length: int = 512  # For tokenized text inputs

    # Model architecture (should match target model)
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    dag_depth: int = 4  # Target DAG depth for pretraining

    # Training hyperparameters
    learning_rate: float = 3e-4
    max_iters: int = 10000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 500
    lr_decay_iters: int = 10000
    min_lr: float = 3e-5

    # Loss weights
    sign_loss_weight: float = 1.0
    log_loss_weight: float = 1.0
    op_loss_weight: float = 1.0

    # Random seeds
    seed: int = 42

    # New options
    # If True, use the full GPT backbone (models.dag_model.GPT) instead of the
    # shallow PredictorOnlyModel. Loss is still computed only on the DAG
    # predictor outputs so the language model head is not used.
    full_backbone: bool = False

    # Number of transformer layers when using the full backbone (ignored when
    # `full_backbone` is False and the shallow predictor is used).
    n_layer: int = 12


def generate_run_name(cfg: DAGTrainConfig) -> str:
    """Generate a run name using RunPod identifier or local random string."""
    # Check if we're running on RunPod
    runpod_id = os.environ.get("RUNPOD_POD_ID")

    if runpod_id and runpod_id.strip():
        # Use RunPod identifier
        base = runpod_id
    else:
        # Generate local identifier with random string
        random_str = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=12)
        )
        base = f"local_{random_str}"

    # Append note if provided
    note_val = getattr(cfg, "note", None)
    if note_val:
        return f"{base} - {note_val}"
    return base


def load_config_file(path: str) -> Dict[str, object]:
    """Load configuration from Python file."""
    namespace = runpy.run_path(path)
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


def update_config(cfg: DAGTrainConfig, data: Dict[str, object]) -> None:
    """Update configuration object with values from dictionary."""
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)


def apply_overrides(cfg: DAGTrainConfig, overrides: List[str]) -> None:
    """Apply command line overrides to configuration."""
    for override in overrides:
        if "=" not in override:
            continue

        key, value = override.split("=", 1)
        key = key.lstrip("-")

        # Convert value to appropriate type
        try:
            value = literal_eval(value)
        except (ValueError, SyntaxError):
            # Keep as string if can't evaluate
            pass

        if hasattr(cfg, key):
            setattr(cfg, key, value)


def parse_args() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DAG predictor on structure prediction"
    )
    parser.add_argument(
        "config", nargs="?", default="config/train_predictor_default.py"
    )
    parser.add_argument("--wandb-api-key", type=str, help="Weights & Biases API key")
    parser.add_argument("--wandb-run-id", type=str, help="Resume existing wandb run")
    parser.add_argument("--dag-depth", type=int, help="Override DAG depth")
    parser.add_argument("--note", type=str, help="Run note for identification")
    parser.add_argument(
        "--keep-alive", action="store_true", help="Keep pod alive after training"
    )
    parser.add_argument(
        "--use-runpod", action="store_true", help="Use RunPod for training"
    )
    parser.add_argument("--gpu-type", type=str, help="GPU type for RunPod")
    return parser


def get_lr(it: int, *, cfg: DAGTrainConfig) -> float:
    """Get learning rate for a given iteration."""
    # 1) linear warmup for warmup_iters steps
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters

    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg.lr_decay_iters:
        return cfg.min_lr

    # 3) in between, use cosine decay as the base
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    base_lr = cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    # 4) apply cyclical modulation if enabled
    if getattr(cfg, "use_cyclical_lr", False):
        progress_in_decay = it - cfg.warmup_iters
        progress_in_cycle = (
            progress_in_decay % cfg.cyclical_lr_period
        ) / cfg.cyclical_lr_period
        # Sinusoidal modulation
        modulation = 1.0 + cfg.cyclical_lr_amplitude * math.sin(
            2 * math.pi * progress_in_cycle
        )
        final_lr = base_lr * modulation
        return max(cfg.min_lr, final_lr)

    return base_lr


def get_checkpoint_filename(
    cfg: DAGTrainConfig, iter_num: int, model_name: str, val_acc: float | None = None
) -> str:
    """Generate checkpoint filename without extension."""
    if val_acc is not None:
        # Format accuracy as percentage with 2 decimal places
        acc_str = f"{val_acc * 100:.2f}acc"
        return f"dag_ckpt_{model_name}_{cfg.name}_{iter_num:06d}_{acc_str}"
    else:
        return f"dag_ckpt_{model_name}_{cfg.name}_{iter_num:06d}"


def clean_previous_checkpoints(cfg: DAGTrainConfig, model_name: str) -> None:
    """Remove previous checkpoints if requested."""
    if not cfg.clear_previous_checkpoints:
        return

    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return

    pattern = f"dag_ckpt_{model_name}_{cfg.name}_*.*"
    removed_count = 0
    for ckpt_file in checkpoint_dir.glob(pattern):
        try:
            ckpt_file.unlink()
            removed_count += 1
        except Exception as e:
            print(f"Warning: Could not remove {ckpt_file}: {e}")

    if removed_count > 0:
        print(f"Removed {removed_count} previous checkpoints")


def find_latest_checkpoint(
    cfg: DAGTrainConfig, model_name: str, any_run: bool = False
) -> Path | None:
    """Find the latest checkpoint for resuming training."""
    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return None

    if any_run:
        pattern = f"dag_ckpt_{model_name}_*_*.*"
    else:
        pattern = f"dag_ckpt_{model_name}_{cfg.name}_*.*"

    checkpoints = list(checkpoint_dir.glob(pattern))

    if not checkpoints:
        return None

    # Sort by iteration number
    def extract_iter(path):
        try:
            parts = path.stem.split("_")
            # Handle both old format (dag_ckpt_model_name_iter) and new format (dag_ckpt_model_name_iter_acc)
            if len(parts) >= 4:
                # Extract iteration number - it's either the last part or second to last
                iter_part = (
                    parts[-2]
                    if len(parts) >= 5 and parts[-1].endswith("acc")
                    else parts[-1]
                )
                return int(iter_part)
            return 0
        except (ValueError, IndexError):
            return 0

    return max(checkpoints, key=extract_iter)


def compute_dag_structure_loss(
    pred_sgn: torch.Tensor,
    pred_log: torch.Tensor,
    pred_ops: torch.Tensor,
    target_sgn: torch.Tensor,
    target_log: torch.Tensor,
    target_ops: torch.Tensor,
    cfg: DAGTrainConfig,
) -> Dict[str, torch.Tensor]:
    """Compute loss for DAG structure prediction.

    Args:
        pred_sgn: (B, T, num_nodes) predicted signs
        pred_log: (B, T, num_nodes) predicted log magnitudes
        pred_ops: (B, T, depth, n_ops) predicted operation probabilities
        target_sgn: (B, T, num_nodes) target signs
        target_log: (B, T, num_nodes) target log magnitudes
        target_ops: (B, T, depth, n_ops) target operation probabilities (one-hot)
        cfg: Training configuration

    Returns:
        Dictionary with loss components
    """

    B, T = pred_sgn.shape[:2]

    # Sign loss (MSE)
    sign_loss = F.mse_loss(pred_sgn, target_sgn, reduction="none")

    # Log magnitude loss (MSE)
    log_loss = F.mse_loss(pred_log, target_log, reduction="none")

    # Operation loss (negative log-likelihood since pred_ops are probabilities, not logits)
    pred_ops_flat = pred_ops.reshape(-1, pred_ops.size(-1))  # (B*T*depth, n_ops)
    target_ops_flat = target_ops.reshape(-1, target_ops.size(-1))  # (B*T*depth, n_ops)
    target_op_indices = target_ops_flat.argmax(dim=-1)  # (B*T*depth,)

    # Use nll_loss since pred_ops are probabilities (need log probabilities)
    log_pred_ops = torch.log(
        pred_ops_flat + 1e-8
    )  # Add small epsilon for numerical stability
    op_loss_flat = F.nll_loss(log_pred_ops, target_op_indices, reduction="none")
    op_loss = op_loss_flat.reshape(B, T, -1)  # (B, T, depth)

    sign_loss = sign_loss.sum() / B
    log_loss = log_loss.sum() / B
    op_loss = op_loss.sum() / B

    # Weighted total loss - ONLY from the main predictions (signs, magnitudes, operations)
    total_loss = (
        cfg.sign_loss_weight * sign_loss
        + cfg.log_loss_weight * log_loss
        + cfg.op_loss_weight * op_loss
    )

    loss_dict = {
        "total_loss": total_loss,
        "sign_loss": sign_loss,
        "log_loss": log_loss,
        "op_loss": op_loss,
    }

    return loss_dict


def evaluate_dag_model(
    model: torch.nn.Module,
    val_loader,
    device: str,
    ctx,
    cfg: DAGTrainConfig,
    eval_iters: int,
    seed: int,
) -> Dict[str, float]:
    """Evaluate DAG model on validation set."""
    model.eval()
    # ------------------------------------------------------------------ #
    # Seed RNG for reproducible validation sample selection and expose the
    # seed in the console output so the exact example can be regenerated
    # later for debugging or unit testing purposes.
    # ------------------------------------------------------------------ #
    eval_sample_seed = seed
    _eval_random.seed(eval_sample_seed)

    total_losses = {
        "total_loss": 0.0,
        "sign_loss": 0.0,
        "log_loss": 0.0,
        "op_loss": 0.0,
    }
    # Additional evaluation metrics we want to track
    total_metrics = {
        "op_accuracy": 0.0,
        "full_dag_op_match": 0.0,
        "sign_accuracy": 0.0,
        "log_magnitude_mape": 0.0,
    }

    num_batches = 0

    with torch.no_grad():
        for i, (texts, structures) in enumerate(val_loader):
            if i >= eval_iters:
                break

            # Move structure tensors to device
            target_sgn = structures["initial_sgn"].to(device)  # (B, num_nodes)
            target_log = structures["initial_log"].to(device)  # (B, num_nodes)
            target_ops = structures["operation_probs"].to(device)  # (B, depth, n_ops)

            # Tokenize texts properly using the mathematical expressions
            input_tokens = tokenize_texts(texts, cfg.sequence_length, device)

            # Forward pass through shallow attention DAG predictor model
            with ctx:
                if cfg.full_backbone and hasattr(model, "dag"):
                    # Obtain hidden states from the GPT backbone and run the DAG plan predictor
                    hidden = model.forward_hidden(input_tokens)
                    pred_sgn, pred_log, pred_ops = model.dag.plan_predictor(hidden)
                else:
                    # Predictor-only model (shallow attention)
                    pred_sgn, pred_log, pred_ops = model(input_tokens)

                # Average predictions over sequence length for structure prediction
                pred_sgn = pred_sgn.mean(dim=1)  # (B, num_nodes_pred)
                pred_log = pred_log.mean(dim=1)  # (B, num_nodes_pred)
                pred_ops = pred_ops.mean(dim=1)  # (B, depth_pred, n_ops)

                # Ensure target and prediction tensors have compatible shapes
                target_nodes = target_sgn.size(1)
                pred_nodes = pred_sgn.size(1)
                target_depth = target_ops.size(1)
                pred_depth = pred_ops.size(1)

                # Predictions should match targets, throw error if not
                if pred_nodes != target_nodes or pred_depth != target_depth:
                    raise ValueError(
                        f"Predictions do not match targets. Pred nodes: {pred_nodes}, Target nodes: {target_nodes}, Pred depth: {pred_depth}, Target depth: {target_depth}"
                    )

                # Add sequence dimension to match loss function expectations
                pred_sgn = pred_sgn.unsqueeze(1)  # (B, 1, num_nodes)
                pred_log = pred_log.unsqueeze(1)  # (B, 1, num_nodes)
                pred_ops = pred_ops.unsqueeze(1)  # (B, 1, depth, n_ops)

                target_sgn = target_sgn.unsqueeze(1)  # (B, 1, num_nodes)
                target_log = target_log.unsqueeze(1)  # (B, 1, num_nodes)
                target_ops = target_ops.unsqueeze(1)  # (B, 1, depth, n_ops)

                # Compute losses
                losses = compute_dag_structure_loss(
                    pred_sgn,
                    pred_log,
                    pred_ops,
                    target_sgn,
                    target_log,
                    target_ops,
                    cfg,
                )

                # Console debug: show one random sample's text and initial values (first batch only)
                if i == 0:
                    batch_size = target_sgn.size(0)  # Get batch size from tensor
                    sample_idx = _eval_random.randrange(batch_size)
                    sample_text = texts[sample_idx]

                    # Gather predicted and target sign/log tensors directly
                    pred_sign_vec = pred_sgn.squeeze(1)[sample_idx]
                    pred_log_vec = pred_log.squeeze(1)[sample_idx]
                    tgt_sign_vec = target_sgn.squeeze(1)[sample_idx]
                    tgt_log_vec = target_log.squeeze(1)[sample_idx]

                    # Convert to real numbers: sign * exp(log_mag)
                    pred_real_vals = (
                        (pred_sign_vec * _torch.exp(pred_log_vec)).cpu().tolist()
                    )
                    tgt_real_vals = (
                        (tgt_sign_vec * _torch.exp(tgt_log_vec)).cpu().tolist()
                    )

                    # Count tokens in the sample text
                    enc = get_encoding("gpt2")
                    sample_tokens = enc.encode_ordinary(sample_text)

                    print("\n=== Validation Sample ===")
                    # Log the RNG seed so we can reproduce this sample exactly
                    print(f"Validation RNG seed: {eval_sample_seed}")
                    print(f"Text: {sample_text}")
                    print(f"Tokens: {len(sample_tokens)}")
                    print("Target initial values:")
                    print([round(v, 4) for v in tgt_real_vals])
                    print("Predicted initial values:")
                    print([round(v, 4) for v in pred_real_vals])

                    # Decode ground-truth operations for this sample
                    gt_ops_onehot = target_ops.squeeze(1)[sample_idx]  # (depth, n_ops)
                    gt_op_indices = gt_ops_onehot.argmax(dim=-1).cpu().tolist()
                    gt_op_names = [OP_NAMES[idx] for idx in gt_op_indices]
                    print("Operations (ground truth):")
                    print(gt_op_names)

                    # Decode predicted operations for this sample
                    pred_ops_sample = pred_ops.squeeze(1)[sample_idx]  # (depth, n_ops)
                    pred_op_indices = pred_ops_sample.argmax(dim=-1).cpu().tolist()
                    pred_op_names = [OP_NAMES[idx] for idx in pred_op_indices]
                    print("Operations (predicted w/ argmax):")
                    print(pred_op_names)
                    print("==========================\n")

                # ------------------------------------------------------------------ #
                # Compute additional evaluation metrics
                # ------------------------------------------------------------------ #
                # Squeeze the sequence dimension which is 1 after earlier unsqueeze
                pred_ops_sq = pred_ops.squeeze(1)  # (B, depth, n_ops)
                target_ops_sq = target_ops.squeeze(1)
                pred_ops_idx = pred_ops_sq.argmax(dim=-1)
                target_ops_idx = target_ops_sq.argmax(dim=-1)

                op_correct = pred_ops_idx.eq(target_ops_idx)  # (B, depth)
                op_acc = op_correct.float().mean()
                full_match = op_correct.all(dim=-1).float().mean()  # (B,)

                pred_sgn_sq = pred_sgn.squeeze(1)  # (B, num_nodes)
                target_sgn_sq = target_sgn.squeeze(1)
                sign_correct = torch.sign(pred_sgn_sq).eq(torch.sign(target_sgn_sq))
                sign_acc = sign_correct.float().mean()

                # Magnitude MAPE using exponentiated logs
                pred_mag = pred_log.squeeze(1).exp()
                target_mag = target_log.squeeze(1).exp()
                log_mape = (
                    (pred_mag - target_mag).abs() / target_mag.clamp_min(1e-8)
                ).mean()

                # Accumulate metrics
                total_metrics["op_accuracy"] += op_acc.item()
                total_metrics["full_dag_op_match"] += full_match.item()
                total_metrics["sign_accuracy"] += sign_acc.item()
                total_metrics["log_magnitude_mape"] += log_mape.item()

                # Accumulate losses
                for key, value in losses.items():
                    total_losses[key] += value.item()

                num_batches += 1

    # Average losses
    if num_batches > 0:
        for key in total_losses:
            total_losses[key] /= num_batches
        for key in total_metrics:
            total_metrics[key] /= num_batches

    model.train()
    # Merge dictionaries before returning for convenience
    return {**total_losses, **total_metrics}


def _get_dag_predictions(
    model: torch.nn.Module,
    input_tokens: torch.Tensor,
    cfg: DAGTrainConfig,
    ctx,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper function to get predictions from the DAG model."""
    with ctx:
        pred_sgn, pred_log, pred_ops = model(input_tokens)

    # Average over sequence dimension
    pred_sgn_avg = pred_sgn.mean(dim=1)  # (B, num_nodes_pred)
    pred_log_avg = pred_log.mean(dim=1)  # (B, num_nodes_pred)
    pred_ops_avg = pred_ops.mean(dim=1)  # (B, depth_pred, n_ops)

    # Ensure target and prediction tensors have compatible shapes
    target_nodes = input_tokens.size(1)  # Assuming input_tokens shape is (B, T, ...)
    pred_nodes = pred_sgn_avg.size(1)
    target_depth = pred_ops_avg.size(
        1
    )  # Assuming pred_ops_avg shape is (B, T, depth, ...)
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

    return pred_sgn_seq, pred_log_seq, pred_ops_seq


def train_predictor(cfg: DAGTrainConfig, wandb_run_id: str | None = None) -> None:
    """Run DAG predictor training loop."""
    # --------------------------------------------------------------------- #
    # Setup
    # --------------------------------------------------------------------- #
    setup_start = time.time()
    print(f"[{time.time() - setup_start:.2f}s] Starting DAG predictor training")
    print(f"[{time.time() - setup_start:.2f}s] PyTorch version: {torch.__version__}")

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
    if cfg.full_backbone:
        model_name = GPT.__name__
        model_config = GPTConfig(
            vocab_size=50304,
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            n_layer=cfg.n_layer,
            dropout=cfg.dropout,
            bias=cfg.bias,
            dag_depth=cfg.dag_depth,
            block_size=cfg.sequence_length,
            softmax_temperature=20.0,
        )
    else:
        model_name = PredictorOnlyModel.__name__
        model_config = PredictorOnlyConfig(
            vocab_size=50304,
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            dropout=cfg.dropout,
            bias=cfg.bias,
            dag_depth=cfg.dag_depth,
            sequence_length=cfg.sequence_length,
            softmax_temperature=20.0,
        )

    # Clean previous checkpoints
    if master_process:
        clean_previous_checkpoints(cfg, model_name=model_name)

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
            print(f"[{time.time() - setup_start:.2f}s] W&B URL: {run.url}")
        except Exception as e:
            print(
                f"[{time.time() - setup_start:.2f}s] Error: Failed to initialize wandb: {e}"
            )
            raise
    else:
        # Non-master processes don't initialize wandb
        run = None

    # Device and dtype setup
    torch.manual_seed(1337 + seed_offset)
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
    print(f"[{time.time() - setup_start:.2f}s] Creating DAG structure dataloaders")

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
    print(
        f"[{time.time() - setup_start:.2f}s] Initializing {'full GPT backbone' if cfg.full_backbone else 'shallow attention'} DAG predictor model"
    )

    # Handle checkpoint loading using shared logic
    expected_keys = ["model", "optimizer", "model_config", "iter_num", "best_val_loss"]
    checkpoint, iter_num, best_val_loss = handle_checkpoint_loading(
        cfg, device, None, expected_keys
    )

    if checkpoint is not None:
        # Loading from checkpoint - create model from saved config
        print(f"[{time.time() - setup_start:.2f}s] Loading model from checkpoint")
        saved_config = checkpoint["model_config"]

        if cfg.full_backbone:
            # Reconstruct GPTConfig from saved model_config
            model_config_dict = {
                "vocab_size": saved_config.get("vocab_size", 50304),
                "n_embd": saved_config.get("n_embd", cfg.n_embd),
                "n_head": saved_config.get("n_head", cfg.n_head),
                "n_layer": saved_config.get("n_layer", cfg.n_layer),
                "dropout": saved_config.get("dropout", cfg.dropout),
                "bias": saved_config.get("bias", cfg.bias),
                "dag_depth": saved_config.get("dag_depth", cfg.dag_depth),
                "block_size": saved_config.get("block_size", cfg.sequence_length),
                "softmax_temperature": saved_config.get("softmax_temperature", 20.0),
            }
            model_config = GPTConfig(**model_config_dict)
            model = GPT(model_config)
        else:
            # Reconstruct PredictorOnlyConfig from saved model_config
            model_config_dict = {
                "vocab_size": saved_config.get("vocab_size", 50304),
                "n_embd": saved_config.get("n_embd", cfg.n_embd),
                "n_head": saved_config.get("n_head", cfg.n_head),
                "dropout": saved_config.get("dropout", cfg.dropout),
                "bias": saved_config.get("bias", cfg.bias),
                "dag_depth": saved_config.get("dag_depth", cfg.dag_depth),
                "sequence_length": saved_config.get(
                    "sequence_length", cfg.sequence_length
                ),
                "softmax_temperature": saved_config.get("softmax_temperature", 20.0),
            }
            model_config = PredictorOnlyConfig(**model_config_dict)
            model = PredictorOnlyModel(model_config)

        # Load model state
        state_dict = {
            k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()
        }
        model.load_state_dict(state_dict)
        print(f"[{time.time() - setup_start:.2f}s] ✅ Model loaded from checkpoint")
    else:
        # Creating new model from scratch
        if cfg.full_backbone:
            model = GPT(model_config)
            print(
                f"[{time.time() - setup_start:.2f}s] ✅ Full backbone model initialized."
            )
        else:
            model = PredictorOnlyModel(model_config)
            print(
                f"[{time.time() - setup_start:.2f}s] ✅ Shallow predictor model initialized."
            )

    model.to(device)

    if master_process:
        print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")

    # Optimizer setup
    print(f"[{time.time() - setup_start:.2f}s] Initializing optimizer")

    # All parameters are trainable in this standalone model
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"[{time.time() - setup_start:.2f}s] Training {len(trainable_params)} parameters"
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    if checkpoint is not None and "optimizer" in checkpoint:
        print(
            f"[{time.time() - setup_start:.2f}s] 🔄 Loading optimizer state from checkpoint"
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            f"[{time.time() - setup_start:.2f}s] ✅ Optimizer state loaded (Adam momentum preserved)"
        )
    else:
        print(
            f"[{time.time() - setup_start:.2f}s] 🆕 Initializing fresh optimizer state"
        )

    # Gradient scaler
    scalar_enabled = actual_dtype == "float16"
    scaler = (
        torch.cuda.amp.GradScaler(enabled=scalar_enabled)
        if TORCH_2_2_1
        else torch.amp.GradScaler("cuda", enabled=scalar_enabled)
    )

    # Model compilation
    if cfg.compile:
        print(f"[{time.time() - setup_start:.2f}s] Compiling model")
        try:
            model = torch.compile(model, mode="reduce-overhead", disable="cudagraphs")
        except TypeError:
            model = torch.compile(model, mode="reduce-overhead")

    if ddp:
        model = DDP(model, device_ids=[int(device.split(":")[-1])])

    # --------------------------------------------------------------------- #
    # Training loop
    # --------------------------------------------------------------------- #
    print(f"[{time.time() - setup_start:.2f}s] 🚀 Training initialization complete!")
    print(f"[{time.time() - setup_start:.2f}s] 📋 Summary:")
    print(f"[{time.time() - setup_start:.2f}s]   - Mode: {cfg.init_from}")
    print(f"[{time.time() - setup_start:.2f}s]   - Starting iteration: {iter_num}")
    print(
        f"[{time.time() - setup_start:.2f}s]   - Best validation loss: {best_val_loss:.4f}"
    )
    print(f"[{time.time() - setup_start:.2f}s]   - Max iterations: {cfg.max_iters}")
    print(f"[{time.time() - setup_start:.2f}s] Starting training loop")

    t0 = time.time()
    raw_model = model.module if ddp else model
    train_start = time.time()

    # Initialize loss accumulator for logging
    loss_accum = {
        "total_loss": 0.0,
        "sign_loss": 0.0,
        "log_loss": 0.0,
        "op_loss": 0.0,
        "op_accuracy": 0.0,
        "full_dag_op_match": 0.0,
        "sign_accuracy": 0.0,
        "log_magnitude_mape": 0.0,
    }

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
                print(
                    f"step {iter_num}: train loss {loss_accum.get('total_loss', 'N/A'):.4f}, "
                    f"val loss {eval_losses['total_loss']:.4f}"
                )

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
                    if cfg.save_best and is_new_best:
                        # Save a single checkpoint for the best model
                        val_acc = eval_losses.get("op_accuracy", None)
                        if val_acc is not None:
                            acc_str = f"{val_acc * 100:.2f}acc"
                            checkpoint_base = f"dag_ckpt_{raw_model.__class__.__name__}_{cfg.name}_best_{acc_str}"
                        else:
                            checkpoint_base = f"dag_ckpt_{raw_model.__class__.__name__}_{cfg.name}_best"
                        checkpoint_path = Path(CHECKPOINT_DIR) / checkpoint_base
                        print(f"saving new best checkpoint to {checkpoint_path}")
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_config": model_config.__dict__,
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                        }
                        _safe_torch_save(checkpoint, checkpoint_path)

                    if cfg.always_save_checkpoint or (
                        not cfg.save_best and is_new_best
                    ):
                        # Save a checkpoint with the iteration number
                        val_acc = eval_losses.get("op_accuracy", None)
                        checkpoint_base = get_checkpoint_filename(
                            cfg, iter_num, raw_model.__class__.__name__, val_acc=val_acc
                        )
                        checkpoint_path = Path(CHECKPOINT_DIR) / checkpoint_base
                        print(f"saving checkpoint to {checkpoint_path}")
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_config": model_config.__dict__,
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                        }
                        _safe_torch_save(checkpoint, checkpoint_path)

                model.train()

            # Early stopping
            if iter_num == 0 and cfg.eval_only:
                break

            # Forward and backward pass
            optimizer.zero_grad(set_to_none=True)

            # Get a batch
            texts, structures = next(train_loader)

            # Move targets to device
            target_sgn = structures["initial_sgn"].to(device)
            target_log = structures["initial_log"].to(device)
            target_ops = structures["operation_probs"].to(device)

            loss_accum = {
                "total_loss": 0.0,
                "sign_loss": 0.0,
                "log_loss": 0.0,
                "op_loss": 0.0,
                "op_accuracy": 0.0,
                "full_dag_op_match": 0.0,
                "sign_accuracy": 0.0,
                "log_magnitude_mape": 0.0,
            }

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
                print(log_msg)

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
        print(f"Fatal error in training loop: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print(
            f"[{time.time() - train_start:.2f}s] Training loop completed in {time.time() - train_start:.2f}s"
        )
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
            print("Stopping RunPod instance due to error...")
            runpod_service.stop_runpod()

        # Re-raise the exception to ensure proper exit code
        raise


if __name__ == "__main__":
    main()
