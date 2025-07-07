"""
Training script for DAG predictor pretraining on structure prediction.

This script trains the DAG predictor components to predict DAG structures
from text descriptions, serving as pretraining before full model training.
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
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import runpod_service
import wandb
from dag_logger import DAGLogger
from dag_model import (MLP, Block, CausalSelfAttention, DAGPlanPredictor,
                       LayerNorm, op_names)
from data.dagset.streaming import create_dag_dataloaders
from python_version_check import check_python_version

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
                _st.save_file(obj, str(tmp_path.with_suffix(".safetensors")))
                tmp_path = tmp_path.with_suffix(".safetensors")
            else:
                fallback_path = tmp_path.with_suffix(".pt")
                torch.save(obj, fallback_path, _use_new_zipfile_serialization=False)
                tmp_path = fallback_path

            tmp_path.replace(path)
            return
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
    import torch

    for v in state.values():
        if isinstance(v, dict):
            if not _all_tensors(v):
                return False
        elif not isinstance(v, torch.Tensor):
            return False
    return True


@dataclass
class DAGTrainConfig:
    """Container for DAG predictor training hyperparameters."""

    eval_interval: int = 100
    log_interval: int = 10
    eval_iters: int = 50
    eval_only: bool = False
    always_save_checkpoint: bool = True
    clear_previous_checkpoints: bool = False
    init_from: str = "scratch"

    name: str = "dag_pretrain"  # Project/run name

    # DAG dataset parameters
    max_dag_depth: int = 8
    train_examples_per_batch: int = 1000
    val_examples_per_batch: int = 100

    gradient_accumulation_steps: int = 4
    batch_size: int = 32
    sequence_length: int = 512  # For tokenized text inputs

    # Model architecture (should match target model)
    n_layer: int = 12
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

    backend: str = "nccl"
    dtype: str = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    compile: bool = True
    keep_alive: bool = False
    check_nans: bool = False

    # Optional run note
    note: str | None = None

    # Loss weights
    sign_loss_weight: float = 1.0
    log_loss_weight: float = 1.0
    op_loss_weight: float = 1.0

    # Random seeds
    train_seed: int = 42
    val_seed: int = 43


# --------------------------------------------------------------------------- #
# Shallow Attention DAG Predictor Model
# --------------------------------------------------------------------------- #


@dataclass
class ShallowAttentionConfig:
    """Configuration for shallow attention model."""

    vocab_size: int = 50304  # GPT-2 vocab size
    n_embd: int = 768
    n_head: int = 12
    dropout: float = 0.0
    bias: bool = False
    dag_depth: int = 4
    sequence_length: int = 512
    softmax_temperature: float = 20.0


class ShallowAttentionDAGPredictor(nn.Module):
    """
    Standalone model that performs shallow attention over token embeddings
    and uses a DAG predictor for structure prediction.

    Architecture:
    Token IDs -> Embeddings -> Position Embeddings -> Single Attention Layer -> DAG Predictor
    """

    def __init__(self, config: ShallowAttentionConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.sequence_length, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Single attention block for shallow attention
        self.attention_block = Block(config)

        # Layer norm for stability
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # DAG predictor
        self.dag_predictor = DAGPlanPredictor(config, config.softmax_temperature)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, return_internal_state: bool = False):
        """
        Forward pass through shallow attention to DAG predictor.

        Args:
            input_ids: (B, T) token IDs
            return_internal_state: Whether to return internal states from DAG predictor

        Returns:
            DAG predictor outputs (signs, log magnitudes, operation probs)
            Optionally internal states if return_internal_state=True
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_emb = self.wte(input_ids)  # (B, T, n_embd)

        # Position embeddings
        pos = torch.arange(T, device=device)
        pos_emb = self.wpe(pos)  # (T, n_embd)

        # Combine embeddings
        hidden = self.drop(token_emb + pos_emb)  # (B, T, n_embd)

        # Single attention layer (shallow attention)
        hidden = self.attention_block(hidden)  # (B, T, n_embd)

        # Final layer norm
        hidden = self.ln_f(hidden)  # (B, T, n_embd)

        # DAG predictor
        return self.dag_predictor(
            hidden, return_internal_state_or_hidden=return_internal_state
        )

    def get_num_params(self, non_embedding=True):
        """Get number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params


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
    parser.add_argument("config", type=str, help="Path to config file")
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
    """Learning rate scheduler with warmup and cosine decay."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def get_checkpoint_filename(cfg: DAGTrainConfig, iter_num: int) -> str:
    """Generate checkpoint filename."""
    return f"dag_ckpt_{cfg.name}_{iter_num:06d}.pt"


def clean_previous_checkpoints(cfg: DAGTrainConfig) -> None:
    """Remove previous checkpoints if requested."""
    if not cfg.clear_previous_checkpoints:
        return

    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return

    pattern = f"dag_ckpt_{cfg.name}_*.pt"
    removed_count = 0
    for ckpt_file in checkpoint_dir.glob(pattern):
        try:
            ckpt_file.unlink()
            removed_count += 1
        except Exception as e:
            print(f"Warning: Could not remove {ckpt_file}: {e}")

    if removed_count > 0:
        print(f"Removed {removed_count} previous checkpoints")


def find_latest_checkpoint(cfg: DAGTrainConfig) -> Path | None:
    """Find the latest checkpoint for resuming training."""
    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return None

    pattern = f"dag_ckpt_{cfg.name}_*.pt"
    checkpoints = list(checkpoint_dir.glob(pattern))

    if not checkpoints:
        return None

    # Sort by iteration number
    def extract_iter(path):
        try:
            return int(path.stem.split("_")[-1])
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
    target_depths: torch.Tensor,
    cfg: DAGTrainConfig,
    internal_state: Dict[str, torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute loss for DAG structure prediction with optional internal state losses.

    Args:
        pred_sgn: (B, T, num_nodes) predicted signs
        pred_log: (B, T, num_nodes) predicted log magnitudes
        pred_ops: (B, T, depth, n_ops) predicted operation probabilities
        target_sgn: (B, T, num_nodes) target signs
        target_log: (B, T, num_nodes) target log magnitudes
        target_ops: (B, T, depth, n_ops) target operation probabilities (one-hot)
        target_depths: (B,) actual depths for each example
        cfg: Training configuration
        internal_state: Optional internal states for enhanced loss computation

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

    # Create masks for valid positions based on actual depths
    # For now, we'll use all positions since we're doing sequence-level training
    sign_mask = torch.ones_like(sign_loss)
    log_mask = torch.ones_like(log_loss)
    op_mask = torch.ones_like(op_loss)

    # Apply masks and compute weighted losses
    sign_loss = (sign_loss * sign_mask).sum() / sign_mask.sum()
    log_loss = (log_loss * log_mask).sum() / log_mask.sum()
    op_loss = (op_loss * op_mask).sum() / op_mask.sum()

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

    # Note: internal_state parameter is kept for debugging/logging purposes only
    # We do NOT add any regularization losses based on internal states

    return loss_dict


def evaluate_dag_model(
    model: torch.nn.Module,
    val_loader,
    device: str,
    ctx,
    cfg: DAGTrainConfig,
    eval_iters: int,
) -> Dict[str, float]:
    """Evaluate DAG model on validation set."""
    model.eval()

    total_losses = {
        "total_loss": 0.0,
        "sign_loss": 0.0,
        "log_loss": 0.0,
        "op_loss": 0.0,
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
            target_depths = structures["depths"].to(device)  # (B,)

            # Tokenize texts (simple approach for now)
            # In practice, you'd want more sophisticated text encoding
            batch_size = len(texts)

            # Create dummy input tokens (we're not using them for DAG prediction)
            # In a real implementation, you'd encode the text descriptions
            input_tokens = torch.randint(
                0, 1000, (batch_size, cfg.sequence_length), device=device
            )

            # Forward pass through shallow attention DAG predictor model
            with ctx:
                # Use the standalone shallow attention model
                pred_sgn, pred_log, pred_ops = model(
                    input_tokens, return_internal_state=False
                )

                # Average predictions over sequence length for structure prediction
                pred_sgn = pred_sgn.mean(dim=1)  # (B, num_nodes_pred)
                pred_log = pred_log.mean(dim=1)  # (B, num_nodes_pred)
                pred_ops = pred_ops.mean(dim=1)  # (B, depth_pred, n_ops)

                # Ensure target and prediction tensors have compatible shapes
                target_nodes = target_sgn.size(1)
                pred_nodes = pred_sgn.size(1)
                target_depth = target_ops.size(1)
                pred_depth = pred_ops.size(1)

                # Resize predictions to match targets if needed
                if pred_nodes != target_nodes:
                    if pred_nodes > target_nodes:
                        # Truncate predictions
                        pred_sgn = pred_sgn[:, :target_nodes]
                        pred_log = pred_log[:, :target_nodes]
                    else:
                        # Pad predictions with zeros
                        pad_nodes = target_nodes - pred_nodes
                        pred_sgn = F.pad(pred_sgn, (0, pad_nodes))
                        pred_log = F.pad(pred_log, (0, pad_nodes))

                if pred_depth != target_depth:
                    if pred_depth > target_depth:
                        # Truncate predictions
                        pred_ops = pred_ops[:, :target_depth]
                    else:
                        # Pad predictions with zeros
                        pad_depth = target_depth - pred_depth
                        pred_ops = F.pad(pred_ops, (0, 0, 0, pad_depth))

                # Add sequence dimension to match loss function expectations
                pred_sgn = pred_sgn.unsqueeze(1)  # (B, 1, num_nodes)
                pred_log = pred_log.unsqueeze(1)  # (B, 1, num_nodes)
                pred_ops = pred_ops.unsqueeze(1)  # (B, 1, depth, n_ops)

                target_sgn = target_sgn.unsqueeze(1)  # (B, 1, num_nodes)
                target_log = target_log.unsqueeze(1)  # (B, 1, num_nodes)
                target_ops = target_ops.unsqueeze(1)  # (B, 1, depth, n_ops)

                # Compute losses (no internal state for evaluation)
                losses = compute_dag_structure_loss(
                    pred_sgn,
                    pred_log,
                    pred_ops,
                    target_sgn,
                    target_log,
                    target_ops,
                    target_depths,
                    cfg,
                    internal_state=None,
                )

                # Accumulate losses
                for key, value in losses.items():
                    total_losses[key] += value.item()

                num_batches += 1

    # Average losses
    if num_batches > 0:
        for key in total_losses:
            total_losses[key] /= num_batches

    model.train()
    return total_losses


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

    # Clean previous checkpoints
    clean_previous_checkpoints(cfg)

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
        train_seed=cfg.train_seed,
        val_seed=cfg.val_seed,
    )

    # --------------------------------------------------------------------- #
    # Model creation
    # --------------------------------------------------------------------- #
    print(
        f"[{time.time() - setup_start:.2f}s] Initializing shallow attention DAG predictor model"
    )

    # Create config for shallow attention model
    model_config = ShallowAttentionConfig(
        vocab_size=50304,  # Standard GPT-2 vocab size
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        dropout=cfg.dropout,
        bias=cfg.bias,
        dag_depth=cfg.dag_depth,
        sequence_length=cfg.sequence_length,
        softmax_temperature=20.0,
    )

    if cfg.init_from == "scratch":
        print(
            f"[{time.time() - setup_start:.2f}s] Initializing shallow attention model from scratch"
        )
        model = ShallowAttentionDAGPredictor(model_config)
        iter_num, best_val_loss = 0, 1e9
    elif cfg.init_from == "resume":
        print(f"[{time.time() - setup_start:.2f}s] Resuming from checkpoint")
        ckpt_path = find_latest_checkpoint(cfg)
        if ckpt_path is None:
            raise ValueError(f"No checkpoint found for resuming")
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Create model with saved config
        saved_config = ShallowAttentionConfig(**checkpoint["model_config"])
        model = ShallowAttentionDAGPredictor(saved_config)
        state_dict = {
            k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()
        }
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    else:
        raise ValueError(f"Unsupported init_from {cfg.init_from}")

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

    if cfg.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])

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
    print(f"[{time.time() - setup_start:.2f}s] Starting training loop")

    t0 = time.time()
    raw_model = model.module if ddp else model
    running_mfu = -1.0

    while iter_num <= cfg.max_iters:
        # Learning rate scheduling
        lr = get_lr(iter_num, cfg=cfg) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluation
        if iter_num % cfg.eval_interval == 0 and master_process:
            print(
                f"[{time.time() - setup_start:.2f}s] Running evaluation at iter {iter_num}"
            )

            val_losses = evaluate_dag_model(
                raw_model, val_loader, device, ctx, cfg, cfg.eval_iters
            )

            # Build validation log message
            val_log_msg = (
                f"step {iter_num}: val_total {val_losses['total_loss']:.4f}, "
                f"val_sign {val_losses['sign_loss']:.4f}, "
                f"val_log {val_losses['log_loss']:.4f}, "
                f"val_op {val_losses['op_loss']:.4f}"
            )
            print(val_log_msg)

            # Log to wandb
            if run is not None:
                log_dict = {
                    "iter": iter_num,
                    "lr": lr,
                    "val/total_loss": val_losses["total_loss"],
                    "val/sign_loss": val_losses["sign_loss"],
                    "val/log_loss": val_losses["log_loss"],
                    "val/op_loss": val_losses["op_loss"],
                }

                wandb.log(log_dict, step=iter_num, commit=False)

            # Save checkpoint
            if val_losses["total_loss"] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = val_losses["total_loss"]
                if iter_num > 0:
                    ckpt = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": model_config.__dict__,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": cfg.__dict__,
                    }
                    checkpoint_filename = get_checkpoint_filename(cfg, iter_num)
                    checkpoint_path = Path(CHECKPOINT_DIR) / checkpoint_filename
                    if master_process:
                        print(f"Saving checkpoint: {checkpoint_path}")
                    _safe_torch_save(ckpt, checkpoint_path)

        if iter_num == 0 and cfg.eval_only:
            break

        # Training step
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Get training batch
        texts, structures = next(train_loader)

        # Move to device
        target_sgn = structures["initial_sgn"].to(device)
        target_log = structures["initial_log"].to(device)
        target_ops = structures["operation_probs"].to(device)
        target_depths = structures["depths"].to(device)

        loss_accum = {
            "total_loss": 0.0,
            "sign_loss": 0.0,
            "log_loss": 0.0,
            "op_loss": 0.0,
        }

        for micro_step in range(cfg.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == cfg.gradient_accumulation_steps - 1
                )

            with ctx:
                # Tokenize texts properly (instead of dummy tokens)
                batch_size = len(texts)

                # For now, create random input tokens (in real implementation, would tokenize texts)
                # TODO: Implement proper text tokenization from DAG structure descriptions
                input_tokens = torch.randint(
                    0,
                    min(1000, raw_model.config.vocab_size),
                    (batch_size, cfg.sequence_length),
                    device=device,
                )

                # Forward pass through shallow attention DAG predictor model
                pred_sgn, pred_log, pred_ops, internal_state = raw_model(
                    input_tokens, return_internal_state=True
                )

                # Average over sequence dimension
                pred_sgn_avg = pred_sgn.mean(dim=1)  # (B, num_nodes_pred)
                pred_log_avg = pred_log.mean(dim=1)  # (B, num_nodes_pred)
                pred_ops_avg = pred_ops.mean(dim=1)  # (B, depth_pred, n_ops)

                # Ensure target and prediction tensors have compatible shapes
                target_nodes = target_sgn.size(1)
                pred_nodes = pred_sgn_avg.size(1)
                target_depth = target_ops.size(1)
                pred_depth = pred_ops_avg.size(1)

                # Resize predictions to match targets if needed
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

                # Compute loss with internal states
                losses = compute_dag_structure_loss(
                    pred_sgn_seq,
                    pred_log_seq,
                    pred_ops_seq,
                    target_sgn_seq,
                    target_log_seq,
                    target_ops_seq,
                    target_depths,
                    cfg,
                    internal_state,
                )

                loss = losses["total_loss"] / cfg.gradient_accumulation_steps

                # Accumulate losses for logging
                for key, value in losses.items():
                    loss_accum[key] += value.item() / cfg.gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

        # Optimization step
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(dag_params, cfg.grad_clip)

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
                log_dict = {
                    "iter": iter_num,
                    "train/total_loss": loss_accum["total_loss"],
                    "train/sign_loss": loss_accum["sign_loss"],
                    "train/log_loss": loss_accum["log_loss"],
                    "train/op_loss": loss_accum["op_loss"],
                    "lr": lr,
                }

                # Add internal losses to wandb logging
                for key, value in loss_accum.items():
                    if (
                        key.endswith("_loss")
                        and key not in log_dict
                        and key != "total_loss"
                    ):
                        log_dict[f"train/{key}"] = value

                wandb.log(log_dict, step=iter_num, commit=True)

        iter_num += 1

    # Cleanup
    if ddp:
        destroy_process_group()
    run.finish()

    # Stop RunPod if needed
    if os.getenv("RUNPOD_POD_ID") and not getattr(cfg, "keep_alive", False):
        runpod_service.stop_runpod()


def main() -> None:
    """Main entry point."""
    check_python_version()

    parser = parse_args()
    args, overrides = parser.parse_known_args()

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

    if args.use_runpod:
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

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    train_predictor(cfg, wandb_run_id=args.wandb_run_id)


if __name__ == "__main__":
    main()
