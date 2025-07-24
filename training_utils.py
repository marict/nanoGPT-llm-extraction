from __future__ import annotations

import argparse
import math
import os
import random
import runpy
import shutil
import string
import subprocess
import time
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------

# Checkpoint directory
CHECKPOINT_DIR = (
    "/runpod-volume/checkpoints" if os.path.exists("/runpod-volume") else "checkpoints"
)


# --------------------------------------------------------------------------- #
# Git utilities for debugging and tracking
# --------------------------------------------------------------------------- #


def log_git_commit_info() -> None:
    """Log current git commit information for debugging and tracking."""
    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        commit_hash = result.stdout.strip()

        # Get current branch
        try:
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            branch = branch_result.stdout.strip()
        except subprocess.CalledProcessError:
            branch = "unknown"

        # Get short commit message (first line only, limit to 60 chars)
        try:
            msg_result = subprocess.run(
                ["git", "log", "-1", "--format=%s"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            commit_msg = msg_result.stdout.strip()[:60]
            if len(msg_result.stdout.strip()) > 60:
                commit_msg += "..."
        except subprocess.CalledProcessError:
            commit_msg = "no message"

        print(f"Repository info: {commit_hash[:8]} on {branch} - {commit_msg}")

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        print(f"Could not retrieve git information: {e}")
    except Exception as e:
        print(f"Unexpected error getting git info: {e}")


# --------------------------------------------------------------------------- #
# Disk space monitoring utilities
# --------------------------------------------------------------------------- #


def get_disk_usage_percent(path: str = ".") -> float:
    """Get disk usage percentage for the given path.

    Args:
        path: Path to check disk usage for (defaults to current directory)

    Returns:
        Disk usage as a percentage (0-100)
    """
    try:
        _, total, free = shutil.disk_usage(path)
        used = total - free
        usage_percent = (used / total) * 100.0
        return usage_percent
    except Exception as e:
        print(f"Warning: Could not check disk usage for {path}: {e}")
        return 0.0


def check_disk_space_emergency(path: str = ".", threshold: float = 95.0) -> bool:
    """Check if disk usage exceeds emergency threshold.

    Args:
        path: Path to check disk usage for
        threshold: Emergency threshold percentage (default: 95%)

    Returns:
        True if disk usage exceeds threshold, False otherwise
    """
    usage_percent = get_disk_usage_percent(path)

    if usage_percent >= threshold:
        print(
            f"🚨 DISK SPACE EMERGENCY: {usage_percent:.1f}% usage exceeds {threshold}% threshold!"
        )
        print(f"📊 Disk usage details for {path}:")

        try:
            _, total, free = shutil.disk_usage(path)
            print(f"  Total: {total / (1024**3):.1f} GB")
            print(f"  Free:  {free / (1024**3):.1f} GB")
            print(f"  Used:  {(total - free) / (1024**3):.1f} GB")
        except Exception as e:
            print(f"  Could not get detailed disk usage: {e}")

        return True

    return False


def cleanup_disk_space_emergency():
    """Emergency disk space cleanup procedures."""
    print("🧹 Attempting emergency disk space cleanup...")

    cleanup_paths = ["/tmp", str(Path.home() / ".cache"), "/var/tmp"]

    cleaned_mb = 0

    for cleanup_path in cleanup_paths:
        if os.path.exists(cleanup_path):
            try:
                # Clean temporary files older than 1 hour
                for root, dirs, files in os.walk(cleanup_path):
                    for file in files:
                        file_path = Path(root) / file
                        try:
                            if file_path.stat().st_mtime < (
                                time.time() - 3600
                            ):  # 1 hour old
                                size_mb = file_path.stat().st_size / (1024 * 1024)
                                file_path.unlink()
                                cleaned_mb += size_mb
                        except Exception:
                            continue  # Skip files we can't delete
            except Exception as e:
                print(f"Warning: Could not clean {cleanup_path}: {e}")

    print(f"🧹 Emergency cleanup freed {cleaned_mb:.1f} MB")
    return cleaned_mb


# --------------------------------------------------------------------------- #
# Configuration utilities
# --------------------------------------------------------------------------- #


def generate_run_name(cfg) -> str:
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


@dataclass
class BaseConfig:
    """Base class for training configurations."""

    # Meta
    name: str
    note: str | None = None

    # Checkpointing
    init_from: str = "scratch"
    always_save_checkpoint: bool = True
    save_best: bool = False
    clear_previous_checkpoints: bool = False
    # When True, continuously overwrite the latest checkpoint file from this
    # run instead of creating a new file each time. A unique sub-directory
    # (based on the current wandb run name) will be used so that checkpoints
    # from other runs are left untouched.
    overwrite_previous: bool = True

    # Evaluation
    eval_once: bool = False
    eval_interval: int = 250
    log_interval: int = 1
    eval_iters: int = 200

    # DDP / System
    backend: str = "nccl"
    dtype: str = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    compile: bool = True

    # Runpod
    keep_alive: bool = False

    # Debugging
    check_nans: bool = False

    # When True, reset the stored iteration counter to zero when reloading a
    # checkpoint. This lets you start a fresh learning-rate schedule while
    # still initialising the model with pre-trained weights.
    reload_reset_iters: bool = False


def load_config_file(path: str) -> Dict[str, object]:
    """Load configuration from a Python file."""
    return runpy.run_path(path)


def update_config(cfg: BaseConfig, data: Dict[str, object]) -> None:
    """Update configuration object with dictionary data."""
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def apply_overrides(cfg: BaseConfig, overrides: List[str]) -> None:
    """Apply command-line overrides to configuration."""
    for override in overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            key = key.lstrip("-")
            if hasattr(cfg, key):
                field_type = type(getattr(cfg, key))
                if field_type == bool:
                    setattr(cfg, key, value.lower() in ("true", "1", "yes"))
                elif field_type in (int, float):
                    setattr(cfg, key, field_type(value))
                elif field_type == list:
                    # Handle list fields by parsing as Python literal
                    try:
                        setattr(cfg, key, literal_eval(value))
                    except (ValueError, SyntaxError):
                        # Fallback: split by comma
                        setattr(cfg, key, [v.strip() for v in value.split(",")])
                else:
                    setattr(cfg, key, value)


def parse_args() -> argparse.ArgumentParser:
    """Create argument parser for training scripts."""
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument("--subset", type=float, help="Fraction of dataset to use")
    parser.add_argument("--dag-depth", type=int, help="DAG depth")
    parser.add_argument("--wandb-api-key", type=str, help="Weights & Biases API key")
    parser.add_argument("--wandb-run-id", type=str, help="Resume specific W&B run")
    parser.add_argument("--note", type=str, help="Optional note for run name")
    parser.add_argument(
        "--use-runpod", action="store_true", help="Use RunPod for training"
    )
    parser.add_argument("--gpu-type", type=str, help="GPU type for RunPod")
    parser.add_argument(
        "--keep-alive", action="store_true", help="Keep instance alive after training"
    )
    return parser


def get_lr(it: int, *, cfg: BaseConfig) -> float:
    """Calculate learning rate with warmup and decay."""
    # Validate configuration first
    if not hasattr(cfg, "learning_rate"):
        raise AttributeError("Config must have 'learning_rate' attribute")
    if not hasattr(cfg, "warmup_iters"):
        raise AttributeError("Config must have 'warmup_iters' attribute")
    if not hasattr(cfg, "lr_decay_iters"):
        raise AttributeError("Config must have 'lr_decay_iters' attribute")
    if not hasattr(cfg, "min_lr"):
        raise AttributeError("Config must have 'min_lr' attribute")

    # 1. Determine base learning rate
    if it < cfg.warmup_iters:
        # Linear warmup
        if cfg.warmup_iters > 0:
            # Using it + 1 to make it 1-based for warmup, matches test expectations
            base_lr = cfg.learning_rate * (it + 1) / cfg.warmup_iters
        else:
            base_lr = cfg.learning_rate
    elif it > cfg.lr_decay_iters:
        # Past decay phase
        base_lr = cfg.min_lr
    else:
        # In between, might be constant or cosine decay
        if not getattr(cfg, "decay_lr", True):
            base_lr = cfg.learning_rate
        else:
            # Cosine decay
            decay_ratio = (it - cfg.warmup_iters) / (
                cfg.lr_decay_iters - cfg.warmup_iters
            )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            base_lr = cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    # 2. Apply cyclical modulation if enabled and not in warmup
    final_lr = base_lr
    if getattr(cfg, "use_cyclical_lr", False) and it >= cfg.warmup_iters:
        period = getattr(cfg, "cyclical_lr_period", 1000)
        amplitude = getattr(cfg, "cyclical_lr_amplitude", 0.1)

        # Cycle starts after warmup
        progress_in_decay = it - cfg.warmup_iters
        cycle_progress = (progress_in_decay % period) / period
        cyclical_factor = 1.0 + amplitude * math.sin(2 * math.pi * cycle_progress)

        final_lr *= cyclical_factor

    # 3. Final clamping
    # During warmup, we want the linear ramp-up, so no min_lr clamping yet
    if it < cfg.warmup_iters:
        return final_lr

    return max(cfg.min_lr, final_lr)


def get_checkpoint_filename(
    cfg: BaseConfig,
    iter_num: int,
    model_name: str | None = None,
    val_acc: float | None = None,
) -> str:
    """Generate checkpoint filename based on config and iteration."""
    base_name = model_name if model_name else cfg.name
    safe_name = "".join(c for c in base_name if c.isalnum() or c in ("-", "_"))

    if val_acc is not None:
        acc_str = f"{val_acc * 100:.2f}acc"
        return f"ckpt_{safe_name}_{iter_num}_{acc_str}.pt"
    else:
        return f"ckpt_{safe_name}_{iter_num}.pt"


def check_for_nonfinite(
    named_tensors_iter: Iterable[tuple[str, torch.Tensor]], label: str
) -> None:
    """Check tensors for NaN/Inf values and print detailed diagnostic info."""
    for name, tensor in named_tensors_iter:
        if tensor is None:
            continue
        if torch.isnan(tensor).any():
            print(
                f"[{label} NAN] {name}  →  min={tensor.min():.3e}  max={tensor.max():.3e}"
            )
        elif torch.isinf(tensor).any():
            print(
                f"[{label} INF] {name}  →  min={tensor.min():.3e}  max={tensor.max():.3e}"
            )
