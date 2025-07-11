from __future__ import annotations

import argparse
import math
import os
import random
import runpy
import string
import time
from ast import literal_eval
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

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


class CheckpointLoadError(Exception):
    """Raised when checkpoint loading fails."""


def _list_available_checkpoints(config_name: str | None = None) -> List[str]:
    """List available checkpoint files in the checkpoint directory.
    
    Args:
        config_name: Optional config name to filter checkpoints. If None, lists all checkpoints.
        
    Returns:
        List of checkpoint filenames (without full path)
    """
    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return []
    
    if config_name:
        safe_name = "".join(c for c in config_name if c.isalnum() or c in ("-", "_"))
        patterns = [
            f"ckpt_{safe_name}_*.pt",
            f"ckpt_{safe_name}_*.safetensors",
            f"dag_ckpt_*_{safe_name}_*.pt",
            f"dag_ckpt_*_{safe_name}_*.safetensors",
        ]
    else:
        patterns = [
            "ckpt_*.pt",
            "ckpt_*.safetensors", 
            "dag_ckpt_*.pt",
            "dag_ckpt_*.safetensors",
        ]
    
    checkpoint_files = []
    for pattern in patterns:
        checkpoint_files.extend(checkpoint_dir.glob(pattern))
    
    return [f.name for f in sorted(checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True)]


def load_checkpoint_from_path(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    expected_keys: Optional[List[str]] = None,
) -> Dict:
    """Load a checkpoint from a specific path with validation.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint on
        expected_keys: List of required keys in the checkpoint

    Returns:
        Dictionary containing the loaded checkpoint data

    Raises:
        CheckpointLoadError: If the checkpoint cannot be loaded or is invalid
    """
    checkpoint_path = Path(checkpoint_path)

    # Check if checkpoint exists
    if not checkpoint_path.exists():
        # List available checkpoints for a helpful error message
        available_checkpoints = _list_available_checkpoints()
        if available_checkpoints:
            checkpoint_list = "\n".join(f"  - {name}" for name in available_checkpoints[:10])
            if len(available_checkpoints) > 10:
                checkpoint_list += f"\n  ... and {len(available_checkpoints) - 10} more"
            error_msg = (
                f"Checkpoint file not found: {checkpoint_path}\n"
                f"Available checkpoints in {CHECKPOINT_DIR}:\n{checkpoint_list}"
            )
        else:
            error_msg = (
                f"Checkpoint file not found: {checkpoint_path}\n"
                f"No checkpoints found in {CHECKPOINT_DIR}"
            )
        
        print(f"ERROR: {error_msg}")

        # Kill runpod instance if running on runpod
        if os.getenv("RUNPOD_POD_ID"):
            try:
                import runpod_service

                print("Stopping RunPod instance due to missing checkpoint...")
                runpod_service.stop_runpod()
            except Exception as e:
                print(f"Warning: Failed to stop RunPod instance: {e}")

        raise CheckpointLoadError(error_msg)

    # Try to load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Successfully loaded checkpoint from: {checkpoint_path}")
    except Exception as e:
        error_msg = f"Failed to load checkpoint from {checkpoint_path}: {e}"
        print(f"ERROR: {error_msg}")

        # Kill runpod instance if running on runpod
        if os.getenv("RUNPOD_POD_ID"):
            try:
                import runpod_service

                print("Stopping RunPod instance due to checkpoint loading error...")
                runpod_service.stop_runpod()
            except Exception as stop_e:
                print(f"Warning: Failed to stop RunPod instance: {stop_e}")

        raise CheckpointLoadError(error_msg) from e

    # Validate checkpoint structure
    if expected_keys:
        missing_keys = [key for key in expected_keys if key not in checkpoint]
        if missing_keys:
            error_msg = f"Checkpoint missing required keys: {missing_keys}"
            print(f"ERROR: {error_msg}")

            # Kill runpod instance if running on runpod
            if os.getenv("RUNPOD_POD_ID"):
                try:
                    import runpod_service

                    print("Stopping RunPod instance due to invalid checkpoint...")
                    runpod_service.stop_runpod()
                except Exception as stop_e:
                    print(f"Warning: Failed to stop RunPod instance: {stop_e}")

            raise CheckpointLoadError(error_msg)

    return checkpoint


def handle_checkpoint_loading(
    cfg: BaseConfig,
    device: str = "cpu",
    gpt_model_class=None,
    expected_keys: Optional[List[str]] = None,
) -> Tuple[Optional[Dict], int, float]:
    """Handle checkpoint loading based on init_from configuration.

    Args:
        cfg: Configuration object with init_from field
        device: Device to load checkpoint on
        gpt_model_class: GPT model class for pretrained loading (optional)
        expected_keys: List of required keys in checkpoint (optional)

    Returns:
        Tuple of (checkpoint_dict, iter_num, best_val_loss)

    Raises:
        CheckpointLoadError: If checkpoint loading fails
        ValueError: If init_from option is unsupported
        
    Supported init_from values:
        - "scratch": Initialize model from scratch
        - "resume" or "latest": Load the latest checkpoint for this config name
        - "gpt2", "gpt2-medium", etc.: Load pretrained GPT-2 weights
        - Any other string: Treat as a direct path to a checkpoint file
    """
    init_from = cfg.init_from

    # Handle different init_from options
    if init_from == "scratch":
        print("Initializing model from scratch")
        return None, 0, 1e9

    elif init_from == "resume" or init_from == "latest":
        print("Resuming from latest checkpoint")
        ckpt_path = find_latest_checkpoint(cfg)
        if ckpt_path is None:
            # List available checkpoints for a helpful error message
            available_checkpoints = _list_available_checkpoints(cfg.name)
            if available_checkpoints:
                checkpoint_list = "\n".join(f"  - {name}" for name in available_checkpoints[:10])
                if len(available_checkpoints) > 10:
                    checkpoint_list += f"\n  ... and {len(available_checkpoints) - 10} more"
                error_msg = (
                    f"No checkpoint found for config name '{cfg.name}' in {CHECKPOINT_DIR}\n"
                    f"Available checkpoints for '{cfg.name}':\n{checkpoint_list}\n"
                    f"You can use init_from='scratch' to start from scratch, or specify a specific checkpoint path."
                )
            else:
                all_checkpoints = _list_available_checkpoints()
                if all_checkpoints:
                    checkpoint_list = "\n".join(f"  - {name}" for name in all_checkpoints[:10])
                    if len(all_checkpoints) > 10:
                        checkpoint_list += f"\n  ... and {len(all_checkpoints) - 10} more"
                    error_msg = (
                        f"No checkpoint found for config name '{cfg.name}' in {CHECKPOINT_DIR}\n"
                        f"Available checkpoints (all configs):\n{checkpoint_list}\n"
                        f"You can use init_from='scratch' to start from scratch, or specify a specific checkpoint path."
                    )
                else:
                    error_msg = (
                        f"No checkpoint found for config name '{cfg.name}' in {CHECKPOINT_DIR}\n"
                        f"No checkpoints found in {CHECKPOINT_DIR}\n"
                        f"You can use init_from='scratch' to start from scratch."
                    )
            
            print(f"ERROR: {error_msg}")

            # Kill runpod instance if running on runpod
            if os.getenv("RUNPOD_POD_ID"):
                try:
                    import runpod_service

                    print("Stopping RunPod instance due to missing checkpoint...")
                    runpod_service.stop_runpod()
                except Exception as e:
                    print(f"Warning: Failed to stop RunPod instance: {e}")

            raise CheckpointLoadError(error_msg)

        checkpoint = load_checkpoint_from_path(ckpt_path, device, expected_keys)
        return (
            checkpoint,
            checkpoint.get("iter_num", 0),
            checkpoint.get("best_val_loss", 1e9),
        )

    elif init_from.startswith("gpt2"):
        print(f"Loading GPT-2 weights: {init_from}")
        if gpt_model_class is None:
            raise ValueError("GPT model class required for GPT-2 loading")

        # This returns a model, not a checkpoint dict
        # The calling code will handle the model directly
        return None, 0, 1e9

    else:
        # Assume it's a direct path to a checkpoint
        print(f"Loading checkpoint from path: {init_from}")
        checkpoint_path = Path(init_from)

        # Support both absolute and relative paths
        if not checkpoint_path.is_absolute():
            # Try relative to checkpoint directory first
            checkpoint_path = Path(CHECKPOINT_DIR) / checkpoint_path
            if not checkpoint_path.exists():
                # Try relative to current working directory
                checkpoint_path = Path(init_from)

        checkpoint = load_checkpoint_from_path(checkpoint_path, device, expected_keys)
        return (
            checkpoint,
            checkpoint.get("iter_num", 0),
            checkpoint.get("best_val_loss", 1e9),
        )


# --------------------------------------------------------------------------- #
# Safe checkpoint saving with retry
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

    # Evaluation
    eval_only: bool = False
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


def load_config_file(path: str) -> Dict[str, object]:
    """Executes a python config file and returns its public symbols."""
    cfg_dict = runpy.run_path(path)
    return {k: v for k, v in cfg_dict.items() if not k.startswith("_")}


def update_config(cfg: BaseConfig, data: Dict[str, object]) -> None:
    """Overwrite fields in <cfg> with matching keys from <data>."""
    for f in fields(cfg):
        if f.name in data:
            setattr(cfg, f.name, data[f.name])


def apply_overrides(cfg: BaseConfig, overrides: List[str]) -> None:
    """Apply --key=value CLI overrides and boolean flags."""
    for arg in overrides:
        if not arg.startswith("--"):
            raise ValueError(f"Invalid override: {arg}")

        if "=" not in arg:
            key = arg[2:].replace("-", "_")
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown config key: {key}")
            cur = getattr(cfg, key)
            if not isinstance(cur, bool):
                raise ValueError(
                    f"Flag {arg} can only be used with boolean config keys"
                )
            setattr(cfg, key, True)
        else:
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
    parser.add_argument("--wandb-run-id", help="Resume existing wandb run with this ID")
    parser.add_argument("--note", help="Optional note to append to config name")
    parser.add_argument("--subset", type=float)
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep pod alive after training (disables auto-stop)",
    )
    return parser


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #
def get_lr(it: int, *, cfg: BaseConfig) -> float:
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
    cfg: BaseConfig,
    iter_num: int,
    model_name: str | None = None,
    val_acc: float | None = None,
) -> str:
    """Generate checkpoint filename with config name, iteration number, and optional validation accuracy."""
    name_part = model_name or cfg.name
    safe_name = "".join(c for c in name_part if c.isalnum() or c in ("-", "_"))

    if val_acc is not None:
        # Format accuracy as percentage with 2 decimal places
        acc_str = f"{val_acc * 100:.2f}acc"
        return f"ckpt_{safe_name}_{iter_num}_{acc_str}.pt"
    else:
        return f"ckpt_{safe_name}_{iter_num}.pt"


def clean_previous_checkpoints(cfg: BaseConfig, model_name: str | None = None) -> None:
    """Remove all previous checkpoint files for this config name."""
    if not cfg.clear_previous_checkpoints:
        print("Skipping checkpoint cleanup")
        return

    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return

    name_part = model_name or cfg.name
    safe_name = "".join(c for c in name_part if c.isalnum() or c in ("-", "_"))
    # Updated pattern to handle both old and new filename formats
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


def find_latest_checkpoint(
    cfg: BaseConfig, model_name: str | None = None, any_run: bool = False
) -> Path | None:
    """Find the latest checkpoint file for this config name."""
    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return None

    if any_run:
        pattern = "ckpt_*.pt"
    else:
        name_part = model_name or cfg.name
        safe_name = "".join(c for c in name_part if c.isalnum() or c in ("-", "_"))
        pattern = f"ckpt_{safe_name}_*.pt"

    checkpoint_files = list(checkpoint_dir.glob(pattern))
    if not checkpoint_files:
        return None

    latest_file = None
    latest_iter = -1

    for ckpt_file in checkpoint_files:
        try:
            parts = ckpt_file.stem.split("_")
            # Handle both old format (ckpt_name_iter) and new format (ckpt_name_iter_acc)
            if len(parts) >= 3:
                # Extract iteration number - it's either the last part or second to last
                iter_part = (
                    parts[-2]
                    if len(parts) >= 4 and parts[-1].endswith("acc")
                    else parts[-1]
                )
                iter_num = int(iter_part)
                if iter_num > latest_iter:
                    latest_iter = iter_num
                    latest_file = ckpt_file
        except (ValueError, IndexError):
            continue

    return latest_file


def _check_for_nonfinite(named_tensors_iter, label: str) -> None:
    """Scan an iterator of (name, tensor) pairs and raise if any tensor
    contains NaN or Inf. Prints summary and a few offending indices first.
    """
    for name, t in named_tensors_iter:
        if t is None:
            continue
        if torch.isfinite(t).all():
            continue
        nan_cnt = int(torch.isnan(t).sum())
        inf_cnt = int(torch.isinf(t).sum())
        print(
            f"[{label} NAN] {name} → NaN:{nan_cnt} Inf:{inf_cnt} shape={tuple(t.shape)}"
        )

        bad_idx = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5]
        for j, coord in enumerate(bad_idx):
            coord_tuple = tuple(coord.tolist())
            bad_val = t[coord_tuple].item()
            print(f"   [{j}] idx={coord_tuple}  val={bad_val}")
        raise RuntimeError(f"{label} contains non-finite values (see log above)")
