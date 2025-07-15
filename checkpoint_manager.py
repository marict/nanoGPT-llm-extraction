"""
Unified checkpoint management for nanoGPT training scripts.

This module provides centralized checkpoint loading, saving, and management
functionality that works for both regular training (train.py) and DAG
predictor training (train_predictor.py).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

# Optional safetensors for pure-tensor checkpoints
try:
    import safetensors.torch as _st  # type: ignore

    _HAVE_ST = True
except ModuleNotFoundError:
    _HAVE_ST = False

# Import model classes for initialization
from models.dag_model import GPT, OP_NAMES, GPTConfig
from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel

# Checkpoint directory
CHECKPOINT_DIR = (
    "/runpod-volume/checkpoints" if os.path.exists("/runpod-volume") else "checkpoints"
)


class CheckpointLoadError(Exception):
    """Raised when checkpoint loading fails."""


class CheckpointSaveError(Exception):
    """Raised when saving a checkpoint fails after retries."""


class CheckpointManager:
    """Unified checkpoint management for all training scripts."""

    def __init__(self, checkpoint_type: str = "regular"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_type: Either "regular" for train.py or "dag" for train_predictor.py
                           Note: Both use the same checkpoint naming pattern (ckpt_)
        """
        self.checkpoint_type = checkpoint_type

    @property
    def checkpoint_dir(self) -> Path:
        """Get the current checkpoint directory (allows for dynamic updates)."""
        return Path(CHECKPOINT_DIR)

    def _get_checkpoint_patterns(
        self, config_name: str, model_name: str = None
    ) -> List[str]:
        """Get checkpoint file patterns based on config name.

        Both regular and DAG training use the same checkpoint naming: ckpt_{config_name}_*
        """
        safe_name = "".join(c for c in config_name if c.isalnum() or c in ("-", "_"))

        patterns = [
            f"ckpt_{safe_name}_*.pt",
            f"ckpt_{safe_name}_*.safetensors",
        ]

        return patterns

    def _list_available_checkpoints(
        self, config_name: str = None, model_name: str = None
    ) -> List[str]:
        """List available checkpoint files in the checkpoint directory."""
        if not self.checkpoint_dir.exists():
            return []

        if config_name:
            patterns = self._get_checkpoint_patterns(config_name, model_name)
        else:
            # List all checkpoints (both regular and DAG use same pattern)
            patterns = ["ckpt_*.pt", "ckpt_*.safetensors"]

        checkpoint_files = []
        for pattern in patterns:
            checkpoint_files.extend(self.checkpoint_dir.glob(pattern))

        return [
            f.name
            for f in sorted(
                checkpoint_files, key=lambda x: x.stat().st_mtime, reverse=True
            )
        ]

    def _extract_iteration_number(self, checkpoint_path: Path) -> int:
        """Extract iteration number from checkpoint filename."""
        try:
            parts = checkpoint_path.stem.split("_")

            # Format: ckpt_config_iter[_acc]
            if len(parts) >= 3:
                iter_part = (
                    parts[-2]
                    if len(parts) >= 4 and parts[-1].endswith("acc")
                    else parts[-1]
                )
                return int(iter_part)
            return 0
        except (ValueError, IndexError):
            return 0

    def find_latest_checkpoint(
        self, config_name: str, model_name: str = None, any_run: bool = False
    ) -> Path | None:
        """Find the latest checkpoint by iteration number."""
        if not self.checkpoint_dir.exists():
            return None

        if any_run:
            patterns = ["ckpt_*.pt", "ckpt_*.safetensors"]
            checkpoint_files = []
            for pattern in patterns:
                checkpoint_files.extend(self.checkpoint_dir.glob(pattern))
        else:
            patterns = self._get_checkpoint_patterns(config_name, model_name)
            checkpoint_files = []
            for pattern in patterns:
                checkpoint_files.extend(self.checkpoint_dir.glob(pattern))

        if not checkpoint_files:
            return None

        # Find checkpoint with highest iteration number
        latest_file = None
        latest_iter = -1

        for ckpt_file in checkpoint_files:
            iter_num = self._extract_iteration_number(ckpt_file)
            if iter_num > latest_iter:
                latest_iter = iter_num
                latest_file = ckpt_file

        return latest_file

    def find_best_checkpoint(
        self, config_name: str, model_name: str = None
    ) -> Path | None:
        """Find the best checkpoint (saved with 'best' in the name)."""
        if not self.checkpoint_dir.exists():
            return None

        patterns = self._get_checkpoint_patterns(config_name, model_name)
        best_patterns = []

        for pattern in patterns:
            # Look for checkpoints with 'best' in the name
            best_pattern = pattern.replace("_*.", "_best*.")
            best_patterns.append(best_pattern)

        checkpoint_files = []
        for pattern in best_patterns:
            checkpoint_files.extend(self.checkpoint_dir.glob(pattern))

        if not checkpoint_files:
            return None

        # If multiple best checkpoints, return the most recent by modification time
        return max(checkpoint_files, key=lambda x: x.stat().st_mtime)

    def load_checkpoint_from_path(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        expected_keys: Optional[List[str]] = None,
    ) -> Dict:
        """Load a checkpoint from a specific path with validation."""
        checkpoint_path = Path(checkpoint_path)

        # Check if checkpoint exists
        if not checkpoint_path.exists():
            # List available checkpoints for a helpful error message
            available_checkpoints = self._list_available_checkpoints()
            if available_checkpoints:
                checkpoint_list = "\n".join(
                    f"  - {name}" for name in available_checkpoints[:10]
                )
                if len(available_checkpoints) > 10:
                    checkpoint_list += (
                        f"\n  ... and {len(available_checkpoints) - 10} more"
                    )
                error_msg = (
                    f"Checkpoint file not found: {checkpoint_path}\n"
                    f"Available {self.checkpoint_type} checkpoints in {CHECKPOINT_DIR}:\n{checkpoint_list}"
                )
            else:
                error_msg = (
                    f"Checkpoint file not found: {checkpoint_path}\n"
                    f"No {self.checkpoint_type} checkpoints found in {CHECKPOINT_DIR}"
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
        self,
        config,
        device: str = "cpu",
        model_name: str = None,
        expected_keys: Optional[List[str]] = None,
        prefer_best: bool = False,
    ) -> Tuple[Optional[Dict], int, float]:
        """Handle checkpoint loading based on init_from configuration.

        Args:
            config: Configuration object with init_from and name fields
            device: Device to load checkpoint on
            model_name: Optional model name for DAG checkpoints
            expected_keys: List of required keys in checkpoint
            prefer_best: If True, prefer best checkpoint over latest

        Returns:
            Tuple of (checkpoint_dict, iter_num, best_val_loss)
        """
        init_from = config.init_from

        # Handle different init_from options
        if init_from == "scratch":
            print("Initializing model from scratch")
            return None, 0, 1e9

        elif init_from in ["resume", "latest", "best"]:
            if init_from == "best" or prefer_best:
                print("Resuming from best checkpoint")
                ckpt_path = self.find_best_checkpoint(config.name, model_name)
                if ckpt_path is None:
                    print("No best checkpoint found, looking for latest checkpoint")
                    ckpt_path = self.find_latest_checkpoint(config.name, model_name)
            else:
                print("Resuming from latest checkpoint")
                ckpt_path = self.find_latest_checkpoint(config.name, model_name)

            if ckpt_path is None:
                # List available checkpoints for a helpful error message
                available_checkpoints = self._list_available_checkpoints(
                    config.name, model_name
                )
                if available_checkpoints:
                    checkpoint_list = "\n".join(
                        f"  - {name}" for name in available_checkpoints[:10]
                    )
                    if len(available_checkpoints) > 10:
                        checkpoint_list += (
                            f"\n  ... and {len(available_checkpoints) - 10} more"
                        )
                    error_msg = (
                        f"No checkpoint found for config name '{config.name}' in {CHECKPOINT_DIR}\n"
                        f"Available {self.checkpoint_type} checkpoints for '{config.name}':\n{checkpoint_list}\n"
                        f"You can use init_from='scratch' to start from scratch, or specify a specific checkpoint path."
                    )
                else:
                    all_checkpoints = self._list_available_checkpoints()
                    if all_checkpoints:
                        checkpoint_list = "\n".join(
                            f"  - {name}" for name in all_checkpoints[:10]
                        )
                        if len(all_checkpoints) > 10:
                            checkpoint_list += (
                                f"\n  ... and {len(all_checkpoints) - 10} more"
                            )
                        error_msg = (
                            f"No checkpoint found for config name '{config.name}' in {CHECKPOINT_DIR}\n"
                            f"Available {self.checkpoint_type} checkpoints (all configs):\n{checkpoint_list}\n"
                            f"You can use init_from='scratch' to start from scratch, or specify a specific checkpoint path."
                        )
                    else:
                        error_msg = (
                            f"No checkpoint found for config name '{config.name}' in {CHECKPOINT_DIR}\n"
                            f"No {self.checkpoint_type} checkpoints found in {CHECKPOINT_DIR}\n"
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

            checkpoint = self.load_checkpoint_from_path(
                ckpt_path, device, expected_keys
            )
            return (
                checkpoint,
                checkpoint.get("iter_num", 0),
                checkpoint.get("best_val_loss", 1e9),
            )

        elif init_from.startswith("gpt2"):
            print(f"Loading GPT-2 weights: {init_from}")
            # GPT-2 loading is handled by the calling code
            return None, 0, 1e9

        else:
            # Assume it's a direct path to a checkpoint
            print(f"Loading checkpoint from path: {init_from}")
            checkpoint_path = Path(init_from)

            # Support both absolute and relative paths
            if not checkpoint_path.is_absolute():
                # Try relative to checkpoint directory first
                checkpoint_path = self.checkpoint_dir / checkpoint_path
                if not checkpoint_path.exists():
                    # Try relative to current working directory
                    checkpoint_path = Path(init_from)

            checkpoint = self.load_checkpoint_from_path(
                checkpoint_path, device, expected_keys
            )
            return (
                checkpoint,
                checkpoint.get("iter_num", 0),
                checkpoint.get("best_val_loss", 1e9),
            )

    def generate_checkpoint_filename(
        self,
        config_name: str,
        iter_num: int,
        model_name: str = None,
        val_acc: float = None,
        is_best: bool = False,
    ) -> str:
        """Generate checkpoint filename based on parameters."""
        safe_name = "".join(c for c in config_name if c.isalnum() or c in ("-", "_"))

        if is_best:
            if val_acc is not None:
                acc_str = f"{val_acc * 100:.2f}acc"
                return f"ckpt_{safe_name}_best_{acc_str}.pt"
            else:
                return f"ckpt_{safe_name}_best.pt"
        else:
            if val_acc is not None:
                acc_str = f"{val_acc * 100:.2f}acc"
                return f"ckpt_{safe_name}_{iter_num}_{acc_str}.pt"
            else:
                return f"ckpt_{safe_name}_{iter_num}.pt"

    def save_checkpoint(
        self, checkpoint_data: Dict, filename: str, retries: int = 1
    ) -> None:
        """Save checkpoint with retry logic."""
        checkpoint_path = self.checkpoint_dir / filename
        tmp_path = checkpoint_path.with_suffix(".tmp")

        for attempt in range(retries + 1):
            try:
                if _HAVE_ST and self._all_tensors(checkpoint_data):
                    # Use safetensors for pure tensor checkpoints
                    final_path = checkpoint_path.with_suffix(".safetensors")
                    _st.save_file(checkpoint_data, str(tmp_path))
                else:
                    # Fallback to torch.save for mixed-type or no-safetensors state
                    final_path = checkpoint_path.with_suffix(".pt")
                    torch.save(
                        checkpoint_data, tmp_path, _use_new_zipfile_serialization=False
                    )

                try:
                    # Perform atomic rename
                    tmp_path.rename(final_path)
                    print(f"Saved checkpoint: {final_path}")
                    return
                except Exception as exc:
                    if attempt >= retries:
                        raise CheckpointSaveError(
                            f"Failed to rename temporary checkpoint {tmp_path} to {final_path}: {exc}"
                        ) from exc
                    print(
                        f"Retrying checkpoint rename ({attempt+1}/{retries}) due to error: {exc}"
                    )
            except Exception as exc:
                if attempt >= retries:
                    raise CheckpointSaveError(
                        f"Failed to save checkpoint {checkpoint_path}: {exc}"
                    ) from exc
                print(
                    f"Retrying checkpoint save ({attempt+1}/{retries}) due to error: {exc}"
                )

    def clean_previous_checkpoints(
        self, config_name: str, model_name: str = None
    ) -> None:
        """Remove previous checkpoint files for this config name."""
        if not self.checkpoint_dir.exists():
            return

        patterns = self._get_checkpoint_patterns(config_name, model_name)
        removed_count = 0

        for pattern in patterns:
            for ckpt_file in self.checkpoint_dir.glob(pattern):
                try:
                    ckpt_file.unlink()
                    removed_count += 1
                except Exception as e:
                    print(f"Warning: Could not remove {ckpt_file}: {e}")

        if removed_count > 0:
            print(
                f"Removed {removed_count} previous {self.checkpoint_type} checkpoint(s)"
            )

    def _all_tensors(self, state):
        """Return True if every leaf value in state dict is a torch.Tensor."""
        for v in state.values():
            if isinstance(v, dict):
                if not self._all_tensors(v):
                    return False
            elif not isinstance(v, torch.Tensor):
                return False
        return True

    def initialize_model(
        self,
        config,
        model_args: Dict,
        ModelConfig,
        ModelClass,
        device: str = "cpu",
        setup_start_time: float = None,
    ) -> Tuple[torch.nn.Module, Optional[Dict]]:
        """Initialize a model based on configuration.

        Args:
            config: Configuration object with init_from field
            model_args: Dictionary of model arguments
            ModelConfig: Model configuration class
            ModelClass: Model class to instantiate
            device: Device to place model on
            setup_start_time: Start time for logging (optional)

        Returns:
            Tuple of (initialized model, checkpoint dict if loaded)
        """
        if setup_start_time is None:
            setup_start_time = time.time()

        master_process = True  # Assume master process for logging

        # Get checkpoint if needed
        expected_keys = ["model_args", "model", "iter_num", "best_val_loss"]
        checkpoint, iter_num, best_val_loss = self.handle_checkpoint_loading(
            config, device, expected_keys=expected_keys
        )

        if config.init_from == "scratch":
            if master_process:
                print(
                    f"[{time.time() - setup_start_time:.2f}s] Initializing model from scratch"
                )
            gptconf = ModelConfig(**model_args)
            model = ModelClass(gptconf)
            return model, None

        elif config.init_from.startswith("gpt2"):
            if master_process:
                print(
                    f"[{time.time() - setup_start_time:.2f}s] Loading GPT-2 weights {config.init_from}"
                )
            base_model = GPT.from_pretrained(
                config.init_from, dict(dropout=config.dropout)
            )
            for k in (
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "bias",
                "vocab_size",
            ):
                model_args[k] = getattr(base_model.config, k)
            # Create new model with desired dag_depth (may be >0)
            gptconf = ModelConfig(**model_args)
            model = ModelClass(gptconf)

            # Load overlapping GPT weights regardless of dag_depth
            try:
                model.load_state_dict(base_model.state_dict(), strict=False)
            except Exception as e:
                print(f"Warning: partial weight loading failed: {e}")
            return model, None

        elif checkpoint is not None:
            # Loading from checkpoint (resume or direct path)
            print(
                f"[{time.time() - setup_start_time:.2f}s] Loading model from checkpoint"
            )
            for k in (
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "bias",
                "vocab_size",
            ):
                model_args[k] = checkpoint["model_args"][k]
            gptconf = ModelConfig(**model_args)
            model = ModelClass(gptconf)
            state_dict = {
                k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()
            }
            model.load_state_dict(state_dict)
            return model, checkpoint

        else:
            raise ValueError(f"Unsupported init_from {config.init_from}")

    def initialize_dag_model(
        self,
        cfg,
        checkpoint: Optional[Dict],
        device: str = "cpu",
        setup_start_time: float | None = None,
    ) -> Tuple[torch.nn.Module, object]:
        """Initialize a DAG predictor (GPT backbone or PredictorOnly) model.

        Args:
            cfg: DAGTrainConfig or compatible namespace with relevant attributes
            checkpoint: Loaded checkpoint dictionary (or None)
            device: Device string ("cpu", "cuda", etc.)
            setup_start_time: Reference start time for consistent logging (optional)

        Returns:
            Tuple (model, model_config) where model_config is the instantiated
            GPTConfig or PredictorOnlyConfig.
        """
        if setup_start_time is None:
            setup_start_time = time.time()

        # ------------------------------------------------------------------ #
        # Decide model type based on cfg.full_backbone flag
        # ------------------------------------------------------------------ #
        use_full_backbone = getattr(cfg, "full_backbone", False)

        # When resuming, prefer the saved model configuration; otherwise derive
        # from the current cfg.
        saved_cfg = checkpoint.get("model_config") if checkpoint is not None else None

        if use_full_backbone:
            # Prepare configuration for GPT backbone
            model_cfg_dict = {
                "vocab_size": (saved_cfg or {}).get("vocab_size", 50304),
                "n_embd": (saved_cfg or {}).get("n_embd", cfg.n_embd),
                "n_head": (saved_cfg or {}).get("n_head", cfg.n_head),
                "n_layer": (saved_cfg or {}).get(
                    "n_layer", getattr(cfg, "n_layer", 12)
                ),
                "dropout": (saved_cfg or {}).get("dropout", cfg.dropout),
                "bias": (saved_cfg or {}).get("bias", cfg.bias),
                "dag_depth": (saved_cfg or {}).get("dag_depth", cfg.dag_depth),
                "block_size": (saved_cfg or {}).get("block_size", cfg.sequence_length),
                "softmax_temperature": (saved_cfg or {}).get(
                    "softmax_temperature", 20.0
                ),
                # Propagate subset or default full list
                "op_names": (saved_cfg or {}).get(
                    "op_names", getattr(cfg, "op_names", OP_NAMES.copy())
                ),
            }
            model_config = GPTConfig(**model_cfg_dict)
            model = GPT(model_config)
        else:
            # Shallow PredictorOnly model
            model_cfg_dict = {
                "vocab_size": (saved_cfg or {}).get("vocab_size", 50304),
                "n_embd": (saved_cfg or {}).get("n_embd", cfg.n_embd),
                "n_head": (saved_cfg or {}).get("n_head", cfg.n_head),
                "dropout": (saved_cfg or {}).get("dropout", cfg.dropout),
                "bias": (saved_cfg or {}).get("bias", cfg.bias),
                "dag_depth": (saved_cfg or {}).get("dag_depth", cfg.dag_depth),
                "sequence_length": (saved_cfg or {}).get(
                    "sequence_length", cfg.sequence_length
                ),
                "softmax_temperature": (saved_cfg or {}).get(
                    "softmax_temperature", 20.0
                ),
                # Propagate subset or default full list
                "op_names": (saved_cfg or {}).get(
                    "op_names", getattr(cfg, "op_names", OP_NAMES.copy())
                ),
            }
            model_config = PredictorOnlyConfig(**model_cfg_dict)
            model = PredictorOnlyModel(model_config)

        # ------------------------------------------------------------------ #
        # Load weights if checkpoint provided
        # ------------------------------------------------------------------ #
        if checkpoint is not None and "model" in checkpoint:
            state_dict = {
                k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()
            }
            model.load_state_dict(state_dict)
            print(
                f"[{time.time() - setup_start_time:.2f}s] ✅ Model loaded from checkpoint"
            )
        else:
            init_msg = "Full backbone" if use_full_backbone else "Shallow predictor"
            print(
                f"[{time.time() - setup_start_time:.2f}s] ✅ {init_msg} model initialized."
            )

        # Caller will move model to device; we just return it.
        return model, model_config
