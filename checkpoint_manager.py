"""
Unified checkpoint management for nanoGPT training scripts.

This module provides centralized checkpoint loading, saving, and management
functionality that works for both regular training (train.py) and DAG
predictor training (train_predictor.py).
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

import wandb

# Import model classes for initialization
from models.dag_model import GPT, GPTConfig
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

    def _get_checkpoint_patterns(self, config_name: str) -> List[str]:
        """Get checkpoint file patterns based on config name.

        Both regular and DAG training use the same checkpoint naming: ckpt_{config_name}_*
        """
        safe_name = "".join(c for c in config_name if c.isalnum() or c in ("-", "_"))

        patterns = [
            f"ckpt_{safe_name}_*.pt",
        ]

        return patterns

    def _list_available_checkpoints(
        self, config_name: str = None, model_name: str = None
    ) -> List[str]:
        """List available checkpoint files in the checkpoint directory."""
        if not self.checkpoint_dir.exists():
            return []

        if config_name:
            patterns = self._get_checkpoint_patterns(config_name)
        else:
            # List all checkpoints (both regular and DAG use same pattern)
            patterns = ["ckpt_*.pt"]

        checkpoint_files = []
        for pattern in patterns:
            checkpoint_files.extend(self.checkpoint_dir.rglob(pattern))

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
        self, config_name: str, any_run: bool = False
    ) -> Path | None:
        """Find the latest checkpoint by iteration number."""
        if not self.checkpoint_dir.exists():
            return None

        if any_run:
            patterns = ["ckpt_*.pt"]
            checkpoint_files = []
            for pattern in patterns:
                checkpoint_files.extend(self.checkpoint_dir.rglob(pattern))
        else:
            patterns = self._get_checkpoint_patterns(config_name)
            checkpoint_files = []
            for pattern in patterns:
                checkpoint_files.extend(self.checkpoint_dir.rglob(pattern))

        if not checkpoint_files:
            return None

        # Prefer newest modification time instead of relying on encoded iteration.
        # Determine highest iteration number among checkpoints
        iters = [self._extract_iteration_number(p) for p in checkpoint_files]
        max_iter = max(iters)

        if max_iter > 0:
            # Filter to checkpoints with this iteration and pick most recent mtime among them
            candidates = [
                p for p, it_n in zip(checkpoint_files, iters) if it_n == max_iter
            ]
            return max(candidates, key=lambda x: x.stat().st_mtime)

        # No iteration info found, choose by modification time
        return max(checkpoint_files, key=lambda x: x.stat().st_mtime)

    def find_best_checkpoint(self, config_name: str) -> Path | None:
        """Find the best checkpoint (saved with 'best' in the name)."""
        if not self.checkpoint_dir.exists():
            return None

        patterns = self._get_checkpoint_patterns(config_name)
        best_patterns = []

        for pattern in patterns:
            # Look for checkpoints with 'best' in the name
            best_pattern = pattern.replace("_*.", "_best*.")
            best_patterns.append(best_pattern)

        checkpoint_files = []
        for pattern in best_patterns:
            checkpoint_files.extend(self.checkpoint_dir.rglob(pattern))

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

            raise CheckpointLoadError(error_msg)

        # Try to load the checkpoint
        try:
            # Use weights_only=False to allow loading custom config objects
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            print(f"Successfully loaded checkpoint from: {checkpoint_path}")
        except Exception as e:
            error_msg = f"Failed to load checkpoint from {checkpoint_path}: {e}"
            print(f"ERROR: {error_msg}")

            raise CheckpointLoadError(error_msg) from e

        # Validate checkpoint structure
        if expected_keys:
            missing_keys = [key for key in expected_keys if key not in checkpoint]
            if missing_keys:
                error_msg = f"Checkpoint missing required keys: {missing_keys}"
                print(f"ERROR: {error_msg}")

                raise CheckpointLoadError(error_msg)

        return checkpoint

    def handle_checkpoint_loading(
        self,
        config,
        device: str = "cpu",
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
                ckpt_path = self.find_best_checkpoint(config.name)
                if ckpt_path is None:
                    print("No best checkpoint found, looking for latest checkpoint")
                    ckpt_path = self.find_latest_checkpoint(config.name)
            else:
                print("Resuming from latest checkpoint")
                ckpt_path = self.find_latest_checkpoint(config.name)

            if ckpt_path is None:
                # List available checkpoints for a helpful error message
                available_checkpoints = self._list_available_checkpoints(config.name)
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

                raise CheckpointLoadError(error_msg)

            checkpoint = self.load_checkpoint_from_path(
                ckpt_path, device, expected_keys
            )
            # Optionally reset the iteration counter when reloading. This is
            # useful when fine-tuning from a pre-trained checkpoint but you
            # want to start a fresh learning-rate schedule.
            iter_num = checkpoint.get("iter_num", 0)
            if getattr(config, "reload_reset_iters", False):
                print("reload_reset_iters is True – resetting iter_num to 0")
                iter_num = 0
                checkpoint["iter_num"] = 0

            best_val_loss = checkpoint.get("best_val_loss", 1e9)
            return (
                checkpoint,
                iter_num,
                best_val_loss,
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
            # Apply the same optional iteration-reset logic for explicit
            # checkpoint paths.
            iter_num = checkpoint.get("iter_num", 0)
            if getattr(config, "reload_reset_iters", False):
                print("reload_reset_iters is True – resetting iter_num to 0")
                iter_num = 0
                checkpoint["iter_num"] = 0

            best_val_loss = checkpoint.get("best_val_loss", 1e9)
            return (
                checkpoint,
                iter_num,
                best_val_loss,
            )

    def generate_checkpoint_filename(
        self,
    ) -> str:
        """Generate checkpoint filename based on parameters."""
        safe_name = "".join(c for c in wandb.run.id if c.isalnum() or c in ("-", "_"))
        return f"ckpt_{safe_name}.pt"

    def save_checkpoint(
        self, checkpoint_data: Dict, filename: str, retries: int = 1
    ) -> None:
        """Save checkpoint to W&B artifacts with retry logic."""

        if wandb.run is None:
            raise CheckpointSaveError(
                "W&B run is not initialized. Make sure to call wandb.init() before saving checkpoints."
            )

        for attempt in range(retries + 1):
            try:
                # Save to temporary file first
                with tempfile.NamedTemporaryFile(
                    suffix=f"-{Path(filename).name}",
                    delete=False,
                ) as tmp_file:
                    torch.save(checkpoint_data, tmp_file.name)
                    tmp_path = tmp_file.name

                # Upload to W&B
                self._save_checkpoint_to_wandb(tmp_path, filename, checkpoint_data)

                # Clean up temporary file
                os.unlink(tmp_path)
                print(f"✅ Checkpoint saved to W&B: {filename}")
                return

            except Exception as exc:
                # Clean up temporary file if it exists
                if "tmp_path" in locals():
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

                if attempt >= retries:
                    raise CheckpointSaveError(
                        f"Failed to save checkpoint {filename}: {exc}"
                    ) from exc
                print(
                    f"Retrying checkpoint save ({attempt+1}/{retries}) due to error: {exc}"
                )

    def _save_checkpoint_to_wandb(
        self,
        tmp_path: str,
        filename: str,
    ) -> None:
        """Save checkpoint to W&B."""
        try:
            wandb.save(tmp_path)
            print(f"📦 Saved checkpoint to W&B: {filename}")

        except Exception as e:
            raise CheckpointSaveError(f"Failed to save checkpoint to W&B: {e}") from e

    def clean_previous_checkpoints(self, config_name: str) -> None:
        """Remove previous checkpoint files for this config name."""
        if not self.checkpoint_dir.exists():
            return

        patterns = self._get_checkpoint_patterns(config_name)
        removed_count = 0

        for pattern in patterns:
            for ckpt_file in self.checkpoint_dir.rglob(pattern):
                try:
                    ckpt_file.unlink()
                    removed_count += 1
                except Exception as e:
                    print(f"Warning: Could not remove {ckpt_file}: {e}")

        if removed_count > 0:
            print(
                f"Removed {removed_count} previous {self.checkpoint_type} checkpoint(s)"
            )

    def clean_previous_best_checkpoints(self, config_name: str) -> None:
        """Remove previous 'best' checkpoint files for this config name, keeping only the latest best."""
        if not self.checkpoint_dir.exists():
            return

        patterns = self._get_checkpoint_patterns(config_name)
        best_checkpoints = []

        # Find all best checkpoints
        for pattern in patterns:
            # Look for checkpoints with 'best' in the name
            best_pattern = pattern.replace("_*.", "_best*.")
            for ckpt_file in self.checkpoint_dir.rglob(best_pattern):
                best_checkpoints.append(ckpt_file)

        if len(best_checkpoints) <= 1:
            return  # No cleanup needed

        # Sort by modification time, keep the most recent
        best_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        checkpoints_to_remove = best_checkpoints[1:]  # Remove all but the most recent

        removed_count = 0
        for ckpt_file in checkpoints_to_remove:
            try:
                ckpt_file.unlink()
                removed_count += 1
                print(f"Removed previous best checkpoint: {ckpt_file.name}")
            except Exception as e:
                print(f"Warning: Could not remove {ckpt_file}: {e}")

        if removed_count > 0:
            print(f"Cleaned up {removed_count} previous best checkpoint(s)")

    def download_checkpoint_from_wandb(
        self,
        run_name: str = None,
        artifact_name: str = None,
        local_dir: Path = None,
        force: bool = False,
    ) -> Path | None:
        """Download checkpoint from W&B artifacts.

        Args:
            run_name: W&B run name (if None, uses current run)
            artifact_name: Specific artifact name (if None, downloads latest best model)
            local_dir: Local directory to save checkpoint
            force: Force re-download even if file exists

        Returns:
            Path to downloaded checkpoint, or None if failed
        """

        # Check if W&B is properly configured
        if not wandb.run and not os.getenv("WANDB_API_KEY"):
            print(
                "❌ W&B not configured. Set WANDB_API_KEY environment variable or run wandb login"
            )
            return None

        try:
            local_dir = local_dir or Path("wandb_checkpoints")
            local_dir.mkdir(parents=True, exist_ok=True)

            # Initialize W&B API with timeout
            print("🔄 Initializing W&B API...")
            api = wandb.Api(timeout=30)  # 30 second timeout

            # Determine run name
            if run_name is None:
                if wandb.run is not None:
                    run_name = wandb.run.name
                else:
                    print("❌ No run name provided and no active W&B run")
                    print(
                        "   Please provide a run_name parameter or run wandb.init() first"
                    )
                    return None

            # Get entity and project from current run or environment
            entity = wandb.run.entity if wandb.run else os.getenv("WANDB_ENTITY")
            project = wandb.run.project if wandb.run else os.getenv("WANDB_PROJECT")

            if not entity or not project:
                print("❌ W&B entity/project not configured")
                print("   Set WANDB_ENTITY and WANDB_PROJECT environment variables")
                return None

            # Find the run with timeout
            print(f"🔄 Looking for W&B run: {entity}/{project}/{run_name}")
            try:
                run = api.run(f"{entity}/{project}/{run_name}")
            except Exception as e:
                print(f"❌ Run not found: {entity}/{project}/{run_name}")
                print(f"   Error: {e}")
                return None

            # Look for artifacts
            print("🔄 Fetching artifacts...")
            artifacts = run.logged_artifacts()

            if artifact_name is None:
                # Look for best model artifact
                best_artifacts = [a for a in artifacts if a.type == "best_model"]
                if not best_artifacts:
                    print("❌ No best model artifacts found")
                    print(
                        "   Available artifact types: "
                        + ", ".join(set(a.type for a in artifacts))
                    )
                    return None
                artifact = best_artifacts[-1]  # Get latest
            else:
                # Look for specific artifact
                matching_artifacts = [a for a in artifacts if artifact_name in a.name]
                if not matching_artifacts:
                    print(f"❌ No artifacts found matching '{artifact_name}'")
                    print(
                        "   Available artifacts: "
                        + ", ".join(a.name for a in artifacts)
                    )
                    return None
                artifact = matching_artifacts[-1]  # Get latest

            print(f"📦 Found artifact: {artifact.name}")

            # Download the artifact with timeout
            print("🔄 Downloading artifact...")
            artifact_dir = artifact.download(root=str(local_dir))

            # Find the checkpoint file
            checkpoint_files = list(Path(artifact_dir).rglob("*.pt"))

            if not checkpoint_files:
                print("❌ No checkpoint files found in artifact")
                return None

            checkpoint_path = checkpoint_files[0]
            print(f"✅ Downloaded checkpoint: {checkpoint_path}")

            return checkpoint_path

        except Exception as e:
            print(f"❌ Error downloading from W&B: {e}")
            return None

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
                raise e
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
        setup_start_time: float | None = None,
    ) -> Tuple[torch.nn.Module, object]:
        """Initialize a DAG predictor (GPT backbone or PredictorOnly) model.

        Args:
            cfg: DAGTrainConfig with required attributes
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
        # Decide model type based on cfg.n_layer (full GPT vs lightweight predictor)
        # ------------------------------------------------------------------ #
        use_full_gpt = getattr(cfg, "n_layer", 1) > 2

        # When resuming, prefer the saved model configuration; otherwise derive
        # from the current cfg.
        saved_cfg = checkpoint.get("model_config") if checkpoint is not None else None

        # ------------------------------------------------------------------ #
        # If the saved model configuration is incompatible with the current
        # cfg (e.g. different dag_depth or digit settings that change tensor
        # shapes), **ignore** the checkpoint and start fresh. This prevents
        # shape-mismatch errors when resuming after hyper-parameter changes.
        # ------------------------------------------------------------------ #
        critical_keys = [
            "dag_depth",
            "n_embd",
            "n_head",
            "n_layer",
            "block_size",
            "vocab_size",
            "max_digits",
            "max_decimal_places",
        ]
        if saved_cfg is not None:
            incompat = []
            for k in critical_keys:
                if k in saved_cfg and hasattr(cfg, k):
                    if saved_cfg[k] != getattr(cfg, k):
                        incompat.append((k, saved_cfg[k], getattr(cfg, k)))
            if incompat:
                err_lines = [
                    "Incompatible checkpoint detected – the following critical parameters differ between the saved model and current config:",
                    *[f"  * {k}: ckpt={old} ≠ cfg={new}" for k, old, new in incompat],
                    "Aborting resume. Either revert your config changes, specify a compatible checkpoint path, or set init_from='scratch'.",
                ]
                raise CheckpointLoadError("\n".join(err_lines))
        # ------------------------------------------------------------------ #

        if use_full_gpt:
            # Prepare configuration for GPT backbone
            model_cfg_dict = {
                "vocab_size": (saved_cfg or {}).get("vocab_size", 50304),
                "n_embd": (saved_cfg or {}).get("n_embd", cfg.n_embd),
                "n_head": (saved_cfg or {}).get("n_head", cfg.n_head),
                "n_layer": (saved_cfg or {}).get("n_layer", cfg.n_layer),
                "dropout": (saved_cfg or {}).get("dropout", cfg.dropout),
                "bias": (saved_cfg or {}).get("bias", cfg.bias),
                "dag_depth": (saved_cfg or {}).get("dag_depth", cfg.dag_depth),
                "block_size": (saved_cfg or {}).get("block_size", cfg.block_size),
                "max_digits": (saved_cfg or {}).get("max_digits", cfg.max_digits),
                "max_decimal_places": (saved_cfg or {}).get(
                    "max_decimal_places", cfg.max_decimal_places
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
                "n_layer": (saved_cfg or {}).get("n_layer", cfg.n_layer),
                "dropout": (saved_cfg or {}).get("dropout", cfg.dropout),
                "bias": (saved_cfg or {}).get("bias", cfg.bias),
                "dag_depth": (saved_cfg or {}).get("dag_depth", cfg.dag_depth),
                "block_size": (saved_cfg or {}).get("block_size", cfg.block_size),
                "max_digits": (saved_cfg or {}).get("max_digits", cfg.max_digits),
                "max_decimal_places": (saved_cfg or {}).get(
                    "max_decimal_places", cfg.max_decimal_places
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
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                raise CheckpointLoadError(
                    f"Failed to load weights from checkpoint due to shape mismatch: {e}\n"
                    "Ensure that your configuration matches the checkpoint or start from scratch with init_from='scratch'."
                )
            print(
                f"[{time.time() - setup_start_time:.2f}s] ✅ Model loaded from checkpoint"
            )
        else:
            init_msg = "Full DAGGPT" if use_full_gpt else "Predictor only"
            print(
                f"[{time.time() - setup_start_time:.2f}s] ✅ {init_msg} model initialized."
            )

        # Caller will move model to device; we just return it.
        return model, model_config
