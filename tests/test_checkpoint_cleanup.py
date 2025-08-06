"""Tests for checkpoint cleanup functionality."""

import tempfile
import time
from pathlib import Path

import pytest
import torch

from checkpoint_manager import CheckpointManager


def test_clean_previous_best_checkpoints():
    """Test that clean_previous_best_checkpoints removes old best checkpoints."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create checkpoint manager with temporary directory
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        # Mock the checkpoint directory
        original_checkpoint_dir = CheckpointManager.checkpoint_dir
        CheckpointManager.checkpoint_dir = checkpoint_dir

        try:
            manager = CheckpointManager()

            # Create some dummy checkpoint files
            config_name = "test_config"

            # Create old best checkpoints
            old_checkpoints = [
                f"ckpt_{config_name}_best_90.00acc.pt",
                f"ckpt_{config_name}_best_92.50acc.pt",
                f"ckpt_{config_name}_best_95.00acc.pt",
            ]

            for i, filename in enumerate(old_checkpoints):
                checkpoint_path = checkpoint_dir / filename
                # Create dummy checkpoint data
                dummy_data = {"iter": i, "accuracy": 90.0 + i * 2.5}
                torch.save(dummy_data, checkpoint_path)
                # Add small delay to ensure different timestamps
                time.sleep(0.01)

            # Create one regular checkpoint (should not be removed)
            regular_checkpoint = checkpoint_dir / f"ckpt_{config_name}_100_97.50acc.pt"
            torch.save({"iter": 100, "accuracy": 97.5}, regular_checkpoint)

            # Verify files exist
            assert len(list(checkpoint_dir.glob("*.pt"))) == 4

            # Clean up previous best checkpoints
            manager.clean_previous_best_checkpoints(config_name)

            # Verify only the most recent best checkpoint remains
            remaining_files = list(checkpoint_dir.glob("*.pt"))
            assert len(remaining_files) == 2  # 1 best + 1 regular

            # Check that the most recent best checkpoint is still there
            best_checkpoints = [f for f in remaining_files if "best" in f.name]
            assert len(best_checkpoints) == 1
            assert best_checkpoints[0].name == "ckpt_test_config_best_95.00acc.pt"

            # Check that regular checkpoint is still there
            regular_checkpoints = [f for f in remaining_files if "best" not in f.name]
            assert len(regular_checkpoints) == 1
            assert regular_checkpoints[0].name == "ckpt_test_config_100_97.50acc.pt"

        finally:
            # Restore original checkpoint directory
            CheckpointManager.checkpoint_dir = original_checkpoint_dir


def test_clean_previous_best_checkpoints_no_cleanup_needed():
    """Test that cleanup doesn't remove anything when only one best checkpoint exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        original_checkpoint_dir = CheckpointManager.checkpoint_dir
        CheckpointManager.checkpoint_dir = checkpoint_dir

        try:
            manager = CheckpointManager()

            # Create only one best checkpoint
            config_name = "test_config"
            checkpoint_path = checkpoint_dir / f"ckpt_{config_name}_best_95.00acc.pt"
            torch.save({"iter": 100, "accuracy": 95.0}, checkpoint_path)

            # Verify file exists
            assert len(list(checkpoint_dir.glob("*.pt"))) == 1

            # Clean up (should do nothing)
            manager.clean_previous_best_checkpoints(config_name)

            # Verify file still exists
            remaining_files = list(checkpoint_dir.glob("*.pt"))
            assert len(remaining_files) == 1
            assert remaining_files[0].name == "ckpt_test_config_best_95.00acc.pt"

        finally:
            CheckpointManager.checkpoint_dir = original_checkpoint_dir


def test_clean_previous_best_checkpoints_empty_directory():
    """Test that cleanup handles empty directory gracefully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        original_checkpoint_dir = CheckpointManager.checkpoint_dir
        CheckpointManager.checkpoint_dir = checkpoint_dir

        try:
            manager = CheckpointManager()

            # Directory is empty
            assert len(list(checkpoint_dir.glob("*.pt"))) == 0

            # Clean up (should do nothing)
            manager.clean_previous_best_checkpoints("test_config")

            # Directory should still be empty
            assert len(list(checkpoint_dir.glob("*.pt"))) == 0

        finally:
            CheckpointManager.checkpoint_dir = original_checkpoint_dir


def test_clean_previous_best_checkpoints_different_configs():
    """Test that cleanup only affects checkpoints for the specified config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        original_checkpoint_dir = CheckpointManager.checkpoint_dir
        CheckpointManager.checkpoint_dir = checkpoint_dir

        try:
            manager = CheckpointManager()

            # Create checkpoints for different configs
            config1_checkpoints = [
                f"ckpt_config1_best_90.00acc.pt",
                f"ckpt_config1_best_95.00acc.pt",
            ]

            config2_checkpoints = [
                f"ckpt_config2_best_85.00acc.pt",
                f"ckpt_config2_best_92.00acc.pt",
            ]

            # Create all checkpoints
            for i, filename in enumerate(config1_checkpoints + config2_checkpoints):
                checkpoint_path = checkpoint_dir / filename
                torch.save({"iter": i, "accuracy": 85.0 + i * 2.5}, checkpoint_path)
                time.sleep(0.01)

            # Verify all files exist
            assert len(list(checkpoint_dir.glob("*.pt"))) == 4

            # Clean up only config1
            manager.clean_previous_best_checkpoints("config1")

            # Verify config1 has only one best checkpoint, config2 unchanged
            remaining_files = list(checkpoint_dir.glob("*.pt"))
            assert len(remaining_files) == 3  # 1 config1 best + 2 config2

            config1_files = [f for f in remaining_files if "config1" in f.name]
            config2_files = [f for f in remaining_files if "config2" in f.name]

            assert len(config1_files) == 1
            assert len(config2_files) == 2

            # Check that the most recent config1 best checkpoint remains
            assert config1_files[0].name == "ckpt_config1_best_95.00acc.pt"

        finally:
            CheckpointManager.checkpoint_dir = original_checkpoint_dir
