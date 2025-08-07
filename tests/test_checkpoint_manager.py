import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from checkpoint_manager import CheckpointManager, CheckpointSaveError


class TestCheckpointManager:
    """Test checkpoint manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("wandb.run")
    @patch("wandb.save")
    def test_save_checkpoint_success(self, mock_wandb_save, mock_wandb_run):
        """Test that checkpoint saving works without argument mismatch errors."""
        # Mock wandb.run to exist
        mock_wandb_run.name = "test_run"
        mock_wandb_run.id = "test_run_id"

        # Create checkpoint manager
        manager = CheckpointManager()

        # Create test checkpoint data
        checkpoint_data = {
            "model": {"layer1.weight": torch.randn(10, 10)},
            "optimizer": {"param_groups": []},
            "model_config": {"n_layer": 2, "n_head": 4},
            "iter_num": 100,
            "best_val_loss": 0.5,
        }

        filename = "test_checkpoint.pt"

        # This should not raise any argument mismatch errors
        try:
            manager.save_checkpoint(checkpoint_data, filename)
        except CheckpointSaveError as e:
            pytest.fail(f"Checkpoint saving failed with error: {e}")

        # Verify wandb.save was called
        mock_wandb_save.assert_called_once()

    @patch("wandb.run")
    @patch("wandb.save")
    def test_save_checkpoint_with_retries(self, mock_wandb_save, mock_wandb_run):
        """Test that checkpoint saving works with retry logic."""
        # Mock wandb.run to exist
        mock_wandb_run.name = "test_run"
        mock_wandb_run.id = "test_run_id"

        # Make wandb.save fail first time, succeed second time
        mock_wandb_save.side_effect = [Exception("Network error"), None]

        # Create checkpoint manager
        manager = CheckpointManager()

        # Create test checkpoint data
        checkpoint_data = {
            "model": {"layer1.weight": torch.randn(10, 10)},
            "optimizer": {"param_groups": []},
            "model_config": {"n_layer": 2, "n_head": 4},
            "iter_num": 100,
            "best_val_loss": 0.5,
        }

        filename = "test_checkpoint.pt"

        # This should succeed on retry
        try:
            manager.save_checkpoint(checkpoint_data, filename, retries=1)
        except CheckpointSaveError as e:
            pytest.fail(f"Checkpoint saving failed with error: {e}")

        # Verify wandb.save was called twice (once failed, once succeeded)
        assert mock_wandb_save.call_count == 2

    @patch("wandb.run", None)
    @patch("wandb.save")
    def test_save_checkpoint_no_wandb_run(self, mock_wandb_save):
        """Test that checkpoint saving fails gracefully when wandb.run is None."""

        # Create checkpoint manager
        manager = CheckpointManager()

        # Create test checkpoint data
        checkpoint_data = {
            "model": {"layer1.weight": torch.randn(10, 10)},
            "optimizer": {"param_groups": []},
            "model_config": {"n_layer": 2, "n_head": 4},
            "iter_num": 100,
            "best_val_loss": 0.5,
        }

        filename = "test_checkpoint.pt"

        # This should raise CheckpointSaveError
        with pytest.raises(CheckpointSaveError, match="W&B run is not initialized"):
            manager.save_checkpoint(checkpoint_data, filename)

    def test_save_checkpoint_argument_mismatch_fixed(self):
        """Test that the _save_checkpoint_to_wandb method signature is correct."""
        manager = CheckpointManager()

        # Check that the method signature matches what we expect
        import inspect

        sig = inspect.signature(manager._save_checkpoint_to_wandb)
        params = list(sig.parameters.keys())

        # Should only have tmp_path and filename (2 parameters, self is not included)
        assert len(params) == 2, f"Expected 2 parameters, got {len(params)}: {params}"
        assert (
            params[0] == "tmp_path"
        ), f"First parameter should be 'tmp_path', got {params[0]}"
        assert (
            params[1] == "filename"
        ), f"Second parameter should be 'filename', got {params[1]}"

        # Test that the method can be called with the correct arguments
        with patch("wandb.save") as mock_save:
            manager._save_checkpoint_to_wandb("/tmp/test.pt", "test.pt")
            mock_save.assert_called_once_with("/tmp/test.pt")

    @patch("wandb.run")
    @patch("wandb.save")
    def test_save_checkpoint_temp_file_cleanup(self, mock_wandb_save, mock_wandb_run):
        """Test that temporary files are properly cleaned up even on failure."""
        # Mock wandb.run to exist
        mock_wandb_run.name = "test_run"
        mock_wandb_run.id = "test_run_id"

        # Make wandb.save fail
        mock_wandb_save.side_effect = Exception("W&B save failed")

        # Create checkpoint manager
        manager = CheckpointManager()

        # Create test checkpoint data
        checkpoint_data = {
            "model": {"layer1.weight": torch.randn(10, 10)},
            "optimizer": {"param_groups": []},
            "model_config": {"n_layer": 2, "n_head": 4},
            "iter_num": 100,
            "best_val_loss": 0.5,
        }

        filename = "test_checkpoint.pt"

        # This should raise CheckpointSaveError
        with pytest.raises(CheckpointSaveError, match="Failed to save checkpoint"):
            manager.save_checkpoint(checkpoint_data, filename, retries=0)

        # Verify that no temporary files are left behind
        temp_files = list(Path(tempfile.gettempdir()).glob("*-test_checkpoint.pt"))
        assert len(temp_files) == 0, f"Temporary files left behind: {temp_files}"

    def test_save_checkpoint_argument_mismatch_regression(self):
        """Test that the original argument mismatch bug is fixed."""
        manager = CheckpointManager()

        # This test specifically checks that the method call doesn't have argument mismatch
        # The original bug was: _save_checkpoint_to_wandb(tmp_path, filename, checkpoint_data)
        # But the method signature was: _save_checkpoint_to_wandb(self, tmp_path, filename)

        # Test that we can call the method with the correct number of arguments
        with patch("wandb.save") as mock_save:
            # This should not raise a TypeError about argument mismatch
            try:
                manager._save_checkpoint_to_wandb("/tmp/test.pt", "test.pt")
            except TypeError as e:
                if "takes" in str(e) and "positional arguments" in str(e):
                    pytest.fail(f"Argument mismatch error detected: {e}")
                else:
                    raise  # Re-raise if it's a different TypeError

            mock_save.assert_called_once_with("/tmp/test.pt")
