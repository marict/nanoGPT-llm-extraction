"""
Tests for checkpoint loading functionality with path-based init_from support.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from training_utils import (BaseConfig, CheckpointLoadError,
                            find_latest_checkpoint, get_checkpoint_filename,
                            handle_checkpoint_loading,
                            load_checkpoint_from_path)


class _TestConfig(BaseConfig):
    """Test configuration class."""

    def __init__(self):
        # Initialize with required name parameter
        super().__init__(name="test_config")


class TestCheckpointLoading:
    """Test cases for checkpoint loading functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for test checkpoints
        self.test_dir = tempfile.mkdtemp()
        self.test_checkpoint_dir = Path(self.test_dir) / "checkpoints"
        self.test_checkpoint_dir.mkdir(exist_ok=True)

        # Sample checkpoint data
        self.sample_checkpoint = {
            "model": {"layer.weight": torch.randn(2, 2)},
            "optimizer": {"state": {}},
            "model_args": {"n_layer": 2, "n_head": 2, "n_embd": 64},
            "model_config": {"n_layer": 2, "n_head": 2, "n_embd": 64},
            "iter_num": 1000,
            "best_val_loss": 0.5,
        }

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_checkpoint(self, filename: str, data: dict = None) -> Path:
        """Create a test checkpoint file."""
        if data is None:
            data = self.sample_checkpoint

        checkpoint_path = self.test_checkpoint_dir / filename
        torch.save(data, checkpoint_path)
        return checkpoint_path

    def test_load_checkpoint_from_path_success(self):
        """Test successful checkpoint loading from a specific path."""
        checkpoint_path = self.create_test_checkpoint("test_checkpoint.pt")

        loaded_checkpoint = load_checkpoint_from_path(checkpoint_path)

        assert "model" in loaded_checkpoint
        assert "iter_num" in loaded_checkpoint
        assert loaded_checkpoint["iter_num"] == 1000
        assert loaded_checkpoint["best_val_loss"] == 0.5

    def test_load_checkpoint_from_path_missing_file(self):
        """Test error handling when checkpoint file is missing."""
        missing_path = self.test_checkpoint_dir / "nonexistent.pt"

        with pytest.raises(CheckpointLoadError, match="Checkpoint file not found"):
            load_checkpoint_from_path(missing_path)

    def test_load_checkpoint_from_path_invalid_file(self):
        """Test error handling when checkpoint file is corrupted."""
        # Create an invalid checkpoint file
        invalid_path = self.test_checkpoint_dir / "invalid.pt"
        with open(invalid_path, "w") as f:
            f.write("not a valid pytorch file")

        with pytest.raises(CheckpointLoadError, match="Failed to load checkpoint"):
            load_checkpoint_from_path(invalid_path)

    def test_load_checkpoint_from_path_missing_keys(self):
        """Test validation of required keys in checkpoint."""
        # Create checkpoint missing required keys
        incomplete_checkpoint = {"model": {"layer.weight": torch.randn(2, 2)}}
        checkpoint_path = self.create_test_checkpoint(
            "incomplete.pt", incomplete_checkpoint
        )

        expected_keys = ["model", "optimizer", "iter_num"]
        with pytest.raises(CheckpointLoadError, match="missing required keys"):
            load_checkpoint_from_path(checkpoint_path, expected_keys=expected_keys)

    @patch.dict(os.environ, {"RUNPOD_POD_ID": "test_pod_123"})
    @patch("runpod_service.stop_runpod")
    def test_runpod_termination_on_missing_checkpoint(self, mock_stop_runpod):
        """Test that RunPod instance is terminated when checkpoint is missing."""
        missing_path = self.test_checkpoint_dir / "nonexistent.pt"

        with pytest.raises(CheckpointLoadError):
            load_checkpoint_from_path(missing_path)

        # Verify that runpod service stop was called
        mock_stop_runpod.assert_called_once()

    @patch.dict(os.environ, {"RUNPOD_POD_ID": "test_pod_123"})
    @patch("runpod_service.stop_runpod")
    def test_runpod_termination_on_invalid_checkpoint(self, mock_stop_runpod):
        """Test that RunPod instance is terminated when checkpoint is invalid."""
        # Create checkpoint missing required keys
        incomplete_checkpoint = {"model": {"layer.weight": torch.randn(2, 2)}}
        checkpoint_path = self.create_test_checkpoint(
            "incomplete.pt", incomplete_checkpoint
        )

        expected_keys = ["model", "optimizer", "iter_num"]
        with pytest.raises(CheckpointLoadError):
            load_checkpoint_from_path(checkpoint_path, expected_keys=expected_keys)

        # Verify that runpod service stop was called
        mock_stop_runpod.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)  # Clear RUNPOD_POD_ID
    def test_no_runpod_termination_when_not_on_runpod(self):
        """Test that RunPod termination is not attempted when not on RunPod."""
        missing_path = self.test_checkpoint_dir / "nonexistent.pt"

        # Should raise error but not attempt runpod termination
        with pytest.raises(CheckpointLoadError):
            load_checkpoint_from_path(missing_path)

    def test_handle_checkpoint_loading_scratch(self):
        """Test handle_checkpoint_loading with init_from='scratch'."""
        cfg = _TestConfig()
        cfg.init_from = "scratch"

        checkpoint, iter_num, best_val_loss = handle_checkpoint_loading(cfg)

        assert checkpoint is None
        assert iter_num == 0
        assert best_val_loss == 1e9

    @patch("training_utils.find_latest_checkpoint")
    def test_handle_checkpoint_loading_resume_success(self, mock_find_latest):
        """Test handle_checkpoint_loading with init_from='resume'."""
        cfg = _TestConfig()
        cfg.init_from = "resume"

        # Create a test checkpoint and set up mock
        checkpoint_path = self.create_test_checkpoint("resume_test.pt")
        mock_find_latest.return_value = checkpoint_path

        checkpoint, iter_num, best_val_loss = handle_checkpoint_loading(cfg)

        assert checkpoint is not None
        assert iter_num == 1000
        assert best_val_loss == 0.5

    @patch("training_utils.find_latest_checkpoint")
    def test_handle_checkpoint_loading_resume_not_found(self, mock_find_latest):
        """Test handle_checkpoint_loading with init_from='resume' when no checkpoint exists."""
        cfg = _TestConfig()
        cfg.init_from = "resume"

        # Mock returning None (no checkpoint found)
        mock_find_latest.return_value = None

        with pytest.raises(CheckpointLoadError, match="No checkpoint found"):
            handle_checkpoint_loading(cfg)

    def test_handle_checkpoint_loading_gpt2(self):
        """Test handle_checkpoint_loading with init_from='gpt2'."""
        cfg = _TestConfig()
        cfg.init_from = "gpt2"

        # Create a mock GPT model class
        class MockGPT:
            pass

        checkpoint, iter_num, best_val_loss = handle_checkpoint_loading(
            cfg, gpt_model_class=MockGPT
        )

        assert checkpoint is None
        assert iter_num == 0
        assert best_val_loss == 1e9

    def test_handle_checkpoint_loading_absolute_path(self):
        """Test handle_checkpoint_loading with absolute path."""
        cfg = _TestConfig()
        checkpoint_path = self.create_test_checkpoint("absolute_test.pt")
        cfg.init_from = str(checkpoint_path)  # Use absolute path

        checkpoint, iter_num, best_val_loss = handle_checkpoint_loading(cfg)

        assert checkpoint is not None
        assert iter_num == 1000
        assert best_val_loss == 0.5

    @patch("training_utils.CHECKPOINT_DIR")
    def test_handle_checkpoint_loading_relative_path(self, mock_checkpoint_dir):
        """Test handle_checkpoint_loading with relative path."""
        mock_checkpoint_dir = str(self.test_checkpoint_dir)

        cfg = _TestConfig()
        # Create checkpoint and use relative filename
        self.create_test_checkpoint("relative_test.pt")
        cfg.init_from = "relative_test.pt"  # Use relative path

        with patch("training_utils.CHECKPOINT_DIR", mock_checkpoint_dir):
            checkpoint, iter_num, best_val_loss = handle_checkpoint_loading(cfg)

        assert checkpoint is not None
        assert iter_num == 1000
        assert best_val_loss == 0.5

    def test_handle_checkpoint_loading_path_not_found(self):
        """Test handle_checkpoint_loading with path that doesn't exist."""
        cfg = _TestConfig()
        cfg.init_from = "/nonexistent/path/checkpoint.pt"

        with pytest.raises(CheckpointLoadError):
            handle_checkpoint_loading(cfg)

    def test_handle_checkpoint_loading_unsupported_option(self):
        """Test handle_checkpoint_loading with unsupported init_from option."""
        cfg = _TestConfig()
        cfg.init_from = "unsupported_option"

        # This should be treated as a path, so it will raise CheckpointLoadError
        with pytest.raises(CheckpointLoadError):
            handle_checkpoint_loading(cfg)

    def test_checkpoint_loading_with_expected_keys(self):
        """Test checkpoint loading with expected keys validation."""
        cfg = _TestConfig()
        checkpoint_path = self.create_test_checkpoint("keys_test.pt")
        cfg.init_from = str(checkpoint_path)

        expected_keys = ["model", "optimizer", "iter_num", "best_val_loss"]
        checkpoint, iter_num, best_val_loss = handle_checkpoint_loading(
            cfg, expected_keys=expected_keys
        )

        assert checkpoint is not None
        assert all(key in checkpoint for key in expected_keys)

    def test_checkpoint_loading_string_vs_path_input(self):
        """Test that both string and Path inputs work for checkpoint loading."""
        checkpoint_path = self.create_test_checkpoint("string_path_test.pt")

        # Test with string input
        checkpoint_str = load_checkpoint_from_path(str(checkpoint_path))

        # Test with Path input
        checkpoint_path_obj = load_checkpoint_from_path(checkpoint_path)

        # Both should work and return the same data
        assert checkpoint_str["iter_num"] == checkpoint_path_obj["iter_num"]
        assert checkpoint_str["best_val_loss"] == checkpoint_path_obj["best_val_loss"]

    def test_checkpoint_filename_with_validation_accuracy(self):
        """Test that checkpoint filenames include validation accuracy when provided."""
        cfg = _TestConfig()
        cfg.name = "test_model"
        iter_num = 1000

        # Test without validation accuracy
        filename_no_acc = get_checkpoint_filename(cfg, iter_num)
        expected_no_acc = "ckpt_test_model_1000.pt"
        assert filename_no_acc == expected_no_acc

        # Test with validation accuracy
        val_acc = 0.8542  # 85.42%
        filename_with_acc = get_checkpoint_filename(cfg, iter_num, val_acc=val_acc)
        expected_with_acc = "ckpt_test_model_1000_85.42acc.pt"
        assert filename_with_acc == expected_with_acc

        # Test with different model name
        model_name = "GPT"
        filename_model_name = get_checkpoint_filename(
            cfg, iter_num, model_name=model_name, val_acc=val_acc
        )
        expected_model_name = "ckpt_GPT_1000_85.42acc.pt"
        assert filename_model_name == expected_model_name

    def test_find_latest_checkpoint_with_accuracy(self):
        """Test that find_latest_checkpoint works with new filename format including accuracy."""
        cfg = _TestConfig()
        cfg.name = "accuracy_test"

        # Create checkpoints with different formats
        old_format_path = self.test_checkpoint_dir / "ckpt_accuracy_test_100.pt"
        new_format_path = (
            self.test_checkpoint_dir / "ckpt_accuracy_test_200_85.42acc.pt"
        )
        newer_format_path = (
            self.test_checkpoint_dir / "ckpt_accuracy_test_300_90.15acc.pt"
        )

        # Create test checkpoint files
        for path in [old_format_path, new_format_path, newer_format_path]:
            self.create_test_checkpoint(path.name)

        # Patch CHECKPOINT_DIR to use our test directory
        with patch("training_utils.CHECKPOINT_DIR", str(self.test_checkpoint_dir)):
            # Should find the latest checkpoint (iter 300) regardless of format
            latest = find_latest_checkpoint(cfg)
            assert latest is not None
            assert latest.name == "ckpt_accuracy_test_300_90.15acc.pt"

    def test_dag_checkpoint_filename_with_accuracy(self):
        """Test that DAG predictor checkpoint filenames include validation accuracy."""
        from train_predictor import DAGTrainConfig
        from train_predictor import \
            get_checkpoint_filename as dag_get_checkpoint_filename

        cfg = DAGTrainConfig()
        cfg.name = "dag_test"
        iter_num = 1000
        model_name = "PredictorOnlyModel"

        # Test without validation accuracy
        filename_no_acc = dag_get_checkpoint_filename(cfg, iter_num, model_name)
        expected_no_acc = "dag_ckpt_PredictorOnlyModel_dag_test_001000"
        assert filename_no_acc == expected_no_acc

        # Test with validation accuracy (op_accuracy)
        val_acc = 0.7823  # 78.23%
        filename_with_acc = dag_get_checkpoint_filename(
            cfg, iter_num, model_name, val_acc=val_acc
        )
        expected_with_acc = "dag_ckpt_PredictorOnlyModel_dag_test_001000_78.23acc"
        assert filename_with_acc == expected_with_acc


class TestCheckpointLoadingIntegration:
    """Integration tests for checkpoint loading with training configurations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_checkpoint_dir = Path(self.test_dir) / "checkpoints"
        self.test_checkpoint_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_train_config_compatibility(self):
        """Test that checkpoint loading works with actual training configs."""
        from train import TrainConfig

        cfg = TrainConfig()
        cfg.init_from = "scratch"
        cfg.name = "test_train"

        # Should work without errors
        checkpoint, iter_num, best_val_loss = handle_checkpoint_loading(cfg)
        assert checkpoint is None
        assert iter_num == 0
        assert best_val_loss == 1e9

    def test_dag_train_config_compatibility(self):
        """Test that checkpoint loading works with DAG training configs."""
        from train_predictor import DAGTrainConfig

        cfg = DAGTrainConfig()
        cfg.init_from = "scratch"
        cfg.name = "test_dag_train"

        # Should work without errors
        checkpoint, iter_num, best_val_loss = handle_checkpoint_loading(cfg)
        assert checkpoint is None
        assert iter_num == 0
        assert best_val_loss == 1e9

    @patch.dict(os.environ, {"RUNPOD_POD_ID": "integration_test_pod"})
    @patch("runpod_service.stop_runpod")
    def test_runpod_integration(self, mock_stop_runpod):
        """Test RunPod integration in a realistic scenario."""
        from train import TrainConfig

        cfg = TrainConfig()
        cfg.init_from = "/nonexistent/checkpoint.pt"
        cfg.name = "runpod_integration_test"

        with pytest.raises(CheckpointLoadError):
            handle_checkpoint_loading(cfg)

        # Verify RunPod termination was attempted
        mock_stop_runpod.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
