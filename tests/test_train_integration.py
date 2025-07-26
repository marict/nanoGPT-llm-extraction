#!/usr/bin/env python
"""Integration tests for train.py script."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestTrainIntegration:
    """Integration tests that run the full training script."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()

    def teardown_method(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

        # Clean up any checkpoints created during test
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            for item in checkpoint_dir.iterdir():
                if item.name.startswith("test-"):
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

    def test_train_script_runs_without_errors(self):
        """Test that train.py can run with minimal config without crashing."""
        # Set up environment
        env = os.environ.copy()
        env["WANDB_API_KEY"] = "test_key_12345"  # Fake API key for testing
        env["WANDB_MODE"] = "disabled"  # Disable wandb for testing

        # Use the minimal config designed for testing
        config_file = "config/train_default.py"

        # Run the training script
        try:
            result = subprocess.run(
                ["python", "train.py", config_file],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                cwd=self.original_cwd,
            )

            # Check that the script completed successfully
            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)

            assert (
                result.returncode == 0
            ), f"Training script failed with return code {result.returncode}"

            # Basic sanity checks on output
            assert "Training Configuration" in result.stdout
            assert "dag_depth:" in result.stdout

            # Should not have any critical errors
            assert "ERROR" not in result.stderr.upper()
            assert "EXCEPTION" not in result.stderr.upper()
            assert "TRACEBACK" not in result.stderr.upper()

        except subprocess.TimeoutExpired:
            pytest.fail(
                "Training script timed out - this suggests an infinite loop or hang"
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Training script failed to run: {e}")

    def test_train_predictor_script_runs_without_errors(self):
        """Test that train_predictor.py can run with minimal config without crashing."""
        # Set up environment
        env = os.environ.copy()
        env["WANDB_API_KEY"] = "test_key_12345"  # Fake API key for testing
        env["WANDB_MODE"] = "disabled"  # Disable wandb for testing

        # Use the minimal config designed for testing
        config_file = "config/train_predictor_default.py"

        # Run the training script
        try:
            result = subprocess.run(
                ["python", "train_predictor.py", config_file],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                cwd=self.original_cwd,
            )

            # Check that the script completed successfully
            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)

            assert (
                result.returncode == 0
            ), f"Predictor training script failed with return code {result.returncode}"

            # Basic sanity checks on output
            assert "Training Configuration" in result.stdout
            assert "dag_depth:" in result.stdout

            # Should not have any critical errors (excluding deprecation warnings)
            stderr_lines = result.stderr.upper().split("\n")
            critical_errors = [
                line
                for line in stderr_lines
                if "ERROR" in line
                and "WARNING" not in line
                and "DEPRECATED" not in line
            ]
            assert not critical_errors, f"Found critical errors: {critical_errors}"
            assert "EXCEPTION" not in result.stderr.upper()
            assert "TRACEBACK" not in result.stderr.upper()

        except subprocess.TimeoutExpired:
            pytest.fail(
                "Predictor training script timed out - this suggests an infinite loop or hang"
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Predictor training script failed to run: {e}")

    def test_train_script_with_overrides(self):
        """Test that train.py works with command line overrides."""
        # Set up environment
        env = os.environ.copy()
        env["WANDB_API_KEY"] = "test_key_12345"  # Fake API key for testing
        env["WANDB_MODE"] = "disabled"  # Disable wandb for testing

        # Use config with overrides to make it even smaller/faster
        config_file = "config/train_default.py"

        try:
            result = subprocess.run(
                [
                    "python",
                    "train.py",
                    config_file,
                    "--max_iters=2",  # Override to make it even shorter
                    "--eval_interval=1",
                    "--log_interval=1",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minutes timeout for shorter run
                cwd=self.original_cwd,
            )

            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)

            assert (
                result.returncode == 0
            ), f"Training script with overrides failed with return code {result.returncode}"

            # Basic sanity checks
            assert "Starting main function" in result.stdout
            assert "Starting training" in result.stdout

        except subprocess.TimeoutExpired:
            pytest.fail("Training script with overrides timed out")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Training script with overrides failed to run: {e}")
