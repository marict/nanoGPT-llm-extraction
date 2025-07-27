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
                timeout=30,  # 30 seconds timeout for ultra-fast run
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
