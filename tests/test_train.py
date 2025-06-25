import os
import pickle
import re
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

import train
from train import (TrainConfig, clean_previous_checkpoints, estimate_loss,
                   find_latest_checkpoint, generate_run_name)

REPO_ROOT = Path(__file__).parent.parent

import pytest


@pytest.mark.parametrize("batch_size", [1, 2])
def test_train_script_runs(tmp_path: Path, batch_size: int):
    """Run ``train.py`` on a tiny synthetic dataset to ensure the script works."""
    # Create a minimal dataset locally to avoid network downloads from prepare.py
    data_dir = tmp_path / "data" / "shakespeare"
    data_dir.mkdir(parents=True)

    vocab_size = 10
    block_size = 32

    # Generate a very small token stream just long enough for ``get_batch``
    arr = np.arange(block_size + 2, dtype=np.uint16) % vocab_size
    arr.tofile(data_dir / "train.bin")
    arr.tofile(data_dir / "val.bin")

    meta = {
        "vocab_size": vocab_size,
        "itos": {i: str(i) for i in range(vocab_size)},
        "stoi": {str(i): i for i in range(vocab_size)},
    }
    with open(data_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    train_script = REPO_ROOT / "train.py"
    config_file = REPO_ROOT / "config" / "train_default.py"

    cmd = [
        sys.executable,
        str(train_script),
        str(config_file),
        "--compile=False",
        "--eval_interval=1",
        "--eval_iters=1",
        "--log_interval=1",
        "--max_iters=0",
        "--dataset=shakespeare",
        f"--batch_size={batch_size}",
        "--n_layer=1",
        "--n_head=1",
        "--n_embd=32",
        "--block_size=32",
        "--subset=0.1",
    ]
    # Provide a minimal stub for the wandb library so training can run without
    # network access.
    (tmp_path / "wandb.py").write_text(
        """\nclass _Run:\n    url = 'http://wandb.local'\n\n"""
        "class Settings:\n    def __init__(self, *args, **kwargs):\n        pass\n\n"
        "def init(*a, **k):\n    return _Run()\n\n"
        "def log(*a, **k):\n    pass\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}:{env.get('PYTHONPATH', '')}"
    env["WANDB_API_KEY"] = "dummy"

    # Run the command and capture output
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,  # Don't raise exception, we'll handle it
    )

    if result.returncode != 0:
        # Format a detailed error message
        error_msg = [
            f"Command failed with return code {result.returncode}",
            f"Command: {' '.join(str(x) for x in cmd)}",
            f"Working directory: {tmp_path}",
            "\nSTDOUT:",
            result.stdout,
            "\nSTDERR:",
            result.stderr,
        ]
        pytest.fail("\n".join(error_msg))


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.training = True

    def forward(self, x, y):
        # Return a fixed loss value for testing
        return torch.tensor(0.0), torch.tensor(1.0)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


def mock_batch_fn(split: str):
    # Return fixed tensors for testing
    x = torch.ones((2, 3))  # batch_size=2, block_size=3
    y = torch.ones((2, 3))
    return x, y


def test_estimate_loss():
    # Setup
    model = MockModel()
    eval_iters = 2
    ctx = nullcontext()

    # Run
    losses = estimate_loss(model, eval_iters, mock_batch_fn, ctx)

    # Assert
    assert isinstance(losses, dict)
    assert set(losses.keys()) == {"train", "val"}
    assert all(isinstance(v, torch.Tensor) for v in losses.values())
    assert all(
        v.item() == 1.0 for v in losses.values()
    )  # Our mock model always returns 1.0
    assert model.training  # Model should be back in training mode


def test_estimate_loss_with_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Setup
    model = MockModel().cuda()
    eval_iters = 2
    ctx = torch.cuda.amp.autocast()

    # Run
    losses = estimate_loss(model, eval_iters, mock_batch_fn, ctx)

    # Assert
    assert isinstance(losses, dict)
    assert set(losses.keys()) == {"train", "val"}
    assert all(isinstance(v, torch.Tensor) for v in losses.values())
    assert all(v.item() == 1.0 for v in losses.values())
    assert model.training  # Model should be back in training mode


def test_generate_run_name_edge_cases(monkeypatch):
    # Test without RUNPOD_POD_ID (local mode)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)

    cfg = SimpleNamespace()  # Config not used in new implementation
    name = generate_run_name(cfg)
    assert name.startswith("local_"), f"Expected local run name, got: {name}"
    assert (
        len(name) == 18
    ), f"Expected 18 characters (local_ + 12 chars), got: {len(name)}"

    # Test with RUNPOD_POD_ID (RunPod mode)
    test_pod_id = "49j146ruxv4k5b"
    monkeypatch.setenv("RUNPOD_POD_ID", test_pod_id)

    name = generate_run_name(cfg)
    assert name == test_pod_id, f"Expected RunPod ID {test_pod_id}, got: {name}"

    # Test with empty RUNPOD_POD_ID (should fallback to local)
    monkeypatch.setenv("RUNPOD_POD_ID", "")
    name = generate_run_name(cfg)
    assert name.startswith(
        "local_"
    ), f"Expected local run name for empty RUNPOD_POD_ID, got: {name}"


def test_subset_config_edge_cases():
    """Test subset configuration with various edge cases and validation."""
    from train import TrainConfig, apply_overrides

    # Test default value
    cfg = TrainConfig()
    assert cfg.subset == 1.0, f"Expected default subset to be 1.0, got {cfg.subset}"

    # Test valid subset values
    test_cases = [
        (0.1, 0.1),  # 10% of dataset
        (0.5, 0.5),  # 50% of dataset
        (0.99, 0.99),  # 99% of dataset
        (1.0, 1.0),  # 100% of dataset
    ]

    for input_val, expected in test_cases:
        cfg = TrainConfig()
        apply_overrides(cfg, [f"--subset={input_val}"])
        assert cfg.subset == expected, f"Expected subset {expected}, got {cfg.subset}"

    # Test that subset is included in config dict for wandb logging
    cfg = TrainConfig()
    cfg.subset = 0.5
    config_dict = cfg.__dict__
    assert "subset" in config_dict, "subset should be in config dict"
    assert (
        config_dict["subset"] == 0.5
    ), "subset value should be preserved in config dict"

    # Test that run name generation works (now uses RunPod ID or local format)
    from train import generate_run_name

    cfg = TrainConfig()
    cfg.subset = 0.25

    run_name = generate_run_name(cfg)
    # The run name should either be a RunPod ID or start with "local_"
    assert (
        run_name.startswith("local_") or len(run_name) > 5
    ), f"Run name should be valid format: {run_name}"

    # Test that subset can be overridden via CLI
    cfg = TrainConfig()
    apply_overrides(cfg, ["--subset=0.75"])
    assert cfg.subset == 0.75, f"Expected subset 0.75 after override, got {cfg.subset}"

    # Test type validation (should raise ValueError for invalid types)
    cfg = TrainConfig()
    try:
        apply_overrides(cfg, ["--subset=invalid"])
        assert False, "Should have raised ValueError for invalid subset type"
    except ValueError:
        pass  # Expected


def test_keep_alive_config():
    """Test keep_alive configuration and argument parsing."""
    from train import TrainConfig, apply_overrides

    # Test default value
    cfg = TrainConfig()
    assert (
        cfg.keep_alive is False
    ), f"Expected default keep_alive to be False, got {cfg.keep_alive}"

    # Test setting keep_alive via config
    cfg = TrainConfig()
    cfg.keep_alive = True
    assert (
        cfg.keep_alive is True
    ), f"Expected keep_alive to be True after setting, got {cfg.keep_alive}"

    # Test that keep_alive can be overridden via CLI
    cfg = TrainConfig()
    apply_overrides(cfg, ["--keep-alive"])
    assert (
        cfg.keep_alive is True
    ), f"Expected keep_alive True after CLI override, got {cfg.keep_alive}"

    # Test that keep_alive is included in config dict for wandb logging
    cfg = TrainConfig()
    cfg.keep_alive = True
    config_dict = cfg.__dict__
    assert "keep_alive" in config_dict, "keep_alive should be in config dict"
    assert (
        config_dict["keep_alive"] is True
    ), "keep_alive value should be preserved in config dict"

    try:
        apply_overrides(cfg, ["--subset=true"])
        assert False, "Should have raised ValueError for boolean subset"
    except ValueError:
        pass  # Expected


def test_math_eval_config_integration():
    """Test that math evaluation config options work correctly."""
    cfg = train.TrainConfig()

    # Test default math eval configuration
    assert cfg.eval_math == True
    assert cfg.math_eval_tasks == ["gsm8k", "svamp"]
    assert cfg.math_eval_max_examples == 50

    # Test config override
    override_data = {
        "eval_math": False,
        "math_eval_tasks": ["gsm8k", "svamp"],
        "math_eval_max_examples": 100,
    }
    train.update_config(cfg, override_data)

    assert cfg.eval_math == False
    assert cfg.math_eval_tasks == ["gsm8k", "svamp"]
    assert cfg.math_eval_max_examples == 100


def test_evaluate_math():
    """Test the evaluate_math function."""
    from dag_model import GPT, GPTConfig

    # Create a small test model
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=50257,
        block_size=64,
        bias=False,
        dag_depth=0,
    )
    model = GPT(config)

    # Mock the run_math_eval.run_eval function
    with mock.patch("train.run_math_eval.run_eval") as mock_run_eval:
        mock_run_eval.return_value = {"gsm8k": 0.25}

        scores = train.evaluate_math(model, "cpu", tasks=["gsm8k"], max_examples=5)

        assert scores == {"gsm8k": 0.25}
        mock_run_eval.assert_called_once_with(
            model, "cpu", tasks=["gsm8k"], max_examples=5
        )


def test_evaluate_math_default_tasks():
    """Test evaluate_math with default tasks (no tasks specified)."""
    from dag_model import GPT, GPTConfig

    # Create a small test model
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=50257,
        block_size=64,
        bias=False,
        dag_depth=0,
    )
    model = GPT(config)

    # Mock the run_math_eval.run_eval function
    with mock.patch("train.run_math_eval.run_eval") as mock_run_eval:
        mock_run_eval.return_value = {"gsm8k": 0.25, "svamp": 0.35}

        # Call with no tasks specified - should use defaults
        scores = train.evaluate_math(model, "cpu", max_examples=5)

        assert scores == {"gsm8k": 0.25, "svamp": 0.35}
        # Verify it was called with both default tasks
        mock_run_eval.assert_called_once_with(
            model, "cpu", tasks=["gsm8k", "svamp"], max_examples=5
        )


def test_evaluate_math_error_handling():
    """Test evaluate_math error handling."""
    from dag_model import GPT, GPTConfig

    # Create a small test model
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=50257,
        block_size=64,
        bias=False,
        dag_depth=0,
    )
    model = GPT(config)

    # Mock the run_math_eval.run_eval function to raise an exception
    with mock.patch("train.run_math_eval.run_eval") as mock_run_eval:
        mock_run_eval.side_effect = Exception("Test error")

        scores = train.evaluate_math(
            model, "cpu", tasks=["gsm8k", "svamp"], max_examples=5
        )

        # Should return -1.0 for all tasks when error occurs
        assert scores == {"gsm8k": -1.0, "svamp": -1.0}


def test_math_eval_config_integration():
    """Test that math evaluation config options work correctly."""
    cfg = train.TrainConfig()

    # Test default math eval configuration
    assert cfg.eval_math == True
    assert cfg.math_eval_tasks == ["gsm8k", "svamp"]
    assert cfg.math_eval_max_examples == 50

    # Test config override
    override_data = {
        "eval_math": False,
        "math_eval_tasks": ["gsm8k", "svamp"],
        "math_eval_max_examples": 100,
    }
    train.update_config(cfg, override_data)

    assert cfg.eval_math == False
    assert cfg.math_eval_tasks == ["gsm8k", "svamp"]
    assert cfg.math_eval_max_examples == 100


def test_checkpoint_filename_generation():
    """Test the new checkpoint filename generation with config name."""
    from train import TrainConfig, get_checkpoint_filename

    # Test basic filename generation
    cfg = TrainConfig()
    cfg.name = "test-project"
    filename = get_checkpoint_filename(cfg, 1000)
    assert filename == "ckpt_test-project_1000.pt"

    # Test with special characters (should be sanitized)
    cfg.name = "test@project#123!"
    filename = get_checkpoint_filename(cfg, 500)
    assert filename == "ckpt_testproject123_500.pt"

    # Test with underscores and hyphens (should be preserved)
    cfg.name = "test_project-v2"
    filename = get_checkpoint_filename(cfg, 250)
    assert filename == "ckpt_test_project-v2_250.pt"


def test_clean_previous_checkpoints(tmp_path):
    """Test the checkpoint cleaning functionality."""

    # Mock the CHECKPOINT_DIR to use temporary directory
    original_dir = train.CHECKPOINT_DIR
    train.CHECKPOINT_DIR = str(tmp_path)

    try:
        # Create some fake checkpoint files
        (tmp_path / "ckpt_testproject_100.pt").touch()
        (tmp_path / "ckpt_testproject_200.pt").touch()
        (tmp_path / "ckpt_otherproject_100.pt").touch()
        (tmp_path / "some_other_file.txt").touch()

        # Test cleaning with clean_previous_runs=False (should do nothing)
        cfg = TrainConfig()
        cfg.name = "testproject"
        cfg.clean_previous_runs = False
        clean_previous_checkpoints(cfg)

        # All files should still exist
        assert (tmp_path / "ckpt_testproject_100.pt").exists()
        assert (tmp_path / "ckpt_testproject_200.pt").exists()
        assert (tmp_path / "ckpt_otherproject_100.pt").exists()
        assert (tmp_path / "some_other_file.txt").exists()

        # Test cleaning with clean_previous_runs=True
        cfg.clean_previous_runs = True
        clean_previous_checkpoints(cfg)

        # Only testproject checkpoints should be removed
        assert not (tmp_path / "ckpt_testproject_100.pt").exists()
        assert not (tmp_path / "ckpt_testproject_200.pt").exists()
        assert (tmp_path / "ckpt_otherproject_100.pt").exists()  # Different project
        assert (tmp_path / "some_other_file.txt").exists()  # Not a checkpoint

    finally:
        # Restore original directory
        train.CHECKPOINT_DIR = original_dir


def test_find_latest_checkpoint(tmp_path):
    """Test finding the latest checkpoint file."""

    # Mock the CHECKPOINT_DIR to use temporary directory
    original_dir = train.CHECKPOINT_DIR
    train.CHECKPOINT_DIR = str(tmp_path)

    try:
        cfg = TrainConfig()
        cfg.name = "testproject"

        # Test with no checkpoints
        result = find_latest_checkpoint(cfg)
        assert result is None

        # Create some checkpoint files with different iteration numbers
        (tmp_path / "ckpt_testproject_100.pt").touch()
        (tmp_path / "ckpt_testproject_500.pt").touch()
        (tmp_path / "ckpt_testproject_300.pt").touch()
        (tmp_path / "ckpt_otherproject_400.pt").touch()  # Different project

        # Should find the latest checkpoint (500)
        result = find_latest_checkpoint(cfg)
        assert result is not None
        assert result.name == "ckpt_testproject_500.pt"

    finally:
        # Restore original directory
        train.CHECKPOINT_DIR = original_dir
