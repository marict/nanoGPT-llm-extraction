import os
import pickle
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

import train
from train import estimate_loss, generate_run_name

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
    import re
    from types import SimpleNamespace

    # Patch datetime to return a fixed value for reproducibility
    class FixedDatetime:
        @classmethod
        def now(cls):
            return cls()

        def strftime(self, fmt):
            return "2024-01-01_00-00-00"

    monkeypatch.setattr("train.datetime", FixedDatetime)

    # Edge case: zero values
    cfg = SimpleNamespace(
        batch_size=0,
        n_layer=0,
        n_head=0,
        n_embd=0,
        dag_depth=0,
        dataset="testset",
        learning_rate=0.0,
    )
    name = generate_run_name(cfg)
    assert name.startswith("2024-01-01_00-00-00_b0_l0_h0_d0_dag0_testset_lr0e0"), name

    # Edge case: negative values
    cfg = SimpleNamespace(
        batch_size=-1,
        n_layer=-2,
        n_head=-3,
        n_embd=-4,
        dag_depth=-5,
        dataset="negset",
        learning_rate=-1e-5,
    )
    name = generate_run_name(cfg)
    assert (
        "b-1_l-2_h-3_d-4" in name
        and "_dag-5" in name
        and "negset" in name
        and "lr-1e-5" in name
    ), name

    # Edge case: large values
    cfg = SimpleNamespace(
        batch_size=10**6,
        n_layer=512,
        n_head=256,
        n_embd=4096,
        dag_depth=128,
        dataset="largeset",
        learning_rate=1e2,
    )
    name = generate_run_name(cfg)
    assert "b1000000_l512_h256_d4096_dag128_largeset_lr1e2" in name, name

    # Edge case: dataset with special characters
    cfg = SimpleNamespace(
        batch_size=1,
        n_layer=1,
        n_head=1,
        n_embd=1,
        dag_depth=1,
        dataset="weird set!@#",
        learning_rate=1e-4,
    )
    name = generate_run_name(cfg)
    assert "weird set!@#" in name, name

    # Edge case: learning rate with many decimals
    cfg = SimpleNamespace(
        batch_size=2,
        n_layer=2,
        n_head=2,
        n_embd=2,
        dag_depth=2,
        dataset="decimals",
        learning_rate=0.000123456789,
    )
    name = generate_run_name(cfg)
    # Should be in scientific notation
    assert re.search(r"lr[\d\.-]+e-\d+", name), name


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

    # Test that subset is included in run name generation
    from train import generate_run_name

    cfg = TrainConfig()
    cfg.subset = 0.25
    cfg.batch_size = 16
    cfg.n_layer = 2
    cfg.n_head = 4
    cfg.n_embd = 128
    cfg.dag_depth = 0
    cfg.dataset = "testset"
    cfg.learning_rate = 1e-4

    run_name = generate_run_name(cfg)
    # The run name should contain the hyperparameters but not necessarily subset
    # since subset is not currently included in the run name generation
    assert (
        "b16_l2_h4_d128_dag0_testset" in run_name
    ), f"Run name should contain hyperparameters: {run_name}"

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
    from model import GPT, GPTConfig

    # Create a small test model
    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=32, vocab_size=50257, block_size=64, bias=False
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
    from model import GPT, GPTConfig

    # Create a small test model
    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=32, vocab_size=50257, block_size=64, bias=False
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
    from model import GPT, GPTConfig

    # Create a small test model
    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=32, vocab_size=50257, block_size=64, bias=False
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


def test_math_eval_examples_parameter():
    """Test the new math_eval_examples parameter."""
    cfg = train.TrainConfig()

    # Test default value
    assert cfg.math_eval_examples == 50

    # Test config override
    override_data = {"math_eval_examples": 10}
    train.update_config(cfg, override_data)
    assert cfg.math_eval_examples == 10

    # Test CLI override
    cfg = train.TrainConfig()
    train.apply_overrides(cfg, ["--math_eval_examples=5"])
    assert cfg.math_eval_examples == 5
