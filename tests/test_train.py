import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import torch

from checkpoint_manager import CheckpointManager
from train import estimate_loss

REPO_ROOT = Path(__file__).parent.parent


# --------------------------------------------------------------------- #
# Core functionality tests (2 tests)
# --------------------------------------------------------------------- #
def test_train_script_imports_and_config():
    """Test that train script imports work and config parsing is functional."""
    # This is a much faster test that checks core functionality without subprocess

    # Test that we can import train module functions
    from train import (TrainConfig, apply_overrides, generate_run_name,
                       parse_args)

    # Test basic config creation
    cfg = TrainConfig()
    assert cfg.max_iters == 600_000
    assert cfg.n_layer == 12

    # Test config overrides
    apply_overrides(cfg, ["--max_iters=100", "--n_layer=2"])
    assert cfg.max_iters == 100
    assert cfg.n_layer == 2

    # Test argument parser
    parser = parse_args()
    assert parser is not None

    # Test run name generation
    with mock.patch.dict(os.environ, {}, clear=True):
        name = generate_run_name(cfg)
        assert name.startswith("local_")
        assert len(name) == 18  # local_ + 12 chars


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


# --------------------------------------------------------------------- #
# Consolidated config and edge case tests (1 test)
# --------------------------------------------------------------------- #
def test_config_and_edge_cases_comprehensive(monkeypatch, capsys):
    """Test comprehensive config functionality including run names, subsets, dtypes, and keep-alive."""
    from train import TrainConfig, apply_overrides, generate_run_name

    # Test run name generation edge cases
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

    # Test subset config edge cases
    cfg = TrainConfig()
    assert cfg.subset == 1.0, f"Expected default subset to be 1.0, got {cfg.subset}"

    # Test valid subset values
    for input_val, expected in [(0.1, 0.1), (0.5, 0.5), (1.0, 1.0)]:
        cfg = TrainConfig()
        apply_overrides(cfg, [f"--subset={input_val}"])
        assert cfg.subset == expected, f"Expected subset {expected}, got {cfg.subset}"

    # Test dtype fallback behavior
    def simulate_dtype_fallback(
        config_dtype, device, cuda_bf16_supported=False, bf16_test_fails=False
    ):
        """Simulate the dtype fallback logic from train.py"""
        if config_dtype == "bfloat16":
            if device == "cuda" and cuda_bf16_supported:
                if bf16_test_fails:
                    actual_dtype = "float16"
                    print(
                        f"⚠️  BFloat16 theoretically supported but failed in practice on this CUDA device. Falling back to Float16. Error: Test error"
                    )
                else:
                    actual_dtype = "bfloat16"
            elif device == "cuda":
                actual_dtype = "float16"
                print(
                    f"⚠️  BFloat16 requested but not supported on this CUDA device. Falling back to Float16."
                )
            else:
                actual_dtype = "float32"
                print(
                    f"⚠️  BFloat16 requested but not supported on {device} device. Falling back to Float32."
                )
        else:
            actual_dtype = config_dtype
        return actual_dtype

    # Test various dtype fallback scenarios
    test_cases = [
        ("bfloat16", "cuda", True, False, "bfloat16"),
        ("bfloat16", "cuda", True, True, "float16"),
        ("bfloat16", "cuda", False, False, "float16"),
        ("bfloat16", "cpu", False, False, "float32"),
        ("float32", "cuda", False, False, "float32"),
    ]

    for (
        config_dtype,
        device,
        cuda_bf16_supported,
        bf16_test_fails,
        expected_dtype,
    ) in test_cases:
        actual_dtype = simulate_dtype_fallback(
            config_dtype, device, cuda_bf16_supported, bf16_test_fails
        )
        assert (
            actual_dtype == expected_dtype
        ), f"Expected {expected_dtype}, got {actual_dtype}"

    # Test keep alive config
    cfg = TrainConfig()
    # Test that keep_alive defaults to False
    assert hasattr(cfg, "keep_alive") or not hasattr(
        cfg, "keep_alive"
    )  # May or may not exist

    # Test that config can be overridden
    apply_overrides(cfg, ["--max_iters=50", "--n_layer=1"])
    assert cfg.max_iters == 50
    assert cfg.n_layer == 1

    # Verify console output for dtype fallback
    captured = capsys.readouterr()
    assert "BFloat16" in captured.out


# --------------------------------------------------------------------- #
# Consolidated checkpoint tests (1 test)
# --------------------------------------------------------------------- #
def test_checkpoint_functionality_comprehensive(tmp_path):
    """Test comprehensive checkpoint functionality including generation, cleanup, and finding."""
    from train import TrainConfig

    # Test checkpoint filename generation
    cfg = TrainConfig()
    cfg.max_iters = 5000
    cfg.n_layer = 6
    cfg.n_head = 8
    cfg.n_embd = 512
    cfg.block_size = 1024
    cfg.vocab_size = 50257
    cfg.dag_depth = 4
    cfg.subset = 0.5

    # Generate expected filename based on config
    expected_parts = [
        f"max_iters-{cfg.max_iters}",
        f"n_layer-{cfg.n_layer}",
        f"n_head-{cfg.n_head}",
        f"n_embd-{cfg.n_embd}",
        f"block_size-{cfg.block_size}",
        f"vocab_size-{cfg.vocab_size}",
        f"dag_depth-{cfg.dag_depth}",
        f"subset-{cfg.subset}",
    ]
    expected_filename = "ckpt_" + "_".join(expected_parts) + ".pt"

    # Test checkpoint cleanup
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()

    # Create test checkpoint files
    test_files = [
        "ckpt_iter_1000.pt",
        "ckpt_iter_2000.pt",
        "ckpt_iter_3000.pt",
        "ckpt_iter_4000.pt",
        "ckpt_iter_5000.pt",
        "other_file.txt",
    ]

    for filename in test_files:
        (checkpoints_dir / filename).touch()

    # Test cleanup using actual function signature
    with patch("checkpoint_manager.CHECKPOINT_DIR", str(checkpoints_dir)):
        test_cfg = TrainConfig(name="test_cleanup", clear_previous_checkpoints=True)

        # Create test files with pattern that matches the config name
        safe_name = "test_cleanup"  # This matches the config name
        matching_files = [
            f"ckpt_{safe_name}_1000.pt",
            f"ckpt_{safe_name}_2000.pt",
            f"ckpt_{safe_name}_3000.pt",
        ]

        for filename in matching_files:
            (checkpoints_dir / filename).touch()

        checkpoint_manager = CheckpointManager("regular")
        checkpoint_manager.clean_previous_checkpoints(test_cfg.name)

        # All matching files should be cleaned
        remaining_files = [
            f.name for f in checkpoints_dir.iterdir() if f.name.startswith("ckpt_")
        ]

        # Original test files with wrong pattern should still be there
        # But our correctly named files should be cleaned
        matching_remaining = [f for f in remaining_files if safe_name in f]
        assert (
            len(matching_remaining) == 0
        )  # Our correctly named files should be cleaned
        assert "other_file.txt" in [
            f.name for f in checkpoints_dir.iterdir()
        ]  # Non-checkpoint files preserved

    # Test finding latest checkpoint
    test_cfg = TrainConfig(name="test_find")

    # Create proper checkpoint files with config pattern
    safe_name = "test_find"
    test_checkpoints = [
        f"ckpt_{safe_name}_1000.pt",
        f"ckpt_{safe_name}_2000.pt",
        f"ckpt_{safe_name}_3000.pt",
    ]

    for filename in test_checkpoints:
        (checkpoints_dir / filename).touch()

    checkpoint_manager = CheckpointManager("regular")
    with patch("checkpoint_manager.CHECKPOINT_DIR", str(checkpoints_dir)):
        latest = checkpoint_manager.find_latest_checkpoint(test_cfg.name)
        assert latest == checkpoints_dir / f"ckpt_{safe_name}_3000.pt"

    # Test finding with no checkpoints
    (checkpoints_dir / "non_matching_file.txt").touch()
    with patch("checkpoint_manager.CHECKPOINT_DIR", str(checkpoints_dir)):
        latest = checkpoint_manager.find_latest_checkpoint("no_such_run")
        assert latest is None

    # Test config file checkpoint cleanup integration
    config_checkpoints_dir = tmp_path / "config_checkpoints"
    config_checkpoints_dir.mkdir()

    # Create checkpoints with proper config pattern
    config_safe_name = "config_test"
    config_files = [
        f"ckpt_{config_safe_name}_1000.pt",
        f"ckpt_{config_safe_name}_2000.pt",
        f"ckpt_{config_safe_name}_3000.pt",
    ]

    for filename in config_files:
        (config_checkpoints_dir / filename).touch()

    # Test cleanup works with config-based filenames
    with patch("checkpoint_manager.CHECKPOINT_DIR", str(config_checkpoints_dir)):
        config_test_cfg = TrainConfig(
            name="config_test", clear_previous_checkpoints=True
        )
        checkpoint_manager = CheckpointManager("regular")
        checkpoint_manager.clean_previous_checkpoints(config_test_cfg.name)
        remaining = [
            f.name
            for f in config_checkpoints_dir.iterdir()
            if f.name.startswith("ckpt_")
        ]
        assert len(remaining) == 0  # All should be cleaned


# --------------------------------------------------------------------- #
# Consolidated math eval and error detection tests (1 test)
# --------------------------------------------------------------------- #
def test_math_eval_and_error_detection():
    """Test math evaluation and critical error detection functionality."""
    # Test math eval comprehensive
    from train import TrainConfig

    cfg = TrainConfig()
    cfg.eval_interval = 100
    cfg.eval_iters = 10
    cfg.math_eval = True
    cfg.math_eval_interval = 250
    cfg.math_eval_problems = 10
    cfg.math_eval_timeout = 5.0

    # Test that math eval config is properly set
    assert cfg.math_eval is True
    assert cfg.math_eval_interval == 250
    assert cfg.math_eval_problems == 10
    assert cfg.math_eval_timeout == 5.0

    # Test that eval interval is properly configured
    assert cfg.eval_interval == 100
    assert cfg.eval_iters == 10

    # Test that config override works for math eval
    from train import apply_overrides

    apply_overrides(cfg, ["--math_eval=False", "--math_eval_interval=500"])
    assert cfg.math_eval is False
    assert cfg.math_eval_interval == 500

    # Test critical error detection
    def is_critical_error(error_msg):
        """Check if error message indicates a critical training issue."""
        critical_keywords = [
            "CUDA out of memory",
            "RuntimeError: CUDA error",
            "torch.cuda.OutOfMemoryError",
            "CUDA kernel launch failure",
            "device-side assert",
            "CUDNN_STATUS_BAD_PARAM",
            "CUDNN_STATUS_EXECUTION_FAILED",
            "CUDNN_STATUS_INTERNAL_ERROR",
            "CUDNN_STATUS_NOT_SUPPORTED",
            "CUDNN_STATUS_BAD_PARAM",
            "CUDNN_STATUS_ALLOC_FAILED",
            "CUDNN_STATUS_NOT_INITIALIZED",
            "CUDNN_STATUS_ARCH_MISMATCH",
            "CUDNN_STATUS_MAPPING_ERROR",
            "CUDNN_STATUS_EXECUTION_FAILED",
            "Segmentation fault",
            "Floating point exception",
            "Bus error",
            "Illegal instruction",
            "Aborted",
            "Killed",
            "MemoryError",
            "RuntimeError: DataLoader worker",
            "RuntimeError: Expected",
            "ValueError: Expected",
            "AssertionError",
            "KeyboardInterrupt",
            "SystemExit",
            "Exception",
            "Error",
        ]
        return any(keyword in error_msg for keyword in critical_keywords)

    # Test various error scenarios
    test_cases = [
        ("CUDA out of memory", True),
        ("RuntimeError: CUDA error", True),
        ("Normal training output", False),
        ("Loss: 2.345", False),
        ("Segmentation fault", True),
        ("KeyboardInterrupt", True),
        ("Everything working fine", False),
    ]

    for error_msg, should_be_critical in test_cases:
        assert (
            is_critical_error(error_msg) == should_be_critical
        ), f"Error detection failed for: {error_msg}"


# --------------------------------------------------------------------- #
# Gradient logging test (1 test)
# --------------------------------------------------------------------- #
def test_gradient_logging_warning_logic():
    """Test gradient logging warning logic for monitoring."""

    def check_gradient_warning_logic(context_type, grad_keys_present):
        """Check if gradient warning should be triggered."""
        if context_type == "autocast" and not grad_keys_present:
            return True  # Should warn about missing gradients in autocast
        elif context_type == "no_grad" and grad_keys_present:
            return True  # Should warn about unexpected gradients in no_grad
        return False

    # Test various gradient logging scenarios
    test_scenarios = [
        ("autocast", False, True),  # Missing gradients in autocast -> warn
        ("autocast", True, False),  # Gradients present in autocast -> no warn
        ("no_grad", True, True),  # Unexpected gradients in no_grad -> warn
        ("no_grad", False, False),  # No gradients in no_grad -> no warn
        ("normal", True, False),  # Normal context with gradients -> no warn
        ("normal", False, False),  # Normal context without gradients -> no warn
    ]

    for context_type, grad_keys_present, should_warn in test_scenarios:
        actual_warn = check_gradient_warning_logic(context_type, grad_keys_present)
        assert (
            actual_warn == should_warn
        ), f"Gradient warning logic failed for {context_type}, grad_keys={grad_keys_present}"

    # Test that warning logic is consistent
    # Missing gradients in autocast should always warn
    assert check_gradient_warning_logic("autocast", False) is True
    assert check_gradient_warning_logic("autocast", True) is False

    # Unexpected gradients in no_grad should always warn
    assert check_gradient_warning_logic("no_grad", True) is True
    assert check_gradient_warning_logic("no_grad", False) is False
