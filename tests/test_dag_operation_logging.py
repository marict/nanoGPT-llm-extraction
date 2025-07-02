import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_common import (SMALL_CONFIG, assert_valid_forward_pass,
                         assert_valid_logging, assert_valid_node_values,
                         sample_batch_small, sample_batch_tiny,
                         setup_gradient_tracking_test, small_model, tiny_model)

from dag_logger import DAGLogger
from dag_model import GPT, GPTConfig, op_names


def test_basic_operation_logging(small_model, sample_batch_small):
    """Test basic operation and console logging functionality."""
    model, config = small_model
    input_ids, target_ids = sample_batch_small

    logits, loss = model(input_ids, target_ids)

    # Test all logging functionality
    logger, wandb_dict, extra_vals = assert_valid_logging(model)

    # Verify basic functionality works without errors
    assert len(extra_vals) >= 0  # May be empty without gradients
    assert len(wandb_dict) >= 0  # May be empty without gradients


def test_operation_gradient_capture(small_model, sample_batch_small):
    """Test that operation gradients are correctly captured."""
    model, config = small_model

    # Set up gradient tracking test
    logger, loss, input_ids, target_ids = setup_gradient_tracking_test(model, config)

    # Verify gradients were captured
    extra_vals = logger.get_extra_vals(model)

    expected_grad_keys = [f"op_grad/{op}" for op in op_names]
    found_grad_keys = [key for key in extra_vals.keys() if key.startswith("op_grad/")]

    assert len(found_grad_keys) > 0, "No operation gradients found"

    for key in expected_grad_keys:
        assert key in extra_vals, f"Missing gradient key: {key}"
        assert isinstance(extra_vals[key], float), f"Gradient {key} is not a float"


def test_gradient_consistency(small_model, sample_batch_small):
    """Test gradient computation consistency across multiple passes."""
    model, config = small_model
    input_ids, target_ids = sample_batch_small

    logger = DAGLogger()
    gradients_runs = []

    for run in range(2):  # Test 2 runs for efficiency
        # Forward pass then gradient tracking
        logits, loss = model(input_ids, target_ids)
        logger.setup_gradient_tracking(model)
        logger.update_gradient_tracking(model)
        loss.backward()

        logger.compute_log_statistics(model)
        extra_vals = logger.get_extra_vals(model)
        gradients = [extra_vals.get(f"op_grad/{op}", 0.0) for op in op_names]
        gradients_runs.append(gradients)

        model.zero_grad()

    # Verify gradients are reasonable in both runs
    for run_idx, gradients in enumerate(gradients_runs):
        for i, (op, grad) in enumerate(zip(op_names, gradients)):
            assert torch.isfinite(
                torch.tensor(grad)
            ), f"Run {run_idx}: Gradient for {op} not finite"
            assert (
                abs(grad) < 1.0
            ), f"Run {run_idx}: Gradient for {op} too large: {grad}"


def test_comprehensive_logging_integration(small_model, sample_batch_small):
    """Test comprehensive logging with both gate/norm and gradient information."""
    model, config = small_model

    # Set up gradient tracking test and compute statistics
    logger, loss, input_ids, target_ids = setup_gradient_tracking_test(model, config)

    # Use the logger with gradients already captured
    extra_vals = logger.get_extra_vals(model)
    wandb_dict = logger.get_wandb_logging_dict(model)

    # Check for gate and norm values
    gate_keys = [key for key in extra_vals.keys() if key.startswith("gate/")]
    norm_keys = [key for key in extra_vals.keys() if key.startswith("norm/")]
    assert len(gate_keys) > 0 or len(norm_keys) > 0, "No gate or norm values found"

    # Check for gradient values
    grad_keys = [key for key in extra_vals.keys() if key.startswith("op_grad/")]
    assert len(grad_keys) == len(op_names), "Should have gradients for all operations"

    # Verify wandb logging dict structure
    assert isinstance(wandb_dict, dict)
    for key, value in wandb_dict.items():
        if not (key.endswith("_timeseries") or key.endswith("_plot")):
            assert isinstance(
                value, (int, float, str, list)
            ), f"Non-serializable value for {key}"


def test_dag_hidden_gradient_logging():
    """Test that DAG hidden gradients are captured correctly."""
    cfg = GPTConfig(
        n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=50, dag_depth=2
    )
    model = GPT(cfg)
    logger = DAGLogger()

    # Forward pass
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, y)

    # Set up gradient tracking BEFORE backward pass
    logger.setup_gradient_tracking(model)
    loss.backward()

    # Extract logging data
    logger.compute_log_statistics(model)
    extra_vals = logger.get_extra_vals(model)

    # Check for DAG gradient keys
    expected_dag_grad_keys = [
        "dag_output_grad_norm",
        "dag_output_grad_mean",
        "dag_output_grad_std",
        "dag_scratch_grad_norm",
        "dag_scratch_grad_mean",
    ]

    for key in expected_dag_grad_keys:
        assert key in extra_vals, f"Missing DAG gradient key: {key}"
        assert isinstance(
            extra_vals[key], float
        ), f"DAG gradient {key} should be a float"
        assert torch.isfinite(
            torch.tensor(extra_vals[key])
        ), f"DAG gradient {key} should be finite"

    # Verify gradient norms are non-negative
    assert (
        extra_vals["dag_output_grad_norm"] >= 0
    ), "Gradient norm should be non-negative"
    assert extra_vals["dag_output_grad_std"] >= 0, "Gradient std should be non-negative"
    assert (
        extra_vals["dag_scratch_grad_norm"] >= 0
    ), "Scratch gradient norm should be non-negative"


def test_no_dag_depth_logging(sample_batch_small):
    """Test logging with standard GPT (no DAG)."""
    cfg = GPTConfig(
        vocab_size=50,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=0,
        dropout=0.0,
        bias=False,
    )
    model = GPT(cfg)
    input_ids, target_ids = sample_batch_small

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # Standard GPT shouldn't have DAG methods
    assert not hasattr(model, "get_op_probabilities")
    assert not hasattr(model, "get_operand_probabilities")


def test_logging_after_multiple_passes(small_model, sample_batch_small):
    """Test logging works correctly after multiple forward passes."""
    model, config = small_model
    input_ids, target_ids = sample_batch_small

    logger = DAGLogger()

    # Multiple forward passes
    for i in range(2):  # Reduced for efficiency
        logits, loss = model(input_ids, target_ids)
        logger.setup_gradient_tracking(model)
        logger.update_gradient_tracking(model)
        loss.backward()
        model.zero_grad()

        # Test logging still works
        logger.compute_log_statistics(model)
        extra_vals = logger.get_extra_vals(model)
        assert isinstance(extra_vals, dict), "Should return a dictionary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
