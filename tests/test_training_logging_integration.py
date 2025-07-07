#!/usr/bin/env python
"""Test training and logging integration."""

import sys
from pathlib import Path

import torch

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from dag_logger import DAGLogger


# --------------------------------------------------------------------- #
# Consolidated training logging tests (2 tests)
# --------------------------------------------------------------------- #
def test_comprehensive_training_and_text_generation_logging(
    small_model, sample_batch_small
):
    """Test comprehensive training logging including text generation and batch processing."""
    model, config = small_model
    batch_x, batch_y = sample_batch_small

    # Test text generation logging integration
    model.eval()

    # Generate some text with temperature and top-k sampling
    max_new_tokens = 10
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)  # Small prompt

    with torch.no_grad():
        generated = model.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=0.8, top_k=10
        )

    # Verify generation worked
    assert generated.shape[1] == prompt.shape[1] + max_new_tokens
    assert generated.dtype == prompt.dtype

    # Test comprehensive training logging
    model.train()

    # Multiple forward passes to test logging stability
    for iteration in range(3):
        # Do forward pass first to create tensors (in training mode)
        logits, loss = model(batch_x, batch_y)

        # Verify shapes and values
        assert logits.shape == (batch_x.size(0), batch_x.size(1), config.vocab_size)
        assert loss.item() > 0
        assert torch.isfinite(loss)

        # Set up gradient tracking AFTER forward pass when tensors exist
        logger = DAGLogger()
        logger.setup_gradient_tracking(model)

        # Test logging functionality
        logger.compute_log_statistics(model)

        # Test console logging works (but don't actually print to reduce noise)
        try:
            logger.format_console_logging(model)
        except Exception as e:
            pytest.fail(f"Console logging failed: {e}")

        # Test wandb logging dict
        wandb_dict = logger.get_wandb_logging_dict(model)
        assert isinstance(wandb_dict, dict)

        # Test extra values
        extra_vals = logger.get_extra_vals(model)
        assert isinstance(extra_vals, dict)

        # Test gradient capture
        loss.backward()

        # Verify gradients exist for key parameters
        grad_found = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_found = True
                assert torch.isfinite(
                    param.grad
                ).all(), f"Non-finite gradient in {name}"

        assert grad_found, "No gradients found in model"

        # Clear gradients for next iteration
        model.zero_grad()

    # Test node values comprehensive
    logger = DAGLogger()

    # Test that node values are properly captured
    with torch.no_grad():
        model(batch_x)

    node_values = logger.get_node_values_list(model)
    assert len(node_values) > 0, "No node values captured"

    # Verify all node values are valid numbers
    for i, value in enumerate(node_values):
        assert isinstance(
            value, (int, float)
        ), f"Node value {i} is not a number: {value}"
        assert torch.isfinite(
            torch.tensor(float(value))
        ), f"Node value {i} is not finite: {value}"


def test_gradient_capture_and_api_integration(small_model, tiny_model):
    """Test gradient capture integration and end-to-end API functionality."""
    model, config = small_model
    tiny_model_obj, tiny_config = tiny_model

    # Test gradient capture integration with small model
    logger = DAGLogger()

    # Create test data
    batch_x = torch.randint(0, config.vocab_size, (2, 4))
    batch_y = torch.randint(0, config.vocab_size, (2, 4))

    # Ensure model is in training mode for gradient tracking
    model.train()

    # Do forward pass first to create tensors (in training mode)
    logits, loss = model(batch_x, batch_y)

    # Set up gradient tracking AFTER forward pass when tensors exist
    logger.setup_gradient_tracking(model)

    # Backward pass
    loss.backward()

    # Check that gradients were captured
    assert len(logger.captured_gradients) > 0, "No gradients captured"

    # Verify captured gradients are reasonable
    for grad_name, grad_value in logger.captured_gradients.items():
        assert isinstance(grad_value, float), f"Gradient {grad_name} is not a float"
        assert torch.isfinite(
            torch.tensor(grad_value)
        ), f"Gradient {grad_name} is not finite"
        assert abs(grad_value) < 100, f"Gradient {grad_name} too large: {grad_value}"

    # Test API integration end-to-end with tiny model
    tiny_model_obj.train()

    # Test multiple scenarios - ensure seq_len < block_size (3)
    test_scenarios = [
        {"batch_size": 1, "seq_len": 2},
        {"batch_size": 2, "seq_len": 2},
        {"batch_size": 3, "seq_len": 2},
    ]

    for scenario in test_scenarios:
        batch_size = scenario["batch_size"]
        seq_len = scenario["seq_len"]

        # Create test batch
        x = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
        y = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits, loss = tiny_model_obj(x, y)

        # Verify outputs
        assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)
        assert loss.shape == ()
        assert loss.item() > 0

        # Test logging integration
        logger_tiny = DAGLogger()
        logger_tiny.compute_log_statistics(tiny_model_obj)
        extra_vals = logger_tiny.get_extra_vals(tiny_model_obj)

        # Should have some logging values
        assert len(extra_vals) > 0, f"No extra values for scenario {scenario}"

        # Test gradient computation
        tiny_model_obj.zero_grad()
        loss.backward()

        # Verify gradients exist
        param_with_grad = False
        for param in tiny_model_obj.parameters():
            if param.grad is not None:
                param_with_grad = True
                assert torch.isfinite(param.grad).all()

        assert (
            param_with_grad
        ), f"No parameters received gradients for scenario {scenario}"

    # Test that API handles edge cases
    tiny_model_obj.eval()

    # Test with different input sizes - ensure seq_len < block_size
    for seq_len in [1, 2]:  # Use seq_len < block_size (3)
        x_test = torch.randint(0, tiny_config.vocab_size, (1, seq_len))

        with torch.no_grad():
            logits_test, _ = tiny_model_obj(x_test)
            assert logits_test.shape == (1, seq_len, tiny_config.vocab_size)

    # Test generation functionality
    with torch.no_grad():
        prompt = torch.tensor([[1]], dtype=torch.long)
        generated = tiny_model_obj.generate(prompt, max_new_tokens=3, temperature=1.0)
        assert generated.shape == (1, 4)  # prompt + 3 new tokens
