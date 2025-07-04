import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_common import (assert_valid_logging, assert_valid_node_values,
                         mock_decode, mock_encode,
                         setup_gradient_tracking_test)

from dag_logger import DAGLogger
from dag_model import GPT, GPTConfig, op_names


def test_text_generation_logging_integration(small_model):
    """Test that logging works correctly with text generation."""
    model, cfg = small_model
    model.eval()

    dag_logger = DAGLogger()

    sample_prompt = "Two plus 5 = "
    encoded = mock_encode(sample_prompt)
    prompt_ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        generated = model.generate(
            prompt_ids,
            max_new_tokens=10,  # Reduced for speed
            temperature=0.8,
            top_k=40,
        )
        generated_sample = mock_decode(generated[0].cpu().tolist())

    # Test logging after generation
    logger, wandb_dict, extra_vals = assert_valid_logging(model)

    assert generated_sample is not None
    assert "decoded_" in generated_sample


def test_comprehensive_training_logging(small_model, sample_batch_small):
    """Test comprehensive logging during training scenario."""
    model, config = small_model

    # Set up gradient tracking test and compute statistics
    logger, loss, _, _ = setup_gradient_tracking_test(model, config)

    # Use the logger with gradients already captured
    extra_vals = logger.get_extra_vals(model)
    wandb_dict = logger.get_wandb_logging_dict(model)

    # Verify specific components
    norm_keys = [k for k in wandb_dict if k.startswith("norm/")]
    assert len(norm_keys) > 0, "Should have norm values"

    # Check operation gradients
    op_grad_keys = [key for key in extra_vals.keys() if key.startswith("grad/op")]
    assert len(op_grad_keys) == len(
        op_names
    ), "Should have gradients for all operations"

    # Verify JSON serializable for wandb
    for key, value in wandb_dict.items():
        if not (key.endswith("_timeseries") or key.endswith("_plot")):
            assert isinstance(
                value, (int, float, str, list)
            ), f"Value for {key} not JSON serializable"


def test_node_values_comprehensive(small_model):
    """Test node values logging comprehensively."""
    model, cfg = small_model
    model.eval()

    # Forward pass
    input_ids = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        _, _ = model(input_ids)

    # Test node values extraction
    logger = DAGLogger()
    node_values = logger.get_node_values_list(model)

    # Comprehensive validation
    assert isinstance(node_values, list), "Node values should be a list"
    assert len(node_values) == cfg.block_size, f"Expected {cfg.block_size} nodes"

    for i, val in enumerate(node_values):
        assert isinstance(val, float), f"Node value {i} should be float"
        assert torch.isfinite(torch.tensor(val)), f"Node value {i} should be finite"

    # Test that logging works after forward pass
    logger.compute_log_statistics(model)
    logger.format_console_logging(model)

    # Test multiple forward passes
    input_ids2 = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        _, _ = model(input_ids2)

    node_values2 = logger.get_node_values_list(model)
    assert len(node_values2) == cfg.block_size, "Second call should have same length"

    # Test error case - model without forward pass
    fresh_model = GPT(cfg)
    fresh_logger = DAGLogger()
    with pytest.raises(
        AssertionError, match="Model missing last_values_list attribute"
    ):
        fresh_logger.get_node_values_list(fresh_model)


def test_gradient_capture_integration(small_model):
    """Test gradient capture after various model operations."""
    model, cfg = small_model

    # 1. Test generation first
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    with torch.no_grad():
        model.generate(prompt, max_new_tokens=2, temperature=0.8, top_k=10)

    # 2. Test training step with gradient capture
    batch_size = 2
    seq_len = cfg.block_size
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    # Forward pass first
    _, loss = model(input_ids, target_ids)

    # Set up gradient tracking
    dag_logger = DAGLogger()
    dag_logger.setup_gradient_tracking(model)

    loss.backward()

    # Test logging after backward pass
    dag_logger.compute_log_statistics(model)
    extra_vals = dag_logger.get_extra_vals(model)

    # Verify gradients were captured
    op_grad_keys = [key for key in extra_vals.keys() if key.startswith("grad/op")]
    assert len(op_grad_keys) > 0, "Should have captured operation gradients"

    # Test console logging works
    dag_logger.format_console_logging(model)


def test_api_integration_end_to_end(tiny_model):
    """Test complete DAG logging API integration."""
    model, cfg = tiny_model
    model.eval()

    # Forward pass
    input_ids = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        _, _ = model(input_ids)

    # Test complete logging workflow
    logger, wandb_dict, extra_vals = assert_valid_logging(model)

    # Verify node values are available
    assert_valid_node_values(model)

    # Verify key logging components exist
    assert len(wandb_dict) > 0, "Should have wandb logging data"
    assert len(extra_vals) > 0, "Should have extra values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
