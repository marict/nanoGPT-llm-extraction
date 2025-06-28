import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dag_logger import DAGLogger
from dag_model import GPT, GPTConfig, op_names


def test_training_loop_text_generation_integration():
    """Test that the training loop evaluation behaves correctly with text generation."""

    def mock_encode(text):
        return [1, 2, 3, 4, 5]

    def mock_decode(tokens):
        return f"decoded_{len(tokens)}_tokens"

    cfg = GPTConfig(
        vocab_size=50,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=2,
        dropout=0.0,
        bias=False,
    )

    model = GPT(cfg)
    model.eval()

    encode = mock_encode
    decode = mock_decode

    dag_logger = DAGLogger()

    sample_prompt = "Two plus 5 is equal to: "
    encoded = encode(sample_prompt)
    prompt_ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        generated = model.generate(
            prompt_ids,
            max_new_tokens=20,
            temperature=0.8,
            top_k=40,
        )
        generated_sample = decode(generated[0].cpu().tolist())

    dag_logger.format_console_logging(model)

    assert generated_sample is not None
    assert "decoded_" in generated_sample


def test_operation_logging_during_training_step():
    """Test that DAG operation logging works correctly during a training-like scenario."""
    cfg = GPTConfig(
        vocab_size=50,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=2,
        dropout=0.0,
        bias=False,
    )

    model = GPT(cfg)
    model.train()

    # Simulate forward pass
    batch_size = 2
    seq_len = cfg.block_size
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    _, loss = model(input_ids, target_ids)

    # Set up gradient tracking before backward pass
    logger = DAGLogger()
    logger.setup_gradient_tracking(model)

    loss.backward()

    # Test console logging works
    try:
        logger.format_console_logging(model)
        # If we get here without exception, console logging works
        assert True
    except Exception as e:
        pytest.fail(f"Console logging failed: {e}")
    # Test that wandb logging dict works
    wandb_dict = logger.get_wandb_logging_dict(model)
    assert isinstance(wandb_dict, dict), "Should return a dictionary"

    # Should have gate and norm values
    gate_keys = [k for k in wandb_dict if k.startswith("gate/")]
    norm_keys = [k for k in wandb_dict if k.startswith("norm/")]
    assert len(gate_keys) > 0 or len(norm_keys) > 0, "Should have gate or norm values"

    # 3. Extra values (including gradients)
    extra_vals = logger.get_extra_vals(model)

    # Check for gate and norm values
    gate_keys = [key for key in extra_vals.keys() if key.startswith("gate/")]
    norm_keys = [key for key in extra_vals.keys() if key.startswith("norm/")]
    assert len(gate_keys) > 0 or len(norm_keys) > 0, "No gate or norm values found"

    # Check for gradient values
    grad_keys = [key for key in extra_vals.keys() if key.startswith("op_grad/")]
    assert len(grad_keys) == len(op_names), "Should have gradients for all operations"

    # 4. Verify the data would be suitable for wandb logging
    wandb_log_dict = logger.get_wandb_logging_dict(model)

    # All values should be JSON serializable (floats, ints, strings, lists) or wandb objects
    for key, value in wandb_log_dict.items():
        if key.endswith("_timeseries") or key.endswith("_plot"):
            # Wandb plot objects are acceptable for wandb logging
            assert value is not None, f"Plot {key} should not be None"
        else:
            # Other values should be JSON serializable
            assert isinstance(
                value, (int, float, str, list)
            ), f"Value for {key} is not JSON serializable: {type(value)}"

            # If it's a list, all elements should be JSON serializable
            if isinstance(value, list):
                for i, item in enumerate(value):
                    assert isinstance(
                        item, (int, float, str)
                    ), f"List item {i} for {key} is not JSON serializable: {type(item)}"


def test_console_logging_format():
    """Test that the logging produces properly formatted console output."""
    cfg = GPTConfig(
        vocab_size=50,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=2,
        dropout=0.0,
        bias=False,
    )

    model = GPT(cfg)
    model.train()

    # Forward and backward pass
    batch_size = 2
    seq_len = cfg.block_size
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    _, loss = model(input_ids, target_ids)

    # Set up gradient tracking before backward pass
    logger = DAGLogger()
    logger.setup_gradient_tracking(model)

    loss.backward()

    # Test that format_console_logging works
    try:
        logger.format_console_logging(model)
        # If we get here without exception, console logging works
        assert True
    except Exception as e:
        pytest.fail(f"Console logging failed: {e}")

    # Test that wandb logging dict works
    wandb_dict = logger.get_wandb_logging_dict(model)
    assert isinstance(wandb_dict, dict), "Should return a dictionary"

    # Should have gate and norm values
    gate_keys = [k for k in wandb_dict if k.startswith("gate/")]
    norm_keys = [k for k in wandb_dict if k.startswith("norm/")]
    assert len(gate_keys) > 0 or len(norm_keys) > 0, "Should have gate or norm values"
    # Test node values
    node_values = logger.get_node_values_list(model)
    assert len(node_values) > 0, "Should have node values"
    assert all(
        isinstance(v, float) for v in node_values
    ), "All node values should be floats"


def test_node_values_logging():
    """Test that node values are properly logged as a list."""
    cfg = GPTConfig(
        vocab_size=50,
        block_size=6,  # Small block size for predictable number of nodes
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=2,
        dropout=0.0,
        bias=False,
    )

    model = GPT(cfg)
    model.eval()  # Use eval mode for deterministic behavior

    # Forward pass
    batch_size = 1
    seq_len = cfg.block_size
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        _, _ = model(input_ids)

    # Test that node values can be retrieved using DAGLogger
    logger = DAGLogger()
    node_values = logger.get_node_values_list(model)

    # Basic validation
    assert isinstance(node_values, list), "Node values should be a list"
    assert len(node_values) > 0, "Should have at least one node value"

    # All values should be floats
    for i, val in enumerate(node_values):
        assert isinstance(
            val, float
        ), f"Node value at index {i} should be float, got {type(val)}"
        assert not torch.isnan(
            torch.tensor(val)
        ), f"Node value at index {i} should not be NaN"
        assert torch.isfinite(
            torch.tensor(val)
        ), f"Node value at index {i} should be finite"

    # Test console logging format
    logger.format_console_logging(model)

    # Test formatting works
    assert node_values is not None, "Should have node values"
    formatted_values = [f"{val:.4f}" for val in node_values]
    assert len(formatted_values) == len(node_values)
    assert all(isinstance(val, str) and "." in val for val in formatted_values)

    # Test that multiple forward passes can be logged
    input_ids2 = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        _, _ = model(input_ids2)

    node_values2 = logger.get_node_values_list(model)
    assert isinstance(node_values2, list), "Second call should also return list"
    assert len(node_values2) > 0, "Second call should have node values"

    # Test that the method handles edge cases
    # Create a model that hasn't done a forward pass yet
    fresh_model = GPT(cfg)
    fresh_logger = DAGLogger()
    empty_values = fresh_logger.get_node_values_list(fresh_model)
    assert empty_values == [], "Model without forward pass should return empty list"


def test_logging_api_integration():
    """Test that DAG logging API works correctly end-to-end."""
    cfg = GPTConfig(
        vocab_size=30,
        block_size=4,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dag_depth=1,
        dropout=0.0,
        bias=False,
    )

    model = GPT(cfg)
    model.eval()

    # Forward pass
    input_ids = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        _, _ = model(input_ids)

    # Test logging API
    logger = DAGLogger()

    # Test console logging
    try:
        logger.format_console_logging(model)
        # If we get here without exception, console logging works
        assert True
    except Exception as e:
        pytest.fail(f"Console logging failed: {e}")

    # Test wandb logging
    wandb_dict = logger.get_wandb_logging_dict(model)
    assert isinstance(wandb_dict, dict), "Should return a dictionary"

    # Should have gate and norm values
    gate_keys = [k for k in wandb_dict if k.startswith("gate/")]
    norm_keys = [k for k in wandb_dict if k.startswith("norm/")]
    assert len(gate_keys) > 0 or len(norm_keys) > 0, "Should have gate or norm values"
    # Test node values
    node_values = logger.get_node_values_list(model)
    assert len(node_values) > 0, "Should have node values"
    assert all(
        isinstance(v, float) for v in node_values
    ), "All node values should be floats"


def test_dag_logging_after_text_generation():
    """Test that DAG logging captures information from text generation phase."""
    cfg = GPTConfig(
        vocab_size=30,
        block_size=6,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=2,
        dropout=0.0,
        bias=False,
    )

    model = GPT(cfg)
    model.eval()

    dag_logger = DAGLogger()

    # Text generation phase
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))

    with torch.no_grad():
        _ = model.generate(prompt, max_new_tokens=2, temperature=0.8, top_k=10)

    # Test console logging after generation
    try:
        dag_logger.format_console_logging(model)
        # If we get here without exception, console logging works
        assert True
    except Exception as e:
        pytest.fail(f"Console logging failed: {e}")

    # Test wandb logging after generation
    wandb_dict = dag_logger.get_wandb_logging_dict(model)
    assert isinstance(wandb_dict, dict), "Should return a dictionary"

    # Should have gate and norm values
    gate_keys = [k for k in wandb_dict if k.startswith("gate/")]
    norm_keys = [k for k in wandb_dict if k.startswith("norm/")]
    assert len(gate_keys) > 0 or len(norm_keys) > 0, "Should have gate or norm values"
    # Test node values
    node_values = dag_logger.get_node_values_list(model)
    assert len(node_values) > 0, "Should have node values"
    assert all(
        isinstance(v, float) for v in node_values
    ), "All node values should be floats"


def test_operation_logits_removed():
    """Test that operation logits and probabilities are no longer available."""
    cfg = GPTConfig(
        vocab_size=30,
        block_size=6,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=2,
        dropout=0.0,
        bias=False,
    )

    model = GPT(cfg)
    model.eval()

    dag_logger = DAGLogger()

    # Forward pass
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    with torch.no_grad():
        model.generate(prompt, max_new_tokens=1)

    # Verify old methods no longer exist
    assert not hasattr(
        dag_logger, "get_op_logits_dict"
    ), "get_op_logits_dict should be removed"
    assert not hasattr(
        dag_logger, "get_op_probabilities"
    ), "get_op_probabilities should be removed"

    # Verify wandb logging dict doesn't contain old keys
    wandb_dict = dag_logger.get_wandb_logging_dict(model)

    # Should not have any operation logits or probabilities
    op_logit_keys = [k for k in wandb_dict.keys() if k.startswith("op_logits/")]
    op_prob_keys = [k for k in wandb_dict.keys() if k.startswith("op_probs/")]
    op_timeseries_keys = [k for k in wandb_dict.keys() if k == "op_probs_timeseries"]

    assert (
        len(op_logit_keys) == 0
    ), f"Should not have operation logits keys, found: {op_logit_keys}"
    assert (
        len(op_prob_keys) == 0
    ), f"Should not have operation probability keys, found: {op_prob_keys}"
    assert (
        len(op_timeseries_keys) == 0
    ), f"Should not have operation timeseries, found: {op_timeseries_keys}"

    # But should have gate and norm values
    gate_keys = [k for k in wandb_dict.keys() if k.startswith("gate/")]
    norm_keys = [k for k in wandb_dict.keys() if k.startswith("norm/")]
    assert len(gate_keys) > 0 or len(norm_keys) > 0, "Should have gate or norm values"

    # Test node values
    node_values = dag_logger.get_node_values_list(model)
    assert len(node_values) > 0, "Should have node values"
    assert all(
        isinstance(v, float) for v in node_values
    ), "All node values should be floats"


def test_gradient_capture_after_text_generation():
    """Test that gradients can be captured after text generation (sampling)."""
    cfg = GPTConfig(
        vocab_size=50,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=32,
        dag_depth=2,
        dropout=0.0,
        bias=False,
    )

    model = GPT(cfg)
    model.train()

    # Set up DAG logger
    dag_logger = DAGLogger()

    # 1. Do text generation first (like in training loop)
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=2, temperature=0.8, top_k=40)

    # Verify generation worked
    assert generated.shape[1] == 5  # 3 original + 2 new tokens

    # 2. Now do a training step with gradient capture
    batch_size = 2
    seq_len = cfg.block_size
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    # Forward pass first to create last_activations
    _, loss = model(input_ids, target_ids)

    # Set up gradient tracking AFTER forward pass but BEFORE backward pass
    dag_logger.setup_gradient_tracking(model)

    # Backward pass
    loss.backward()

    # 3. Verify gradients were captured
    extra_vals = dag_logger.get_extra_vals(model)
    grad_keys = [key for key in extra_vals.keys() if key.startswith("op_grad/")]

    # Should have gradients for all operations
    expected_grad_keys = [f"op_grad/{op}" for op in op_names]

    assert len(grad_keys) == len(
        expected_grad_keys
    ), f"Expected {len(expected_grad_keys)} gradient keys, got {len(grad_keys)}"

    for expected_key in expected_grad_keys:
        assert expected_key in extra_vals, f"Missing gradient key: {expected_key}"
        grad_val = extra_vals[expected_key]
        assert isinstance(
            grad_val, float
        ), f"Gradient {expected_key} is not a float: {type(grad_val)}"
        assert abs(grad_val) < 1.0, f"Gradient {expected_key} is too large: {grad_val}"

    # 4. Verify wandb logging dict includes gradients
    wandb_dict = dag_logger.get_wandb_logging_dict(model)
    wandb_grad_keys = [key for key in wandb_dict.keys() if key.startswith("op_grad/")]
    assert len(wandb_grad_keys) == len(
        expected_grad_keys
    ), f"Wandb dict missing gradient keys"

    # 5. Verify console logging works
    dag_logger.format_console_logging(model)

    print(f"✓ Successfully captured {len(grad_keys)} gradients after text generation")


if __name__ == "__main__":
    pytest.main([__file__])
