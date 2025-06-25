# tests/test_training_logging_integration.py
import sys
from pathlib import Path

import pytest
import torch

# --------------------------------------------------------------------- #
# import library code
# --------------------------------------------------------------------- #
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dag_logger import DAGLogger
from dag_model import GPT, GPTConfig, op_names


# --------------------------------------------------------------------- #
# Test integration with training loop functionality
# --------------------------------------------------------------------- #
def test_training_loop_text_generation_integration():
    """Test that the training loop evaluation behaves correctly with text generation."""
    from unittest.mock import Mock

    # Mock functions to simulate train.py environment
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

    # Mock the encoding/decoding
    encode = mock_encode
    decode = mock_decode

    # Simulate the evaluation process
    dag_logger = DAGLogger()

    # Text generation (as done in train.py evaluation)
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

    # DAG logging after generation (as done in train.py)
    dag_logger.format_console_logging(model)

    # Verify the integration worked
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

    # Test all logging functions work together
    # 1. Operation probabilities
    op_probs = logger.get_op_probabilities(model)
    assert len(op_probs) == len(
        op_names
    ), "Should have probabilities for all operations"

    for op_name in op_names:
        key = f"op_probs/{op_name}"
        assert key in op_probs, f"Missing probability for {op_name}"
        assert (
            0.0 <= op_probs[key] <= 1.0
        ), f"Invalid probability for {op_name}: {op_probs[key]}"

    # 2. Operand probabilities
    operand_probs = logger.get_operand_probabilities(model)
    assert len(operand_probs) > 0, "Should have operand probabilities"

    # Check that operand probabilities are valid
    for key, prob in operand_probs.items():
        assert isinstance(prob, float), f"Operand probability for {key} should be float"
        assert 0.0 <= prob <= 1.0, f"Operand probability should be in [0,1]: {prob}"

    # 3. Extra values (including gradients)
    extra_vals = logger.get_extra_vals(model)

    # Check for entropy values
    entropy_keys = [key for key in extra_vals.keys() if key.startswith("dag_entropy/")]
    assert len(entropy_keys) > 0, "Should have entropy values"

    # Check for gradient values
    grad_keys = [key for key in extra_vals.keys() if key.startswith("op_grad/")]
    assert len(grad_keys) == len(op_names), "Should have gradients for all operations"

    # 4. Verify the data would be suitable for wandb logging
    wandb_log_dict = logger.get_wandb_logging_dict(model)

    # All values should be JSON serializable (floats, ints, strings, lists)
    for key, value in wandb_log_dict.items():
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
    logger.format_console_logging(model)

    # Also test individual methods for backward compatibility
    op_probs = logger.get_op_probabilities(model)
    operand_probs = logger.get_operand_probabilities(model)

    # Verify the data is properly formatted
    for op_name in op_names:
        key = f"op_probs/{op_name}"
        assert key in op_probs, f"Missing probability for {op_name}"
        assert 0.0 <= op_probs[key] <= 1.0, f"Invalid probability: {op_probs[key]}"

    # Verify operand probabilities are properly formatted
    assert len(operand_probs) > 0, "Should have operand probabilities"
    for key, prob in operand_probs.items():
        assert isinstance(
            prob, float
        ), f"Operand probability should be float: {type(prob)}"
        assert 0.0 <= prob <= 1.0, f"Operand probability should be in [0,1]: {prob}"


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
    if node_values:
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
    logger.format_console_logging(model)

    # Verify all components work
    op_probs = logger.get_op_probabilities(model)
    operand_probs = logger.get_operand_probabilities(model)
    node_values = logger.get_node_values_list(model)

    assert len(node_values) > 0, "Should have node values"
    assert len(op_probs) > 0, "Should have operation probabilities"
    assert len(operand_probs) > 0, "Should have operand probabilities"


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

        generated = model.generate(prompt, max_new_tokens=2, temperature=0.8, top_k=10)

    # DAG logging after generation
    op_probs = dag_logger.get_op_probabilities(model)
    operand_probs = dag_logger.get_operand_probabilities(model)
    node_values = dag_logger.get_node_values_list(model)

    # Verify we have the expected logging information
    assert len(op_probs) == 9, "Should have probabilities for all 9 operations"
    assert len(operand_probs) > 0, "Should have operand probabilities"
    assert len(node_values) > 0, "Should have node values"

    # Verify operation probabilities sum to 1
    op_prob_sum = sum(op_probs.values())
    assert (
        abs(op_prob_sum - 1.0) < 1e-5
    ), f"Operation probabilities should sum to 1, got {op_prob_sum}"

    # Verify operand probabilities are valid
    for key, prob in operand_probs.items():
        assert 0.0 <= prob <= 1.0, f"Operand probability {prob} for {key} not in [0,1]"

    # Verify operand probabilities sum correctly for each operand
    operand1_keys = [k for k in operand_probs.keys() if k.startswith("operand1_probs/")]
    operand2_keys = [k for k in operand_probs.keys() if k.startswith("operand2_probs/")]

    if operand1_keys:
        operand1_sum = sum(operand_probs[k] for k in operand1_keys)
        assert (
            abs(operand1_sum - 1.0) < 1e-5
        ), f"Operand1 probabilities should sum to 1, got {operand1_sum}"

    if operand2_keys:
        operand2_sum = sum(operand_probs[k] for k in operand2_keys)
        assert (
            abs(operand2_sum - 1.0) < 1e-5
        ), f"Operand2 probabilities should sum to 1, got {operand2_sum}"


def test_operation_logits_removed():
    """Test that operation logits are no longer available."""
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

    # Verify operation logits method no longer exists
    assert not hasattr(
        dag_logger, "get_op_logits_dict"
    ), "get_op_logits_dict should be removed"

    # Verify wandb logging dict doesn't contain operation logits
    wandb_dict = dag_logger.get_wandb_logging_dict(model)

    op_logit_keys = [k for k in wandb_dict.keys() if k.startswith("op_logits/")]
    assert (
        len(op_logit_keys) == 0
    ), f"Should not have operation logits keys, found: {op_logit_keys}"

    # But should contain operation probabilities and operand probabilities
    op_prob_keys = [k for k in wandb_dict.keys() if k.startswith("op_probs/")]
    operand_prob_keys = [k for k in wandb_dict.keys() if k.startswith("operand")]

    assert len(op_prob_keys) > 0, "Should have operation probability keys"
    assert len(operand_prob_keys) > 0, "Should have operand probability keys"


if __name__ == "__main__":
    pytest.main([__file__])
