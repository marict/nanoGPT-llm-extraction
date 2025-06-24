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
    """Test that the training loop can successfully generate text samples."""
    # Create a small model for testing
    cfg = GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=2,
        dropout=0.0,
        bias=False,
    )

    model = GPT(cfg)
    model.eval()  # Set to eval mode for generation

    # Mock encode/decode functions (simple character-level)
    def mock_encode(text):
        return [ord(c) % cfg.vocab_size for c in text]

    def mock_decode(tokens):
        return "".join(
            [chr(t + ord("a")) for t in tokens[:20]]
        )  # Limit length for readability

    # Test text generation functionality similar to what's in train.py
    sample_prompt = "Two plus 5 is equal to: "
    encoded = mock_encode(sample_prompt)
    prompt_ids = torch.tensor(encoded[: cfg.block_size], dtype=torch.long).unsqueeze(0)

    # Generate text
    with torch.no_grad():
        try:
            generated = model.generate(
                prompt_ids, max_new_tokens=10, temperature=0.8, top_k=40
            )
            generated_tokens = generated[0].cpu().tolist()
            generated_text = mock_decode(generated_tokens)

            # Basic checks
            assert isinstance(generated_text, str), "Generated text should be a string"
            assert len(generated_tokens) > len(
                encoded[: cfg.block_size]
            ), "Generated tokens should be longer than prompt tokens"
            assert len(generated_text) > 0, "Generated text should not be empty"

        except Exception as e:
            pytest.fail(f"Text generation failed: {e}")


def test_operation_logging_during_training_step():
    """Test that operation logging works correctly during a simulated training step."""
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

    # Simulate training step
    batch_size = 2
    seq_len = cfg.block_size
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # Set up gradient tracking before backward pass (as done in training loop)
    logger = DAGLogger()
    logger.setup_gradient_tracking(model)

    # Backward pass
    loss.backward()

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

    # 2. Operation logits
    op_logits = logger.get_op_logits_dict(model)
    assert len(op_logits) == len(op_names), "Should have logits for all operations"

    for op_name in op_names:
        key = f"op_logits/{op_name}"
        assert key in op_logits, f"Missing logit for {op_name}"
        assert isinstance(op_logits[key], float), f"Logit for {op_name} should be float"

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
    op_logits = logger.get_op_logits_dict(model)

    # Verify the data is properly formatted
    for op_name in op_names:
        key = f"op_probs/{op_name}"
        assert key in op_probs, f"Missing probability for {op_name}"
        assert 0.0 <= op_probs[key] <= 1.0, f"Invalid probability: {op_probs[key]}"

    for op_name in op_names:
        key = f"op_logits/{op_name}"
        assert key in op_logits, f"Missing logit for {op_name}"
        assert isinstance(
            op_logits[key], float
        ), f"Logit should be float: {type(op_logits[key])}"


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

    # Test console logging format (simulating train.py)
    logger.format_console_logging(model)  # This includes node values

    # Also test manual formatting for verification
    if node_values:
        formatted_values = [f"{val:.4f}" for val in node_values]
        print(f"Node values: {formatted_values}")

        # Verify the formatting works
        assert len(formatted_values) == len(node_values), "Should format all values"
        for formatted_val in formatted_values:
            assert isinstance(formatted_val, str), "Formatted values should be strings"
            # Check that the string contains a decimal point (formatted as float)
            assert (
                "." in formatted_val
            ), f"Formatted value should contain decimal: {formatted_val}"

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


def test_node_values_end_to_end_demo():
    """Demo test showing node values logging functionality working end-to-end."""
    print("\n=== Node Values Logging Demo ===")

    cfg = GPTConfig(
        vocab_size=30,
        block_size=4,  # Very small for clear output
        n_layer=1,
        n_head=1,
        n_embd=16,
        dag_depth=1,
        dropout=0.0,
        bias=False,
    )

    model = GPT(cfg)
    model.eval()

    # Set manual seed for reproducible output
    torch.manual_seed(42)

    # Forward pass
    input_ids = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    print(f"Input tokens: {input_ids[0].tolist()}")

    with torch.no_grad():
        _, _ = model(input_ids)

    # Get all logging information using DAGLogger
    logger = DAGLogger()

    # Demonstrate the clean API
    logger.format_console_logging(model)

    # Also get individual components for validation
    op_probs = logger.get_op_probabilities(model)
    op_logits = logger.get_op_logits_dict(model)
    node_values = logger.get_node_values_list(model)

    print("=== End Demo ===\n")

    # Basic assertions
    assert len(node_values) > 0, "Should have node values"
    assert len(op_probs) > 0, "Should have operation probabilities"
    assert len(op_logits) > 0, "Should have operation logits"


if __name__ == "__main__":
    pytest.main([__file__])
