# tests/test_training_logging_integration.py
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

# --------------------------------------------------------------------- #
# import library code
# --------------------------------------------------------------------- #
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dag_model import DAGGPT, DAGGPTConfig, op_names


# --------------------------------------------------------------------- #
# Test integration with training loop functionality
# --------------------------------------------------------------------- #
def test_training_loop_text_generation_integration():
    """Test that the training loop can successfully generate text samples."""
    # Create a small model for testing
    cfg = DAGGPTConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=2,
        dropout=0.0,
        bias=False,
    )

    model = DAGGPT(cfg)
    model.eval()  # Set to eval mode for generation

    # Mock encode/decode functions (simple character-level)
    def mock_encode(text):
        return [ord(c) % cfg.vocab_size for c in text]

    def mock_decode(tokens):
        return "".join(
            [chr(t + ord("a")) for t in tokens[:20]]
        )  # Limit length for readability

    # Test text generation functionality similar to what's in train.py
    sample_prompt = "The answer to the math problem is"
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
    cfg = DAGGPTConfig(
        vocab_size=50,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=2,
        dropout=0.0,
        bias=False,
    )

    model = DAGGPT(cfg)
    model.train()

    # Simulate training step
    batch_size = 2
    seq_len = cfg.block_size
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # Backward pass
    loss.backward()

    # Test that all logging methods work and return expected data
    # This simulates what happens in the training loop

    # 1. Operation probabilities
    op_probs = model.get_op_probabilities()
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
    op_logits = model.get_op_logits_dict()
    assert len(op_logits) == len(op_names), "Should have logits for all operations"

    for op_name in op_names:
        key = f"op_logits/{op_name}"
        assert key in op_logits, f"Missing logit for {op_name}"
        assert isinstance(op_logits[key], float), f"Logit for {op_name} should be float"

    # 3. Extra values (including gradients)
    extra_vals = model.extra_vals()

    # Check for entropy values
    entropy_keys = [key for key in extra_vals.keys() if key.startswith("dag_entropy/")]
    assert len(entropy_keys) > 0, "Should have entropy values"

    # Check for gradient values
    grad_keys = [key for key in extra_vals.keys() if key.startswith("op_grad/")]
    assert len(grad_keys) == len(op_names), "Should have gradients for all operations"

    # 4. Verify the data would be suitable for wandb logging
    wandb_log_dict = {
        **op_probs,
        **op_logits,
        **extra_vals,
    }

    # All values should be JSON serializable (floats, ints, strings)
    for key, value in wandb_log_dict.items():
        assert isinstance(
            value, (int, float, str)
        ), f"Value for {key} is not JSON serializable: {type(value)}"


def test_console_logging_format():
    """Test that the logging produces properly formatted console output."""
    cfg = DAGGPTConfig(
        vocab_size=50,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=2,
        dropout=0.0,
        bias=False,
    )

    model = DAGGPT(cfg)
    model.train()

    # Forward and backward pass
    batch_size = 2
    seq_len = cfg.block_size
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    logits, loss = model(input_ids, target_ids)
    loss.backward()

    # Get logging data
    op_probs = model.get_op_probabilities()
    op_logits = model.get_op_logits_dict()

    # Test console logging format (simulating what's in train.py)
    if op_probs:
        print("Operation probabilities:")
        for op_name, prob in op_probs.items():
            formatted_name = op_name.replace("op_probs/", "")
            print(f"  {formatted_name}: {prob:.4f}")

            # Verify formatting
            assert (
                formatted_name in op_names
            ), f"Unexpected operation name: {formatted_name}"
            assert 0.0 <= prob <= 1.0, f"Invalid probability: {prob}"

    if op_logits:
        print("Operation logits:")
        for op_name, logit in op_logits.items():
            formatted_name = op_name.replace("op_logits/", "")
            print(f"  {formatted_name}: {logit:.4f}")

            # Verify formatting
            assert (
                formatted_name in op_names
            ), f"Unexpected operation name: {formatted_name}"
            assert isinstance(logit, float), f"Logit should be float: {type(logit)}"


if __name__ == "__main__":
    pytest.main([__file__])
