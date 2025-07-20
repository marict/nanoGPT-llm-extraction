from pathlib import Path

import pytest
import torch

from models.dag_model import DAGPlanPredictor, GPTConfig


def test_initial_predictions_favor_zero():
    """Test that the digit bias initialization causes the model to initially predict zeros."""
    # Create a small test config
    config = GPTConfig(
        block_size=128,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=3,
        max_digits=2,
        max_decimal_places=2,
    )

    # Initialize the predictor
    predictor = DAGPlanPredictor(config)

    # Create a random hidden state (batch_size=1, seq_len=1)
    batch_size, seq_len = 1, 1
    hidden = torch.randn(batch_size, seq_len, config.n_embd)

    # Forward pass to get predictions
    with torch.no_grad():
        initial_sgn, digit_probs, _ = predictor(hidden)

    # Check that digit probabilities strongly favor zero
    # For each digit position, the probability of digit 0 should be much higher
    for node in range(config.dag_depth + 1):
        for digit_pos in range(config.max_digits + config.max_decimal_places):
            # Get probabilities for this digit position
            probs = digit_probs[0, 0, node, digit_pos]

            # The probability for digit 0 should be much higher than others
            zero_prob = probs[0].item()
            other_probs = probs[1:].max().item()

            # Print for debugging
            print(
                f"Node {node}, Digit {digit_pos}: Zero prob={zero_prob:.4f}, Max other={other_probs:.4f}"
            )

            # Zero digit should have significantly higher probability
            assert zero_prob > 0.9, f"Expected zero probability > 0.9, got {zero_prob}"
            assert (
                other_probs < 0.1
            ), f"Expected other digit max probability < 0.1, got {other_probs}"


if __name__ == "__main__":
    test_initial_predictions_favor_zero()
