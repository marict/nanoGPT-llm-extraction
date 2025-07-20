from pathlib import Path

import pytest
import torch

from models.dag_model import DAGPlanPredictor, GPTConfig


def test_digit_bias_initialization():
    """Test that the digit biases are properly initialized to favor zero."""
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

    # Get the bias from the final layer of the initial_values_predictor
    bias = predictor.initial_values_predictor[-1].bias

    # Calculate expected dimensions
    num_scratch_nodes = config.dag_depth + 1  # 4
    digits_per_number = config.max_digits + config.max_decimal_places  # 4

    # Check that the digit biases are set correctly
    digit_bias_start = num_scratch_nodes

    # For each node
    for node in range(num_scratch_nodes):
        # For each digit position
        for dig in range(digits_per_number):
            # Calculate offset for this digit
            offset = digit_bias_start + (node * digits_per_number + dig) * 10

            # First digit (0) should have bias of 3.0
            assert bias[offset].item() == pytest.approx(3.0)

            # Remaining digits (1-9) should have bias of -3.0
            for i in range(1, 10):
                assert bias[offset + i].item() == pytest.approx(-3.0)


if __name__ == "__main__":
    test_digit_bias_initialization()
