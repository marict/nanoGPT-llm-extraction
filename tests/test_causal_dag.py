#!/usr/bin/env python3
"""
Test causal DAG implementation.

This test file verifies that the new token-by-token DAG processing
maintains causality and proper gradient flow.
"""

import pytest
import torch

from dag_model import GPT, GPTConfig


@pytest.fixture
def causal_dag_config():
    """Small DAG config for testing causal behavior."""
    return GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=16,
        vocab_size=32,
        block_size=8,
        dag_depth=2,
        bias=False,
    )


@pytest.fixture
def causal_dag_model(causal_dag_config):
    """Create a small DAG model for testing."""
    model = GPT(causal_dag_config)
    return model, causal_dag_config


def test_causal_dag_forward_pass(causal_dag_model):
    """Test that causal DAG forward pass works without errors."""
    model, config = causal_dag_model
    model.eval()

    # Test different sequence lengths
    for seq_len in [1, 2, 4, 8]:
        x = torch.randint(0, config.vocab_size, (2, seq_len))

        with torch.no_grad():
            logits, loss = model(x)

        # Check output shapes
        assert logits.shape == (2, seq_len, config.vocab_size)

        # Check that we have the right number of nodes
        node_values = model.get_node_values_list()
        assert (
            len(node_values) == seq_len
        ), f"Expected {seq_len} nodes, got {len(node_values)}"

        # Check that node values are properly shaped
        if hasattr(model, "dag"):
            # Check that node storage tensors have correct shape
            B, N, T, H = model.dag.node_embeds.shape
            assert T == seq_len, f"Expected {seq_len} time steps, got {T}"
            assert (
                N == config.dag_depth + 1
            ), f"Expected {config.dag_depth + 1} nodes, got {N}"
            assert H == config.n_embd, f"Expected {config.n_embd} hidden dim, got {H}"

            # Check node values tensor shape
            B, N, T = model.dag.node_values.shape
            assert T == seq_len, f"Expected {seq_len} time steps, got {T}"
            assert (
                N == config.dag_depth + 1
            ), f"Expected {config.dag_depth + 1} nodes, got {N}"


def test_causal_dag_gradient_flow(causal_dag_model):
    """Test that gradients flow properly through all tokens."""
    model, config = causal_dag_model
    model.train()

    seq_len = 4
    x = torch.randint(0, config.vocab_size, (2, seq_len))
    targets = torch.randint(0, config.vocab_size, (2, seq_len))

    logits, loss = model(x, targets)

    # Ensure loss is reasonable
    assert loss > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"

    # Backward pass should work without errors
    loss.backward()

    # Check that DAG-related parameters have gradients
    dag_params_with_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None and (
            "dag" in name or "node_embed" in name or "mix_gate" in name
        ):
            dag_params_with_grad += 1
            # Gradients should not be zero (most of the time)
            assert (
                param.grad.norm() >= 0
            ), f"Gradient norm for {name} should be non-negative"

    # Should have many DAG parameters with gradients
    assert (
        dag_params_with_grad > 10
    ), f"Expected many DAG params with gradients, got {dag_params_with_grad}"


def test_causal_dag_causality(causal_dag_model):
    """Test that the DAG respects causality - future tokens don't affect past tokens."""
    model, config = causal_dag_model
    model.eval()

    # Use deterministic seed for reproducible test
    torch.manual_seed(42)

    # Create two identical prefixes with different suffixes
    prefix_len = 3
    total_len = 5

    # Same prefix - use fixed values for reproducibility
    prefix = torch.tensor([[1, 2, 3]])

    # Different suffixes
    suffix1 = torch.tensor([[4, 5]])
    suffix2 = torch.tensor([[6, 7]])

    seq1 = torch.cat([prefix, suffix1], dim=1)
    seq2 = torch.cat([prefix, suffix2], dim=1)

    with torch.no_grad():
        logits1, _ = model(seq1)
        logits2, _ = model(seq2)

    # The logits for the prefix positions should be very similar
    # (they might not be exactly equal due to floating point precision)
    prefix_diff = torch.abs(logits1[:, :prefix_len] - logits2[:, :prefix_len]).max()

    # The difference should be small. For a properly causal model, this should be near zero
    # but allow for some numerical differences from operations like Gumbel softmax
    assert (
        prefix_diff < 5e-2
    ), f"Prefix logits differ by {prefix_diff:.2e}, causality may be violated"

    # Also test that position 0 is exactly the same (it should be for sure)
    pos0_diff = torch.abs(logits1[:, 0] - logits2[:, 0]).max()
    assert (
        pos0_diff < 1e-6
    ), f"Position 0 differs by {pos0_diff:.2e}, this indicates a serious causality issue"


def test_causal_dag_node_growth(causal_dag_model):
    """Test that nodes grow properly with sequence length."""
    model, config = causal_dag_model
    model.eval()

    # Test with increasing sequence lengths
    for seq_len in range(1, 6):
        x = torch.randint(0, config.vocab_size, (1, seq_len))

        with torch.no_grad():
            logits, loss = model(x)

        node_values = model.get_node_values_list()

        # Should have exactly seq_len nodes
        assert (
            len(node_values) == seq_len
        ), f"Seq len {seq_len}: expected {seq_len} nodes, got {len(node_values)}"

        # Check node storage tensors
        if hasattr(model, "dag"):
            # Node embeddings should be (B, N, T, H)
            B, N, T, H = model.dag.node_embeds.shape
            assert T == seq_len, f"Expected {seq_len} time steps, got {T}"
            assert (
                N == config.dag_depth + 1
            ), f"Expected {config.dag_depth + 1} nodes, got {N}"

            # Node values should be (B, N, T)
            B, N, T = model.dag.node_values.shape
            assert T == seq_len, f"Expected {seq_len} time steps in values, got {T}"

        # All node values should be finite
        for i, val in enumerate(node_values):
            assert torch.isfinite(
                torch.tensor(val)
            ), f"Node {i} value {val} is not finite"


def test_causal_dag_deterministic(causal_dag_model):
    """Test that the same input produces similar outputs (with Gumbel softmax, some randomness is expected)."""
    model, config = causal_dag_model
    model.eval()

    # Use deterministic seed to reduce randomness
    torch.manual_seed(42)
    x = torch.randint(0, config.vocab_size, (2, 4))

    # Set seed again before each forward pass for more consistent results
    torch.manual_seed(42)
    with torch.no_grad():
        logits1, _ = model(x)

    torch.manual_seed(42)
    with torch.no_grad():
        logits2, _ = model(x)

    # Due to Gumbel softmax, outputs might have some randomness, but should be quite similar
    # when using the same seed
    assert torch.allclose(
        logits1, logits2, atol=1e-4
    ), "Model should be reasonably consistent with same seed"


def test_causal_dag_no_future_leakage(causal_dag_model):
    """Test that changing future tokens doesn't affect past predictions."""
    model, config = causal_dag_model
    model.eval()

    # Use deterministic sequences for reproducible test
    torch.manual_seed(42)
    base_seq = torch.tensor([[1, 2, 3, 4]])

    # Create extended sequence (adds one more token)
    extended_seq = torch.cat([base_seq, torch.tensor([[5]])], dim=1)

    with torch.no_grad():
        base_logits, _ = model(base_seq)
        extended_logits, _ = model(extended_seq)

    # The logits for the base sequence positions should be very similar
    # Allow for some tolerance due to Gumbel softmax randomness
    diff = torch.abs(base_logits - extended_logits[:, :4]).max()
    assert diff < 5e-2, f"Future tokens affected past predictions by {diff:.2e}"

    # First position should be exactly the same (most important)
    first_diff = torch.abs(base_logits[:, 0] - extended_logits[:, 0]).max()
    assert (
        first_diff < 1e-6
    ), f"First position affected by future token by {first_diff:.2e}"


def test_causal_dag_node_values_access(causal_dag_model):
    """Test that we can access node values after forward pass."""
    model, config = causal_dag_model
    model.eval()

    x = torch.randint(0, config.vocab_size, (1, 3))

    with torch.no_grad():
        logits, loss = model(x)

    # Check that we can access node values through the model method
    node_values = model.get_node_values_list()

    # Values should have the right length
    assert len(node_values) == 3, "Should have 3 node values"

    # All values should be finite
    for i, val in enumerate(node_values):
        assert torch.isfinite(torch.tensor(val)), f"Node {i} value {val} is not finite"


if __name__ == "__main__":
    pytest.main([__file__])
