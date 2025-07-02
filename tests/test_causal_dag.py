#!/usr/bin/env python3
"""
Test causal DAG implementation.

This test file verifies that the new token-by-token DAG processing
maintains causality and proper gradient flow.
"""

import pytest
import torch

from dag_logger import DAGLogger
from dag_model import GPT, GPTConfig

torch.manual_seed(1)


@pytest.fixture
def causal_dag_config():
    """Small DAG config for testing."""
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


def test_basic_dag_functionality(causal_dag_model):
    """Test basic DAG operations including forward pass, node growth, and determinism."""
    model, config = causal_dag_model
    model.eval()

    # Test with different sequence lengths
    for seq_len in [1, 2, 4, 8]:
        x = torch.randint(0, config.vocab_size, (2, seq_len))

        with torch.no_grad():
            # Test determinism
            with torch.random.fork_rng():
                torch.manual_seed(1)
                logits1, loss = model(x)
            with torch.random.fork_rng():
                torch.manual_seed(1)
                logits2, _ = model(x)

            # Check output shapes and determinism
            assert logits1.shape == (2, seq_len, config.vocab_size)
            assert torch.allclose(
                logits1, logits2, atol=1e-10
            ), "Model should be deterministic"

            # Check node count and values
            logger = DAGLogger()
            node_values = logger.get_node_values_list(model)
            assert (
                len(node_values) == seq_len
            ), f"Expected {seq_len} nodes, got {len(node_values)}"

            # Check node values are finite
            for i, val in enumerate(node_values):
                assert torch.isfinite(
                    torch.tensor(val)
                ), f"Node {i} value {val} is not finite"

            # Check detailed node storage
            if hasattr(model, "dag"):
                detailed_values = logger.get_detailed_node_values(model)
                assert detailed_values, "Should have detailed node values"
                assert (
                    len(detailed_values["values_per_token"]) == seq_len
                ), "Incorrect number of token values"
                assert (
                    detailed_values["scratch_nodes"] == config.dag_scratch_nodes
                ), "Incorrect number of scratch nodes"
                assert detailed_values["batch_size"] == 2, "Incorrect batch size"


def test_comprehensive_causality(causal_dag_model):
    """Comprehensive test for causality across different contexts and modifications.

    Note we have to use torch.random.fork_rng() to ensure determinism.
    """
    model, config = causal_dag_model
    model.eval()

    # Test 1: Basic causality with different suffixes
    prefix = torch.tensor([[1, 2, 3]])
    suffix1 = torch.tensor([[4, 5]])
    suffix2 = torch.tensor([[6, 7]])

    seq1 = torch.cat([prefix, suffix1], dim=1)
    seq2 = torch.cat([prefix, suffix2], dim=1)

    with torch.no_grad():
        with torch.random.fork_rng():
            torch.manual_seed(1)
            logits1, _ = model(seq1)
        with torch.random.fork_rng():
            torch.manual_seed(1)
            logits2, _ = model(seq2)

        # Check prefix logits are identical
        prefix_diff = torch.abs(logits1[:, :3] - logits2[:, :3]).max()
        assert prefix_diff < 1e-5, f"Prefix logits differ by {prefix_diff:.2e}"

    # Test 2: Incremental context
    base_tokens = torch.randint(0, config.vocab_size, (1, 5))
    prev_logits = None

    for prefix_len in range(1, 6):
        prefix_seq = base_tokens[:, :prefix_len]
        with torch.no_grad():
            with torch.random.fork_rng():
                torch.manual_seed(1)
                logits, _ = model(prefix_seq)

            if prev_logits is not None:
                # Check consistency with previous context
                for pos in range(prefix_len - 1):
                    diff = torch.abs(prev_logits[:, pos] - logits[:, pos]).max()
                    assert (
                        diff < 1e-2
                    ), f"Position {pos} changed with longer context: diff={diff:.2e}"

            prev_logits = logits

    # Test 3: End-to-end stress test
    base_seq = torch.tensor([[1, 2, 3, 4, 5]])
    modifications = [
        torch.tensor([[1, 2, 3, 4, 9]]),  # Change last token
        torch.tensor([[1, 2, 3, 8, 9]]),  # Change last two tokens
        torch.tensor([[1, 2, 7, 8, 9]]),  # Change last three tokens
    ]

    with torch.no_grad():
        with torch.random.fork_rng():
            torch.manual_seed(1)
            base_logits, _ = model(base_seq)

        for mod_seq in modifications:
            with torch.random.fork_rng():
                torch.manual_seed(1)
                mod_logits, _ = model(mod_seq)

            # Find common prefix length
            common_len = 0
            for i in range(min(base_seq.shape[1], mod_seq.shape[1])):
                if base_seq[0, i] == mod_seq[0, i]:
                    common_len += 1
                else:
                    break

            # Check logits for common prefix
            if common_len > 0:
                diff = torch.abs(
                    base_logits[:, :common_len] - mod_logits[:, :common_len]
                ).max()
                assert diff < 1e-5, f"Common prefix logits differ by {diff:.2e}"


if __name__ == "__main__":
    pytest.main([__file__])
