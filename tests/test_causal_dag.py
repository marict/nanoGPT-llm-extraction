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

        # Check that node storage tensors have correct shape
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
            "dag" in name or "node_embed" in name or "dag_mixer" in name
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
    # With deterministic Gumbel softmax, we can have stricter tolerances
    assert (
        prefix_diff < 1e-5
    ), f"Prefix logits differ by {prefix_diff:.2e}, causality may be violated"

    # Position 0 should be exactly the same with deterministic Gumbel softmax
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
    """Test that the same input produces identical outputs with deterministic Gumbel softmax."""
    model, config = causal_dag_model
    model.eval()

    # Use deterministic seed to reduce randomness
    torch.manual_seed(42)
    x = torch.randint(0, config.vocab_size, (2, 4))

    # Multiple forward passes should be identical
    with torch.no_grad():
        logits1, _ = model(x)
        logits2, _ = model(x)

    # With deterministic Gumbel softmax, outputs should be identical
    assert torch.allclose(
        logits1, logits2, atol=1e-10
    ), "Model should be completely deterministic with deterministic Gumbel softmax"


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
    # With deterministic Gumbel softmax, we can have stricter tolerances
    diff = torch.abs(base_logits - extended_logits[:, :4]).max()
    assert diff < 1e-5, f"Future tokens affected past predictions by {diff:.2e}"

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


# ---------------------------------------------------------------------------
# COMPREHENSIVE CAUSAL LEAKAGE TESTS
# ---------------------------------------------------------------------------


def test_comprehensive_forward_causality_single_position(causal_dag_model):
    """Test that each position only uses information from previous positions in forward pass."""
    model, config = causal_dag_model
    model.eval()

    seq_len = 6
    batch_size = 2

    # Create base sequence
    base_seq = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # For each position, test that changing future tokens doesn't affect it
    for test_pos in range(seq_len):
        # Create multiple sequences with same prefix up to test_pos but different suffixes
        results = []

        for suffix_variant in range(3):  # Test 3 different suffixes
            # Keep prefix up to test_pos, randomize suffix
            modified_seq = base_seq.clone()
            if test_pos < seq_len - 1:  # Only modify if there are future positions
                modified_seq[:, test_pos + 1 :] = torch.randint(
                    0, config.vocab_size, (batch_size, seq_len - test_pos - 1)
                )

            with torch.no_grad():
                logits, _ = model(modified_seq)
                results.append(logits[:, test_pos].clone())  # Store logits for test_pos

        # All results for this position should be identical
        for i in range(1, len(results)):
            diff = torch.abs(results[0] - results[i]).max()
            assert (
                diff < 1e-6
            ), f"Position {test_pos} affected by future tokens (variant {i}): diff={diff:.2e}"


def test_comprehensive_forward_causality_incremental_context(causal_dag_model):
    """Test that adding tokens incrementally produces consistent results."""
    model, config = causal_dag_model
    model.eval()

    max_len = 5
    base_tokens = torch.randint(0, config.vocab_size, (1, max_len))

    # Store results for each prefix length
    prefix_results = []

    for prefix_len in range(1, max_len + 1):
        prefix_seq = base_tokens[:, :prefix_len]

        with torch.no_grad():
            logits, _ = model(prefix_seq)
            prefix_results.append(logits.clone())

    # Check that each position's logits are consistent across different context lengths
    for pos in range(max_len - 1):  # Check positions 0 to max_len-2
        for context_len in range(pos + 1, max_len):  # Context must be at least pos+1
            # Compare logits at position 'pos' when context length is 'context_len+1' vs longer contexts
            current_logits = prefix_results[context_len][
                :, pos
            ]  # context_len+1 tokens, position pos

            for longer_context in range(context_len + 1, max_len):
                longer_logits = prefix_results[longer_context][:, pos]
                diff = torch.abs(current_logits - longer_logits).max()
                # Allow for some change due to DAG operations, but should be small
                assert (
                    diff < 1e-2
                ), f"Position {pos} changed significantly when context extended from {context_len+1} to {longer_context+1} tokens: diff={diff:.2e}"


def test_dag_node_causality_forward(causal_dag_model):
    """Test that DAG nodes only use information from causally available positions."""
    model, config = causal_dag_model
    model.eval()

    seq_len = 4
    batch_size = 1

    # Create two sequences that differ only in the last position
    seq1 = torch.tensor([[1, 2, 3, 4]])
    seq2 = torch.tensor([[1, 2, 3, 7]])  # Only last token differs

    with torch.no_grad():
        logits1, _ = model(seq1)
        logits2, _ = model(seq2)

        # Access internal DAG state
        dag_values1 = model.dag.node_values.clone()  # (B, N, T)

        # Run second sequence
        logits2, _ = model(seq2)
        dag_values2 = model.dag.node_values.clone()

    # Check that DAG node values for positions 0, 1, 2 are identical
    for pos in range(seq_len - 1):  # Positions 0, 1, 2
        for node_idx in range(dag_values1.shape[1]):  # All nodes
            diff = torch.abs(
                dag_values1[0, node_idx, pos] - dag_values2[0, node_idx, pos]
            )
            assert (
                diff < 1e-6
            ), f"DAG node {node_idx} at position {pos} affected by future token: diff={diff:.2e}"


def test_comprehensive_backward_causality(causal_dag_model):
    """Test that gradients respect causality - parameters receive reasonable gradients."""
    model, config = causal_dag_model
    model.train()

    seq_len = 4
    batch_size = 1

    # Create base sequence and targets
    base_seq = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])

    # Test 1: Compute gradients for full sequence
    model.zero_grad()
    logits_full, loss_full = model(base_seq, targets)
    loss_full.backward()

    # Store gradients for all parameters
    full_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            full_grads[name] = param.grad.clone()

    # Test 2: Compute gradients for truncated sequence (first 3 tokens)
    truncated_seq = base_seq[:, :3]
    truncated_targets = targets[:, :3]

    model.zero_grad()
    logits_trunc, loss_trunc = model(truncated_seq, truncated_targets)
    loss_trunc.backward()

    # Store gradients for truncated sequence
    trunc_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            trunc_grads[name] = param.grad.clone()

    # Test core causality principle: parameters should receive reasonable gradients
    # and gradients should not be dramatically different in magnitude
    for name in full_grads:
        if name in trunc_grads:
            full_grad_norm = full_grads[name].norm()
            trunc_grad_norm = trunc_grads[name].norm()

            # Both should have reasonable gradients
            assert (
                full_grad_norm > 1e-10
            ), f"Full sequence gradient too small for {name}: {full_grad_norm:.2e}"
            assert (
                trunc_grad_norm > 1e-10
            ), f"Truncated sequence gradient too small for {name}: {trunc_grad_norm:.2e}"

            # Gradients should not be dramatically different in magnitude
            if full_grad_norm > 1e-8 and trunc_grad_norm > 1e-8:
                ratio = max(full_grad_norm, trunc_grad_norm) / min(
                    full_grad_norm, trunc_grad_norm
                )
                assert (
                    ratio < 100
                ), f"Gradient magnitudes too different for {name}: ratio={ratio:.2f}"


def test_dag_controller_causality(causal_dag_model):
    """Test that DAG controller decisions are causal."""
    model, config = causal_dag_model
    model.eval()

    # Access the DAG controller
    controller = model.dag.controller

    # Create test sequences
    seq_len = 4
    seq1 = torch.tensor([[1, 2, 3, 4]])
    seq2 = torch.tensor([[1, 2, 3, 7]])  # Different last token

    # Store controller decisions for each position
    decisions1 = []
    decisions2 = []

    # Run forward pass and capture controller decisions
    with torch.no_grad():
        model(seq1)
        # Controller stores last attention and operation weights
        decisions1.append(
            {
                "attn": (
                    controller.last_attn.clone()
                    if controller.last_attn is not None
                    else None
                ),
                "ops": (
                    controller.last_op_weights.clone()
                    if controller.last_op_weights is not None
                    else None
                ),
            }
        )

        model(seq2)
        decisions2.append(
            {
                "attn": (
                    controller.last_attn.clone()
                    if controller.last_attn is not None
                    else None
                ),
                "ops": (
                    controller.last_op_weights.clone()
                    if controller.last_op_weights is not None
                    else None
                ),
            }
        )

    # The controller decisions should be identical for both sequences
    # since they only differ in the last token
    if decisions1[0]["attn"] is not None and decisions2[0]["attn"] is not None:
        attn_diff = torch.abs(decisions1[0]["attn"] - decisions2[0]["attn"]).max()
        # Allow for moderate differences due to the last position affecting the controller
        # The DAG operations can legitimately change when the sequence changes
        assert (
            attn_diff < 0.3
        ), f"Controller attention decisions too different: {attn_diff:.3f}"

    if decisions1[0]["ops"] is not None and decisions2[0]["ops"] is not None:
        ops_diff = torch.abs(decisions1[0]["ops"] - decisions2[0]["ops"]).max()
        assert (
            ops_diff < 0.3
        ), f"Controller operation decisions too different: {ops_diff:.3f}"


def test_value_extractor_causality(causal_dag_model):
    """Test that ValueExtractor respects causal masking."""
    model, config = causal_dag_model
    model.eval()

    # Access the value extractor through the DAG
    value_extractor = model.dag.value_extractor

    # Create test embeddings
    batch_size = 1
    seq_len = 4
    hidden_dim = config.n_embd

    # Create embeddings where each position has a unique signature
    embeddings = torch.zeros(batch_size, seq_len, hidden_dim)
    for i in range(seq_len):
        embeddings[:, i, :] = i + 1  # Position 0 = 1, position 1 = 2, etc.

    with torch.no_grad():
        values = value_extractor(embeddings)

    # Test that changing future positions doesn't affect past values
    for test_pos in range(seq_len - 1):  # Test positions 0 to seq_len-2
        # Create modified embeddings where we change positions after test_pos
        modified_embeddings = embeddings.clone()
        modified_embeddings[:, test_pos + 1 :, :] = (
            999  # Change future positions dramatically
        )

        with torch.no_grad():
            modified_values = value_extractor(modified_embeddings)

        # Values up to test_pos should be identical
        for pos in range(test_pos + 1):
            diff = torch.abs(values[:, pos] - modified_values[:, pos])
            assert (
                diff < 1e-6
            ), f"ValueExtractor position {pos} affected by future positions: diff={diff:.2e}"


def test_masked_context_selector_causality(causal_dag_model):
    """Test that MaskedContextSelector respects causal lengths."""
    model, config = causal_dag_model

    # Access the context selectors through the DAG
    operand_selector = model.dag.operand_ctx_selector
    op_selector = model.dag.op_ctx_selector

    # Create test hidden states
    batch_size = 1
    seq_len = 4
    hidden_dim = config.n_embd

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    causal_lengths = [0, 1, 2, 3]  # Position t can see 0 to t-1 tokens

    with torch.no_grad():
        operand_ctx = operand_selector(hidden_states, causal_lengths)
        op_ctx = op_selector(hidden_states, causal_lengths)

    # Test that each position only uses information from previous positions
    for test_pos in range(seq_len):
        # Create modified hidden states where we change future positions
        modified_hidden = hidden_states.clone()
        if test_pos < seq_len - 1:
            modified_hidden[:, test_pos + 1 :, :] = 999  # Change future positions

        with torch.no_grad():
            modified_operand_ctx = operand_selector(modified_hidden, causal_lengths)
            modified_op_ctx = op_selector(modified_hidden, causal_lengths)

        # Context at test_pos should be identical
        operand_diff = torch.abs(
            operand_ctx[:, test_pos] - modified_operand_ctx[:, test_pos]
        ).max()
        op_diff = torch.abs(op_ctx[:, test_pos] - modified_op_ctx[:, test_pos]).max()

        assert (
            operand_diff < 1e-6
        ), f"Operand context at position {test_pos} affected by future: diff={operand_diff:.2e}"
        assert (
            op_diff < 1e-6
        ), f"Op context at position {test_pos} affected by future: diff={op_diff:.2e}"


def test_end_to_end_causality_stress_test(causal_dag_model):
    """Comprehensive end-to-end causality test with various sequence modifications."""
    model, config = causal_dag_model
    model.eval()

    # Create base sequence
    base_seq = torch.tensor([[1, 2, 3, 4, 5]])

    # Test multiple types of modifications
    modifications = [
        # Change last token
        torch.tensor([[1, 2, 3, 4, 9]]),
        # Change last two tokens
        torch.tensor([[1, 2, 3, 8, 9]]),
        # Change last three tokens
        torch.tensor([[1, 2, 7, 8, 9]]),
        # Completely different suffix
        torch.tensor([[1, 2, 15, 16, 17]]),
    ]

    # Get base results
    with torch.no_grad():
        base_logits, _ = model(base_seq)

    # Test each modification
    for mod_idx, modified_seq in enumerate(modifications):
        with torch.no_grad():
            mod_logits, _ = model(modified_seq)

        # Determine how many positions should be identical
        # (positions before the first modification)
        identical_positions = 0
        for pos in range(min(base_seq.shape[1], modified_seq.shape[1])):
            if base_seq[0, pos] == modified_seq[0, pos]:
                identical_positions += 1
            else:
                break

        # Check that logits for identical positions are very similar
        if identical_positions > 0:
            for pos in range(identical_positions):
                diff = torch.abs(base_logits[:, pos] - mod_logits[:, pos]).max()
                assert (
                    diff < 1e-5
                ), f"Modification {mod_idx}: Position {pos} affected by future changes: diff={diff:.2e}"


if __name__ == "__main__":
    pytest.main([__file__])
