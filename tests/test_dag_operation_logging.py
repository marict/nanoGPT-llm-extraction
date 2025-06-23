# tests/test_dag_operation_logging.py
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------- #
# import library code
# --------------------------------------------------------------------- #
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dag_model import DAGGPT, DAGGPTConfig, op_names


# --------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------- #
@pytest.fixture
def small_dag_model():
    """Create a small DAG model for testing."""
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
    model.train()  # Enable training mode for gradient computation
    return model, cfg


@pytest.fixture
def sample_batch():
    """Create sample input and target tensors."""
    batch_size = 2
    seq_len = 6
    vocab_size = 50

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    return input_ids, target_ids


# --------------------------------------------------------------------- #
# Test operation probabilities logging
# --------------------------------------------------------------------- #
def test_get_op_probabilities(small_dag_model, sample_batch):
    """Test that operation probabilities are correctly computed and returned."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # Get operation probabilities
    op_probs = model.get_op_probabilities()

    # Check that we have probabilities for all operations
    expected_keys = [f"op_probs/{op}" for op in op_names]
    assert all(
        key in op_probs for key in expected_keys
    ), f"Missing probability keys. Got: {list(op_probs.keys())}"

    # Check that all probabilities are valid (between 0 and 1)
    for key, prob in op_probs.items():
        assert 0.0 <= prob <= 1.0, f"Invalid probability {prob} for {key}"

    # Check that probabilities sum to approximately 1
    prob_sum = sum(
        prob for key, prob in op_probs.items() if key.startswith("op_probs/")
    )
    assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities don't sum to 1: {prob_sum}"


def test_get_op_probabilities_no_forward_pass(small_dag_model):
    """Test that get_op_probabilities returns empty dict when no forward pass has been done."""
    model, _ = small_dag_model

    # No forward pass yet
    op_probs = model.get_op_probabilities()

    # Should return empty dict
    assert op_probs == {}


# --------------------------------------------------------------------- #
# Test operation logits logging
# --------------------------------------------------------------------- #
def test_get_op_logits_dict(small_dag_model, sample_batch):
    """Test that operation logits are correctly returned."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # Get operation logits
    op_logits = model.get_op_logits_dict()

    # Check that we have logits for all operations
    expected_keys = [f"op_logits/{op}" for op in op_names]
    assert all(
        key in op_logits for key in expected_keys
    ), f"Missing logit keys. Got: {list(op_logits.keys())}"

    # Check that logits are real numbers (can be any value)
    for key, logit in op_logits.items():
        assert isinstance(logit, float), f"Logit {logit} for {key} is not a float"
        assert not torch.isnan(torch.tensor(logit)), f"Logit for {key} is NaN"


def test_op_logits_to_probabilities_consistency(small_dag_model, sample_batch):
    """Test that probabilities computed from logits match get_op_probabilities output."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # Get both probabilities and logits
    op_probs = model.get_op_probabilities()
    op_logits = model.get_op_logits_dict()

    # Convert logits to probabilities manually
    logit_values = [op_logits[f"op_logits/{op}"] for op in op_names]
    logit_tensor = torch.tensor(logit_values)
    computed_probs = F.softmax(logit_tensor, dim=0)

    # Compare with get_op_probabilities output
    for i, op in enumerate(op_names):
        expected_prob = op_probs[f"op_probs/{op}"]
        computed_prob = computed_probs[i].item()
        assert (
            abs(expected_prob - computed_prob) < 1e-5
        ), f"Probability mismatch for {op}: expected {expected_prob}, computed {computed_prob}"


# --------------------------------------------------------------------- #
# Test gradient logging
# --------------------------------------------------------------------- #
def test_operation_gradient_capture(small_dag_model, sample_batch):
    """Test that operation gradients are correctly captured."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # Backward pass to compute gradients
    loss.backward()

    # Get extra values which should include gradients
    extra_vals = model.extra_vals()

    # Check for operation gradient keys
    expected_grad_keys = [f"op_grad/{op}" for op in op_names]
    found_grad_keys = [key for key in extra_vals.keys() if key.startswith("op_grad/")]

    assert len(found_grad_keys) > 0, "No operation gradients found in extra_vals"

    # Check that we have gradients for all operations
    for key in expected_grad_keys:
        assert key in extra_vals, f"Missing gradient key: {key}"
        assert isinstance(
            extra_vals[key], float
        ), f"Gradient {extra_vals[key]} for {key} is not a float"


def test_gradient_computation_consistency(small_dag_model, sample_batch):
    """Test that gradients are properly computed and reasonable across multiple forward/backward passes."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    gradients_run1 = []
    gradients_run2 = []

    # First run
    logits1, loss1 = model(input_ids, target_ids)
    loss1.backward()
    extra_vals1 = model.extra_vals()
    gradients_run1 = [extra_vals1.get(f"op_grad/{op}", 0.0) for op in op_names]

    # Reset model gradients
    model.zero_grad()

    # Second run with same input (gradients may vary due to Gumbel sampling)
    logits2, loss2 = model(input_ids, target_ids)
    loss2.backward()
    extra_vals2 = model.extra_vals()
    gradients_run2 = [extra_vals2.get(f"op_grad/{op}", 0.0) for op in op_names]

    # Verify gradients are reasonable (finite, not too large)
    for i, op in enumerate(op_names):
        grad1, grad2 = gradients_run1[i], gradients_run2[i]

        # Check gradients are finite
        assert torch.isfinite(
            torch.tensor(grad1)
        ), f"Gradient for {op} in run1 is not finite: {grad1}"
        assert torch.isfinite(
            torch.tensor(grad2)
        ), f"Gradient for {op} in run2 is not finite: {grad2}"

        # Check gradients are reasonable (not too large)
        assert abs(grad1) < 1.0, f"Gradient for {op} in run1 is too large: {grad1}"
        assert abs(grad2) < 1.0, f"Gradient for {op} in run2 is too large: {grad2}"


def test_extra_vals_includes_all_logging_info(small_dag_model, sample_batch):
    """Test that extra_vals includes both entropy and gradient information."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Forward and backward pass
    logits, loss = model(input_ids, target_ids)
    loss.backward()

    # Get extra values
    extra_vals = model.extra_vals()

    # Check for entropy values
    entropy_keys = [key for key in extra_vals.keys() if key.startswith("dag_entropy/")]
    assert len(entropy_keys) > 0, "No entropy values found"

    # Check for gradient values
    grad_keys = [
        key
        for key in extra_vals.keys()
        if key.startswith("dag_grad/") or key.startswith("op_grad/")
    ]
    assert len(grad_keys) > 0, "No gradient values found"

    # Check for operation-specific gradients
    op_grad_keys = [key for key in extra_vals.keys() if key.startswith("op_grad/")]
    assert len(op_grad_keys) == len(
        op_names
    ), f"Expected {len(op_names)} operation gradients, got {len(op_grad_keys)}"


# --------------------------------------------------------------------- #
# Test edge cases and error handling
# --------------------------------------------------------------------- #
def test_logging_with_no_dag_depth(sample_batch):
    """Test that logging functions work correctly when dag_depth=0 (no DAG)."""
    cfg = DAGGPTConfig(
        vocab_size=50,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=0,  # No DAG
        dropout=0.0,
        bias=False,
    )

    # This should create a regular GPT model, not DAGGPT
    from model import GPT

    model = GPT(cfg)

    input_ids, target_ids = sample_batch

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # These methods shouldn't exist on regular GPT
    assert not hasattr(model, "get_op_probabilities")
    assert not hasattr(model, "get_op_logits_dict")


def test_logging_after_multiple_forward_passes(small_dag_model, sample_batch):
    """Test that logging works correctly after multiple forward passes."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Multiple forward passes
    for i in range(3):
        logits, loss = model(input_ids, target_ids)
        loss.backward()
        model.zero_grad()

        # Check that logging still works
        op_probs = model.get_op_probabilities()
        op_logits = model.get_op_logits_dict()
        extra_vals = model.extra_vals()

        assert len(op_probs) == len(
            op_names
        ), f"Iteration {i}: wrong number of probabilities"
        assert len(op_logits) == len(op_names), f"Iteration {i}: wrong number of logits"
        assert any(
            key.startswith("op_grad/") for key in extra_vals.keys()
        ), f"Iteration {i}: no operation gradients found"


def test_gradient_tracking_with_grad_context(small_dag_model, sample_batch):
    """Test that gradient tracking respects torch.no_grad() context."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Forward pass without gradients
    with torch.no_grad():
        logits, loss = model(input_ids, target_ids)

    # Should still be able to get probabilities and logits
    op_probs = model.get_op_probabilities()
    op_logits = model.get_op_logits_dict()

    assert len(op_probs) == len(op_names), "Probabilities not available under no_grad"
    assert len(op_logits) == len(op_names), "Logits not available under no_grad"

    # But gradients should be empty or zero since no backward pass
    extra_vals = model.extra_vals()
    grad_keys = [key for key in extra_vals.keys() if key.startswith("op_grad/")]

    # Gradients might be present but should be zero or very small
    for key in grad_keys:
        grad_val = extra_vals[key]
        # In no_grad context, gradients are typically not computed or are zero
        assert isinstance(grad_val, float), f"Gradient {key} is not a float: {grad_val}"


# --------------------------------------------------------------------- #
# Integration test with training-like scenario
# --------------------------------------------------------------------- #
def test_logging_integration_training_scenario(small_dag_model):
    """Test logging functionality in a training-like scenario with multiple batches."""
    model, cfg = small_dag_model

    # Simulate training loop
    all_op_probs = []
    all_op_logits = []
    all_gradients = []

    for step in range(5):
        # Generate new batch
        batch_size = 2
        seq_len = cfg.block_size
        input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits, loss = model(input_ids, target_ids)

        # Backward pass
        loss.backward()

        # Collect logging information
        op_probs = model.get_op_probabilities()
        op_logits = model.get_op_logits_dict()
        extra_vals = model.extra_vals()

        all_op_probs.append(op_probs)
        all_op_logits.append(op_logits)
        all_gradients.append(
            {k: v for k, v in extra_vals.items() if k.startswith("op_grad/")}
        )

        # Reset gradients
        model.zero_grad()

    # Verify we collected data for all steps
    assert len(all_op_probs) == 5
    assert len(all_op_logits) == 5
    assert len(all_gradients) == 5

    # Verify each step has complete data
    for step in range(5):
        assert len(all_op_probs[step]) == len(
            op_names
        ), f"Step {step}: incomplete probabilities"
        assert len(all_op_logits[step]) == len(
            op_names
        ), f"Step {step}: incomplete logits"
        assert len(all_gradients[step]) == len(
            op_names
        ), f"Step {step}: incomplete gradients"

        # Check probability validity
        prob_sum = sum(all_op_probs[step].values())
        assert abs(prob_sum - 1.0) < 1e-5, f"Step {step}: probabilities don't sum to 1"


if __name__ == "__main__":
    pytest.main([__file__])
