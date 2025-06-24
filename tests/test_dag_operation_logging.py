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
from dag_logger import DAGLogger
from dag_model import GPT, GPTConfig, op_names


# --------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------- #
@pytest.fixture
def small_dag_model():
    """Create a small DAG model for testing."""
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

    # Get operation probabilities using DAGLogger
    logger = DAGLogger()
    op_probs = logger.get_op_probabilities(model)

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
    logger = DAGLogger()
    op_probs = logger.get_op_probabilities(model)

    # Should return empty dict
    assert op_probs == {}


# --------------------------------------------------------------------- #
# Test operand probabilities logging
# --------------------------------------------------------------------- #
def test_get_operand_probabilities(small_dag_model, sample_batch):
    """Test that operand selection probabilities are correctly returned."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # Get operand probabilities using DAGLogger
    logger = DAGLogger()
    operand_probs = logger.get_operand_probabilities(model)

    # Check that we have probabilities for both operands across all nodes
    # For a model with initial embeddings, we should have at least some nodes
    assert len(operand_probs) > 0, "No operand probabilities found"

    # Check that operand probabilities sum to 1 for each operand
    operand1_probs = [
        v for k, v in operand_probs.items() if k.startswith("operand1_probs/")
    ]
    operand2_probs = [
        v for k, v in operand_probs.items() if k.startswith("operand2_probs/")
    ]

    if operand1_probs:
        operand1_sum = sum(operand1_probs)
        assert (
            abs(operand1_sum - 1.0) < 1e-5
        ), f"Operand1 probabilities don't sum to 1: {operand1_sum}"

    if operand2_probs:
        operand2_sum = sum(operand2_probs)
        assert (
            abs(operand2_sum - 1.0) < 1e-5
        ), f"Operand2 probabilities don't sum to 1: {operand2_sum}"

    # Check that all probabilities are valid
    for key, prob in operand_probs.items():
        assert isinstance(prob, float), f"Probability {prob} for {key} is not a float"
        assert 0.0 <= prob <= 1.0, f"Probability {prob} for {key} is not in [0,1]"
        assert not torch.isnan(torch.tensor(prob)), f"Probability for {key} is NaN"


def test_get_operand_probabilities_no_forward_pass(small_dag_model):
    """Test that get_operand_probabilities returns empty dict when no forward pass has been done."""
    model, _ = small_dag_model

    # Get operand probabilities without forward pass
    logger = DAGLogger()
    operand_probs = logger.get_operand_probabilities(model)

    # Should return empty dict
    assert operand_probs == {}


# --------------------------------------------------------------------- #
# Test gradient logging
# --------------------------------------------------------------------- #
def test_operation_gradient_capture(small_dag_model, sample_batch):
    """Test that operation gradients are correctly captured."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # Set up gradient tracking before backward pass
    logger = DAGLogger()
    logger.setup_gradient_tracking(model)

    # Backward pass to compute gradients
    loss.backward()

    # Get extra values which should include gradients
    extra_vals = logger.get_extra_vals(model)

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

    # Set up logger
    logger = DAGLogger()

    # First run
    logits1, loss1 = model(input_ids, target_ids)
    logger.setup_gradient_tracking(model)
    loss1.backward()
    extra_vals1 = logger.get_extra_vals(model)
    gradients_run1 = [extra_vals1.get(f"op_grad/{op}", 0.0) for op in op_names]

    # Reset model gradients
    model.zero_grad()

    # Second run with same input (gradients may vary due to Gumbel sampling)
    logits2, loss2 = model(input_ids, target_ids)
    logger.setup_gradient_tracking(model)
    loss2.backward()
    extra_vals2 = logger.get_extra_vals(model)
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

    # Set up gradient tracking before backward pass
    logger = DAGLogger()
    logger.setup_gradient_tracking(model)

    loss.backward()

    # Get extra values using DAGLogger
    extra_vals = logger.get_extra_vals(model)

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
    cfg = GPTConfig(
        vocab_size=50,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dag_depth=0,  # No DAG
        dropout=0.0,
        bias=False,
    )

    # This should create a regular GPT model
    from dag_model import GPT

    model = GPT(cfg)

    input_ids, target_ids = sample_batch

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # These methods shouldn't exist on regular GPT
    assert not hasattr(model, "get_op_probabilities")
    assert not hasattr(model, "get_operand_probabilities")


def test_logging_after_multiple_forward_passes(small_dag_model, sample_batch):
    """Test that logging works correctly after multiple forward passes."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Set up logger
    logger = DAGLogger()

    # Multiple forward passes
    for i in range(3):
        logits, loss = model(input_ids, target_ids)
        logger.setup_gradient_tracking(model)
        loss.backward()
        model.zero_grad()

        # Check that logging still works using DAGLogger
        op_probs = logger.get_op_probabilities(model)
        operand_probs = logger.get_operand_probabilities(model)
        extra_vals = logger.get_extra_vals(model)

        assert len(op_probs) == len(
            op_names
        ), f"Iteration {i}: wrong number of probabilities"
        assert len(operand_probs) > 0, f"Iteration {i}: no operand probabilities found"
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

    # Should still be able to get probabilities and operand probs using DAGLogger
    logger = DAGLogger()
    op_probs = logger.get_op_probabilities(model)
    operand_probs = logger.get_operand_probabilities(model)

    assert len(op_probs) == len(op_names), "Probabilities not available under no_grad"
    assert len(operand_probs) > 0, "Operand probabilities not available under no_grad"

    # But gradients should be empty or zero since no backward pass
    extra_vals = logger.get_extra_vals(model)
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

    # Set up logger
    logger = DAGLogger()

    # Simulate training loop
    all_op_probs = []
    all_operand_probs = []
    all_gradients = []

    for step in range(5):
        # Generate new batch
        batch_size = 2
        seq_len = cfg.block_size
        input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits, loss = model(input_ids, target_ids)

        # Set up gradient tracking before backward pass
        logger.setup_gradient_tracking(model)

        # Backward pass
        loss.backward()

        # Collect logging information using DAGLogger
        op_probs = logger.get_op_probabilities(model)
        operand_probs = logger.get_operand_probabilities(model)
        extra_vals = logger.get_extra_vals(model)

        all_op_probs.append(op_probs)
        all_operand_probs.append(operand_probs)
        all_gradients.append(
            {k: v for k, v in extra_vals.items() if k.startswith("op_grad/")}
        )

        # Reset gradients
        model.zero_grad()

    # Verify we collected data for all steps
    assert len(all_op_probs) == 5
    assert len(all_operand_probs) == 5
    assert len(all_gradients) == 5

    # Verify each step has complete data
    for step in range(5):
        assert len(all_op_probs[step]) == len(
            op_names
        ), f"Step {step}: incomplete probabilities"
        assert (
            len(all_operand_probs[step]) > 0
        ), f"Step {step}: no operand probabilities"
        assert len(all_gradients[step]) == len(
            op_names
        ), f"Step {step}: incomplete gradients"

        # Check probability validity
        prob_sum = sum(all_op_probs[step].values())
        assert abs(prob_sum - 1.0) < 1e-5, f"Step {step}: probabilities don't sum to 1"


if __name__ == "__main__":
    pytest.main([__file__])
