# tests/test_dag_operation_logging.py
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import wandb
from dag_logger import DAGLogger
from dag_model import GPT, GPTConfig, op_names

# Add the parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
def test_operation_console_logging(small_dag_model, sample_batch):
    """Test that operation probabilities can be displayed in console format."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # Get console logging via DAGLogger (should not crash)
    logger = DAGLogger()
    try:
        logger.format_console_logging(model)
        # If we get here without exception, console logging works
        assert True
    except Exception as e:
        pytest.fail(f"Console logging failed: {e}")


# --------------------------------------------------------------------- #
# Test operand probabilities logging
# --------------------------------------------------------------------- #
def test_operand_console_logging(small_dag_model, sample_batch):
    """Test that operand selection information can be displayed in console format."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Forward pass
    logits, loss = model(input_ids, target_ids)

    # Get console logging via DAGLogger (should not crash)
    logger = DAGLogger()
    try:
        logger.format_console_logging(model)
        # If we get here without exception, operand logging works
        assert True
    except Exception as e:
        pytest.fail(f"Operand console logging failed: {e}")


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
    """Test that extra_vals includes gate/norm and gradient information."""
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

    # Check for gate and norm values
    gate_keys = [key for key in extra_vals.keys() if key.startswith("gate/")]
    norm_keys = [key for key in extra_vals.keys() if key.startswith("norm/")]
    assert len(gate_keys) > 0 or len(norm_keys) > 0, "No gate or norm values found"

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

        # Check logging still works
        extra_vals = logger.get_extra_vals(model)

        # Check that we have meaningful logging data
        gate_keys = [k for k in extra_vals if k.startswith("gate/")]
        norm_keys = [k for k in extra_vals if k.startswith("norm/")]
        grad_keys = [k for k in extra_vals if k.startswith("op_grad/")]

        assert (
            len(gate_keys) > 0 or len(norm_keys) > 0
        ), f"Iteration {i}: no gate or norm values found"
        assert len(grad_keys) > 0, f"Iteration {i}: no operation gradients found"


def test_gradient_tracking_with_grad_context(small_dag_model, sample_batch):
    """Test that gradient tracking respects torch.no_grad() context."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Forward pass without gradients
    with torch.no_grad():
        _, _ = model(input_ids, target_ids)

    # Should still be able to get console logging
    logger = DAGLogger()
    try:
        logger.format_console_logging(model)
        # If we get here without exception, console logging works under no_grad
        assert True
    except Exception as e:
        pytest.fail(f"Console logging failed under no_grad: {e}")


# --------------------------------------------------------------------- #
# Integration test with training-like scenario
# --------------------------------------------------------------------- #
def test_logging_integration_training_scenario(small_dag_model):
    """Test logging functionality in a training-like scenario with multiple batches."""
    model, cfg = small_dag_model

    # Set up logger
    logger = DAGLogger()

    # Simulate training loop
    all_gate_vals = []
    all_norm_vals = []
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

        # Collect logging information
        extra_vals = logger.get_extra_vals(model)

        all_gate_vals.append(
            {k: v for k, v in extra_vals.items() if k.startswith("gate/")}
        )
        all_norm_vals.append(
            {k: v for k, v in extra_vals.items() if k.startswith("norm/")}
        )
        all_gradients.append(
            {k: v for k, v in extra_vals.items() if k.startswith("op_grad/")}
        )

        # Reset gradients
        model.zero_grad()

    # Verify we collected data for all steps
    assert len(all_gate_vals) == 5
    assert len(all_norm_vals) == 5
    assert len(all_gradients) == 5

    # Verify each step has complete data
    for step in range(5):
        # Should have either gate or norm values
        assert (
            len(all_gate_vals[step]) > 0 or len(all_norm_vals[step]) > 0
        ), f"Step {step}: no gate or norm values found"

        # Should have complete gradient data
        assert len(all_gradients[step]) == len(
            op_names
        ), f"Step {step}: incomplete gradients"

        # Gate values should be in [0,1]
        for gate_key, gate_val in all_gate_vals[step].items():
            assert isinstance(
                gate_val, float
            ), f"Step {step}: gate {gate_key} is not float"
            assert (
                0.0 <= gate_val <= 1.0
            ), f"Step {step}: gate {gate_key} not in [0,1]: {gate_val}"

        # Norm values should be positive
        for norm_key, norm_val in all_norm_vals[step].items():
            assert isinstance(
                norm_val, float
            ), f"Step {step}: norm {norm_key} is not float"
            assert (
                norm_val >= 0
            ), f"Step {step}: norm {norm_key} is negative: {norm_val}"

        # Gradients should be reasonable values
        for grad_key, grad_val in all_gradients[step].items():
            assert isinstance(
                grad_val, float
            ), f"Step {step}: gradient {grad_key} is not float"
            assert (
                abs(grad_val) < 1.0
            ), f"Step {step}: gradient {grad_key} too large: {grad_val}"


def test_gate_and_norm_logging():
    """Test that gate values and norm values are properly captured and logged."""
    from dag_model import GPT, GPTConfig

    # Create a small DAG-enabled model
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=8,
        vocab_size=32,
        block_size=8,
        bias=False,
        dag_depth=2,
    )
    model = GPT(config)
    model.eval()

    # Create input
    x = torch.randint(0, config.vocab_size, (1, config.block_size))

    # Forward pass to generate activations
    with torch.no_grad():
        logits, loss = model(x)

    # Check matrix shapes
    if hasattr(model, "dag"):
        # Node embeddings should be (B, N, T, H)
        B, N, T, H = model.dag.node_embeds.shape
        assert (
            T == config.block_size
        ), f"Expected {config.block_size} time steps, got {T}"
        assert (
            N == config.dag_depth + 1
        ), f"Expected {config.dag_depth + 1} nodes, got {N}"
        assert H == config.n_embd, f"Expected {config.n_embd} hidden dim, got {H}"

        # Node values should be (B, N, T)
        B, N, T = model.dag.node_values.shape
        assert (
            T == config.block_size
        ), f"Expected {config.block_size} time steps in values, got {T}"

    # Get gate and norm values using DAGLogger
    logger = DAGLogger()
    extra_vals = logger.get_extra_vals(model)

    # Check that gate values are present and properly shaped
    assert "gate/mean" in extra_vals, "Gate mean should be logged"
    assert "gate/min" in extra_vals, "Gate min should be logged"
    assert "gate/max" in extra_vals, "Gate max should be logged"
    assert (
        "gate/close_to_zero_ratio" in extra_vals
    ), "Gate close_to_zero_ratio should be logged"

    # Check that norm values are present and properly shaped
    assert "norm/hidden" in extra_vals, "Hidden norm should be logged"
    # Note: dag_sem_raw is no longer available since DAG now returns post-processed hidden states
    assert "norm/dag_sem" in extra_vals, "DAG semantic normalized norm should be logged"
    assert "norm/fused" in extra_vals, "Fused norm should be logged"

    # Verify gate values are in reasonable range [0, 1]
    assert 0.0 <= extra_vals["gate/mean"] <= 1.0, "Gate mean should be in [0, 1]"
    assert 0.0 <= extra_vals["gate/min"] <= 1.0, "Gate min should be in [0, 1]"
    assert 0.0 <= extra_vals["gate/max"] <= 1.0, "Gate max should be in [0, 1]"
    assert (
        0.0 <= extra_vals["gate/close_to_zero_ratio"] <= 1.0
    ), "Close to zero ratio should be in [0, 1]"

    # Verify norm values are positive and properly shaped
    assert extra_vals["norm/hidden"] > 0, "Hidden norm should be positive"
    assert extra_vals["norm/dag_sem"] > 0, "DAG semantic raw norm should be positive"
    assert (
        extra_vals["norm/dag_sem"] > 0
    ), "DAG semantic normalized norm should be positive"
    assert extra_vals["norm/fused"] > 0, "Fused norm should be positive"

    # Test that we can get node values list
    node_values = model.get_node_values_list()
    assert (
        len(node_values) == config.block_size
    ), f"Expected {config.block_size} node values, got {len(node_values)}"
    assert all(
        torch.isfinite(torch.tensor(val)) for val in node_values
    ), "All node values should be finite"


if __name__ == "__main__":
    pytest.main([__file__])
