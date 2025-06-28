import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import wandb
from dag_logger import DAGLogger
from dag_model import GPT, GPTConfig, op_names

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
    model.train()
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


def test_operation_console_logging(small_dag_model, sample_batch):
    """Test that operation probabilities can be displayed in console format."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    logits, loss = model(input_ids, target_ids)

    logger = DAGLogger()
    try:
        logger.format_console_logging(model)
        assert True
    except Exception as e:
        pytest.fail(f"Console logging failed: {e}")


def test_operand_console_logging(small_dag_model, sample_batch):
    """Test that operand selection information can be displayed in console format."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    logits, loss = model(input_ids, target_ids)

    logger = DAGLogger()
    try:
        logger.format_console_logging(model)
        assert True
    except Exception as e:
        pytest.fail(f"Operand console logging failed: {e}")


def test_operation_gradient_capture(small_dag_model, sample_batch):
    """Test that operation gradients are correctly captured."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    logger = DAGLogger()
    logger.setup_gradient_tracking(model)

    logits, loss = model(input_ids, target_ids)

    # Update gradient tracking after forward pass when tensors are available
    logger.update_gradient_tracking(model)

    loss.backward()

    extra_vals = logger.get_extra_vals(model)

    expected_grad_keys = [f"op_grad/{op}" for op in op_names]
    found_grad_keys = [key for key in extra_vals.keys() if key.startswith("op_grad/")]

    assert len(found_grad_keys) > 0, "No operation gradients found in extra_vals"

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

    logger = DAGLogger()

    logger.setup_gradient_tracking(model)
    logits1, loss1 = model(input_ids, target_ids)
    loss1.backward()
    extra_vals1 = logger.get_extra_vals(model)
    gradients_run1 = [extra_vals1.get(f"op_grad/{op}", 0.0) for op in op_names]

    model.zero_grad()

    logger.setup_gradient_tracking(model)
    logits2, loss2 = model(input_ids, target_ids)
    loss2.backward()
    extra_vals2 = logger.get_extra_vals(model)
    gradients_run2 = [extra_vals2.get(f"op_grad/{op}", 0.0) for op in op_names]

    for i, op in enumerate(op_names):
        grad1, grad2 = gradients_run1[i], gradients_run2[i]

        assert torch.isfinite(
            torch.tensor(grad1)
        ), f"Gradient for {op} in run1 is not finite: {grad1}"
        assert torch.isfinite(
            torch.tensor(grad2)
        ), f"Gradient for {op} in run2 is not finite: {grad2}"

        assert abs(grad1) < 1.0, f"Gradient for {op} in run1 is too large: {grad1}"
        assert abs(grad2) < 1.0, f"Gradient for {op} in run2 is too large: {grad2}"


def test_extra_vals_includes_all_logging_info(small_dag_model, sample_batch):
    """Test that extra_vals includes gate/norm and gradient information."""
    model, _ = small_dag_model
    input_ids, target_ids = sample_batch

    # Set up gradient tracking before forward pass
    logger = DAGLogger()
    logger.setup_gradient_tracking(model)

    # Forward and backward pass
    logits, loss = model(input_ids, target_ids)

    # Update gradient tracking after forward pass when tensors are available
    logger.update_gradient_tracking(model)

    loss.backward()

    # Extract logging data including gate and norm values
    # Create dummy floating-point tensors for logging (since input_ids are integer token IDs)
    dummy_hidden = torch.randn(
        input_ids.shape[0], input_ids.shape[1], model.config.n_embd
    )
    dummy_dag_hidden = (
        torch.randn(input_ids.shape[0], input_ids.shape[1], model.config.n_embd) * 0.01
    )
    dummy_mixed = dummy_hidden * 0.5 + dummy_dag_hidden * 0.5
    logger.extract_all_logging_data(
        model, dummy_hidden, dummy_dag_hidden, None, dummy_mixed
    )

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
        logger.setup_gradient_tracking(model)
        logits, loss = model(input_ids, target_ids)

        # Update gradient tracking after forward pass when tensors are available
        logger.update_gradient_tracking(model)

        loss.backward()
        model.zero_grad()

        # Extract logging data including gate and norm values
        # Create dummy floating-point tensors for logging (since input_ids are integer token IDs)
        dummy_hidden = torch.randn(
            input_ids.shape[0], input_ids.shape[1], model.config.n_embd
        )
        dummy_dag_hidden = (
            torch.randn(input_ids.shape[0], input_ids.shape[1], model.config.n_embd)
            * 0.01
        )
        dummy_mixed = dummy_hidden * 0.5 + dummy_dag_hidden * 0.5
        logger.extract_all_logging_data(
            model, dummy_hidden, dummy_dag_hidden, None, dummy_mixed
        )

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

        # Set up gradient tracking before forward pass
        logger.setup_gradient_tracking(model)

        # Forward pass
        logits, loss = model(input_ids, target_ids)

        # Update gradient tracking after forward pass when tensors are available
        logger.update_gradient_tracking(model)

        # Backward pass
        loss.backward()

        # Extract logging data including gate and norm values
        # Create dummy floating-point tensors for logging (since input_ids are integer token IDs)
        dummy_hidden = torch.randn(input_ids.shape[0], input_ids.shape[1], cfg.n_embd)
        dummy_dag_hidden = (
            torch.randn(input_ids.shape[0], input_ids.shape[1], cfg.n_embd) * 0.01
        )
        dummy_mixed = dummy_hidden * 0.5 + dummy_dag_hidden * 0.5
        logger.extract_all_logging_data(
            model, dummy_hidden, dummy_dag_hidden, None, dummy_mixed
        )

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
            N == config.dag_scratch_nodes
        ), f"Expected {config.dag_scratch_nodes} nodes, got {N}"
        assert (
            H == config.dag_node_dim
        ), f"Expected {config.dag_node_dim} hidden dim, got {H}"

        # Node values should be (B, N, T)
        B, N, T = model.dag.node_values.shape
        assert (
            T == config.block_size
        ), f"Expected {config.block_size} time steps in values, got {T}"

    # NEW API: Extract logging data using DAGLogger
    logger = DAGLogger()

    # Simulate what happens in training loop - extract logging data
    # Get the internal values (this would be done in the actual training code)
    if hasattr(model, "dag"):
        # Simulate the values that would be passed from the model's forward pass
        original_hidden = torch.randn(1, config.block_size, config.n_embd)
        dag_hidden = torch.randn(1, config.block_size, config.n_embd) * 0.01
        gate_values = [torch.rand(1, 1) for _ in range(config.block_size)]
        mixed_hidden = original_hidden * 0.5 + dag_hidden * 0.5

        # Extract all logging data in one centralized call
        logger.extract_all_logging_data(
            model,
            original_hidden=original_hidden,
            dag_hidden=dag_hidden,
            gate_values=gate_values,
            mixed_hidden=mixed_hidden,
        )

    # Now get the metrics using the new centralized approach
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

    # Test that we can get node values list using DAGLogger
    node_values = logger.get_node_values_list(model)
    assert (
        len(node_values) == config.block_size
    ), f"Expected {config.block_size} node values, got {len(node_values)}"
    assert all(
        torch.isfinite(torch.tensor(val)) for val in node_values
    ), "All node values should be finite"

    # Test detailed node values using DAGLogger
    detailed_values = logger.get_detailed_node_values(model)
    assert detailed_values, "Should have detailed node values"
    assert (
        len(detailed_values["values_per_token"]) == config.block_size
    ), "Should have values for all tokens"


def test_dag_hidden_gradient_logging():
    """Test that DAG hidden gradients are captured and logged correctly."""
    cfg = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100, dag_depth=2
    )
    model = GPT(cfg)
    logger = DAGLogger()

    # Forward pass
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, y)

    loss.backward()

    # Extract logging data including DAG hidden gradients
    # We need to simulate the values that would be passed in a real training scenario
    original_hidden = torch.randn(2, 8, cfg.n_embd, requires_grad=True)
    dag_hidden = torch.randn(2, 8, cfg.n_embd, requires_grad=True)
    mixed_hidden = original_hidden * 0.5 + dag_hidden * 0.5

    logger.extract_all_logging_data(
        model, original_hidden, dag_hidden, None, mixed_hidden
    )

    # Setup gradient tracking AFTER extract_all_logging_data creates model.last_dag_hidden
    logger.setup_gradient_tracking(model)

    # Create a fake loss to establish gradients for the DAG hidden tensor
    fake_loss = dag_hidden.sum() * 0.001  # Small multiplier to ensure gradients exist
    fake_loss.backward(retain_graph=True)

    # Check that DAG hidden gradients are captured
    extra_vals = logger.get_extra_vals(model)

    expected_dag_grad_keys = [
        "dag_grad/dag_hidden_grad_norm",
        "dag_grad/dag_hidden_grad_mean",
        "dag_grad/dag_hidden_grad_std",
        "dag_grad/dag_hidden_grad_max",
        "dag_grad/dag_hidden_grad_min",
    ]

    for key in expected_dag_grad_keys:
        assert key in extra_vals, f"Missing DAG hidden gradient key: {key}"
        assert isinstance(
            extra_vals[key], float
        ), f"DAG gradient {key} should be a float"
        assert not torch.isnan(
            torch.tensor(extra_vals[key])
        ), f"DAG gradient {key} should not be NaN"
        assert torch.isfinite(
            torch.tensor(extra_vals[key])
        ), f"DAG gradient {key} should be finite"

    # Verify gradient norm is positive
    grad_norm = extra_vals["dag_grad/dag_hidden_grad_norm"]
    assert grad_norm >= 0, "Gradient norm should be non-negative"

    # Verify std is non-negative
    grad_std = extra_vals["dag_grad/dag_hidden_grad_std"]
    assert grad_std >= 0, "Gradient std should be non-negative"

    # Verify max >= min
    grad_max = extra_vals["dag_grad/dag_hidden_grad_max"]
    grad_min = extra_vals["dag_grad/dag_hidden_grad_min"]
    assert grad_max >= grad_min, "Gradient max should be >= gradient min"


if __name__ == "__main__":
    pytest.main([__file__])
