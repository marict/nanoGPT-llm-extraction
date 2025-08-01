"""Test that exec losses are properly computed and flow through gradients."""

import pytest
import torch

from models.dag_model import DAGExecutor
from predictor_utils import compute_dag_loss


@pytest.fixture
def device():
    """Return the device to use for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def test_dag_config():
    """Return test configuration for DAG model."""
    return {
        "dag_depth": 3,
        "total_nodes": 7,  # (3 + 1) initial + 3 intermediate = 4 + 3 = 7
        "batch_size": 2,
        "seq_len": 4,
    }


@pytest.fixture
def dag_executor(test_dag_config):
    """Create a DAGExecutor for testing."""
    return DAGExecutor(dag_depth=test_dag_config["dag_depth"])


def create_test_tensors(config, device):
    """Create test prediction tensors that require gradients."""
    B, T = config["batch_size"], config["seq_len"]
    total_nodes = config["total_nodes"]
    dag_depth = config["dag_depth"]

    # Create prediction tensors with gradients enabled
    pred_V_mag = torch.randn(B, T, total_nodes, device=device, requires_grad=True)
    pred_V_sign = torch.randn(B, T, total_nodes, device=device, requires_grad=True)
    pred_O = torch.randn(
        B, T, dag_depth, total_nodes, device=device, requires_grad=True
    )
    pred_G = torch.randn(B, T, dag_depth, device=device, requires_grad=True)

    return pred_V_mag, pred_V_sign, pred_O, pred_G


def create_test_targets(config, device):
    """Create test target tensors and valid mask."""
    B, T = config["batch_size"], config["seq_len"]
    total_nodes = config["total_nodes"]
    dag_depth = config["dag_depth"]

    target_tensors = []
    for b in range(B):
        batch_targets = []
        for t in range(T):
            target_dict = {
                "target_V_mag": torch.randn(total_nodes, device=device),
                "target_V_sign": torch.randn(total_nodes, device=device),
                "target_O": torch.randn(dag_depth, total_nodes, device=device),
                "target_G": torch.sigmoid(
                    torch.randn(dag_depth, device=device)
                ),  # Valid gates [0,1]
                "target_final_exec": torch.randn(
                    1, device=device
                ).item(),  # Scalar value
            }
            batch_targets.append(target_dict)
        target_tensors.append(batch_targets)

    # Create valid mask - mark some positions as valid
    valid_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    valid_mask[0, 1] = True  # Valid position at batch 0, token 1
    valid_mask[1, 2] = True  # Valid position at batch 1, token 2

    return target_tensors, valid_mask


def test_exec_loss_computed_with_dag_executor(test_dag_config, dag_executor, device):
    """Test that exec loss is computed when dag_executor is provided."""
    pred_V_mag, pred_V_sign, pred_O, pred_G = create_test_tensors(
        test_dag_config, device
    )
    target_tensors, valid_mask = create_test_targets(test_dag_config, device)

    # Move dag_executor to correct device
    dag_executor = dag_executor.to(device)

    # Compute losses WITH dag_executor
    losses_with_executor = compute_dag_loss(
        pred_V_mag,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=dag_executor,
    )

    # Compute losses WITHOUT dag_executor
    losses_without_executor = compute_dag_loss(
        pred_V_mag,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=None,
    )

    # With executor, exec_loss should be non-zero
    assert "exec_loss" in losses_with_executor
    assert (
        losses_with_executor["exec_loss"].item() > 0.0
    ), "Exec loss should be non-zero when executor is provided"

    # Without executor, exec_loss should be zero
    assert "exec_loss" in losses_without_executor
    assert (
        losses_without_executor["exec_loss"].item() == 0.0
    ), "Exec loss should be zero when executor is None"

    # Total loss with executor should be higher than without executor
    total_with = losses_with_executor["total_loss"].item()
    total_without = losses_without_executor["total_loss"].item()
    assert (
        total_with > total_without
    ), "Total loss should be higher when exec loss is included"


def test_exec_loss_gradient_flow(test_dag_config, dag_executor, device):
    """Test that exec loss contributes to gradients of prediction tensors."""
    pred_V_mag, pred_V_sign, pred_O, pred_G = create_test_tensors(
        test_dag_config, device
    )
    target_tensors, valid_mask = create_test_targets(test_dag_config, device)

    # Move dag_executor to correct device
    dag_executor = dag_executor.to(device)

    # Compute losses with dag_executor
    losses = compute_dag_loss(
        pred_V_mag,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=dag_executor,
    )

    # Check that exec_loss requires gradients
    exec_loss = losses["exec_loss"]
    assert exec_loss.requires_grad, "Exec loss should require gradients"

    # Backward pass through exec_loss only
    exec_loss.backward(retain_graph=True)

    # Check that prediction tensors have gradients from exec loss
    assert (
        pred_V_mag.grad is not None
    ), "pred_V_mag should have gradients from exec loss"
    assert (
        pred_V_sign.grad is not None
    ), "pred_V_sign should have gradients from exec loss"
    assert pred_O.grad is not None, "pred_O should have gradients from exec loss"
    assert pred_G.grad is not None, "pred_G should have gradients from exec loss"

    # Check that gradients are non-zero for at least some parameters
    # Note: At least V_mag and V_sign should have gradients since they directly affect execution
    assert torch.any(
        pred_V_mag.grad != 0
    ), "Some pred_V_mag gradients should be non-zero"
    assert torch.any(
        pred_V_sign.grad != 0
    ), "Some pred_V_sign gradients should be non-zero"

    # Verify that exec loss gradients flow to the computation graph
    assert (
        exec_loss.grad_fn is not None
    ), "Exec loss should be part of computation graph"

    # The key test: verify that exec loss can affect at least some of the model parameters
    # This ensures that exec loss is not being cut off from the gradient flow
    total_nonzero_grads = 0
    total_nonzero_grads += torch.sum(pred_V_mag.grad != 0).item()
    total_nonzero_grads += torch.sum(pred_V_sign.grad != 0).item()
    total_nonzero_grads += torch.sum(pred_O.grad != 0).item()
    total_nonzero_grads += torch.sum(pred_G.grad != 0).item()

    assert (
        total_nonzero_grads > 0
    ), "Exec loss should create gradients for at least some parameters"


def test_exec_loss_in_total_loss(test_dag_config, dag_executor, device):
    """Test that exec loss is properly included in total loss calculation."""
    pred_V_mag, pred_V_sign, pred_O, pred_G = create_test_tensors(
        test_dag_config, device
    )
    target_tensors, valid_mask = create_test_targets(test_dag_config, device)

    # Move dag_executor to correct device
    dag_executor = dag_executor.to(device)

    # Compute losses
    losses = compute_dag_loss(
        pred_V_mag,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=dag_executor,
    )

    # Verify that total loss includes all component losses
    # Note: exec_loss is already weighted in predictor_utils.py
    expected_total = (
        losses["V_mag_loss"]
        + losses["V_sign_loss"]
        + losses["O_loss"]
        + losses["G_loss"]
        + losses["exec_loss"]  # Already weighted
    )

    actual_total = losses["total_loss"]

    # Check that total loss equals sum of components (within floating point tolerance)
    assert torch.allclose(actual_total, expected_total, atol=1e-6), (
        f"Total loss should equal sum of components. "
        f"Expected: {expected_total.item():.6f}, Got: {actual_total.item():.6f}"
    )

    # Verify exec loss contributes meaningfully to total loss
    # (exec_loss is already weighted in predictor_utils.py)
    exec_contribution = losses["exec_loss"] / losses["total_loss"]
    assert (
        exec_contribution.item() > 0.001
    ), (  # Lower threshold due to weighting in predictor_utils
        f"Exec loss should contribute meaningfully to total loss. "
        f"Contribution: {exec_contribution.item():.4f}"
    )


def test_exec_loss_not_cut_off_with_extreme_values(
    test_dag_config, dag_executor, device
):
    """Test that exec loss is not cut off when dealing with extreme execution values."""
    pred_V_mag, pred_V_sign, pred_O, pred_G = create_test_tensors(
        test_dag_config, device
    )
    target_tensors, valid_mask = create_test_targets(test_dag_config, device)

    # Move dag_executor to correct device
    dag_executor = dag_executor.to(device)

    # Modify targets to have extreme execution values
    for b in range(len(target_tensors)):
        for t in range(len(target_tensors[b])):
            if valid_mask[b, t]:
                # Set extreme target values to test robust loss computation
                target_tensors[b][t]["target_final_exec"] = 1e8  # Very large value

    # Compute losses - should not fail or return NaN/Inf
    losses = compute_dag_loss(
        pred_V_mag,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=dag_executor,
    )

    # Verify exec loss is finite and non-zero
    exec_loss = losses["exec_loss"]
    assert torch.isfinite(
        exec_loss
    ), "Exec loss should be finite even with extreme values"
    assert (
        exec_loss.item() > 0.0
    ), "Exec loss should be non-zero with extreme target values"

    # Verify total loss is also finite
    total_loss = losses["total_loss"]
    assert torch.isfinite(total_loss), "Total loss should be finite"

    # Test with very small values
    for b in range(len(target_tensors)):
        for t in range(len(target_tensors[b])):
            if valid_mask[b, t]:
                target_tensors[b][t]["target_final_exec"] = 1e-8  # Very small value

    # Should still work with small values
    losses_small = compute_dag_loss(
        pred_V_mag,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=dag_executor,
    )

    assert torch.isfinite(
        losses_small["exec_loss"]
    ), "Exec loss should handle small values"
    assert losses_small["exec_loss"].item() >= 0.0, "Exec loss should be non-negative"


def test_exec_loss_no_valid_positions(test_dag_config, dag_executor, device):
    """Test that exec loss handling when no valid positions exist."""
    pred_V_mag, pred_V_sign, pred_O, pred_G = create_test_tensors(
        test_dag_config, device
    )
    target_tensors, _ = create_test_targets(test_dag_config, device)

    # Create empty valid mask (no valid positions)
    B, T = test_dag_config["batch_size"], test_dag_config["seq_len"]
    valid_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    # Move dag_executor to correct device
    dag_executor = dag_executor.to(device)

    # Compute losses with no valid positions
    losses = compute_dag_loss(
        pred_V_mag,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=dag_executor,
    )

    # All losses should be zero when no valid positions
    assert (
        losses["exec_loss"].item() == 0.0
    ), "Exec loss should be zero with no valid positions"
    assert (
        losses["total_loss"].item() == 0.0
    ), "Total loss should be zero with no valid positions"
    assert (
        losses["V_mag_loss"].item() == 0.0
    ), "V_mag_loss should be zero with no valid positions"
    assert (
        losses["V_sign_loss"].item() == 0.0
    ), "V_sign_loss should be zero with no valid positions"
    assert (
        losses["O_loss"].item() == 0.0
    ), "O_loss should be zero with no valid positions"
    assert (
        losses["G_loss"].item() == 0.0
    ), "G_loss should be zero with no valid positions"
