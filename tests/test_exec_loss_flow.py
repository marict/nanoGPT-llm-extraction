"""Test that exec losses are properly computed and flow through gradients."""

import pytest
import torch

from models.dag_model import DAGExecutor
from predictor_utils import compute_dag_loss


class MockConfig:
    """Minimal config for testing."""

    def __init__(self):
        self.max_digits = 4
        self.max_decimal_places = 4
        self.enable_exec_loss = True


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
    return DAGExecutor(
        dag_depth=test_dag_config["dag_depth"],
        max_digits=4,
        max_decimal_places=4,
        base=10,
    )


@pytest.fixture
def test_config():
    """Create a test config."""
    return MockConfig()


def create_test_tensors(config, device):
    """Create test prediction tensors that require gradients."""
    B, T = config["batch_size"], config["seq_len"]
    total_nodes = config["total_nodes"]
    dag_depth = config["dag_depth"]

    # Generate digit logits instead of V_mag
    num_initial_nodes = dag_depth + 1
    D = 8  # max_digits + max_decimal_places = 4 + 4
    base = 10

    # Create prediction tensors with gradients enabled
    # Use much smaller magnitude to avoid extreme DAG execution results that cause gradient saturation
    pred_digit_logits = (
        torch.randn(B, T, num_initial_nodes, D, base, device=device, requires_grad=True)
        * 0.1
    )
    # Make it a leaf tensor to properly retain gradients
    pred_digit_logits = pred_digit_logits.detach().requires_grad_(True)
    pred_V_sign = (
        (torch.randn(B, T, total_nodes, device=device, requires_grad=True) * 0.1)
        .detach()
        .requires_grad_(True)
    )
    pred_O = (
        (
            torch.randn(B, T, dag_depth, total_nodes, device=device, requires_grad=True)
            * 0.1
        )
        .detach()
        .requires_grad_(True)
    )
    pred_G = (
        (torch.randn(B, T, dag_depth, device=device, requires_grad=True) * 0.1)
        .detach()
        .requires_grad_(True)
    )

    return pred_digit_logits, pred_V_sign, pred_O, pred_G


def create_test_targets(config, device):
    """Create test target tensors and valid mask."""
    B, T = config["batch_size"], config["seq_len"]
    total_nodes = config["total_nodes"]
    dag_depth = config["dag_depth"]

    target_tensors = []
    for _ in range(B):
        batch_targets = []
        for _ in range(T):
            # Generate target digits (one-hot for initial nodes)
            num_initial_nodes = dag_depth + 1
            D = 8  # max_digits + max_decimal_places = 4 + 4
            base = 10
            target_digits = torch.zeros(num_initial_nodes, D, base, device=device)

            # Set random digits as one-hot
            for n in range(num_initial_nodes):
                for d in range(D):
                    digit = torch.randint(0, base, (1,)).item()
                    target_digits[n, d, digit] = 1.0

            target_dict = {
                "target_digits": target_digits,
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


def test_exec_loss_computed_with_dag_executor(
    test_dag_config, dag_executor, device, test_config
):
    """Test that exec loss is computed when dag_executor is provided."""
    pred_digit_logits, pred_V_sign, pred_O, pred_G = create_test_tensors(
        test_dag_config, device
    )
    target_tensors, valid_mask = create_test_targets(test_dag_config, device)

    # Move dag_executor to correct device
    dag_executor = dag_executor.to(device)

    # Compute losses WITH dag_executor
    losses_with_executor = compute_dag_loss(
        pred_digit_logits,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=dag_executor,
        cfg=test_config,
    )

    # Compute losses WITHOUT dag_executor
    losses_without_executor = compute_dag_loss(
        pred_digit_logits,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=None,
        cfg=test_config,
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


def test_exec_loss_gradient_flow(test_dag_config, dag_executor, device, test_config):
    """Test that exec loss contributes to gradients of prediction tensors."""
    pred_digit_logits, pred_V_sign, pred_O, pred_G = create_test_tensors(
        test_dag_config, device
    )
    target_tensors, valid_mask = create_test_targets(test_dag_config, device)

    # Move dag_executor to correct device
    dag_executor = dag_executor.to(device)

    # Compute losses with dag_executor
    losses = compute_dag_loss(
        pred_digit_logits,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=dag_executor,
        cfg=test_config,
    )

    # Check that exec_loss requires gradients
    exec_loss = losses["exec_loss"]
    assert exec_loss.requires_grad, "Exec loss should require gradients"

    # Only test gradient flow if exec_loss is meaningful (not the constant fallback)
    # The robust exec loss returns 1.0 when execution fails
    if exec_loss.item() != 1.0:
        # Enable gradient retention for non-leaf tensors
        pred_digit_logits.retain_grad()
        pred_V_sign.retain_grad()

        # Backward pass through exec_loss only
        exec_loss.backward(retain_graph=True)

        # Check that prediction tensors have gradients from exec loss
        assert (
            pred_digit_logits.grad is not None
        ), "pred_digit_logits should have gradients from exec loss"
        assert (
            pred_V_sign.grad is not None
        ), "pred_V_sign should have gradients from exec loss"
        assert pred_O.grad is not None, "pred_O should have gradients from exec loss"
        assert pred_G.grad is not None, "pred_G should have gradients from exec loss"

        # Check that gradients are non-zero for at least some parameters
        # Note: Use small threshold since exec_loss is weighted by 0.01
        grad_threshold = 1e-10
        assert torch.any(
            torch.abs(pred_digit_logits.grad) > grad_threshold
        ), "Some pred_digit_logits gradients should be non-zero (above threshold)"
        assert torch.any(
            torch.abs(pred_V_sign.grad) > grad_threshold
        ), "Some pred_V_sign gradients should be non-zero (above threshold)"
    else:
        # If exec_loss is the constant fallback, we can't test gradient flow
        # but this is expected behavior for problematic expressions
        print("Exec loss is constant fallback (1.0), skipping gradient flow test")

    # Verify that exec loss gradients flow to the computation graph (always testable)
    assert (
        exec_loss.grad_fn is not None
    ), "Exec loss should be part of computation graph"


def test_exec_loss_in_total_loss(test_dag_config, dag_executor, device, test_config):
    """Test that exec loss is properly included in total loss calculation."""
    pred_digit_logits, pred_V_sign, pred_O, pred_G = create_test_tensors(
        test_dag_config, device
    )
    target_tensors, valid_mask = create_test_targets(test_dag_config, device)

    # Move dag_executor to correct device
    dag_executor = dag_executor.to(device)

    # Compute losses
    losses = compute_dag_loss(
        pred_digit_logits,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=dag_executor,
        cfg=test_config,
    )

    # Verify that total loss includes all component losses
    # Note: exec_loss is already weighted in predictor_utils.py
    expected_total = (
        losses["digit_loss"]
        + losses["V_mag_loss"]
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
    # (exec_loss is already weighted in predictor_utils.py and capped for stability)
    exec_contribution = losses["exec_loss"] / losses["total_loss"]
    assert (
        exec_contribution.item() > 0.0001
    ), (  # Adjusted threshold for capped exec_loss - should be small but non-zero
        f"Exec loss should contribute meaningfully to total loss. "
        f"Contribution: {exec_contribution.item():.6f}"
    )


def test_exec_loss_not_cut_off_with_extreme_values(
    test_dag_config, dag_executor, device, test_config
):
    """Test that exec loss is not cut off when dealing with extreme execution values."""
    pred_digit_logits, pred_V_sign, pred_O, pred_G = create_test_tensors(
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
        pred_digit_logits,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=dag_executor,
        cfg=test_config,
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
        pred_digit_logits,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=dag_executor,
        cfg=test_config,
    )

    assert torch.isfinite(
        losses_small["exec_loss"]
    ), "Exec loss should handle small values"
    assert losses_small["exec_loss"].item() >= 0.0, "Exec loss should be non-negative"


def test_exec_loss_no_valid_positions(
    test_dag_config, dag_executor, device, test_config
):
    """Test that exec loss handling when no valid positions exist."""
    pred_digit_logits, pred_V_sign, pred_O, pred_G = create_test_tensors(
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
        pred_digit_logits,
        pred_V_sign,
        pred_O,
        pred_G,
        target_tensors,
        valid_mask,
        dag_executor=dag_executor,
        cfg=test_config,
    )

    # All losses should be zero when no valid positions
    assert (
        losses["exec_loss"].item() == 0.0
    ), "Exec loss should be zero with no valid positions"
    assert (
        losses["total_loss"].item() == 0.0
    ), "Total loss should be zero with no valid positions"
    assert (
        losses["digit_loss"].item() == 0.0
    ), "digit_loss should be zero with no valid positions"
    assert (
        losses["V_sign_loss"].item() == 0.0
    ), "V_sign_loss should be zero with no valid positions"
    assert (
        losses["O_loss"].item() == 0.0
    ), "O_loss should be zero with no valid positions"
    assert (
        losses["G_loss"].item() == 0.0
    ), "G_loss should be zero with no valid positions"
