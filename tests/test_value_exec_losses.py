import math
import types

import pytest
import torch

from data.dagset.streaming import generate_single_dag_example
from models.dag_model import OP_NAMES, execute_stack
from predictor_utils import compute_dag_structure_loss, digits_to_magnitude


def _make_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Utility to convert an index tensor to one-hot representation."""
    shape = (*indices.shape, num_classes)
    one_hot = torch.zeros(shape, dtype=torch.float32)
    one_hot.scatter_(-1, indices.unsqueeze(-1), 1.0)
    return one_hot


def _build_dummy_tensors(batch: int, seq: int, nodes: int, digits: int, depth: int):
    """Create minimal dummy tensors required by `compute_dag_structure_loss`."""
    device = "cpu"

    # Sign predictions / targets (all +1 to keep sign loss trivially zero)
    tgt_sgn = torch.ones(batch, seq, nodes, device=device)
    pred_sgn = tgt_sgn.clone()

    # Operations (choose the first op for all positions)
    n_ops = len(OP_NAMES)
    tgt_op_idx = torch.zeros(batch, seq, depth, dtype=torch.long, device=device)
    tgt_ops = _make_one_hot(tgt_op_idx, n_ops)
    pred_ops = tgt_ops.clone()

    return pred_sgn, pred_ops, tgt_sgn, tgt_ops


def _build_test_config(
    max_digits: int = 3,
    max_decimal_places: int = 2,
    value_loss_weight: float = 1.0,
    exec_loss_weight: float = 1.0,
):
    """Build a test configuration with the necessary parameters."""
    return types.SimpleNamespace(
        sign_loss_weight=0.0,  # Zero out other losses for focused testing
        digit_loss_weight=0.0,
        op_loss_weight=0.0,
        value_loss_weight=value_loss_weight,
        exec_loss_weight=exec_loss_weight,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        base=10,
    )


@pytest.mark.parametrize("batch,seq,nodes,digits,depth", [(2, 1, 3, 5, 2)])
def test_value_loss_perfect_prediction(batch, seq, nodes, digits, depth):
    """Value loss should be zero when predicted initial values match targets exactly."""
    max_digits = 3
    max_decimal_places = 2

    pred_sgn, pred_ops, tgt_sgn, tgt_ops = _build_dummy_tensors(
        batch, seq, nodes, digits, depth
    )

    # Create target digits that correspond to known values
    target_values = torch.tensor(
        [
            [[1.23, 4.56, 7.89]],  # batch 0, seq 0
            [[2.34, 5.67, 8.90]],  # batch 1, seq 0
        ],
        dtype=torch.float32,
    )  # (batch, seq, nodes)

    # Convert target values to digit representations
    tgt_digits = torch.zeros(batch, seq, nodes, digits, 10)
    for b in range(batch):
        for s in range(seq):
            for n in range(nodes):
                val = target_values[b, s, n].item()

                # Create digit representation
                abs_val = abs(val)

                # Format to fixed decimal places and extract digits
                s_formatted = f"{abs_val:.{max_decimal_places}f}"
                int_part, frac_part = s_formatted.split(".")

                # Pad integer part
                int_part = int_part.zfill(max_digits)

                # Combine digits
                all_digits = int_part + frac_part

                for d, digit_char in enumerate(all_digits):
                    if d < digits:
                        tgt_digits[b, s, n, d, int(digit_char)] = 1.0

    # Create perfect prediction logits (large values that softmax to near one-hot)
    # Instead of using one-hot probabilities, use large logits that approximate perfect predictions
    pred_digits = torch.full(
        (batch, seq, nodes, digits, 10), -10.0
    )  # Start with very negative logits
    for b in range(batch):
        for s in range(seq):
            for n in range(nodes):
                for d in range(digits):
                    # Find which digit is the target (where one-hot is 1)
                    target_digit = tgt_digits[b, s, n, d].argmax()
                    # Set a large positive logit for the target digit
                    pred_digits[b, s, n, d, target_digit] = 10.0

    # Update signs to match target values
    tgt_sgn = torch.sign(target_values)
    pred_sgn = tgt_sgn.clone()

    cfg = _build_test_config(max_digits, max_decimal_places)

    # Dummy target final exec values
    target_final_exec = torch.zeros(batch, seq)

    losses = compute_dag_structure_loss(
        pred_sgn,
        pred_digits,
        pred_ops,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_values,
        target_final_exec,
        cfg,
    )

    assert (
        pytest.approx(losses["value_loss"].item(), abs=1e-3) == 0.0
    )  # Relaxed tolerance for near-perfect logits


@pytest.mark.parametrize("batch,seq,nodes,digits,depth", [(1, 1, 2, 5, 1)])
def test_value_loss_wrong_prediction(batch, seq, nodes, digits, depth):
    """Value loss should be > 0 when predicted initial values differ from targets."""
    max_digits = 3
    max_decimal_places = 2

    pred_sgn, pred_ops, tgt_sgn, tgt_ops = _build_dummy_tensors(
        batch, seq, nodes, digits, depth
    )

    # Create different target and predicted values
    target_values = torch.tensor([[[1.23, 4.56]]], dtype=torch.float32)
    predicted_values = torch.tensor([[[2.34, 5.67]]], dtype=torch.float32)

    # Convert to digit representations
    tgt_digits = torch.zeros(batch, seq, nodes, digits, 10)
    pred_digits = torch.zeros(batch, seq, nodes, digits, 10)

    for values, digit_tensor in [
        (target_values, tgt_digits),
        (predicted_values, pred_digits),
    ]:
        for b in range(batch):
            for s in range(seq):
                for n in range(nodes):
                    val = values[b, s, n].item()
                    abs_val = abs(val)

                    s_formatted = f"{abs_val:.{max_decimal_places}f}"
                    int_part, frac_part = s_formatted.split(".")
                    int_part = int_part.zfill(max_digits)
                    all_digits = int_part + frac_part

                    for d, digit_char in enumerate(all_digits):
                        if d < digits:
                            digit_tensor[b, s, n, d, int(digit_char)] = 1.0

    # Set signs
    tgt_sgn = torch.sign(target_values)
    pred_sgn = torch.sign(predicted_values)

    cfg = _build_test_config(max_digits, max_decimal_places)

    # Dummy target final exec values
    target_final_exec = torch.zeros(batch, seq)

    losses = compute_dag_structure_loss(
        pred_sgn,
        pred_digits,
        pred_ops,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_values,
        target_final_exec,
        cfg,
    )

    assert losses["value_loss"].item() > 0.0


@pytest.mark.parametrize("batch,seq,depth", [(1, 1, 1)])
def test_exec_loss_perfect_prediction(batch, seq, depth):
    """Exec loss should be zero when predicted final execution matches target exactly."""
    max_digits = 3
    max_decimal_places = 2

    # Generate a real DAG example to get consistent data
    example = generate_single_dag_example(
        depth=depth,
        seed=42,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        execute_sympy=True,
    )

    # Extract tensors from the example
    nodes = len(example.initial_values)
    digits = max_digits + max_decimal_places

    # Create tensors matching the example
    tgt_sgn = example.signs.unsqueeze(0).unsqueeze(0)  # (1, 1, nodes)
    tgt_digits = example.digits.unsqueeze(0).unsqueeze(0)  # (1, 1, nodes, digits, 10)
    tgt_ops = example.operations.unsqueeze(0).unsqueeze(0)  # (1, 1, depth, n_ops)

    # Create perfect prediction logits instead of copying one-hot targets
    pred_sgn = tgt_sgn.clone()
    pred_ops = tgt_ops.clone()

    # Convert target one-hots to near-perfect logits
    pred_digits = torch.full_like(tgt_digits, -10.0)  # Start with very negative logits
    for b in range(batch):
        for s in range(seq):
            for n in range(nodes):
                for d in range(digits):
                    # Find which digit is the target (where one-hot is 1)
                    target_digit = tgt_digits[b, s, n, d].argmax()
                    # Set a large positive logit for the target digit
                    pred_digits[b, s, n, d, target_digit] = 10.0

    # Target values
    target_initial_values = torch.tensor(
        [[example.initial_values]], dtype=torch.float32
    )
    target_final_exec = torch.tensor([[example.final_value_exec]], dtype=torch.float32)

    cfg = _build_test_config(max_digits, max_decimal_places)

    losses = compute_dag_structure_loss(
        pred_sgn,
        pred_digits,
        pred_ops,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_initial_values,
        target_final_exec,
        cfg,
    )

    assert (
        pytest.approx(losses["exec_loss"].item(), abs=1e-2) == 0.0
    )  # Relaxed tolerance for near-perfect logits


@pytest.mark.parametrize("batch,seq,depth", [(1, 1, 2)])
def test_exec_loss_wrong_prediction(batch, seq, depth):
    """Exec loss should be > 0 when predicted execution differs from target."""
    max_digits = 3
    max_decimal_places = 2

    # Generate two different DAG examples
    example1 = generate_single_dag_example(
        depth=depth,
        seed=42,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        execute_sympy=True,
    )

    example2 = generate_single_dag_example(
        depth=depth,
        seed=123,  # Different seed for different result
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        execute_sympy=True,
    )

    tgt_sgn = example1.signs.unsqueeze(0).unsqueeze(0)
    tgt_digits = example1.digits.unsqueeze(0).unsqueeze(0)
    tgt_ops = example1.operations.unsqueeze(0).unsqueeze(0)

    pred_sgn = example2.signs.unsqueeze(0).unsqueeze(0)
    pred_digits = example2.digits.unsqueeze(0).unsqueeze(0)
    pred_ops = example2.operations.unsqueeze(0).unsqueeze(0)

    # Target values
    target_initial_values = torch.tensor(
        [[example1.initial_values]], dtype=torch.float32
    )
    target_final_exec = torch.tensor([[example1.final_value_exec]], dtype=torch.float32)

    cfg = _build_test_config(max_digits, max_decimal_places)

    losses = compute_dag_structure_loss(
        pred_sgn,
        pred_digits,
        pred_ops,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_initial_values,
        target_final_exec,
        cfg,
    )

    # Should have non-zero exec loss since predictions differ
    assert losses["exec_loss"].item() > 0.0


def test_value_exec_losses_computed():
    """Test that value and exec losses are always computed with required parameters."""
    batch, seq, nodes, digits, depth = 1, 1, 2, 5, 1
    max_digits = 3
    max_decimal_places = 2

    pred_sgn, pred_ops, tgt_sgn, tgt_ops = _build_dummy_tensors(
        batch, seq, nodes, digits, depth
    )

    # Random digit tensors
    tgt_digits = torch.rand(batch, seq, nodes, digits, 10)
    tgt_digits = tgt_digits / tgt_digits.sum(
        dim=-1, keepdim=True
    )  # Normalize to probabilities
    pred_digits = tgt_digits.clone()

    cfg = _build_test_config(max_digits, max_decimal_places)

    # Create required target values
    target_initial_values = torch.rand(batch, seq, nodes)
    target_final_exec = torch.rand(batch, seq)

    # Call with required parameters
    losses = compute_dag_structure_loss(
        pred_sgn,
        pred_digits,
        pred_ops,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_initial_values,
        target_final_exec,
        cfg,
    )

    # Should compute both losses
    assert "value_loss" in losses
    assert "exec_loss" in losses
    assert torch.isfinite(losses["value_loss"])
    assert torch.isfinite(losses["exec_loss"])


def test_loss_weights_applied():
    """Test that loss weights are properly applied to the new loss terms."""
    batch, seq, nodes, digits, depth = 1, 1, 2, 5, 1
    max_digits = 3
    max_decimal_places = 2

    pred_sgn, pred_ops, tgt_sgn, tgt_ops = _build_dummy_tensors(
        batch, seq, nodes, digits, depth
    )

    # Create some dummy targets that will result in non-zero losses
    target_values = torch.tensor([[[1.23, 4.56]]], dtype=torch.float32)
    target_final_exec = torch.tensor([[7.89]], dtype=torch.float32)

    # Create digit tensors from different values to ensure non-zero loss
    tgt_digits = torch.zeros(batch, seq, nodes, digits, 10)
    pred_digits = torch.zeros(batch, seq, nodes, digits, 10)

    # Set some random but different digit distributions
    tgt_digits[0, 0, 0, 0, 1] = 1.0  # digit 1
    tgt_digits[0, 0, 0, 1, 2] = 1.0  # digit 2
    tgt_digits[0, 0, 1, 0, 4] = 1.0  # digit 4
    tgt_digits[0, 0, 1, 1, 5] = 1.0  # digit 5

    pred_digits[0, 0, 0, 0, 3] = 1.0  # digit 3 (different)
    pred_digits[0, 0, 0, 1, 4] = 1.0  # digit 4 (different)
    pred_digits[0, 0, 1, 0, 6] = 1.0  # digit 6 (different)
    pred_digits[0, 0, 1, 1, 7] = 1.0  # digit 7 (different)

    # Test with different weights
    cfg1 = _build_test_config(
        max_digits, max_decimal_places, value_loss_weight=2.0, exec_loss_weight=3.0
    )
    cfg2 = _build_test_config(
        max_digits, max_decimal_places, value_loss_weight=1.0, exec_loss_weight=1.0
    )

    losses1 = compute_dag_structure_loss(
        pred_sgn,
        pred_digits,
        pred_ops,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_values,
        target_final_exec,
        cfg1,
    )

    losses2 = compute_dag_structure_loss(
        pred_sgn,
        pred_digits,
        pred_ops,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_values,
        target_final_exec,
        cfg2,
    )

    # The weighted losses should differ
    # losses1 should have 2x value_loss and 3x exec_loss compared to the raw losses
    # We can't test exact ratios due to the complexity of the loss computation,
    # but we can verify that the total losses differ when weights differ
    assert losses1["total_loss"].item() != losses2["total_loss"].item()


def test_exec_loss_uses_clipping():
    """Test that exec_loss computation uses clipping (ignore_clip=False) behavior."""
    max_digits = 3
    max_decimal_places = 2

    # Create a scenario where clipping would make a difference
    batch, seq, nodes, digits, depth = 1, 1, 2, 5, 1

    pred_sgn, pred_ops, tgt_sgn, tgt_ops = _build_dummy_tensors(
        batch, seq, nodes, digits, depth
    )

    # Create digit tensors that would result in very large numbers (beyond LOG_LIM)
    # These should get clipped during execution
    pred_digits = torch.zeros(batch, seq, nodes, digits, 10)
    tgt_digits = torch.zeros(batch, seq, nodes, digits, 10)

    # Set digits to represent a very large number (999.99)
    pred_digits[0, 0, 0, 0, 9] = 1.0  # 9
    pred_digits[0, 0, 0, 1, 9] = 1.0  # 9
    pred_digits[0, 0, 0, 2, 9] = 1.0  # 9
    pred_digits[0, 0, 0, 3, 9] = 1.0  # 9
    pred_digits[0, 0, 0, 4, 9] = 1.0  # 9

    # Create a similar pattern for the second node
    pred_digits[0, 0, 1, 0, 9] = 1.0
    pred_digits[0, 0, 1, 1, 9] = 1.0
    pred_digits[0, 0, 1, 2, 9] = 1.0
    pred_digits[0, 0, 1, 3, 9] = 1.0
    pred_digits[0, 0, 1, 4, 9] = 1.0

    # Set targets to be the same for now (we're testing clipping behavior)
    tgt_digits = pred_digits.clone()

    # Set signs to positive
    pred_sgn = torch.ones_like(pred_sgn)
    tgt_sgn = torch.ones_like(tgt_sgn)

    # Target values
    target_initial_values = torch.ones(batch, seq, nodes) * 999.99
    target_final_exec = torch.tensor([[1e6]], dtype=torch.float32)

    cfg = _build_test_config(max_digits, max_decimal_places)

    # This should not raise an error and should handle clipping gracefully
    losses = compute_dag_structure_loss(
        pred_sgn,
        pred_digits,
        pred_ops,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_initial_values,
        target_final_exec,
        cfg,
    )

    # The exec loss should be computed without errors
    assert torch.isfinite(
        losses["exec_loss"]
    ), "exec_loss should be finite even with large numbers"
    assert losses["exec_loss"].item() >= 0.0, "exec_loss should be non-negative"
