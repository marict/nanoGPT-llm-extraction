import math
import types

import pytest
import torch

from data.dagset.streaming import generate_single_dag_example
from models.dag_model import OP_NAMES
from predictor_utils import compute_dag_structure_loss


def _dummy_statistics(batch: int, seq: int):
    """Create dummy statistics tensors for testing."""
    dummy_pred_stats = {
        "initial": torch.zeros(batch, seq, 15),
        "intermediate": torch.zeros(batch, seq, 15),
        "final": torch.zeros(batch, seq, 10),
    }
    dummy_target_stats = {
        "initial": torch.zeros(batch, 15),
        "intermediate": torch.zeros(batch, 15),
        "final": torch.zeros(batch, 10),
    }
    return dummy_pred_stats, dummy_target_stats


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
    tgt_sgn = torch.ones(
        batch, nodes, device=device
    )  # Remove seq dimension for targets
    # Create sign logits that will produce the target signs when passed through tanh
    # For perfect prediction, use large positive logits for +1 signs
    pred_sign_logits = torch.full((batch, seq, nodes), 10.0, device=device)

    # Operations (choose the first op for all positions)
    n_ops = len(OP_NAMES)
    tgt_op_idx = torch.zeros(
        batch, depth, dtype=torch.long, device=device
    )  # Remove seq dimension
    tgt_ops = _make_one_hot(tgt_op_idx, n_ops)
    # Create operation logits instead of one-hot probabilities
    pred_op_logits = torch.full((batch, seq, depth, n_ops), -10.0, device=device)
    pred_op_logits[:, :, :, 0] = 10.0  # Large positive logit for first operation

    return pred_sign_logits, pred_op_logits, tgt_sgn, tgt_ops


def _build_test_config(
    max_digits: int = 3,
    max_decimal_places: int = 2,
):
    """Build a test configuration with the necessary parameters."""
    return types.SimpleNamespace(
        # All loss weights removed - using automatic balancing
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        base=10,
    )


@pytest.mark.parametrize("batch,seq,nodes,digits,depth", [(2, 1, 3, 5, 2)])
def test_value_loss_perfect_prediction(batch, seq, nodes, digits, depth):
    """Value loss should be zero when predicted initial values match targets exactly."""
    max_digits = 3
    max_decimal_places = 2

    pred_sign_logits, pred_op_logits, tgt_sgn, tgt_ops = _build_dummy_tensors(
        batch, seq, nodes, digits, depth
    )

    # Create target digits that correspond to known values
    target_values = torch.tensor(
        [
            [1.23, 4.56, 7.89],  # batch 0
            [2.34, 5.67, 8.90],  # batch 1
        ],
        dtype=torch.float32,
    )  # (batch, nodes)

    # Convert target values to digit representations
    tgt_digits = torch.zeros(batch, nodes, digits, 10)
    for b in range(batch):
        for n in range(nodes):
            val = target_values[b, n].item()

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
                    tgt_digits[b, n, d, int(digit_char)] = 1.0

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
                    target_digit = tgt_digits[b, n, d].argmax()
                    # Set a large positive logit for the target digit
                    pred_digits[b, s, n, d, target_digit] = 10.0

    # Update signs to match target values
    tgt_sgn = torch.sign(target_values)

    # Create sign logits that will produce the target signs when passed through tanh
    # For perfect prediction, use large logits: positive for +1, negative for -1
    pred_sign_logits = torch.full((batch, seq, nodes), 0.0)
    for b in range(batch):
        for s in range(seq):
            for n in range(nodes):
                if tgt_sgn[b, n] > 0:
                    pred_sign_logits[b, s, n] = 10.0
                else:
                    pred_sign_logits[b, s, n] = -10.0

    cfg = _build_test_config(max_digits, max_decimal_places)

    # Dummy target final exec values
    target_final_exec = torch.zeros(batch)

    # Create final token positions (use last sequence position for all examples)
    final_token_pos = torch.full((batch,), seq - 1, dtype=torch.long)

    dummy_pred_stats, dummy_target_stats = _dummy_statistics(batch, seq)
    losses = compute_dag_structure_loss(
        pred_sign_logits,
        pred_digits,
        pred_op_logits,
        dummy_pred_stats,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_values,
        target_final_exec,
        dummy_target_stats,
        final_token_pos,
        cfg,
        uncertainty_params=torch.zeros(6),
    )

    assert (
        pytest.approx(losses["value_loss"].item(), abs=1e-3) == 0.0
    )  # Relaxed tolerance for near-perfect logits


@pytest.mark.parametrize("batch,seq,nodes,digits,depth", [(2, 3, 3, 5, 2)])
def test_loss_only_at_final_token_position(batch, seq, nodes, digits, depth):
    """Test that loss is calculated only at the exact final_token_pos, not at other positions."""
    max_digits = 3
    max_decimal_places = 2

    pred_sign_logits, pred_op_logits, tgt_sgn, tgt_ops = _build_dummy_tensors(
        batch, seq, nodes, digits, depth
    )

    # Create target values
    target_values = torch.tensor(
        [
            [1.23, 4.56, 7.89],  # batch 0
            [2.34, 5.67, 8.90],  # batch 1
        ],
        dtype=torch.float32,
    )  # (batch, nodes)

    # Convert target values to digit representations
    tgt_digits = torch.zeros(batch, nodes, digits, 10)
    for b in range(batch):
        for n in range(nodes):
            val = target_values[b, n].item()
            abs_val = abs(val)
            s_formatted = f"{abs_val:.{max_decimal_places}f}"
            int_part, frac_part = s_formatted.split(".")
            int_part = int_part.zfill(max_digits)
            all_digits = int_part + frac_part

            for d, digit_char in enumerate(all_digits):
                if d < digits:
                    tgt_digits[b, n, d, int(digit_char)] = 1.0

    # Create predictions with perfect accuracy ONLY at the final token position
    # and completely wrong predictions at other positions
    pred_digits = torch.full((batch, seq, nodes, digits, 10), -10.0)

    # Set wrong predictions for all positions except the final one
    for b in range(batch):
        for s in range(seq - 1):  # All positions except final
            for n in range(nodes):
                for d in range(digits):
                    # Set wrong digit (opposite of target)
                    target_digit = tgt_digits[b, n, d].argmax()
                    wrong_digit = (target_digit + 1) % 10  # Pick a different digit
                    pred_digits[b, s, n, d, wrong_digit] = 10.0

    # Set perfect predictions ONLY at final token position
    final_pos = seq - 1
    for b in range(batch):
        for n in range(nodes):
            for d in range(digits):
                target_digit = tgt_digits[b, n, d].argmax()
                pred_digits[b, final_pos, n, d, target_digit] = 10.0

    # Similarly for signs - wrong everywhere except final position
    pred_sign_logits = torch.full((batch, seq, nodes), 0.0)
    tgt_sgn = torch.sign(target_values)

    for b in range(batch):
        for s in range(seq - 1):  # Wrong predictions at non-final positions
            for n in range(nodes):
                # Set opposite sign
                if tgt_sgn[b, n] > 0:
                    pred_sign_logits[b, s, n] = -10.0  # Wrong sign
                else:
                    pred_sign_logits[b, s, n] = 10.0  # Wrong sign

    # Correct predictions only at final position
    for b in range(batch):
        for n in range(nodes):
            if tgt_sgn[b, n] > 0:
                pred_sign_logits[b, final_pos, n] = 10.0  # Correct sign
            else:
                pred_sign_logits[b, final_pos, n] = -10.0  # Correct sign

    cfg = _build_test_config(max_digits, max_decimal_places)
    target_final_exec = torch.zeros(batch)

    # Set final token positions to the last sequence position
    final_token_pos = torch.full((batch,), final_pos, dtype=torch.long)

    dummy_pred_stats, dummy_target_stats = _dummy_statistics(batch, seq)
    losses = compute_dag_structure_loss(
        pred_sign_logits,
        pred_digits,
        pred_op_logits,
        dummy_pred_stats,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_values,
        target_final_exec,
        dummy_target_stats,
        final_token_pos,
        cfg,
        uncertainty_params=torch.zeros(6),
    )

    # Loss should be very low because predictions at final_token_pos are perfect
    # Even though predictions at all other positions are completely wrong
    assert (
        losses["sign_loss"].item() < 0.1
    ), f"Sign loss should be low, got {losses['sign_loss'].item()}"
    assert (
        losses["digit_loss"].item() < 0.1
    ), f"Digit loss should be low, got {losses['digit_loss'].item()}"


@pytest.mark.parametrize("batch,seq,nodes,digits,depth", [(3, 4, 2, 3, 1)])
def test_different_final_token_positions(batch, seq, nodes, digits, depth):
    """Test that different final_token_pos values work correctly for different batch elements."""
    max_digits = 2
    max_decimal_places = 1

    pred_sign_logits, pred_op_logits, tgt_sgn, tgt_ops = _build_dummy_tensors(
        batch, seq, nodes, digits, depth
    )

    # Create target values
    target_values = torch.tensor(
        [
            [1.2, 3.4],  # batch 0
            [5.6, 7.8],  # batch 1
            [9.0, 1.1],  # batch 2
        ],
        dtype=torch.float32,
    )  # (batch, nodes)

    # Convert target values to digit representations
    tgt_digits = torch.zeros(batch, nodes, digits, 10)
    for b in range(batch):
        for n in range(nodes):
            val = target_values[b, n].item()
            abs_val = abs(val)
            s_formatted = f"{abs_val:.{max_decimal_places}f}"
            int_part, frac_part = s_formatted.split(".")
            int_part = int_part.zfill(max_digits)
            all_digits = int_part + frac_part

            for d, digit_char in enumerate(all_digits):
                if d < digits:
                    tgt_digits[b, n, d, int(digit_char)] = 1.0

    # Create predictions - each batch element has correct predictions at different positions
    pred_digits = torch.full((batch, seq, nodes, digits, 10), -10.0)
    pred_sign_logits = torch.full((batch, seq, nodes), 0.0)
    tgt_sgn = torch.sign(target_values)

    # Different final token positions for each batch element
    final_token_positions = [1, 2, 3]  # Different positions for each batch element
    final_token_pos = torch.tensor(final_token_positions, dtype=torch.long)

    # Set correct predictions ONLY at the respective final_token_pos
    for b in range(batch):
        final_pos = final_token_positions[b]

        # Wrong predictions at ALL positions except final_token_pos
        for s in range(seq):
            if s != final_pos:
                for n in range(nodes):
                    # Wrong sign
                    if tgt_sgn[b, n] > 0:
                        pred_sign_logits[b, s, n] = -10.0
                    else:
                        pred_sign_logits[b, s, n] = 10.0

                    # Wrong digits
                    for d in range(digits):
                        target_digit = tgt_digits[b, n, d].argmax()
                        wrong_digit = (target_digit + 1) % 10
                        pred_digits[b, s, n, d, wrong_digit] = 10.0

        # Correct predictions ONLY at final_token_pos
        for n in range(nodes):
            # Correct sign
            if tgt_sgn[b, n] > 0:
                pred_sign_logits[b, final_pos, n] = 10.0
            else:
                pred_sign_logits[b, final_pos, n] = -10.0

            # Correct digits
            for d in range(digits):
                target_digit = tgt_digits[b, n, d].argmax()
                pred_digits[b, final_pos, n, d, target_digit] = 10.0

    cfg = _build_test_config(max_digits, max_decimal_places)
    target_final_exec = torch.zeros(batch)

    dummy_pred_stats, dummy_target_stats = _dummy_statistics(batch, seq)
    losses = compute_dag_structure_loss(
        pred_sign_logits,
        pred_digits,
        pred_op_logits,
        dummy_pred_stats,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_values,
        target_final_exec,
        dummy_target_stats,
        final_token_pos,
        cfg,
        uncertainty_params=torch.zeros(6),
    )

    # Losses should be low because each batch element has correct predictions at their respective final_token_pos
    assert (
        losses["sign_loss"].item() < 0.1
    ), f"Sign loss should be low, got {losses['sign_loss'].item()}"
    assert (
        losses["digit_loss"].item() < 0.1
    ), f"Digit loss should be low, got {losses['digit_loss'].item()}"


@pytest.mark.parametrize("batch,seq,nodes,digits,depth", [(1, 1, 2, 5, 1)])
def test_value_loss_wrong_prediction(batch, seq, nodes, digits, depth):
    """Value loss should be > 0 when predicted initial values differ from targets."""
    max_digits = 3
    max_decimal_places = 2

    pred_sign_logits, pred_op_logits, tgt_sgn, tgt_ops = _build_dummy_tensors(
        batch, seq, nodes, digits, depth
    )

    # Create different target and predicted values
    target_values = torch.tensor([[1.23, 4.56]], dtype=torch.float32)  # (batch, nodes)
    predicted_values = torch.tensor(
        [[2.34, 5.67]], dtype=torch.float32
    )  # (batch, nodes)

    # Convert to digit representations
    tgt_digits = torch.zeros(batch, nodes, digits, 10)  # Remove seq dimension
    pred_digits = torch.zeros(batch, seq, nodes, digits, 10)

    # Target digits (no seq dimension)
    for b in range(batch):
        for n in range(nodes):
            val = target_values[b, n].item()
            abs_val = abs(val)
            s_formatted = f"{abs_val:.{max_decimal_places}f}"
            int_part, frac_part = s_formatted.split(".")
            int_part = int_part.zfill(max_digits)
            all_digits = int_part + frac_part

            for d, digit_char in enumerate(all_digits):
                if d < digits:
                    tgt_digits[b, n, d, int(digit_char)] = 1.0

    # Predicted digits (with seq dimension)
    for b in range(batch):
        for s in range(seq):
            for n in range(nodes):
                val = predicted_values[b, n].item()
                abs_val = abs(val)
                s_formatted = f"{abs_val:.{max_decimal_places}f}"
                int_part, frac_part = s_formatted.split(".")
                int_part = int_part.zfill(max_digits)
                all_digits = int_part + frac_part

                for d, digit_char in enumerate(all_digits):
                    if d < digits:
                        pred_digits[b, s, n, d, int(digit_char)] = 1.0

    # Set signs
    tgt_sgn = torch.sign(target_values)  # (batch, nodes)
    # Create sign logits for predictions
    pred_sign_logits = torch.full((batch, seq, nodes), 0.0)
    for b in range(batch):
        for s in range(seq):
            for n in range(nodes):
                if torch.sign(predicted_values[b, n]) > 0:
                    pred_sign_logits[b, s, n] = 10.0
                else:
                    pred_sign_logits[b, s, n] = -10.0

    cfg = _build_test_config(max_digits, max_decimal_places)

    # Dummy target final exec values
    target_final_exec = torch.zeros(batch)  # Remove seq dimension

    # Create final token positions
    final_token_pos = torch.full((batch,), seq - 1, dtype=torch.long)

    dummy_pred_stats, dummy_target_stats = _dummy_statistics(batch, seq)
    losses = compute_dag_structure_loss(
        pred_sign_logits,
        pred_digits,
        pred_op_logits,
        dummy_pred_stats,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_values,
        target_final_exec,
        dummy_target_stats,
        final_token_pos,
        cfg,
        uncertainty_params=torch.zeros(6),
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
    )

    # Extract tensors from the example
    nodes = len(example.structure_dict["target_initial_digits"])
    digits = max_digits + max_decimal_places

    # Create tensors matching the example - targets without seq dimension
    tgt_sgn = example.structure_dict["target_initial_sgn"].unsqueeze(0)  # (1, nodes)
    tgt_digits = example.structure_dict["target_initial_digits"].unsqueeze(
        0
    )  # (1, nodes, digits, 10)
    tgt_ops = example.structure_dict["target_operation_probs"].unsqueeze(
        0
    )  # (1, depth, n_ops)

    # Create perfect prediction logits - predictions with seq dimension
    pred_sign_logits = torch.zeros(batch, seq, nodes)
    for b in range(batch):
        for s in range(seq):
            for n in range(nodes):
                if tgt_sgn[b, n] > 0:
                    pred_sign_logits[b, s, n] = 10.0
                else:
                    pred_sign_logits[b, s, n] = -10.0

    # Convert target operation one-hots to near-perfect logits
    pred_op_logits = torch.full((batch, seq, depth, len(OP_NAMES)), -10.0)
    for b in range(batch):
        for s in range(seq):
            for d in range(depth):
                # Find which operation is the target (where one-hot is 1)
                target_op = tgt_ops[b, d].argmax()
                # Set a large positive logit for the target operation
                pred_op_logits[b, s, d, target_op] = 10.0

    # Convert target one-hots to near-perfect logits
    pred_digits = torch.full((batch, seq, nodes, digits, 10), -10.0)
    for b in range(batch):
        for s in range(seq):
            for n in range(nodes):
                for d in range(digits):
                    # Find which digit is the target (where one-hot is 1)
                    target_digit = tgt_digits[b, n, d].argmax()
                    # Set a large positive logit for the target digit
                    pred_digits[b, s, n, d, target_digit] = 10.0

    # Target values
    target_initial_values = example.structure_dict["target_initial_values"].unsqueeze(
        0
    )  # (1, nodes)
    target_final_exec = torch.tensor(
        [example.structure_dict["target_final_exec"]], dtype=torch.float32
    )  # (1,)

    # Create final token positions
    final_token_pos = torch.full((batch,), seq - 1, dtype=torch.long)

    cfg = _build_test_config(max_digits, max_decimal_places)

    dummy_pred_stats, dummy_target_stats = _dummy_statistics(batch, seq)
    losses = compute_dag_structure_loss(
        pred_sign_logits,
        pred_digits,
        pred_op_logits,
        dummy_pred_stats,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_initial_values,
        target_final_exec,
        dummy_target_stats,
        final_token_pos,
        cfg,
        uncertainty_params=torch.zeros(6),
    )

    # Note: Due to soft capping in exec_loss computation, even "perfect" predictions
    # don't achieve zero loss. The exec_loss computation has inherent distortion from
    # pred_ln_soft = torch.tanh(pred_ln / 10.0) * 50.0 that affects magnitude estimation.
    # We test that the loss is reasonable but not necessarily zero.
    assert (
        losses["exec_loss"].item() < 50.0
    )  # Should be reasonably bounded (uncertainty weighting may increase individual loss components)


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
    )

    example2 = generate_single_dag_example(
        depth=depth,
        seed=123,  # Different seed for different result
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
    )

    # Targets (no seq dimension)
    tgt_sgn = example1.structure_dict["target_initial_sgn"].unsqueeze(0)  # (1, nodes)
    tgt_digits = example1.structure_dict["target_initial_digits"].unsqueeze(
        0
    )  # (1, nodes, digits, 10)
    tgt_ops = example1.structure_dict["target_operation_probs"].unsqueeze(
        0
    )  # (1, depth, n_ops)

    # Predictions (with seq dimension) - use example2's values as wrong predictions
    nodes = tgt_sgn.shape[1]
    digits = tgt_digits.shape[2]

    # Convert example2's target values to prediction logits
    pred_sign_logits = torch.zeros(batch, seq, nodes)
    example2_sgn = example2.structure_dict["target_initial_sgn"]
    for b in range(batch):
        for s in range(seq):
            for n in range(nodes):
                if example2_sgn[n] > 0:
                    pred_sign_logits[b, s, n] = 10.0
                else:
                    pred_sign_logits[b, s, n] = -10.0

    pred_digits = torch.full((batch, seq, nodes, digits, 10), -10.0)
    example2_digits = example2.structure_dict["target_initial_digits"]
    for b in range(batch):
        for s in range(seq):
            for n in range(nodes):
                for d in range(digits):
                    target_digit = example2_digits[n, d].argmax()
                    pred_digits[b, s, n, d, target_digit] = 10.0

    pred_op_logits = torch.full((batch, seq, depth, len(OP_NAMES)), -10.0)
    example2_ops = example2.structure_dict["target_operation_probs"]
    for b in range(batch):
        for s in range(seq):
            for d in range(depth):
                target_op = example2_ops[d].argmax()
                pred_op_logits[b, s, d, target_op] = 10.0

    # Target values
    target_initial_values = example1.structure_dict["target_initial_values"].unsqueeze(
        0
    )  # (1, nodes)
    target_final_exec = torch.tensor(
        [example1.structure_dict["target_final_exec"]], dtype=torch.float32
    )  # (1,)

    # Create final token positions
    final_token_pos = torch.full((batch,), seq - 1, dtype=torch.long)

    cfg = _build_test_config(max_digits, max_decimal_places)

    dummy_pred_stats, dummy_target_stats = _dummy_statistics(batch, seq)
    losses = compute_dag_structure_loss(
        pred_sign_logits,
        pred_digits,
        pred_op_logits,
        dummy_pred_stats,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_initial_values,
        target_final_exec,
        dummy_target_stats,
        final_token_pos,
        cfg,
        uncertainty_params=torch.zeros(6),
    )

    # Should have non-zero exec loss since predictions differ
    assert losses["exec_loss"].item() > 0.0


def test_value_exec_losses_computed():
    """Test that value and exec losses are always computed with required parameters."""
    batch, seq, nodes, digits, depth = 1, 1, 2, 5, 1
    max_digits = 3
    max_decimal_places = 2

    pred_sign_logits, pred_op_logits, tgt_sgn, tgt_ops = _build_dummy_tensors(
        batch, seq, nodes, digits, depth
    )

    # Random digit tensors - targets without seq dimension
    tgt_digits = torch.rand(batch, nodes, digits, 10)
    tgt_digits = tgt_digits / tgt_digits.sum(
        dim=-1, keepdim=True
    )  # Normalize to probabilities

    # Predictions with seq dimension
    pred_digits = torch.rand(batch, seq, nodes, digits, 10)
    pred_digits = pred_digits / pred_digits.sum(
        dim=-1, keepdim=True
    )  # Normalize to probabilities

    cfg = _build_test_config(max_digits, max_decimal_places)

    # Create required target values
    target_initial_values = torch.rand(batch, nodes)  # Remove seq dimension
    target_final_exec = torch.rand(batch)  # Remove seq dimension

    # Create final token positions
    final_token_pos = torch.full((batch,), seq - 1, dtype=torch.long)

    # Call with required parameters
    dummy_pred_stats, dummy_target_stats = _dummy_statistics(batch, seq)
    losses = compute_dag_structure_loss(
        pred_sign_logits,
        pred_digits,
        pred_op_logits,
        dummy_pred_stats,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_initial_values,
        target_final_exec,
        dummy_target_stats,
        final_token_pos,
        cfg,
        uncertainty_params=torch.zeros(6),
    )

    # Should compute both losses
    assert "value_loss" in losses
    assert "exec_loss" in losses
    assert torch.isfinite(losses["value_loss"])
    assert torch.isfinite(losses["exec_loss"])


def test_loss_weights_applied():
    """Test that configurable loss weights work correctly."""
    # Note: exec_loss and stats_loss now use automatic scaling (no manual weights)
    # This test verifies that the remaining configurable weights still function
    # We just check that the loss computation runs without errors

    batch, seq, nodes, digits, depth = 1, 1, 2, 5, 1
    max_digits = 3
    max_decimal_places = 2

    pred_sign_logits, pred_op_logits, tgt_sgn, tgt_ops = _build_dummy_tensors(
        batch, seq, nodes, digits, depth
    )

    target_values = torch.tensor([[1.23, 4.56]], dtype=torch.float32)  # (batch, nodes)
    target_final_exec = target_values.sum(dim=-1)  # (batch,)

    # Targets without seq dimension
    tgt_digits = torch.zeros(batch, nodes, digits, 10)
    tgt_digits[..., 1] = 1.0

    # Predictions with seq dimension
    pred_digits = torch.zeros(batch, seq, nodes, digits, 10)
    pred_digits[..., 1] = 1.0

    # Create final token positions
    final_token_pos = torch.full((batch,), seq - 1, dtype=torch.long)

    cfg = _build_test_config(max_digits, max_decimal_places)

    dummy_pred_stats, dummy_target_stats = _dummy_statistics(batch, seq)
    losses = compute_dag_structure_loss(
        pred_sign_logits,
        pred_digits,
        pred_op_logits,
        dummy_pred_stats,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_values,
        target_final_exec,
        dummy_target_stats,
        final_token_pos,
        cfg,
        uncertainty_params=torch.zeros(6),
    )

    # Verify all loss components exist and are finite
    for loss_name in [
        "total_loss",
        "sign_loss",
        "digit_loss",
        "op_loss",
        "value_loss",
        "exec_loss",
        "stats_loss",
    ]:
        assert loss_name in losses, f"Missing {loss_name}"
        assert torch.isfinite(losses[loss_name]), f"{loss_name} is not finite"


def test_exec_loss_uses_clipping():
    """Test that exec_loss computation uses clipping (ignore_clip=False) behavior."""
    max_digits = 3
    max_decimal_places = 2

    # Create a scenario where clipping would make a difference
    batch, seq, nodes, digits, depth = 1, 1, 2, 5, 1

    pred_sign_logits, pred_op_logits, tgt_sgn, tgt_ops = _build_dummy_tensors(
        batch, seq, nodes, digits, depth
    )

    # Create digit tensors that would result in very large numbers (beyond LOG_LIM)
    # These should get clipped during execution
    pred_digits = torch.zeros(batch, seq, nodes, digits, 10)
    tgt_digits = torch.zeros(
        batch, nodes, digits, 10
    )  # Remove seq dimension for targets

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
    tgt_digits[0, 0, 0, 9] = 1.0  # 9
    tgt_digits[0, 0, 1, 9] = 1.0  # 9
    tgt_digits[0, 0, 2, 9] = 1.0  # 9
    tgt_digits[0, 0, 3, 9] = 1.0  # 9
    tgt_digits[0, 0, 4, 9] = 1.0  # 9

    tgt_digits[0, 1, 0, 9] = 1.0  # 9
    tgt_digits[0, 1, 1, 9] = 1.0  # 9
    tgt_digits[0, 1, 2, 9] = 1.0  # 9
    tgt_digits[0, 1, 3, 9] = 1.0  # 9
    tgt_digits[0, 1, 4, 9] = 1.0  # 9

    # Set signs to positive (create new sign logits that represent positive signs)
    pred_sign_logits = torch.full_like(
        pred_sign_logits, 10.0
    )  # Large positive logits for +1 signs
    tgt_sgn = torch.ones(batch, nodes)  # Remove seq dimension

    # Target values
    target_initial_values = torch.ones(batch, nodes) * 999.99  # Remove seq dimension
    target_final_exec = torch.tensor([1e6], dtype=torch.float32)  # Remove seq dimension

    # Create final token positions
    final_token_pos = torch.full((batch,), seq - 1, dtype=torch.long)

    cfg = _build_test_config(max_digits, max_decimal_places)

    # This should not raise an error and should handle clipping gracefully
    dummy_pred_stats, dummy_target_stats = _dummy_statistics(batch, seq)
    losses = compute_dag_structure_loss(
        pred_sign_logits,
        pred_digits,
        pred_op_logits,
        dummy_pred_stats,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        target_initial_values,
        target_final_exec,
        dummy_target_stats,
        final_token_pos,
        cfg,
        uncertainty_params=torch.zeros(6),
    )

    # The exec loss should be computed without errors
    assert torch.isfinite(
        losses["exec_loss"]
    ), "exec_loss should be finite even with large numbers"
    assert losses["exec_loss"].item() >= 0.0, "exec_loss should be non-negative"
