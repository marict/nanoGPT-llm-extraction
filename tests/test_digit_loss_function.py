import types

import pytest
import torch

from models.dag_model import OP_NAMES
from predictor_utils import compute_dag_structure_loss


def _dummy_statistics(batch_size, seq_len=1):
    """Create dummy statistics for testing."""
    dummy_pred_stats = {
        "initial": torch.zeros(batch_size, seq_len, 15),
        "intermediate": torch.zeros(batch_size, seq_len, 15),
        "final": torch.zeros(batch_size, seq_len, 10),
    }
    dummy_target_stats = {
        "initial": torch.zeros(batch_size, 15),
        "intermediate": torch.zeros(batch_size, 15),
        "final": torch.zeros(batch_size, 10),
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

    # Sign predictions (with seq dimension) / targets (without seq dimension)
    tgt_sgn = torch.ones(
        batch, nodes, device=device
    )  # Remove seq dimension for targets
    pred_sign_logits = torch.full(
        (batch, seq, nodes), 10.0, device=device
    )  # Large positive logits for +1 signs

    # Operations (choose the first op for all positions)
    n_ops = len(OP_NAMES)
    tgt_op_idx = torch.zeros(
        batch, depth, dtype=torch.long, device=device
    )  # Remove seq dimension for targets
    tgt_ops = _make_one_hot(tgt_op_idx, n_ops)
    # Create operation logits instead of one-hot probabilities
    pred_op_logits = torch.full((batch, seq, depth, n_ops), -10.0, device=device)
    pred_op_logits[:, :, :, 0] = 10.0  # Large positive logit for first operation

    # Target initial values and final execution values (dummy values)
    tgt_initial_values = torch.ones(batch, nodes, device=device)  # Remove seq dimension
    tgt_final_exec = torch.ones(batch, device=device)  # Remove seq and node dimensions

    return (
        pred_sign_logits,
        pred_op_logits,
        tgt_sgn,
        tgt_ops,
        tgt_initial_values,
        tgt_final_exec,
    )


@pytest.mark.parametrize("batch,seq,nodes,digits,depth", [(2, 1, 3, 4, 2)])
def test_digit_loss_zero_and_nonzero(batch, seq, nodes, digits, depth):
    """Digit loss should be zero for perfect predictions and >0 for completely wrong ones."""
    (
        pred_sign_logits,
        pred_op_logits,
        tgt_sgn,
        tgt_ops,
        tgt_initial_values,
        tgt_final_exec,
    ) = _build_dummy_tensors(batch, seq, nodes, digits, depth)

    # Build target digit indices randomly, then one-hot encode (targets without seq dimension)
    tgt_digit_idx = torch.randint(0, 10, (batch, nodes, digits))  # Remove seq dimension
    tgt_digits = _make_one_hot(tgt_digit_idx, 10)

    # 1) Perfect predictions – convert targets to logits that would produce the same one-hot after softmax
    # Predictions have seq dimension
    pred_digits_correct = torch.full((batch, seq, nodes, digits, 10), -10.0)
    # Set large positive logits where targets are 1, broadcast across seq dimension
    for b in range(batch):
        for s in range(seq):
            for n in range(nodes):
                for d in range(digits):
                    target_digit = tgt_digits[b, n, d].argmax()
                    pred_digits_correct[b, s, n, d, target_digit] = 10.0

    cfg = types.SimpleNamespace(
        max_digits=3,
        max_decimal_places=1,
        base=10,
    )

    # Create final token positions
    final_token_pos = torch.full((batch,), seq - 1, dtype=torch.long)

    dummy_pred_stats, dummy_target_stats = _dummy_statistics(batch, seq)
    losses_correct = compute_dag_structure_loss(
        pred_sign_logits,
        pred_digits_correct,
        pred_op_logits,
        dummy_pred_stats,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        tgt_initial_values,
        tgt_final_exec,
        dummy_target_stats,
        final_token_pos,
        cfg,
        uncertainty_params=torch.zeros(6),
    )

    assert pytest.approx(losses_correct["digit_loss"].item(), abs=1e-6) == 0.0

    # 2) Completely wrong predictions – shift indices by +1 (mod 10)
    wrong_digit_idx = (tgt_digit_idx + 1) % 10
    # Convert to logits
    pred_digits_wrong = torch.full((batch, seq, nodes, digits, 10), -10.0)
    for b in range(batch):
        for s in range(seq):
            for n in range(nodes):
                for d in range(digits):
                    wrong_digit = wrong_digit_idx[b, n, d]
                    pred_digits_wrong[b, s, n, d, wrong_digit] = 10.0

    losses_wrong = compute_dag_structure_loss(
        pred_sign_logits,
        pred_digits_wrong,
        pred_op_logits,
        dummy_pred_stats,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        tgt_initial_values,
        tgt_final_exec,
        dummy_target_stats,
        final_token_pos,
        cfg,
        uncertainty_params=torch.zeros(6),
    )

    assert losses_wrong["digit_loss"].item() > 0.0
