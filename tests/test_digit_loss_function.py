import types

import pytest
import torch

from models.dag_model import OP_NAMES
from predictor_utils import compute_dag_structure_loss


def _dummy_statistics(batch_size, seq_len=1):
    """Create dummy statistics for testing."""
    return {
        "initial": torch.zeros(batch_size, seq_len, 15),
        "intermediate": torch.zeros(batch_size, seq_len, 15),
        "final": torch.zeros(batch_size, seq_len, 10),
    }


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

    # Target initial values and final execution values (dummy values)
    tgt_initial_values = torch.ones(batch, seq, nodes, device=device)
    tgt_final_exec = torch.ones(
        batch, seq, 1, device=device
    )  # Single final execution result

    return pred_sgn, pred_ops, tgt_sgn, tgt_ops, tgt_initial_values, tgt_final_exec


@pytest.mark.parametrize("batch,seq,nodes,digits,depth", [(2, 1, 3, 4, 2)])
def test_digit_loss_zero_and_nonzero(batch, seq, nodes, digits, depth):
    """Digit loss should be zero for perfect predictions and >0 for completely wrong ones."""
    pred_sgn, pred_ops, tgt_sgn, tgt_ops, tgt_initial_values, tgt_final_exec = (
        _build_dummy_tensors(batch, seq, nodes, digits, depth)
    )

    # Build target digit indices randomly, then one-hot encode
    tgt_digit_idx = torch.randint(0, 10, (batch, seq, nodes, digits))
    tgt_digits = _make_one_hot(tgt_digit_idx, 10)

    # 1) Perfect predictions – convert targets to logits that would produce the same one-hot after softmax
    pred_digits_correct = torch.full_like(
        tgt_digits, -10.0
    )  # Start with very negative logits
    # Set large positive logits where targets are 1
    pred_digits_correct = (
        pred_digits_correct + tgt_digits * 20.0
    )  # This gives 10.0 where target is 1, -10.0 elsewhere

    cfg = types.SimpleNamespace(
        max_digits=3,
        max_decimal_places=1,
        base=10,
    )

    dummy_stats = _dummy_statistics(batch, seq)
    losses_correct = compute_dag_structure_loss(
        pred_sgn,
        pred_digits_correct,
        pred_ops,
        dummy_stats,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        tgt_initial_values,
        tgt_final_exec,
        dummy_stats,
        cfg,
        log_vars=torch.zeros(6),
    )

    assert pytest.approx(losses_correct["digit_loss"].item(), abs=1e-6) == 0.0

    # 2) Completely wrong predictions – shift indices by +1 (mod 10)
    wrong_digit_idx = (tgt_digit_idx + 1) % 10
    wrong_digits_one_hot = _make_one_hot(wrong_digit_idx, 10)
    # Convert to logits
    pred_digits_wrong = torch.full_like(wrong_digits_one_hot, -10.0)
    pred_digits_wrong = pred_digits_wrong + wrong_digits_one_hot * 20.0

    losses_wrong = compute_dag_structure_loss(
        pred_sgn,
        pred_digits_wrong,
        pred_ops,
        dummy_stats,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        tgt_initial_values,
        tgt_final_exec,
        dummy_stats,
        cfg,
        log_vars=torch.zeros(6),
    )

    assert losses_wrong["digit_loss"].item() > 0.0
