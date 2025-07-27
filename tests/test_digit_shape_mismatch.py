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


def _make_one_hot(indices: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Utility to convert an index tensor to one-hot representation."""
    shape = (*indices.shape, num_classes)
    one_hot = torch.zeros(shape, dtype=torch.float32)
    one_hot.scatter_(-1, indices.unsqueeze(-1), 1.0)
    return one_hot


def _build_common_tensors(batch: int, seq: int, nodes: int, depth: int):
    """Return sign/op tensors shared by both tests."""
    # Signs – targets without seq dimension, predictions with seq dimension
    tgt_sgn = torch.ones(batch, nodes)  # Remove seq dimension for targets
    pred_sign_logits = torch.full(
        (batch, seq, nodes), 10.0
    )  # Large positive logits for +1 signs

    # Operations – pick the first op everywhere
    n_ops = len(OP_NAMES)
    # Targets without seq dimension
    op_idx = torch.zeros(
        batch, depth, dtype=torch.long
    )  # Remove seq dimension for targets
    tgt_ops = _make_one_hot(op_idx, n_ops)
    # Predictions with seq dimension
    pred_op_logits = torch.full((batch, seq, depth, n_ops), -10.0)
    pred_op_logits[:, :, :, 0] = 10.0  # Large positive logit for first operation

    return pred_sign_logits, pred_op_logits, tgt_sgn, tgt_ops


def _dummy_cfg():
    """Minimal namespace carrying the config parameters."""
    return types.SimpleNamespace(
        # All loss weights removed - using automatic balancing
        max_digits=4,
        max_decimal_places=2,
        base=10,
    )


def test_digit_shape_match_no_error():
    """compute_dag_structure_loss should run without error when digit counts match."""
    batch, seq, nodes, digits, depth = 2, 1, 3, 6, 2

    pred_sign_logits, pred_op_logits, tgt_sgn, tgt_ops = _build_common_tensors(
        batch, seq, nodes, depth
    )

    # Random target digits - targets without seq dimension
    tgt_digit_idx = torch.randint(0, 10, (batch, nodes, digits))  # Remove seq dimension
    tgt_digits = _make_one_hot(tgt_digit_idx)
    # Predictions with seq dimension - convert targets to prediction logits
    pred_digits = torch.full((batch, seq, nodes, digits, 10), -10.0)
    for b in range(batch):
        for s in range(seq):
            for n in range(nodes):
                for d in range(digits):
                    target_digit = tgt_digits[b, n, d].argmax()
                    pred_digits[b, s, n, d, target_digit] = 10.0

    # Add dummy targets for the new losses
    target_initial_values = torch.ones(batch, nodes)  # Remove seq dimension
    target_final_exec = torch.ones(batch)  # Remove seq and node dimensions

    # Create final token positions
    final_token_pos = torch.full((batch,), seq - 1, dtype=torch.long)

    dummy_pred_stats, dummy_target_stats = _dummy_statistics(batch, seq)
    # Should not raise
    compute_dag_structure_loss(
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
        _dummy_cfg(),
        uncertainty_params=torch.zeros(6),
    )


def test_digit_shape_mismatch_raises_error():
    """When D differs between prediction and target, a ValueError should be raised."""
    batch, seq, nodes, tgt_digits_count, pred_digits_count, depth = 1, 1, 2, 6, 4, 1

    pred_sign_logits, pred_op_logits, tgt_sgn, tgt_ops = _build_common_tensors(
        batch, seq, nodes, depth
    )

    # Target has 6 digit slots - targets without seq dimension
    tgt_digit_idx = torch.randint(
        0, 10, (batch, nodes, tgt_digits_count)
    )  # Remove seq dimension
    tgt_digits = _make_one_hot(tgt_digit_idx)

    # Prediction intentionally has only 4 digit slots -> mismatch - predictions with seq dimension
    pred_digit_idx = torch.randint(0, 10, (batch, seq, nodes, pred_digits_count))
    pred_digits = _make_one_hot(pred_digit_idx)

    # Add dummy targets for the new losses
    target_initial_values = torch.ones(batch, nodes)  # Remove seq dimension
    target_final_exec = torch.ones(batch)  # Remove seq and node dimensions

    # Create final token positions
    final_token_pos = torch.full((batch,), seq - 1, dtype=torch.long)

    dummy_pred_stats, dummy_target_stats = _dummy_statistics(batch, seq)
    with pytest.raises(ValueError):
        compute_dag_structure_loss(
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
            _dummy_cfg(),
            uncertainty_params=torch.zeros(6),
        )
