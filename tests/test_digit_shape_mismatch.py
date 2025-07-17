import types

import pytest
import torch

from models.dag_model import OP_NAMES
from predictor_utils import compute_dag_structure_loss


def _make_one_hot(indices: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Utility to convert an index tensor to one-hot representation."""
    shape = (*indices.shape, num_classes)
    one_hot = torch.zeros(shape, dtype=torch.float32)
    one_hot.scatter_(-1, indices.unsqueeze(-1), 1.0)
    return one_hot


def _build_common_tensors(batch: int, seq: int, nodes: int, depth: int):
    """Return sign/op tensors shared by both tests."""
    # Signs – all +1 for simplicity
    tgt_sgn = torch.ones(batch, seq, nodes)
    pred_sgn = tgt_sgn.clone()

    # Operations – pick the first op everywhere
    n_ops = len(OP_NAMES)
    op_idx = torch.zeros(batch, seq, depth, dtype=torch.long)
    tgt_ops = _make_one_hot(op_idx, n_ops)
    pred_ops = tgt_ops.clone()

    return pred_sgn, pred_ops, tgt_sgn, tgt_ops


def _dummy_cfg():
    """Minimal namespace carrying the three loss weights."""
    return types.SimpleNamespace(
        sign_loss_weight=0.0,
        digit_loss_weight=1.0,
        op_loss_weight=0.0,
    )


def test_digit_shape_match_no_error():
    """compute_dag_structure_loss should run without error when digit counts match."""
    batch, seq, nodes, digits, depth = 2, 1, 3, 6, 2

    pred_sgn, pred_ops, tgt_sgn, tgt_ops = _build_common_tensors(
        batch, seq, nodes, depth
    )

    # Random target digits and identical predictions with *matching* D
    tgt_digit_idx = torch.randint(0, 10, (batch, seq, nodes, digits))
    tgt_digits = _make_one_hot(tgt_digit_idx)
    pred_digits = tgt_digits.clone()

    # Should not raise
    compute_dag_structure_loss(
        pred_sgn,
        pred_digits,
        pred_ops,
        tgt_sgn,
        tgt_digits,
        tgt_ops,
        _dummy_cfg(),
    )


def test_digit_shape_mismatch_raises_error():
    """When D differs between prediction and target, a ValueError should be raised."""
    batch, seq, nodes, tgt_digits_count, pred_digits_count, depth = 1, 1, 2, 6, 4, 1

    pred_sgn, pred_ops, tgt_sgn, tgt_ops = _build_common_tensors(
        batch, seq, nodes, depth
    )

    # Target has 6 digit slots
    tgt_digit_idx = torch.randint(0, 10, (batch, seq, nodes, tgt_digits_count))
    tgt_digits = _make_one_hot(tgt_digit_idx)

    # Prediction intentionally has only 4 digit slots -> mismatch
    pred_digit_idx = torch.randint(0, 10, (batch, seq, nodes, pred_digits_count))
    pred_digits = _make_one_hot(pred_digit_idx)

    with pytest.raises(ValueError):
        compute_dag_structure_loss(
            pred_sgn,
            pred_digits,
            pred_ops,
            tgt_sgn,
            tgt_digits,
            tgt_ops,
            _dummy_cfg(),
        )
