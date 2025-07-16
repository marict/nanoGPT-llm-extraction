import types

import pytest
import torch

from data.dagset.streaming import float_to_digit_onehot
from models.dag_model import OP_NAMES
from predictor_utils import compute_dag_structure_loss, digits_to_magnitude


def _make_one_hot(idx_tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Utility: convert an index tensor to one-hot representation."""
    shape = (*idx_tensor.shape, num_classes)
    one_hot = torch.zeros(shape, dtype=torch.float32)
    one_hot.scatter_(-1, idx_tensor.unsqueeze(-1), 1.0)
    return one_hot


def test_digit_representation_and_loss_for_negative_value():
    """Ensure that -83.34 is encoded as -0083.3400 and yields zero digit loss when matched."""
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    value = -83.34
    max_digits = 4  # integer digits
    max_decimal_places = 4  # fractional digits
    expected_digits = [0, 0, 8, 3, 3, 4, 0, 0]  # "-0083.3400"

    # ------------------------------------------------------------------
    # Build target digit one-hot tensor
    # ------------------------------------------------------------------
    one_hot = float_to_digit_onehot(value, max_digits, max_decimal_places)

    # 1) Check that each slot encodes the expected digit
    digit_indices = one_hot.argmax(dim=-1).tolist()
    assert digit_indices == expected_digits, "Digit encoding mismatch for -83.34"

    # 2) Verify the magnitude reconstruction helper
    magnitude = digits_to_magnitude(one_hot, max_digits, max_decimal_places).item()
    assert pytest.approx(magnitude, abs=1e-4) == abs(value)

    # ------------------------------------------------------------------
    # Build minimal tensors for compute_dag_structure_loss
    # ------------------------------------------------------------------
    # Shapes: (B=1, T=1, N=1, D, 10)
    target_digits = one_hot.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,D,10)
    pred_digits = target_digits.clone()

    # Sign tensors (B,T,N)
    target_sgn = torch.tensor([[[-1.0]]])
    pred_sgn = target_sgn.clone()

    # Operation tensors – single identity op to keep depth > 0
    depth = 1
    n_ops = len(OP_NAMES)
    op_idx = torch.tensor([[[OP_NAMES.index("identity")]]])  # (1,1,1)
    target_ops = _make_one_hot(op_idx, n_ops)
    pred_ops = target_ops.clone()

    # ------------------------------------------------------------------
    # Compute digit loss; should be ~0 for perfect prediction
    # ------------------------------------------------------------------
    cfg = types.SimpleNamespace(
        sign_loss_weight=0.0,
        digit_loss_weight=1.0,
        op_loss_weight=0.0,
    )

    losses = compute_dag_structure_loss(
        pred_sgn,
        pred_digits,
        pred_ops,
        target_sgn,
        target_digits,
        target_ops,
        cfg,
    )

    assert pytest.approx(losses["digit_loss"].item(), abs=1e-6) == 0.0
