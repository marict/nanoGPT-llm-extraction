import unittest

import torch
import torch.nn.functional as F

from data.dagset.streaming import float_to_digit_onehot
from models.dag_model import OP_NAMES
from predictor_utils import compute_dag_structure_loss
from train_predictor import DAGTrainConfig


class TestDigitPaddingLoss(unittest.TestCase):
    """Ensure compute_dag_structure_loss correctly handles numbers shorter than *max_digits* (leading-zero padding)."""

    def _make_onehot_tensor(
        self, value: float, max_digits: int, max_decimals: int
    ) -> torch.Tensor:
        """Utility: encode *value* with float_to_digit_onehot and return (1,1,1,D,10) tensor."""
        oh = float_to_digit_onehot(value, max_digits, max_decimals)  # (D,10)
        return oh.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,D,10)

    def test_leading_zero_encoding_and_loss(self):
        """65 should be encoded as 0065 for max_digits=4 and produce near-zero digit loss when predicted correctly."""
        max_digits, max_decimals = (
            4,
            1,
        )  # use one fractional digit to avoid split error in helper
        depth = 1  # dag_depth
        n_ops = len(OP_NAMES)

        # ------------------------------------------------------------------
        # Targets (ground truth)
        # ------------------------------------------------------------------
        target_digits = self._make_onehot_tensor(
            65, max_digits, max_decimals
        )  # (B=1,T=1,N=1,D,10)
        B, T, N, D, _ = target_digits.shape

        # Signs (+1)
        target_sgn = torch.ones(B, T, N)

        # Operation targets – choose 'add' (index 0) just for shape completeness
        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[..., 0] = 1.0

        # ------------------------------------------------------------------
        # Case 1: perfect prediction – digit logits equal to target one-hots
        # ------------------------------------------------------------------
        pred_digits_correct = target_digits.clone()
        pred_sgn = target_sgn.clone()
        pred_ops = target_ops.clone()

        cfg = DAGTrainConfig()
        cfg.dag_depth = depth
        cfg.max_digits = max_digits
        cfg.max_decimal_places = max_decimals

        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_digits_correct,
            pred_ops,
            target_sgn,
            target_digits,
            target_ops,
            cfg,
        )
        self.assertLess(
            losses["digit_loss"].item(),
            1e-6,
            "Digit loss should be ~0 for perfect leading-zero predictions",
        )

        # ------------------------------------------------------------------
        # Case 2: wrong prediction – digits without leading zeros (e.g. encode '65' as '6500')
        # ------------------------------------------------------------------
        wrong_digits = torch.zeros_like(target_digits)
        # Place 6 and 5 in the first two integer slots (00 65 -> 65 00)
        wrong_digits[..., 0, 6] = 1.0  # thousands place
        wrong_digits[..., 1, 5] = 1.0  # hundreds place

        losses_wrong = compute_dag_structure_loss(
            pred_sgn,
            wrong_digits,
            pred_ops,
            target_sgn,
            target_digits,
            target_ops,
            cfg,
        )

        self.assertGreater(
            losses_wrong["digit_loss"].item(),
            losses["digit_loss"].item() + 0.1,
            "Incorrect leading-zero handling should incur noticeably higher digit loss",
        )


if __name__ == "__main__":
    unittest.main()
