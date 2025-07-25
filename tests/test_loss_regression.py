import unittest

import pytest
import torch

from models.dag_model import OP_NAMES
from predictor_config import DAGTrainConfig
from predictor_utils import compute_dag_structure_loss

N_OPS = len(OP_NAMES)


def _dummy_statistics(batch_size, seq_len=1):
    """Create dummy statistics for testing."""
    return {
        "initial": torch.zeros(batch_size, seq_len, 15),
        "intermediate": torch.zeros(batch_size, seq_len, 15),
        "final": torch.zeros(batch_size, seq_len, 10),
    }


class TestLossRegressions(unittest.TestCase):
    """Regression tests for historical loss-function bugs."""

    def _make_cfg(self):
        cfg = DAGTrainConfig()
        cfg.dag_depth = 2
        cfg.max_digits = 4
        cfg.max_decimal_places = 2
        cfg.base = 10
        return cfg

    def test_negative_log_magnitude_support(self):
        """Perfect prediction should yield ~zero digit_loss."""
        cfg = self._make_cfg()
        B, T, nodes, depth, n_ops = 1, 1, cfg.dag_depth + 1, cfg.dag_depth, N_OPS

        target_sgn = torch.ones(B, T, nodes)
        pred_sgn = target_sgn.clone()

        # Digits tensor – all zeros except one example digit 3
        D_total = cfg.max_digits + cfg.max_decimal_places
        target_digits = torch.zeros(B, T, nodes, D_total, 10)
        target_digits[..., 0] = 1.0  # Initialize all positions as digit 0
        target_digits[..., 0, 0] = 0.0  # Clear the 0 for first position
        target_digits[..., 0, 3] = 1.0  # first slot digit 3
        # Convert target to logits for prediction
        pred_digits = torch.full_like(target_digits, -10.0)
        pred_digits = pred_digits + target_digits * 20.0

        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[..., 0] = 1.0
        pred_ops = target_ops.clone()

        # Add dummy targets for the new losses
        target_initial_values = torch.ones(B, T, nodes)
        target_final_exec = torch.ones(B, T, 1)

        dummy_stats = _dummy_statistics(1, 1)
        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_digits,
            pred_ops,
            dummy_stats,
            target_sgn,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            dummy_stats,
            cfg,
            log_vars=torch.zeros(6),
        )

        self.assertLess(losses["digit_loss"].item(), 1e-6)

    def test_op_loss_zero_probability_stability(self):
        """Loss should remain finite even when some predicted op probabilities are zero."""
        cfg = self._make_cfg()
        B, T, nodes, depth, n_ops = 1, 1, cfg.dag_depth + 1, cfg.dag_depth, N_OPS

        # Dummy signs/logs (not used for this test)
        pred_sgn = torch.ones(B, T, nodes)
        target_sgn = pred_sgn.clone()
        D_total = cfg.max_digits + cfg.max_decimal_places
        pred_digits = torch.zeros(B, T, nodes, D_total, 10)
        pred_digits[..., 0] = 1.0  # Initialize all positions as digit 0
        target_digits = pred_digits.clone()

        # Target op: always "add"
        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[..., 0] = 1.0

        # Predicted ops: deliberately set some zeros
        pred_ops = torch.zeros(B, T, depth, n_ops)
        pred_ops[..., 0] = 1.0  # correct op gets prob=1; others zero

        # Add dummy targets for the new losses
        target_initial_values = torch.ones(B, T, nodes)
        target_final_exec = torch.ones(B, T, 1)

        dummy_stats = _dummy_statistics(1, 1)
        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_digits,
            pred_ops,
            dummy_stats,
            target_sgn,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            dummy_stats,
            cfg,
            log_vars=torch.zeros(6),
        )

        # All returned losses must be finite numbers
        for key, value in losses.items():
            self.assertTrue(
                torch.isfinite(value).all(), f"Non-finite loss detected in {key}"
            )


if __name__ == "__main__":
    unittest.main()
