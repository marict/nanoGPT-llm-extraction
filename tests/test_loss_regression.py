import unittest

import pytest
import torch

from models.dag_model import OP_NAMES
from predictor_config import DAGTrainConfig
from predictor_utils import compute_dag_structure_loss

N_OPS = len(OP_NAMES)


class TestLossRegressions(unittest.TestCase):
    """Regression tests for historical loss-function bugs."""

    def _make_cfg(self):
        cfg = DAGTrainConfig()
        cfg.dag_depth = 2
        cfg.sign_loss_weight = 1.0
        cfg.digit_loss_weight = 1.0
        cfg.op_loss_weight = 1.0
        cfg.value_loss_weight = 1.0
        cfg.exec_loss_weight = 1.0
        cfg.max_digits = 4
        cfg.max_decimal_places = 2
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
        target_digits[..., 0, 3] = 1.0  # first slot digit 3
        pred_digits = target_digits.clone()

        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[..., 0] = 1.0
        pred_ops = target_ops.clone()

        # Add dummy targets for the new losses
        target_initial_values = torch.ones(B, T, nodes)
        target_final_exec = torch.ones(B, T, 1)

        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_digits,
            pred_ops,
            target_sgn,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            cfg,
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

        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_digits,
            pred_ops,
            target_sgn,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            cfg,
        )

        # All returned losses must be finite numbers
        for key, value in losses.items():
            self.assertTrue(
                torch.isfinite(value).all(), f"Non-finite loss detected in {key}"
            )


if __name__ == "__main__":
    unittest.main()
