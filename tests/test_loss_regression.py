import unittest

import torch

from predictor_config import DAGTrainConfig
from predictor_utils import compute_dag_structure_loss


class TestLossRegressions(unittest.TestCase):
    """Regression tests for historical loss-function bugs."""

    def _make_cfg(self):
        cfg = DAGTrainConfig()
        cfg.dag_depth = 2
        cfg.sign_loss_weight = 1.0
        cfg.log_loss_weight = 1.0
        cfg.op_loss_weight = 1.0
        return cfg

    def test_negative_log_magnitude_support(self):
        """Perfect prediction with negative log values should yield ~zero log_loss."""
        cfg = self._make_cfg()
        B, T, nodes, depth, n_ops = 1, 1, cfg.dag_depth + 1, cfg.dag_depth, 5

        # Signs (all +1 for simplicity)
        target_sgn = torch.ones(B, T, nodes)
        pred_sgn = target_sgn.clone()

        # Negative log-magnitudes (simulate |v| < 1)
        target_log = torch.tensor(
            [[[-0.1, -1.0, -2.0]]], dtype=torch.float32
        )  # shape (1,1,3)
        pred_log = target_log.clone()

        # Operations – one-hot on first op ("add")
        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[..., 0] = 1.0
        pred_ops = target_ops.clone()

        losses = compute_dag_structure_loss(
            pred_sgn, pred_log, pred_ops, target_sgn, target_log, target_ops, cfg
        )

        # Expect virtually zero magnitude loss (<1e-6) when prediction is perfect
        self.assertLess(losses["log_loss"].item(), 1e-6)

    def test_op_loss_zero_probability_stability(self):
        """Loss should remain finite even when some predicted op probabilities are zero."""
        cfg = self._make_cfg()
        B, T, nodes, depth, n_ops = 1, 1, cfg.dag_depth + 1, cfg.dag_depth, 5

        # Dummy signs/logs (not used for this test)
        pred_sgn = torch.ones(B, T, nodes)
        target_sgn = pred_sgn.clone()
        pred_log = torch.zeros(B, T, nodes)
        target_log = pred_log.clone()

        # Target op: always "add"
        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[..., 0] = 1.0

        # Predicted ops: deliberately set some zeros
        pred_ops = torch.zeros(B, T, depth, n_ops)
        pred_ops[..., 0] = 1.0  # correct op gets prob=1; others zero

        losses = compute_dag_structure_loss(
            pred_sgn, pred_log, pred_ops, target_sgn, target_log, target_ops, cfg
        )

        # All returned losses must be finite numbers
        for key, value in losses.items():
            self.assertTrue(
                torch.isfinite(value).all(), f"Non-finite loss detected in {key}"
            )


if __name__ == "__main__":
    unittest.main()
