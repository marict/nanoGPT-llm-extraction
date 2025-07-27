import unittest

import pytest
import torch

from models.dag_model import OP_NAMES
from predictor_config import DAGTrainConfig
from predictor_utils import compute_dag_structure_loss

N_OPS = len(OP_NAMES)


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

        # Targets without seq dimension
        target_sgn = torch.ones(B, nodes)
        # Convert to sign logits for predictions (with seq dimension)
        pred_sign_logits = torch.full(
            (B, T, nodes), 10.0
        )  # Large positive logits for +1 signs

        # Digits tensor – all zeros except one example digit 3
        D_total = cfg.max_digits + cfg.max_decimal_places
        target_digits = torch.zeros(B, nodes, D_total, 10)  # Remove seq dimension
        target_digits[..., 0] = 1.0  # Initialize all positions as digit 0
        target_digits[..., 0, 0] = 0.0  # Clear the 0 for first position
        target_digits[..., 0, 3] = 1.0  # first slot digit 3

        # Convert target to logits for prediction (with seq dimension)
        pred_digits = torch.full((B, T, nodes, D_total, 10), -10.0)
        for b in range(B):
            for s in range(T):
                for n in range(nodes):
                    for d in range(D_total):
                        target_digit = target_digits[b, n, d].argmax()
                        pred_digits[b, s, n, d, target_digit] = 10.0

        # Operations
        target_ops = torch.zeros(B, depth, n_ops)  # Remove seq dimension
        target_ops[..., 0] = 1.0
        # Convert to operation logits for predictions (with seq dimension)
        pred_op_logits = torch.full((B, T, depth, n_ops), -10.0)
        pred_op_logits[:, :, :, 0] = 10.0  # Large positive logit for first operation

        # Add dummy targets for the new losses
        target_initial_values = torch.ones(B, nodes)  # Remove seq dimension
        target_final_exec = torch.ones(B)  # Remove seq and node dimensions

        # Create final token positions
        final_token_pos = torch.full((B,), T - 1, dtype=torch.long)

        dummy_pred_stats, dummy_target_stats = _dummy_statistics(1, 1)
        losses = compute_dag_structure_loss(
            pred_sign_logits,
            pred_digits,
            pred_op_logits,
            dummy_pred_stats,
            target_sgn,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            dummy_target_stats,
            final_token_pos,
            cfg,
            uncertainty_params=torch.zeros(6),
        )

        self.assertLess(losses["digit_loss"].item(), 1e-6)

    def test_op_loss_zero_probability_stability(self):
        """Loss should remain finite even when some predicted op probabilities are zero."""
        cfg = self._make_cfg()
        B, T, nodes, depth, n_ops = 1, 1, cfg.dag_depth + 1, cfg.dag_depth, N_OPS

        # Dummy signs/logs (not used for this test)
        # Targets without seq dimension
        target_sgn = torch.ones(B, nodes)
        # Convert to sign logits for predictions (with seq dimension)
        pred_sign_logits = torch.full(
            (B, T, nodes), 10.0
        )  # Large positive logits for +1 signs

        D_total = cfg.max_digits + cfg.max_decimal_places
        # Predictions with seq dimension
        pred_digits = torch.zeros(B, T, nodes, D_total, 10)
        pred_digits[..., 0] = 1.0  # Initialize all positions as digit 0
        # Targets without seq dimension
        target_digits = torch.zeros(B, nodes, D_total, 10)
        target_digits[..., 0] = 1.0  # Initialize all positions as digit 0

        # Target op: always "add"
        target_ops = torch.zeros(B, depth, n_ops)  # Remove seq dimension
        target_ops[..., 0] = 1.0

        # Predicted ops: deliberately set some zeros - convert to logits
        pred_op_logits = torch.full(
            (B, T, depth, n_ops), float("-inf")
        )  # Start with -inf
        pred_op_logits[..., 0] = 0.0  # correct op gets large logit; others -inf

        # Add dummy targets for the new losses
        target_initial_values = torch.ones(B, nodes)  # Remove seq dimension
        target_final_exec = torch.ones(B)  # Remove seq and node dimensions

        # Create final token positions
        final_token_pos = torch.full((B,), T - 1, dtype=torch.long)

        dummy_pred_stats, dummy_target_stats = _dummy_statistics(1, 1)
        losses = compute_dag_structure_loss(
            pred_sign_logits,
            pred_digits,
            pred_op_logits,
            dummy_pred_stats,
            target_sgn,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            dummy_target_stats,
            final_token_pos,
            cfg,
            uncertainty_params=torch.zeros(6),
        )

        # All returned losses must be finite numbers
        for key, value in losses.items():
            self.assertTrue(
                torch.isfinite(value).all(), f"Non-finite loss detected in {key}"
            )


if __name__ == "__main__":
    unittest.main()
