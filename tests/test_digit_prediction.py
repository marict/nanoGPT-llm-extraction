import math
import unittest

import torch
import torch.nn.functional as F

from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel
from predictor_utils import compute_dag_structure_loss, digits_to_magnitude
from train_predictor import DAGTrainConfig


def _dummy_statistics(batch_size, seq_len=1):
    """Create dummy statistics for testing."""
    return {
        "initial": torch.zeros(batch_size, seq_len, 15),
        "intermediate": torch.zeros(batch_size, seq_len, 15),
        "final": torch.zeros(batch_size, seq_len, 10),
    }


class TestDigitPrediction(unittest.TestCase):
    """Unit-tests for digit distribution pathway inside DAGPlanPredictor."""

    def _build_model(self, max_digits: int = 3, max_decimals: int = 2):
        """Helper – returns a fresh PredictorOnlyModel with custom digit settings."""
        cfg = PredictorOnlyConfig(
            vocab_size=1000,
            n_layer=1,
            n_embd=64,
            n_head=4,
            dropout=0.0,
            bias=False,
            dag_depth=2,
            block_size=32,
        )

        # Attach the digit-related attributes expected by DAGPlanPredictor at runtime
        cfg.max_digits = max_digits
        cfg.max_decimal_places = max_decimals
        return PredictorOnlyModel(cfg)

    # ------------------------------------------------------------------
    # Shape / size checks
    # ------------------------------------------------------------------
    def test_digit_logits_shape(self):
        """digit_logits should have expected shape (B,T,N,D,10)."""
        max_digits, max_decimals = 3, 2
        model = self._build_model(max_digits, max_decimals)
        model.eval()

        B, T = 2, 16
        input_ids = torch.randint(0, 1000, (B, T))

        with torch.no_grad():
            model(input_ids)

        # Retrieve digit logits cached by plan_predictor
        digit_logits = model.dag_predictor.last_digit_logits
        self.assertIsNotNone(digit_logits, "digit_logits attribute missing")

        N = model.config.dag_depth + 1  # scratch nodes
        D_total = max_digits + max_decimals
        expected_shape = (B, T, N, D_total, 10)
        self.assertEqual(digit_logits.shape, expected_shape)

    # ------------------------------------------------------------------
    # Reconstruction consistency
    # ------------------------------------------------------------------
    def test_digit_to_magnitude_consistency(self):
        """Magnitude reconstructed from digits should match log-based value."""
        max_digits, max_decimals = 4, 3
        model = self._build_model(max_digits, max_decimals)
        model.eval()

        B, T = 2, 20
        input_ids = torch.randint(0, 1000, (B, T))

        with torch.no_grad():
            pred_sgn, digit_probs, pred_ops, _ = model(input_ids)

            # Compute log magnitudes for comparison
            magnitude_from_log = torch.log(
                digits_to_magnitude(
                    digit_probs,
                    max_digits=max_digits,
                    max_decimal_places=max_decimals,
                    base=10,
                ).clamp_min(1e-6)
            ) / math.log(10.0)

        # Reconstruct absolute magnitude from digits
        reconstructed_abs = digits_to_magnitude(
            digit_probs,
            max_digits=max_digits,
            max_decimal_places=max_decimals,
            base=10,
        )  # (B,T,N)

        # They should match closely – both derive from the same probabilities.
        self.assertTrue(
            torch.allclose(
                reconstructed_abs, (10**magnitude_from_log).abs(), rtol=1e-3, atol=1e-3
            )
        )

    # ------------------------------------------------------------------
    # Gradient propagation
    # ------------------------------------------------------------------
    def test_digit_logits_gradients(self):
        """Loss should propagate gradients back to digit_logits tensor."""
        max_digits, max_decimals = 3, 2
        model = self._build_model(max_digits, max_decimals)
        model.train()

        B, T = 2, 24
        input_ids = torch.randint(0, 1000, (B, T))

        # Forward pass
        pred_sgn, _, pred_ops, _ = model(input_ids)
        digit_logits = model.dag_predictor.last_digit_logits
        self.assertIsNotNone(digit_logits)

        # Enable gradient capture on non-leaf tensor
        digit_logits.retain_grad()

        # ------------------------------------------------------------------
        # Build synthetic targets matching tensor shapes
        # ------------------------------------------------------------------
        N = model.config.dag_depth + 1
        D_total = max_digits + max_decimals
        sign_target = torch.randint(0, 2, (B, T, N)).float() * 2 - 1  # ±1

        # One-hot targets for digits – choose argmax of current probs for determinism
        digit_probs = F.softmax(digit_logits.detach(), dim=-1)
        digit_idx = digit_probs.argmax(dim=-1)  # (B,T,N,D)
        target_digits = F.one_hot(digit_idx, num_classes=10).float()

        # Operation targets: always choose first op for simplicity
        depth, n_ops = model.config.dag_depth, pred_ops.shape[-1]
        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[..., 0] = 1.0

        # Training config with appropriate digit settings
        cfg = DAGTrainConfig()
        cfg.dag_depth = model.config.dag_depth
        cfg.max_digits = max_digits
        cfg.max_decimal_places = max_decimals

        # Add dummy targets for the new losses
        target_initial_values = torch.ones(B, T, N)
        target_final_exec = torch.ones(B, T, 1)

        dummy_stats = _dummy_statistics(B, T)
        losses = compute_dag_structure_loss(
            pred_sgn,
            digit_logits,
            pred_ops,
            dummy_stats,
            sign_target,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            dummy_stats,
            cfg,
            log_vars=torch.zeros(6),
        )
        total_loss = losses["total_loss"]
        self.assertTrue(torch.isfinite(total_loss))

        # Backward
        total_loss.backward()

        # Verify gradients propagated to digit_logits
        self.assertIsNotNone(digit_logits.grad, "digit_logits gradient missing")
        self.assertTrue(torch.isfinite(digit_logits.grad).all())
        self.assertGreater(digit_logits.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
