"""Test gradient cosine similarity computation.

Gradient cosines are automatically computed during training at log_interval
to show how each loss component aligns with the overall optimization direction.
"""

import unittest

import torch
import torch.nn as nn

from models.dag_model import OP_NAMES
from predictor_config import DAGTrainConfig
from predictor_utils import compute_dag_structure_loss, compute_gradient_cosines


def _dummy_statistics(batch_size, seq_len=1):
    """Create dummy statistics for testing."""
    return {
        "initial": torch.zeros(batch_size, seq_len, 15),
        "intermediate": torch.zeros(batch_size, seq_len, 15),
        "final": torch.zeros(batch_size, seq_len, 10),
    }


N_OPS = len(OP_NAMES)


class SimpleTestModel(nn.Module):
    """Simple model for testing gradient cosine computation."""

    def __init__(self, n_params=10):
        super().__init__()
        self.linear = nn.Linear(n_params, 1)

    def forward(self, x):
        return self.linear(x)


class TestGradientCosines(unittest.TestCase):
    """Test gradient cosine similarity computation."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.cfg = DAGTrainConfig()
        self.cfg.dag_depth = 2
        self.cfg.max_digits = 4
        self.cfg.max_decimal_places = 2

    def test_gradient_cosine_computation(self):
        """Test that gradient cosines are computed correctly."""
        B, T = 2, 4
        num_nodes = self.cfg.dag_depth + 1
        depth = self.cfg.dag_depth
        n_ops = N_OPS

        # Create a simple model and use it to generate predictions (so they're connected to model params)
        model = SimpleTestModel(n_params=100)
        dummy_input = torch.randn(B * T * num_nodes, 100)
        model_output = model(dummy_input)

        # Create test tensors that are connected to model parameters
        pred_sgn = torch.tanh(model_output.view(B, T, num_nodes))
        D_total = self.cfg.max_digits + self.cfg.max_decimal_places

        # Create digit logits by reshaping model output
        dummy_input2 = torch.randn(B * T * num_nodes * D_total * 10, 100)
        pred_digit_logits = model(dummy_input2).view(B, T, num_nodes, D_total, 10)

        # Create ops predictions
        dummy_input3 = torch.randn(B * T * depth * n_ops, 100)
        pred_ops = torch.softmax(model(dummy_input3).view(B, T, depth, n_ops), dim=-1)

        target_sgn = torch.randn(B, T, num_nodes)
        target_digits = torch.nn.functional.one_hot(
            torch.randint(0, 10, (B, T, num_nodes, D_total)), num_classes=10
        ).float()
        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[:, :, :, 0] = 1

        target_initial_values = torch.ones(B, T, num_nodes)
        target_final_exec = torch.ones(B, T, 1)

        model_params = list(model.parameters())

        # Test without gradient cosines
        dummy_stats = _dummy_statistics(B, T)
        losses_without = compute_dag_structure_loss(
            pred_sgn,
            pred_digit_logits,
            pred_ops,
            dummy_stats,
            target_sgn,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            dummy_stats,
            self.cfg,
        )

        # Test with gradient cosines - compute them separately
        dummy_stats = _dummy_statistics(B, T)
        losses_with = compute_dag_structure_loss(
            pred_sgn,
            pred_digit_logits,
            pred_ops,
            dummy_stats,
            target_sgn,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            dummy_stats,
            self.cfg,
        )

        # Compute gradient cosines separately (all losses use automatic balancing)
        weighted_losses = {
            "sign_loss": losses_with["sign_loss"],
            "digit_loss": losses_with["digit_loss"],
            "op_loss": losses_with["op_loss"],
            "value_loss": losses_with["value_loss"],
            "exec_loss": losses_with["exec_loss"],
        }
        gradient_cosines = compute_gradient_cosines(
            weighted_losses,
            losses_with["total_loss"],
            model_params,
        )
        losses_with.update(gradient_cosines)

        # Check that basic losses are the same
        for key in [
            "total_loss",
            "sign_loss",
            "digit_loss",
            "op_loss",
            "value_loss",
            "exec_loss",
        ]:
            self.assertTrue(
                torch.allclose(losses_without[key], losses_with[key], atol=1e-6)
            )

        # Check that gradient cosines are present and reasonable
        expected_cosine_keys = [
            "grad_cosine_sign_loss",
            "grad_cosine_digit_loss",
            "grad_cosine_op_loss",
            "grad_cosine_value_loss",
            "grad_cosine_exec_loss",
        ]

        for key in expected_cosine_keys:
            self.assertIn(key, losses_with)
            cosine_value = losses_with[key]
            self.assertIsInstance(cosine_value, float)
            # Cosine values should be between -1 and 1
            self.assertGreaterEqual(
                cosine_value, -1.1
            )  # Small tolerance for numerical errors
            self.assertLessEqual(cosine_value, 1.1)

    def test_gradient_cosine_with_zero_gradient(self):
        """Test gradient cosine computation when total gradient is near zero."""
        B, T = 1, 1
        num_nodes = self.cfg.dag_depth + 1
        depth = self.cfg.dag_depth
        n_ops = N_OPS

        # Create a simple model
        model = SimpleTestModel(n_params=50)

        # Create perfect predictions (should lead to near-zero gradients) connected to model
        target_sgn = torch.ones(B, T, num_nodes)
        dummy_input = torch.randn(B * T * num_nodes, 50)
        # Keep pred_sgn in valid range [-1, 1] by using tanh
        pred_sgn = torch.tanh(
            target_sgn.clone() + 0.01 * model(dummy_input).view(B, T, num_nodes)
        )

        D_total = self.cfg.max_digits + self.cfg.max_decimal_places
        target_digits = torch.zeros(B, T, num_nodes, D_total, 10)
        target_digits[..., 0] = 1.0  # All digits are 0

        # Create logits that strongly predict the correct digits
        pred_digit_logits = torch.full((B, T, num_nodes, D_total, 10), -10.0)
        # Make a copy and modify to avoid in-place operation
        pred_digit_logits = pred_digit_logits.clone()
        pred_digit_logits[..., 0] = 10.0  # Strong prediction for digit 0

        # Add small connection to model parameters
        dummy_input2 = torch.randn(B * T * num_nodes * D_total * 10, 50)
        pred_digit_logits = pred_digit_logits + 0.01 * model(dummy_input2).view(
            B, T, num_nodes, D_total, 10
        )

        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[..., 0] = 1.0
        dummy_input3 = torch.randn(B * T * depth * n_ops, 50)
        pred_ops = target_ops.clone() + 0.01 * model(dummy_input3).view(
            B, T, depth, n_ops
        )

        target_initial_values = torch.ones(B, T, num_nodes)
        target_final_exec = torch.ones(B, T, 1)

        model_params = list(model.parameters())

        dummy_stats = _dummy_statistics(B, T)
        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_digit_logits,
            pred_ops,
            dummy_stats,
            target_sgn,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            dummy_stats,
            self.cfg,
        )

        # Compute gradient cosines separately (all losses use automatic balancing)
        weighted_losses = {
            "sign_loss": losses["sign_loss"],
            "digit_loss": losses["digit_loss"],
            "op_loss": losses["op_loss"],
            "value_loss": losses["value_loss"],
            "exec_loss": losses["exec_loss"],
        }
        gradient_cosines = compute_gradient_cosines(
            weighted_losses,
            losses["total_loss"],
            model_params,
        )
        losses.update(gradient_cosines)

        # Check that gradient cosines are computed (might be 0.0 due to near-zero gradients)
        expected_cosine_keys = [
            "grad_cosine_sign_loss",
            "grad_cosine_digit_loss",
            "grad_cosine_op_loss",
            "grad_cosine_value_loss",
            "grad_cosine_exec_loss",
        ]

        for key in expected_cosine_keys:
            self.assertIn(key, losses)
            self.assertIsInstance(losses[key], float)

    def test_gradient_cosine_disabled(self):
        """Test that gradient cosines are not computed when disabled."""
        B, T = 1, 1
        num_nodes = self.cfg.dag_depth + 1
        depth = self.cfg.dag_depth
        n_ops = N_OPS

        # Simple tensors with gradient computation (don't need to be connected to model)
        pred_sgn = torch.tanh(torch.randn(B, T, num_nodes, requires_grad=True))
        D_total = self.cfg.max_digits + self.cfg.max_decimal_places
        pred_digit_logits = torch.randn(
            B, T, num_nodes, D_total, 10, requires_grad=True
        )
        pred_ops = torch.softmax(
            torch.randn(B, T, depth, n_ops, requires_grad=True), dim=-1
        )

        target_sgn = torch.randn(B, T, num_nodes)
        target_digits = torch.nn.functional.one_hot(
            torch.randint(0, 10, (B, T, num_nodes, D_total)), num_classes=10
        ).float()
        target_ops = torch.zeros(B, T, depth, n_ops)
        target_ops[:, :, :, 0] = 1

        target_initial_values = torch.ones(B, T, num_nodes)
        target_final_exec = torch.ones(B, T, 1)

        dummy_stats = _dummy_statistics(B, T)
        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_digit_logits,
            pred_ops,
            dummy_stats,
            target_sgn,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            dummy_stats,
            self.cfg,
        )

        # Check that gradient cosines are not present
        gradient_cosine_keys = [
            key for key in losses.keys() if key.startswith("grad_cosine_")
        ]
        self.assertEqual(len(gradient_cosine_keys), 0)


if __name__ == "__main__":
    unittest.main()
