"""Integration test for curriculum learning in training context."""

import unittest

import torch

from predictor_config import DAGTrainConfig
from predictor_utils import compute_dag_structure_loss


class TestCurriculumIntegration(unittest.TestCase):
    """Test curriculum learning integration with training."""

    def test_curriculum_in_training_context(self):
        """Test that curriculum learning works in a realistic training context."""
        # Create a config with curriculum parameters
        cfg = DAGTrainConfig()
        cfg.max_digits = 2
        cfg.max_decimal_places = 2
        cfg.base = 10

        # Add curriculum parameters
        cfg.value_curriculum_beta_start = 1.5
        cfg.value_curriculum_beta_end = 0.3
        cfg.value_curriculum_steps = 100
        cfg.sign_penalty_start = 0.05
        cfg.sign_penalty_end = 0.15

        cfg.exec_curriculum_beta_start = 1.5
        cfg.exec_curriculum_beta_end = 0.2
        cfg.exec_curriculum_steps = 100
        cfg.exec_rel_weight_start = 0.01
        cfg.exec_rel_weight_end = 0.04

        cfg.digit_entropy_weight_start = 0.0
        cfg.digit_entropy_weight_end = 0.03
        cfg.digit_entropy_curriculum_steps = 100

        # Create realistic training tensors
        torch.manual_seed(123)
        batch_size, seq_len, num_nodes = 8, 1, 4
        dag_depth, n_ops = 3, 5

        # Predictions (as would come from model)
        pred_sgn = torch.tanh(torch.randn(batch_size, seq_len, num_nodes) * 0.5)
        pred_digit_logits = torch.randn(
            batch_size,
            seq_len,
            num_nodes,
            cfg.max_digits + cfg.max_decimal_places,
            cfg.base,
        )
        pred_ops = torch.softmax(
            torch.randn(batch_size, seq_len, dag_depth, n_ops), dim=-1
        )

        # Targets
        target_sgn = torch.sign(torch.randn(batch_size, seq_len, num_nodes))
        target_digits = torch.nn.functional.one_hot(
            torch.randint(
                0,
                cfg.base,
                (
                    batch_size,
                    seq_len,
                    num_nodes,
                    cfg.max_digits + cfg.max_decimal_places,
                ),
            ),
            cfg.base,
        ).float()
        target_ops = torch.nn.functional.one_hot(
            torch.randint(0, n_ops, (batch_size, seq_len, dag_depth)), n_ops
        ).float()
        target_initial_values = torch.randn(batch_size, seq_len, num_nodes) * 5
        target_final_exec = torch.randn(batch_size, seq_len) * 10

        # Test training progression
        losses_at_different_iters = []

        for iter_num in [0, 25, 50, 75, 100, 150]:
            losses = compute_dag_structure_loss(
                pred_sgn,
                pred_digit_logits,
                pred_ops,
                target_sgn,
                target_digits,
                target_ops,
                target_initial_values,
                target_final_exec,
                cfg,
                iter_num,
            )

            # Verify all losses are finite
            for key, loss in losses.items():
                self.assertTrue(
                    torch.isfinite(loss),
                    f"Loss {key} at iter {iter_num} is not finite: {loss}",
                )

            losses_at_different_iters.append(
                {
                    "iter": iter_num,
                    "total": losses["total_loss"].item(),
                    "value": losses["value_loss"].item(),
                    "exec": losses["exec_loss"].item(),
                    "digit": losses["digit_loss"].item(),
                }
            )

        # Print progression for verification
        print("\nCurriculum Learning Progression:")
        print("Iter | Total  | Value  | Exec   | Digit")
        print("-" * 40)
        for loss_dict in losses_at_different_iters:
            print(
                f"{loss_dict['iter']:4d} | {loss_dict['total']:.4f} | "
                f"{loss_dict['value']:.4f} | {loss_dict['exec']:.4f} | "
                f"{loss_dict['digit']:.4f}"
            )

        # Verify curriculum behavior: later iterations should generally have higher
        # value-based losses (stricter requirements)
        start_losses = losses_at_different_iters[0]
        mid_losses = losses_at_different_iters[2]  # iter 50
        end_losses = losses_at_different_iters[4]  # iter 100
        beyond_losses = losses_at_different_iters[5]  # iter 150

        # Value and exec losses should increase with curriculum progression
        self.assertGreater(
            end_losses["value"],
            start_losses["value"],
            "Value loss should increase with curriculum",
        )
        self.assertGreater(
            end_losses["exec"],
            start_losses["exec"],
            "Exec loss should increase with curriculum",
        )

        # Losses should plateau after curriculum ends
        self.assertAlmostEqual(
            beyond_losses["value"],
            end_losses["value"],
            places=3,
            msg="Value loss should plateau after curriculum",
        )
        self.assertAlmostEqual(
            beyond_losses["exec"],
            end_losses["exec"],
            places=3,
            msg="Exec loss should plateau after curriculum",
        )

    def test_curriculum_with_gradients(self):
        """Test that curriculum learning preserves gradient flow."""
        cfg = DAGTrainConfig()
        cfg.max_digits = 2
        cfg.max_decimal_places = 1
        cfg.base = 10

        # Simple curriculum setup
        cfg.value_curriculum_steps = 50
        cfg.exec_curriculum_steps = 50
        cfg.digit_entropy_curriculum_steps = 50

        # Create tensors requiring gradients
        batch_size, seq_len, num_nodes = (
            4,
            1,
            3,
        )  # Need at least 3 nodes for 2 operations
        dag_depth, n_ops = 2, 5

        # Create leaf tensors with requires_grad
        pred_sgn = torch.randn(batch_size, seq_len, num_nodes, requires_grad=True)
        pred_sgn = torch.tanh(pred_sgn)  # Apply transformation
        pred_sgn.retain_grad()  # Retain gradients for non-leaf tensor

        pred_digit_logits = torch.randn(
            batch_size,
            seq_len,
            num_nodes,
            cfg.max_digits + cfg.max_decimal_places,
            cfg.base,
            requires_grad=True,
        )

        pred_ops_logits = torch.randn(
            batch_size, seq_len, dag_depth, n_ops, requires_grad=True
        )
        pred_ops = torch.softmax(pred_ops_logits, dim=-1)
        pred_ops.retain_grad()  # Retain gradients for non-leaf tensor

        # Targets
        target_sgn = torch.sign(torch.randn(batch_size, seq_len, num_nodes))
        target_digits = torch.nn.functional.one_hot(
            torch.randint(
                0,
                cfg.base,
                (
                    batch_size,
                    seq_len,
                    num_nodes,
                    cfg.max_digits + cfg.max_decimal_places,
                ),
            ),
            cfg.base,
        ).float()
        target_ops = torch.nn.functional.one_hot(
            torch.randint(0, n_ops, (batch_size, seq_len, dag_depth)), n_ops
        ).float()
        target_initial_values = torch.randn(batch_size, seq_len, num_nodes)
        target_final_exec = torch.randn(batch_size, seq_len)

        # Forward pass with curriculum
        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_digit_logits,
            pred_ops,
            target_sgn,
            target_digits,
            target_ops,
            target_initial_values,
            target_final_exec,
            cfg,
            iter_num=25,  # Middle of curriculum
        )

        # Backward pass
        losses["total_loss"].backward()

        # Verify gradients exist and are finite
        self.assertIsNotNone(pred_sgn.grad)
        self.assertIsNotNone(pred_digit_logits.grad)
        self.assertIsNotNone(pred_ops.grad)

        self.assertTrue(torch.isfinite(pred_sgn.grad).all())
        self.assertTrue(torch.isfinite(pred_digit_logits.grad).all())
        self.assertTrue(torch.isfinite(pred_ops.grad).all())

        # Verify gradients are non-zero (learning should happen)
        self.assertTrue(torch.abs(pred_sgn.grad).sum() > 1e-6)
        self.assertTrue(torch.abs(pred_digit_logits.grad).sum() > 1e-6)
        self.assertTrue(torch.abs(pred_ops.grad).sum() > 1e-6)


if __name__ == "__main__":
    unittest.main()
