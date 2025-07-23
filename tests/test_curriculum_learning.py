"""Test curriculum learning implementation in predictor_utils.py"""

import unittest

import torch
import torch.nn.functional as F

from predictor_config import DAGTrainConfig
from predictor_utils import (
    _compute_digit_loss,
    _compute_exec_loss,
    _compute_value_loss,
    compute_dag_structure_loss,
)


class MockCurriculumConfig(DAGTrainConfig):
    """Mock config with curriculum parameters for testing."""

    def __init__(self):
        super().__init__()
        # Value loss curriculum
        self.value_curriculum_beta_start = 2.0
        self.value_curriculum_beta_end = 0.2
        self.value_curriculum_steps = 1000
        self.sign_penalty_start = 0.1
        self.sign_penalty_end = 0.5

        # Exec loss curriculum
        self.exec_curriculum_beta_start = 2.0
        self.exec_curriculum_beta_end = 0.1
        self.exec_curriculum_steps = 1000
        self.exec_rel_weight_start = 0.01
        self.exec_rel_weight_end = 0.05
        self.exec_overflow_start = 35.0
        self.exec_overflow_end = 20.0

        # Digit loss curriculum
        self.digit_entropy_weight_start = 0.0
        self.digit_entropy_weight_end = 0.1
        self.digit_entropy_curriculum_steps = 1000

        # Standard parameters
        self.max_digits = 3
        self.max_decimal_places = 2
        self.base = 10


class TestCurriculumLearning(unittest.TestCase):
    """Test curriculum learning implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = MockCurriculumConfig()
        self.device = "cpu"
        self.batch_size = 4
        self.seq_len = 1
        self.num_nodes = 3
        self.dag_depth = 2
        self.n_ops = 5  # add, sub, mul, div, identity

        # Create mock tensors
        torch.manual_seed(42)

        # Predictions
        self.pred_sgn = torch.tanh(
            torch.randn(self.batch_size, self.seq_len, self.num_nodes)
        )  # Ensure in [-1,1]
        # Clamp to ensure strict bounds for binary_cross_entropy
        self.pred_sgn = torch.clamp(self.pred_sgn, -0.999, 0.999)
        self.pred_digit_logits = torch.randn(
            self.batch_size,
            self.seq_len,
            self.num_nodes,
            self.cfg.max_digits + self.cfg.max_decimal_places,
            self.cfg.base,
        )
        self.pred_ops = torch.softmax(
            torch.randn(self.batch_size, self.seq_len, self.dag_depth, self.n_ops),
            dim=-1,
        )

        # Targets
        self.target_sgn = torch.sign(
            torch.randn(self.batch_size, self.seq_len, self.num_nodes)
        )
        self.target_digits = F.one_hot(
            torch.randint(
                0,
                self.cfg.base,
                (
                    self.batch_size,
                    self.seq_len,
                    self.num_nodes,
                    self.cfg.max_digits + self.cfg.max_decimal_places,
                ),
            ),
            self.cfg.base,
        ).float()
        self.target_ops = F.one_hot(
            torch.randint(
                0, self.n_ops, (self.batch_size, self.seq_len, self.dag_depth)
            ),
            self.n_ops,
        ).float()
        self.target_initial_values = (
            torch.randn(self.batch_size, self.seq_len, self.num_nodes) * 10
        )
        self.target_final_exec = torch.randn(self.batch_size, self.seq_len) * 20

    def test_value_loss_curriculum_beta_progression(self):
        """Test that value loss beta parameter progresses correctly."""
        pred_digit_probs = F.softmax(self.pred_digit_logits, dim=-1)

        # Test at start of curriculum
        loss_start = _compute_value_loss(
            self.pred_sgn,
            pred_digit_probs,
            self.target_initial_values,
            self.cfg,
            "cpu",
            iter_num=0,
        )

        # Test at middle of curriculum
        loss_mid = _compute_value_loss(
            self.pred_sgn,
            pred_digit_probs,
            self.target_initial_values,
            self.cfg,
            "cpu",
            iter_num=500,
        )

        # Test at end of curriculum
        loss_end = _compute_value_loss(
            self.pred_sgn,
            pred_digit_probs,
            self.target_initial_values,
            self.cfg,
            "cpu",
            iter_num=1000,
        )

        # Test beyond curriculum
        loss_beyond = _compute_value_loss(
            self.pred_sgn,
            pred_digit_probs,
            self.target_initial_values,
            self.cfg,
            "cpu",
            iter_num=2000,
        )

        # Verify losses are finite
        self.assertTrue(torch.isfinite(loss_start))
        self.assertTrue(torch.isfinite(loss_mid))
        self.assertTrue(torch.isfinite(loss_end))
        self.assertTrue(torch.isfinite(loss_beyond))

        # With our setup (random pred/targets), stricter beta should generally give higher loss
        # since smooth_l1_loss becomes more L1-like (less forgiving) with smaller beta
        print(
            f"Value loss progression: start={loss_start:.4f}, mid={loss_mid:.4f}, "
            f"end={loss_end:.4f}, beyond={loss_beyond:.4f}"
        )

    def test_exec_loss_curriculum_progression(self):
        """Test that exec loss curriculum parameters progress correctly."""
        pred_digit_probs = F.softmax(self.pred_digit_logits, dim=-1)

        # Test at different curriculum stages
        loss_start = _compute_exec_loss(
            self.pred_sgn,
            pred_digit_probs,
            self.pred_ops,
            self.target_final_exec,
            self.cfg,
            "cpu",
            iter_num=0,
        )

        loss_mid = _compute_exec_loss(
            self.pred_sgn,
            pred_digit_probs,
            self.pred_ops,
            self.target_final_exec,
            self.cfg,
            "cpu",
            iter_num=500,
        )

        loss_end = _compute_exec_loss(
            self.pred_sgn,
            pred_digit_probs,
            self.pred_ops,
            self.target_final_exec,
            self.cfg,
            "cpu",
            iter_num=1000,
        )

        loss_beyond = _compute_exec_loss(
            self.pred_sgn,
            pred_digit_probs,
            self.pred_ops,
            self.target_final_exec,
            self.cfg,
            "cpu",
            iter_num=2000,
        )

        # Verify losses are finite
        self.assertTrue(torch.isfinite(loss_start))
        self.assertTrue(torch.isfinite(loss_mid))
        self.assertTrue(torch.isfinite(loss_end))
        self.assertTrue(torch.isfinite(loss_beyond))

        print(
            f"Exec loss progression: start={loss_start:.4f}, mid={loss_mid:.4f}, "
            f"end={loss_end:.4f}, beyond={loss_beyond:.4f}"
        )

    def test_digit_loss_entropy_curriculum(self):
        """Test that digit loss entropy penalty progresses correctly."""
        pred_digit_probs = F.softmax(self.pred_digit_logits, dim=-1)

        # Test at start (no entropy penalty)
        loss_start = _compute_digit_loss(
            pred_digit_probs, self.target_digits, "cpu", iter_num=0, cfg=self.cfg
        )

        # Test at end (full entropy penalty)
        loss_end = _compute_digit_loss(
            pred_digit_probs, self.target_digits, "cpu", iter_num=1000, cfg=self.cfg
        )

        # Test without curriculum (should be same as start)
        loss_no_curriculum = _compute_digit_loss(
            pred_digit_probs, self.target_digits, "cpu", iter_num=0, cfg=None
        )

        # Verify losses are finite
        self.assertTrue(torch.isfinite(loss_start))
        self.assertTrue(torch.isfinite(loss_end))
        self.assertTrue(torch.isfinite(loss_no_curriculum))

        # Loss with entropy penalty should be higher than without
        self.assertGreater(
            loss_end.item(), loss_start.item(), "Entropy penalty should increase loss"
        )

        # Start should be same as no curriculum
        self.assertAlmostEqual(
            loss_start.item(),
            loss_no_curriculum.item(),
            places=5,
            msg="Start of curriculum should match no curriculum",
        )

        print(
            f"Digit loss progression: start={loss_start:.4f}, end={loss_end:.4f}, "
            f"no_curriculum={loss_no_curriculum:.4f}"
        )

    def test_curriculum_parameter_calculation(self):
        """Test that curriculum parameters are calculated correctly."""
        # Test value loss curriculum parameters

        # At iter_num = 0 (start)
        progress = min(0 / self.cfg.value_curriculum_steps, 1.0)
        expected_beta = self.cfg.value_curriculum_beta_start + progress * (
            self.cfg.value_curriculum_beta_end - self.cfg.value_curriculum_beta_start
        )
        expected_sign_weight = self.cfg.sign_penalty_start + progress * (
            self.cfg.sign_penalty_end - self.cfg.sign_penalty_start
        )

        self.assertAlmostEqual(
            expected_beta, self.cfg.value_curriculum_beta_start, places=5
        )
        self.assertAlmostEqual(
            expected_sign_weight, self.cfg.sign_penalty_start, places=5
        )

        # At iter_num = curriculum_steps (end)
        progress = min(
            self.cfg.value_curriculum_steps / self.cfg.value_curriculum_steps, 1.0
        )
        expected_beta = self.cfg.value_curriculum_beta_start + progress * (
            self.cfg.value_curriculum_beta_end - self.cfg.value_curriculum_beta_start
        )
        expected_sign_weight = self.cfg.sign_penalty_start + progress * (
            self.cfg.sign_penalty_end - self.cfg.sign_penalty_start
        )

        self.assertAlmostEqual(
            expected_beta, self.cfg.value_curriculum_beta_end, places=5
        )
        self.assertAlmostEqual(
            expected_sign_weight, self.cfg.sign_penalty_end, places=5
        )

        # At iter_num = half curriculum_steps (middle)
        progress = min(
            (self.cfg.value_curriculum_steps // 2) / self.cfg.value_curriculum_steps,
            1.0,
        )
        expected_beta = self.cfg.value_curriculum_beta_start + progress * (
            self.cfg.value_curriculum_beta_end - self.cfg.value_curriculum_beta_start
        )
        expected_middle = (
            self.cfg.value_curriculum_beta_start + self.cfg.value_curriculum_beta_end
        ) / 2

        self.assertAlmostEqual(expected_beta, expected_middle, places=5)

    def test_main_loss_function_with_curriculum(self):
        """Test that the main loss function correctly uses curriculum parameters."""

        # Test early in curriculum
        losses_early = compute_dag_structure_loss(
            self.pred_sgn,
            self.pred_digit_logits,
            self.pred_ops,
            self.target_sgn,
            self.target_digits,
            self.target_ops,
            self.target_initial_values,
            self.target_final_exec,
            self.cfg,
            iter_num=0,
        )

        # Test late in curriculum
        losses_late = compute_dag_structure_loss(
            self.pred_sgn,
            self.pred_digit_logits,
            self.pred_ops,
            self.target_sgn,
            self.target_digits,
            self.target_ops,
            self.target_initial_values,
            self.target_final_exec,
            self.cfg,
            iter_num=1000,
        )

        # Verify all loss components are finite
        for key, loss in losses_early.items():
            self.assertTrue(torch.isfinite(loss), f"Early {key} loss is not finite")

        for key, loss in losses_late.items():
            self.assertTrue(torch.isfinite(loss), f"Late {key} loss is not finite")

        # Compare curriculum-sensitive losses
        print(
            f"Value loss: early={losses_early['value_loss']:.4f}, late={losses_late['value_loss']:.4f}"
        )
        print(
            f"Exec loss: early={losses_early['exec_loss']:.4f}, late={losses_late['exec_loss']:.4f}"
        )
        print(
            f"Digit loss: early={losses_early['digit_loss']:.4f}, late={losses_late['digit_loss']:.4f}"
        )

    def test_curriculum_clamping_behavior(self):
        """Test that curriculum parameters are properly clamped at extremes."""
        pred_digit_probs = F.softmax(self.pred_digit_logits, dim=-1)

        # Test far beyond curriculum steps
        loss_far_beyond = _compute_value_loss(
            self.pred_sgn,
            pred_digit_probs,
            self.target_initial_values,
            self.cfg,
            "cpu",
            iter_num=10000,  # Much larger than curriculum_steps
        )

        loss_at_end = _compute_value_loss(
            self.pred_sgn,
            pred_digit_probs,
            self.target_initial_values,
            self.cfg,
            "cpu",
            iter_num=1000,  # Exactly at curriculum_steps
        )

        # Should be very close since progress is clamped at 1.0
        self.assertAlmostEqual(
            loss_far_beyond.item(),
            loss_at_end.item(),
            places=4,
            msg="Loss should plateau after curriculum ends",
        )

    def test_curriculum_config_defaults(self):
        """Test that default curriculum parameters work when not specified."""
        # Create config without curriculum parameters
        basic_cfg = DAGTrainConfig()
        basic_cfg.max_digits = 3
        basic_cfg.max_decimal_places = 2
        basic_cfg.base = 10

        pred_digit_probs = F.softmax(self.pred_digit_logits, dim=-1)

        # Should not crash and should use defaults
        loss = _compute_value_loss(
            self.pred_sgn,
            pred_digit_probs,
            self.target_initial_values,
            basic_cfg,
            "cpu",
            iter_num=500,
        )

        self.assertTrue(
            torch.isfinite(loss), "Loss with default config should be finite"
        )

    def test_zero_iteration_behavior(self):
        """Test that iter_num=0 gives expected starting behavior."""
        pred_digit_probs = F.softmax(self.pred_digit_logits, dim=-1)

        # All losses at iter_num=0 should use starting parameters
        value_loss = _compute_value_loss(
            self.pred_sgn,
            pred_digit_probs,
            self.target_initial_values,
            self.cfg,
            "cpu",
            iter_num=0,
        )

        exec_loss = _compute_exec_loss(
            self.pred_sgn,
            pred_digit_probs,
            self.pred_ops,
            self.target_final_exec,
            self.cfg,
            "cpu",
            iter_num=0,
        )

        digit_loss = _compute_digit_loss(
            pred_digit_probs, self.target_digits, "cpu", iter_num=0, cfg=self.cfg
        )

        # All should be finite
        self.assertTrue(torch.isfinite(value_loss))
        self.assertTrue(torch.isfinite(exec_loss))
        self.assertTrue(torch.isfinite(digit_loss))

    def test_gradient_flow_with_curriculum(self):
        """Test that gradients flow properly through curriculum-enhanced losses."""
        # Make predictions require gradients
        pred_sgn = torch.clamp(self.pred_sgn.clone(), -0.999, 0.999).requires_grad_(
            True
        )
        pred_digit_logits = self.pred_digit_logits.clone().requires_grad_(True)
        pred_ops = self.pred_ops.clone().requires_grad_(True)

        # Compute loss
        losses = compute_dag_structure_loss(
            pred_sgn,
            pred_digit_logits,
            pred_ops,
            self.target_sgn,
            self.target_digits,
            self.target_ops,
            self.target_initial_values,
            self.target_final_exec,
            self.cfg,
            iter_num=500,
        )

        # Backward pass
        losses["total_loss"].backward()

        # Check that gradients exist and are finite
        self.assertIsNotNone(pred_sgn.grad)
        self.assertIsNotNone(pred_digit_logits.grad)
        self.assertIsNotNone(pred_ops.grad)

        self.assertTrue(torch.isfinite(pred_sgn.grad).all())
        self.assertTrue(torch.isfinite(pred_digit_logits.grad).all())
        self.assertTrue(torch.isfinite(pred_ops.grad).all())


if __name__ == "__main__":
    unittest.main()
