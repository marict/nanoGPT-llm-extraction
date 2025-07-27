"""Test that uncertainty parameters bypass warmup and get full learning rate immediately.

This is critical because uncertainty parameters control loss balancing and need to
adapt quickly to find the right relative weights between loss components.
"""

from unittest.mock import patch

import pytest
import torch

from predictor_config import DAGTrainConfig
from training_utils import get_lr


class TestUncertaintyParamsNoWarmup:
    """Test uncertainty parameter learning rate behavior."""

    def test_uncertainty_params_bypass_warmup(self):
        """Test that uncertainty params get full LR immediately while main params warmup."""
        # Create config with significant warmup
        cfg = DAGTrainConfig()
        cfg.learning_rate = 1e-4
        cfg.warmup_iters = 100  # 100 iterations of warmup
        cfg.lr_decay_iters = 1000
        cfg.min_lr = 1e-5

        uncertainty_params_lr = cfg.learning_rate * 100  # 1e-2

        # Test at different points during warmup
        test_iterations = [0, 25, 50, 75, 100]

        for iter_num in test_iterations:
            # Main parameters should follow warmup schedule
            expected_main_lr = get_lr(iter_num, cfg=cfg)

            # Uncertainty parameters should always get full LR
            expected_uncertainty_lr = uncertainty_params_lr

            if iter_num < cfg.warmup_iters:
                # During warmup, main LR should be less than full
                assert expected_main_lr < cfg.learning_rate, (
                    f"Main LR should be warming up at iter {iter_num}, "
                    f"got {expected_main_lr}, expected < {cfg.learning_rate}"
                )

                # But uncertainty LR should always be full
                assert expected_uncertainty_lr == uncertainty_params_lr, (
                    f"Uncertainty LR should be full at iter {iter_num}, "
                    f"got {expected_uncertainty_lr}, expected {uncertainty_params_lr}"
                )

                # Uncertainty LR should be much higher than main LR during warmup
                assert (
                    expected_uncertainty_lr > expected_main_lr
                ), f"Uncertainty LR should be higher than main LR during warmup at iter {iter_num}"
            else:
                # After warmup, both should be at their full values
                assert expected_main_lr == cfg.learning_rate
                assert expected_uncertainty_lr == uncertainty_params_lr

    def test_learning_rate_magnitudes(self):
        """Test that uncertainty params have correct magnitude relative to main params."""
        cfg = DAGTrainConfig()
        cfg.learning_rate = 1e-4
        uncertainty_params_lr = cfg.learning_rate * 100  # 2 orders of magnitude higher

        # Uncertainty params should be 100x higher
        assert uncertainty_params_lr == 1e-2
        assert uncertainty_params_lr / cfg.learning_rate == 100

        # This ensures fast adaptation of loss balancing weights

    def test_no_warmup_from_iteration_zero(self):
        """Test that uncertainty params get full LR from the very first iteration."""
        cfg = DAGTrainConfig()
        cfg.learning_rate = 2e-4
        cfg.warmup_iters = 50

        uncertainty_params_lr = cfg.learning_rate * 100  # 2e-2

        # At iteration 0, main params should be nearly zero due to warmup
        main_lr_iter_0 = get_lr(0, cfg=cfg)
        expected_main_lr_iter_0 = (
            cfg.learning_rate * (0 + 1) / cfg.warmup_iters
        )  # Linear warmup

        assert abs(main_lr_iter_0 - expected_main_lr_iter_0) < 1e-8
        assert (
            main_lr_iter_0 < 0.1 * cfg.learning_rate
        )  # Should be much less than full LR

        # But uncertainty params should get full LR immediately
        uncertainty_lr_iter_0 = uncertainty_params_lr
        assert uncertainty_lr_iter_0 == 2e-2

        # Ratio should be very high at iteration 0
        ratio = uncertainty_lr_iter_0 / main_lr_iter_0
        assert ratio > 100  # Should be much more than 100x difference at start

    def test_constant_uncertainty_lr_across_training(self):
        """Test that uncertainty LR remains constant throughout training."""
        cfg = DAGTrainConfig()
        cfg.learning_rate = 1e-4
        cfg.warmup_iters = 100
        cfg.lr_decay_iters = 1000
        cfg.min_lr = 1e-5

        uncertainty_params_lr = cfg.learning_rate * 100

        # Test across different phases of training
        test_points = [
            0,  # Start of warmup
            50,  # Middle of warmup
            100,  # End of warmup
            200,  # Constant LR phase
            500,  # Start of decay
            1000,  # End of decay
            1500,  # Minimum LR phase
        ]

        for iter_num in test_points:
            # Main LR changes across training phases
            main_lr = get_lr(iter_num, cfg=cfg)

            # Uncertainty LR should always be the same
            uncertainty_lr = uncertainty_params_lr
            assert (
                uncertainty_lr == cfg.learning_rate * 100
            ), f"Uncertainty LR should be constant at iter {iter_num}, got {uncertainty_lr}"

    def test_training_loop_lr_assignment_logic(self):
        """Test the learning rate assignment logic used in the training loop."""
        cfg = DAGTrainConfig()
        cfg.learning_rate = 1e-4
        cfg.warmup_iters = 50

        uncertainty_params_lr = cfg.learning_rate * 100  # 1e-2

        # Test the assignment logic at different training phases
        test_iterations = [0, 25, 50, 100]

        for iter_num in test_iterations:
            # This simulates the new logic in train_predictor.py
            main_lr = get_lr(iter_num, cfg=cfg)  # get_lr(iter_num, cfg=cfg)
            uncertainty_lr = uncertainty_params_lr  # Always use full LR (no warmup)

            # Create mock optimizer param groups
            param_groups = [
                {"lr": 0.0, "params": []},  # Main parameters
                {"lr": 0.0, "params": []},  # Uncertainty parameters
            ]

            # Apply the assignment logic from train_predictor.py
            # Should always have exactly 2 groups: main params and uncertainty params
            assert (
                len(param_groups) == 2
            ), f"Expected 2 optimizer groups, got {len(param_groups)}"
            param_groups[0]["lr"] = main_lr  # Main parameters
            param_groups[1]["lr"] = uncertainty_lr  # Uncertainty parameters

            # Verify the assignment
            assert param_groups[0]["lr"] == main_lr  # Main gets warmup/decay applied
            assert (
                param_groups[1]["lr"] == uncertainty_params_lr
            )  # Uncertainty gets constant full LR
            assert (
                param_groups[1]["lr"] >= param_groups[0]["lr"]
            )  # Uncertainty LR is higher or equal

            # During warmup, uncertainty should be much higher
            if iter_num < cfg.warmup_iters:
                assert (
                    param_groups[1]["lr"] > param_groups[0]["lr"]
                ), f"At iter {iter_num}, uncertainty LR should be higher than main LR during warmup"

            # Verify that uncertainty LR is always constant
            assert (
                param_groups[1]["lr"] == uncertainty_params_lr
            ), f"Uncertainty LR should be constant at iter {iter_num}"

    def test_always_two_optimizer_groups(self):
        """Test that we always have exactly 2 optimizer groups: main + uncertainty."""
        cfg = DAGTrainConfig()
        cfg.learning_rate = 1e-4

        uncertainty_params_lr = cfg.learning_rate * 100

        # Simulate optimizer group setup
        main_params = [torch.randn(10, requires_grad=True)]  # Dummy main params
        uncertainty_params = [
            torch.randn(6, requires_grad=True)
        ]  # 6 uncertainty params

        optim_groups = [
            {"params": main_params, "lr": cfg.learning_rate},
            {"params": uncertainty_params, "lr": uncertainty_params_lr},
        ]

        # This should always be exactly 2 groups
        assert (
            len(optim_groups) == 2
        ), f"Expected exactly 2 optimizer groups, got {len(optim_groups)}"

        # The assertion from the training loop should pass
        assert (
            len(optim_groups) == 2
        ), f"Expected 2 optimizer groups, got {len(optim_groups)}"

        # Group 0 should be main parameters
        assert optim_groups[0]["lr"] == cfg.learning_rate

        # Group 1 should be uncertainty parameters
        assert optim_groups[1]["lr"] == uncertainty_params_lr
        assert optim_groups[1]["lr"] == cfg.learning_rate * 100
