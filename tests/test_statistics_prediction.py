#!/usr/bin/env python
"""
Tests for auxiliary statistics prediction functionality.
"""

import numpy as np
import pytest
import torch

from data.dagset.streaming import plan_to_tensors
from models.dag_model import (
    MULTI_VALUE_STAT_NAMES,
    SINGLE_VALUE_STAT_NAMES,
    DAGPlanPredictor,
    GPTConfig,
    compute_multi_value_statistics,
    compute_single_value_statistics,
    execute_stack,
)
from predictor_config import DAGTrainConfig
from predictor_utils import _compute_statistics_loss, compute_dag_structure_loss


class TestStatisticsComputation:
    """Test the statistics computation functions."""

    def test_multi_value_statistics_basic(self):
        """Test multi-value statistics computation with basic inputs."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_multi_value_statistics(values)

        assert stats.shape == (15,), f"Expected 15 stats, got {stats.shape}"
        assert len(MULTI_VALUE_STAT_NAMES) == 15, "Stat names count mismatch"

        # Check some basic statistics
        assert abs(stats[0] - 3.0) < 1e-6, f"Mean should be 3.0, got {stats[0]}"  # mean
        assert abs(stats[2] - 1.0) < 1e-6, f"Min should be 1.0, got {stats[2]}"  # min
        assert abs(stats[3] - 5.0) < 1e-6, f"Max should be 5.0, got {stats[3]}"  # max

    def test_multi_value_statistics_edge_cases(self):
        """Test multi-value statistics with edge cases."""
        # Test with negative values
        values = [-2.0, -1.0, 0.0, 1.0, 2.0]
        stats = compute_multi_value_statistics(values)
        assert stats.shape == (15,)
        assert abs(stats[0] - 0.0) < 1e-6, "Mean should be 0.0"

        # Test with all same values
        values = [5.0, 5.0, 5.0]
        stats = compute_multi_value_statistics(values)
        assert abs(stats[1]) < 1e-6, "Std should be 0 for identical values"

    def test_single_value_statistics_basic(self):
        """Test single-value statistics computation."""
        value = 3.14159
        stats = compute_single_value_statistics(value)

        assert stats.shape == (10,), f"Expected 10 stats, got {stats.shape}"
        assert len(SINGLE_VALUE_STAT_NAMES) == 10, "Stat names count mismatch"

        # Check basic values
        assert abs(stats[0] - value) < 1e-6, "Raw value should match"
        assert abs(stats[1] - abs(value)) < 1e-6, "Abs value should match"
        assert stats[9] == 1.0, "Sign should be positive"

    def test_single_value_statistics_negative(self):
        """Test single-value statistics with negative value."""
        value = -2.5
        stats = compute_single_value_statistics(value)

        assert abs(stats[0] - value) < 1e-6, "Raw value should match"
        assert abs(stats[1] - abs(value)) < 1e-6, "Abs value should match"
        assert stats[9] == -1.0, "Sign should be negative"


class TestModelIntegration:
    """Test statistics integration with the model."""

    def test_predictor_statistics_output(self):
        """Test that DAGPlanPredictor outputs statistics correctly."""
        config = GPTConfig(n_embd=64, n_head=2, dag_depth=2)
        predictor = DAGPlanPredictor(config)

        B, T, H = 2, 4, 64
        hidden = torch.randn(B, T, H)

        initial_sgn, digit_probs, operation_probs, statistics = predictor(hidden)

        # Check statistics structure
        assert isinstance(statistics, dict), "Statistics should be a dictionary"
        assert set(statistics.keys()) == {
            "initial",
            "intermediate",
            "final",
        }, f"Unexpected statistics keys: {statistics.keys()}"

        # Check shapes - now per-token (B, T, num_stats)
        assert statistics["initial"].shape == (
            B,
            T,
            15,
        ), f"Initial stats shape mismatch: {statistics['initial'].shape}"
        assert statistics["intermediate"].shape == (
            B,
            T,
            15,
        ), f"Intermediate stats shape mismatch: {statistics['intermediate'].shape}"
        assert statistics["final"].shape == (
            B,
            T,
            10,
        ), f"Final stats shape mismatch: {statistics['final'].shape}"

    def test_predictor_statistics_heads_exist(self):
        """Test that statistics heads are properly initialized."""
        config = GPTConfig(n_embd=64, n_head=2, dag_depth=2)
        predictor = DAGPlanPredictor(config)

        # Check heads exist
        assert hasattr(predictor, "initial_stats_head"), "Missing initial_stats_head"
        assert hasattr(
            predictor, "intermediate_stats_head"
        ), "Missing intermediate_stats_head"
        assert hasattr(predictor, "final_stats_head"), "Missing final_stats_head"

        # Check specialized hidden state exists
        assert hasattr(
            predictor, "stats_structure_hidden"
        ), "Missing stats_structure_hidden"

        # Check stat names
        assert predictor.initial_stat_names == MULTI_VALUE_STAT_NAMES
        assert predictor.intermediate_stat_names == MULTI_VALUE_STAT_NAMES
        assert predictor.final_stat_names == SINGLE_VALUE_STAT_NAMES


class TestExecuteStackIntermediates:
    """Test execute_stack intermediate value tracking."""

    def test_execute_stack_without_intermediates(self):
        """Test execute_stack normal operation without intermediates."""
        from data.dagset.streaming import float_to_digit_onehot

        # Simple test case: 2 + 3 = 5
        initial_values = [2.0, 3.0]
        operations = ["add"]

        # Create tensors
        signs = torch.tensor([1.0, 1.0]).view(1, 1, -1)
        digits = (
            torch.stack(
                [
                    float_to_digit_onehot(2.0, 4, 6, 10),
                    float_to_digit_onehot(3.0, 4, 6, 10),
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        ops = torch.zeros(1, 1, 1, 5)
        ops[0, 0, 0, 0] = 1.0  # add operation

        final_sgn, final_log = execute_stack(
            signs, digits, ops, max_digits=4, max_decimal_places=6, base=10
        )

        result = final_sgn.item() * torch.exp(final_log).item()
        assert abs(result - 5.0) < 0.1, f"Expected ~5.0, got {result}"

    def test_execute_stack_with_intermediates(self):
        """Test execute_stack with intermediate value tracking."""
        from data.dagset.streaming import float_to_digit_onehot

        # Test case: 2 + 3 = 5, then 5 * 1 = 5
        initial_values = [2.0, 3.0, 1.0]
        operations = ["add", "multiply"]

        # Create tensors
        signs = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, -1)
        digits = (
            torch.stack(
                [
                    float_to_digit_onehot(2.0, 4, 6, 10),
                    float_to_digit_onehot(3.0, 4, 6, 10),
                    float_to_digit_onehot(1.0, 4, 6, 10),
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        ops = torch.zeros(1, 1, 2, 5)
        ops[0, 0, 0, 0] = 1.0  # add operation (first applied: identity)
        ops[0, 0, 1, 2] = 1.0  # multiply operation (then: add)

        final_sgn, final_log, intermediates = execute_stack(
            signs,
            digits,
            ops,
            max_digits=4,
            max_decimal_places=6,
            base=10,
            return_intermediates=True,
        )

        # Check final result
        result = final_sgn.item() * torch.exp(final_log).item()
        assert abs(result - 5.0) < 1e-1, f"Expected ~5.0, got {result}"

        # Check intermediates
        assert (
            len(intermediates) == 2
        ), f"Expected 2 intermediates, got {len(intermediates)}"

    def test_execute_stack_intermediates_batch_constraint(self):
        """Test that execute_stack raises error for return_intermediates=True with B>1 or T>1."""
        from data.dagset.streaming import float_to_digit_onehot

        # Test with B > 1 (should fail)
        signs = torch.tensor([[1.0, 1.0], [1.0, 1.0]]).view(2, 1, -1)  # B=2, T=1
        digits = (
            torch.stack(
                [
                    float_to_digit_onehot(2.0, 4, 6, 10),
                    float_to_digit_onehot(3.0, 4, 6, 10),
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(2, -1, -1, -1, -1)
        )  # Expand to B=2
        ops = torch.zeros(2, 1, 1, 5)
        ops[:, 0, 0, 0] = 1.0  # add operation

        with pytest.raises(
            ValueError, match="return_intermediates=True only supports batch_size=1"
        ):
            execute_stack(
                signs,
                digits,
                ops,
                max_digits=4,
                max_decimal_places=6,
                base=10,
                return_intermediates=True,
            )

        # Test with T > 1 (should fail)
        signs = torch.tensor([1.0, 1.0]).view(1, 1, -1).expand(1, 2, -1)  # B=1, T=2
        digits = (
            torch.stack(
                [
                    float_to_digit_onehot(2.0, 4, 6, 10),
                    float_to_digit_onehot(3.0, 4, 6, 10),
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, 2, -1, -1, -1)
        )  # Expand to T=2
        ops = torch.zeros(1, 2, 1, 5)
        ops[0, :, 0, 0] = 1.0  # add operation

        with pytest.raises(
            ValueError,
            match="return_intermediates=True only supports.*sequence_length=1",
        ):
            execute_stack(
                signs,
                digits,
                ops,
                max_digits=4,
                max_decimal_places=6,
                base=10,
                return_intermediates=True,
            )


class TestStreamingIntegration:
    """Test statistics integration with streaming dataset."""

    def test_plan_to_tensors_includes_statistics(self):
        """Test that plan_to_tensors includes statistics targets."""
        initial_values = [2.5, 1.5, 3.0]
        operations = ["add", "subtract"]

        structure_dict = plan_to_tensors(
            initial_values=initial_values,
            operations=operations,
            max_digits=4,
            max_decimal_places=6,
            base=10,
        )

        # Check statistics are included
        stats_keys = [
            "target_initial_stats",
            "target_intermediate_stats",
            "target_final_stats",
        ]
        for key in stats_keys:
            assert key in structure_dict, f"Missing {key} in structure_dict"
            assert isinstance(
                structure_dict[key], torch.Tensor
            ), f"{key} should be a tensor"

        # Check shapes
        assert structure_dict["target_initial_stats"].shape == (
            15,
        ), f"Initial stats shape: {structure_dict['target_initial_stats'].shape}"
        assert structure_dict["target_intermediate_stats"].shape == (
            15,
        ), f"Intermediate stats shape: {structure_dict['target_intermediate_stats'].shape}"
        assert structure_dict["target_final_stats"].shape == (
            10,
        ), f"Final stats shape: {structure_dict['target_final_stats'].shape}"


class TestLossComputation:
    """Test statistics loss computation."""

    def test_statistics_loss_function(self):
        """Test _compute_statistics_loss function."""
        B, T = 2, 4

        # Create dummy predictions and targets (per-token format)
        pred_statistics = {
            "initial": torch.randn(B, T, 15),
            "intermediate": torch.randn(B, T, 15),
            "final": torch.randn(B, T, 10),
        }

        target_statistics = {
            "initial": torch.randn(B, T, 15),
            "intermediate": torch.randn(B, T, 15),
            "final": torch.randn(B, T, 10),
        }

        loss = _compute_statistics_loss(pred_statistics, target_statistics, "cpu")

        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.numel() == 1, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_statistics_loss_gradient_flow(self):
        """Test that gradients flow through statistics loss."""
        B, T = 1, 3

        pred_statistics = {
            "initial": torch.randn(B, T, 15, requires_grad=True),
            "intermediate": torch.randn(B, T, 15, requires_grad=True),
            "final": torch.randn(B, T, 10, requires_grad=True),
        }

        target_statistics = {
            "initial": torch.randn(B, T, 15),
            "intermediate": torch.randn(B, T, 15),
            "final": torch.randn(B, T, 10),
        }

        loss = _compute_statistics_loss(pred_statistics, target_statistics, "cpu")
        loss.backward()

        # Check gradients exist
        for key in pred_statistics:
            assert pred_statistics[key].grad is not None, f"No gradient for {key}"
            assert pred_statistics[key].grad.shape == pred_statistics[key].shape


if __name__ == "__main__":
    pytest.main([__file__])
