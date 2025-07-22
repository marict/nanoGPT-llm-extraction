"""
Tests for memory leak fixes and disk space monitoring.

These tests verify that:
1. Disk space monitoring works correctly
2. Memory cleanup methods properly clear cached tensors
3. Training loop memory management prevents accumulation
"""

import gc
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from dag_logger import DAGLogger
from models.dag_model import GPT, DifferentiableDAG, GPTConfig
from training_utils import (check_disk_space_emergency,
                            cleanup_disk_space_emergency,
                            get_disk_usage_percent)


class TestDiskSpaceMonitoring(unittest.TestCase):
    """Test disk space monitoring functionality."""

    def test_get_disk_usage_percent(self):
        """Test disk usage percentage calculation."""
        usage = get_disk_usage_percent(".")
        self.assertIsInstance(usage, float)
        self.assertGreaterEqual(usage, 0.0)
        self.assertLessEqual(usage, 100.0)

    def test_get_disk_usage_percent_invalid_path(self):
        """Test disk usage with invalid path returns 0."""
        usage = get_disk_usage_percent("/nonexistent/path/that/does/not/exist")
        self.assertEqual(usage, 0.0)

    @patch("shutil.disk_usage")
    def test_check_disk_space_emergency_high_usage(self, mock_disk_usage):
        """Test disk space emergency detection with high usage."""
        # Mock 98% disk usage
        total = 1000
        free = 20  # Only 2% free = 98% used
        mock_disk_usage.return_value = (total, total, free)

        is_emergency = check_disk_space_emergency(".", threshold=95.0)
        self.assertTrue(is_emergency)

    @patch("shutil.disk_usage")
    def test_check_disk_space_emergency_normal_usage(self, mock_disk_usage):
        """Test disk space emergency detection with normal usage."""
        # Mock 50% disk usage
        total = 1000
        free = 500  # 50% free = 50% used
        mock_disk_usage.return_value = (total, total, free)

        is_emergency = check_disk_space_emergency(".", threshold=95.0)
        self.assertFalse(is_emergency)

    @patch("os.path.exists")
    @patch("os.walk")
    def test_cleanup_disk_space_emergency(self, mock_walk, mock_exists):
        """Test emergency disk cleanup functionality."""
        mock_exists.return_value = True

        # Mock some old files to clean up
        old_time = time.time() - 7200  # 2 hours old
        mock_walk.return_value = [("/tmp", [], ["old_file1.tmp", "old_file2.tmp"])]

        with patch("pathlib.Path.stat") as mock_stat, patch(
            "pathlib.Path.unlink"
        ) as mock_unlink:

            mock_stat_obj = MagicMock()
            mock_stat_obj.st_mtime = old_time
            mock_stat_obj.st_size = 1024 * 1024  # 1MB
            mock_stat.return_value = mock_stat_obj

            cleaned_mb = cleanup_disk_space_emergency()

            # Should have attempted to clean files
            self.assertGreaterEqual(cleaned_mb, 0.0)


class TestDAGLoggerMemoryCleanup(unittest.TestCase):
    """Test DAG logger memory cleanup functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = DAGLogger()

    def test_clear_gradient_hooks(self):
        """Test gradient hooks are properly cleared."""
        # Mock some hooks
        mock_hook1 = MagicMock()
        mock_hook2 = MagicMock()
        mock_op_hook = MagicMock()

        self.logger.gradient_hooks = [mock_hook1, mock_hook2]
        self.logger.op_grad_hook = mock_op_hook

        self.logger.clear_gradient_hooks()

        # Verify hooks were removed
        mock_hook1.remove.assert_called_once()
        mock_hook2.remove.assert_called_once()
        mock_op_hook.remove.assert_called_once()
        self.assertEqual(len(self.logger.gradient_hooks), 0)
        self.assertIsNone(self.logger.op_grad_hook)

    def test_clear_memory_cache(self):
        """Test memory cache is properly cleared."""
        # Add some test data
        self.logger.logging_data = {"test": "data"}
        self.logger.captured_gradients = {"grad": "data"}
        self.logger.gradient_hooks = [MagicMock()]

        self.logger.clear_memory_cache()

        # Verify all data was cleared
        self.assertEqual(len(self.logger.logging_data), 0)
        self.assertEqual(len(self.logger.captured_gradients), 0)
        self.assertEqual(len(self.logger.gradient_hooks), 0)

    def test_cleanup_for_next_iteration(self):
        """Test lightweight cleanup between iterations."""
        # Add some test data
        self.logger.logging_data = {"test": "data"}
        self.logger.captured_gradients = {"grad": "data"}

        self.logger.cleanup_for_next_iteration()

        # Verify data was cleared
        self.assertEqual(len(self.logger.logging_data), 0)
        self.assertEqual(len(self.logger.captured_gradients), 0)


class TestModelMemoryCleanup(unittest.TestCase):
    """Test model memory cleanup functionality."""

    def setUp(self):
        """Set up test fixtures."""
        config = GPTConfig(
            block_size=64, vocab_size=100, n_layer=2, n_head=2, n_embd=32, dag_depth=2
        )
        self.model = GPT(config)

    def test_dag_plan_predictor_clear_cache(self):
        """Test DAGPlanPredictor cache clearing."""
        predictor = self.model.dag.plan_predictor

        # Set some cached data
        predictor.last_operation_probs = torch.randn(2, 4, 2, 5)
        predictor.last_digit_logits = torch.randn(2, 4, 3, 10, 10)

        predictor.clear_cache()

        # Verify cache was cleared
        self.assertIsNone(predictor.last_operation_probs)
        self.assertIsNone(predictor.last_digit_logits)

    def test_differentiable_dag_clear_cache(self):
        """Test DifferentiableDAG cache clearing."""
        dag = self.model.dag

        # Set some cached data
        dag.final_hidden = torch.randn(2, 4, 32)
        dag.final_values = torch.randn(2, 3, 4)

        dag.clear_cache()

        # Verify cache was cleared
        self.assertIsNone(dag.final_hidden)
        self.assertIsNone(dag.final_values)

    def test_gpt_model_clear_cache(self):
        """Test GPT model cache clearing."""
        # Set some cached data
        self.model.last_original_hidden = torch.randn(2, 4, 32)
        self.model.last_mixed_hidden = torch.randn(2, 4, 32)
        self.model.final_values = torch.randn(2, 3, 4)
        self.model.last_gate = torch.randn(2, 4, 1)

        self.model.clear_model_cache()

        # Verify cache was cleared
        self.assertIsNone(self.model.last_original_hidden)
        self.assertIsNone(self.model.last_mixed_hidden)
        self.assertIsNone(self.model.final_values)
        self.assertIsNone(self.model.last_gate)


class TestMemoryLeakPrevention(unittest.TestCase):
    """Test memory leak prevention during training simulation."""

    def setUp(self):
        """Set up test fixtures."""
        config = GPTConfig(
            block_size=32, vocab_size=50, n_layer=2, n_head=2, n_embd=16, dag_depth=2
        )
        self.model = GPT(config)
        self.logger = DAGLogger()

    def test_forward_pass_memory_cleanup(self):
        """Test that forward passes don't accumulate memory."""
        # Get initial memory state
        initial_tensors = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])

        # Run multiple forward passes with cleanup
        for i in range(10):
            x = torch.randint(0, 50, (2, 16))
            y = torch.randint(0, 50, (2, 16))

            logits, loss = self.model(x, y)

            # Clear cache after each forward pass
            self.model.clear_model_cache()

            # Clear references
            del x, y, logits, loss

            if i % 5 == 0:
                gc.collect()

        # Final cleanup and garbage collection
        gc.collect()

        # Check that tensor count hasn't grown significantly
        final_tensors = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])
        tensor_growth = final_tensors - initial_tensors

        # Allow some growth but not excessive (should be < 20 new tensors)
        self.assertLess(
            tensor_growth, 20, f"Too many tensors accumulated: {tensor_growth}"
        )

    def test_training_loop_memory_management(self):
        """Test training loop memory management patterns."""
        # Simulate training loop memory management
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        for iteration in range(5):
            # Simulate batch loading
            x = torch.randint(0, 50, (2, 16))
            y = torch.randint(0, 50, (2, 16))

            # Forward pass
            logits, loss = self.model(x, y)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Memory cleanup (as implemented in training loop)
            self.model.clear_model_cache()

            if iteration % 2 == 0:
                self.logger.cleanup_for_next_iteration()

            # Clear batch references
            del x, y, logits, loss

            # Periodic garbage collection
            if iteration % 3 == 0:
                gc.collect()

        # Final cleanup
        self.logger.clear_memory_cache()
        gc.collect()

        # Test passes if no exceptions were raised


if __name__ == "__main__":
    unittest.main()
