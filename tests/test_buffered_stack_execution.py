"""
Comprehensive tests for the buffered stack execution optimization.
Tests both forward pass correctness and gradient preservation.
"""

import sys

import numpy as np
import pytest
import torch

sys.path.append(".")
from dag_model import (GPT, GPTConfig, stack_based_execution_buffered,
                       stack_based_execution_original)


class TestBufferedStackExecution:
    """Test suite for buffered stack execution optimization."""

    # --------------------------------------------------------------------- #
    # Core functionality tests (2 tests)
    # --------------------------------------------------------------------- #
    def test_forward_pass_equivalence(self):
        """Test that buffered and original implementations produce identical results."""
        torch.manual_seed(42)

        # Test with various configurations
        test_configs = [
            (2, 3, 2),  # (batch_size, seq_len, dag_depth)
            (1, 5, 3),
            (4, 2, 1),
            (3, 4, 4),
        ]

        for batch_size, seq_len, dag_depth in test_configs:
            num_initial = dag_depth + 1

            # Create test inputs
            initial_sgn = torch.randn(batch_size, seq_len, num_initial)
            initial_log = torch.abs(torch.randn(batch_size, seq_len, num_initial))

            # Create operation probabilities (should sum to 1.0 along last dim)
            ops_logits = torch.randn(batch_size, seq_len, dag_depth, 5)
            ops = torch.softmax(ops_logits, dim=-1)

            # Run both implementations
            result_sgn_orig, result_log_orig = stack_based_execution_original(
                initial_sgn, initial_log, ops
            )
            result_sgn_buff, result_log_buff = stack_based_execution_buffered(
                initial_sgn, initial_log, ops
            )

            # Check results are identical
            assert torch.allclose(
                result_sgn_orig, result_sgn_buff, atol=1e-6
            ), f"Signs differ for config {(batch_size, seq_len, dag_depth)}"
            assert torch.allclose(
                result_log_orig, result_log_buff, atol=1e-6
            ), f"Logs differ for config {(batch_size, seq_len, dag_depth)}"

    def test_gradient_preservation(self):
        """Test that gradients flow correctly through buffered implementation."""
        torch.manual_seed(42)

        batch_size, seq_len, dag_depth = 2, 3, 2
        num_initial = dag_depth + 1

        # Create test inputs for original implementation
        initial_sgn_orig = torch.randn(
            batch_size, seq_len, num_initial, requires_grad=True
        )
        initial_log_orig = torch.abs(
            torch.randn(batch_size, seq_len, num_initial)
        ).requires_grad_(True)
        ops_logits_orig = torch.randn(
            batch_size, seq_len, dag_depth, 5, requires_grad=True
        )
        ops_orig = torch.softmax(ops_logits_orig, dim=-1)

        # Create identical test inputs for buffered implementation
        initial_sgn_buff = initial_sgn_orig.clone().detach().requires_grad_(True)
        initial_log_buff = initial_log_orig.clone().detach().requires_grad_(True)
        ops_logits_buff = ops_logits_orig.clone().detach().requires_grad_(True)
        ops_buff = torch.softmax(ops_logits_buff, dim=-1)

        # Test original implementation
        result_sgn_orig, result_log_orig = stack_based_execution_original(
            initial_sgn_orig, initial_log_orig, ops_orig
        )
        loss_orig = result_sgn_orig.sum() + result_log_orig.sum()
        loss_orig.backward()

        # Save gradients
        grad_sgn_orig = initial_sgn_orig.grad.clone()
        grad_log_orig = initial_log_orig.grad.clone()
        grad_ops_orig = ops_logits_orig.grad.clone()

        # Test buffered implementation
        result_sgn_buff, result_log_buff = stack_based_execution_buffered(
            initial_sgn_buff, initial_log_buff, ops_buff
        )
        loss_buff = result_sgn_buff.sum() + result_log_buff.sum()
        loss_buff.backward()

        # Check gradients are identical
        assert torch.allclose(
            grad_sgn_orig, initial_sgn_buff.grad, atol=1e-6
        ), "initial_sgn gradients differ"
        assert torch.allclose(
            grad_log_orig, initial_log_buff.grad, atol=1e-6
        ), "initial_log gradients differ"
        assert torch.allclose(
            grad_ops_orig, ops_logits_buff.grad, atol=1e-6
        ), "ops gradients differ"

    # --------------------------------------------------------------------- #
    # Consolidated edge cases and validation (1 test)
    # --------------------------------------------------------------------- #
    def test_edge_cases_and_validation(self):
        """Test edge cases and stack underflow detection."""
        torch.manual_seed(42)

        # Test with dag_depth = 1 (minimum)
        batch_size, seq_len, dag_depth = 1, 2, 1
        num_initial = dag_depth + 1  # 2 initial values

        initial_sgn = torch.randn(batch_size, seq_len, num_initial)
        initial_log = torch.abs(torch.randn(batch_size, seq_len, num_initial))

        # Single operation step
        ops_logits = torch.randn(batch_size, seq_len, dag_depth, 5)
        ops = torch.softmax(ops_logits, dim=-1)

        # Both implementations should work
        result_sgn_orig, result_log_orig = stack_based_execution_original(
            initial_sgn, initial_log, ops
        )
        result_sgn_buff, result_log_buff = stack_based_execution_buffered(
            initial_sgn, initial_log, ops
        )

        assert torch.allclose(result_sgn_orig, result_sgn_buff, atol=1e-6)
        assert torch.allclose(result_log_orig, result_log_buff, atol=1e-6)

        # Test stack underflow detection
        # Create invalid configuration: dag_depth >= num_initial
        batch_size, seq_len = 1, 2
        dag_depth = 3
        num_initial = 2  # Invalid: need at least dag_depth + 1 = 4

        initial_sgn = torch.randn(batch_size, seq_len, num_initial)
        initial_log = torch.abs(torch.randn(batch_size, seq_len, num_initial))
        ops_logits = torch.randn(batch_size, seq_len, dag_depth, 5)
        ops = torch.softmax(ops_logits, dim=-1)

        # Both implementations should raise RuntimeError
        with pytest.raises(RuntimeError, match="Stack underflow"):
            stack_based_execution_original(initial_sgn, initial_log, ops)

        with pytest.raises(RuntimeError, match="Stack underflow"):
            stack_based_execution_buffered(initial_sgn, initial_log, ops)

    # --------------------------------------------------------------------- #
    # Consolidated device and dtype compatibility (1 test)
    # --------------------------------------------------------------------- #
    def test_device_and_dtype_compatibility(self):
        """Test device and dtype compatibility for buffered implementation."""
        torch.manual_seed(42)

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        dtypes = [torch.float32, torch.float16]
        if torch.cuda.is_available():
            dtypes.append(torch.bfloat16)

        # Test device compatibility
        for device in devices:
            batch_size, seq_len, dag_depth = 2, 3, 2
            num_initial = dag_depth + 1

            initial_sgn = torch.randn(batch_size, seq_len, num_initial, device=device)
            initial_log = torch.abs(
                torch.randn(batch_size, seq_len, num_initial, device=device)
            )
            ops_logits = torch.randn(batch_size, seq_len, dag_depth, 5, device=device)
            ops = torch.softmax(ops_logits, dim=-1)

            # Should work without errors
            result_sgn, result_log = stack_based_execution_buffered(
                initial_sgn, initial_log, ops
            )

            assert result_sgn.device == device
            assert result_log.device == device

        # Test dtype preservation
        for dtype in dtypes:
            batch_size, seq_len, dag_depth = 2, 3, 2
            num_initial = dag_depth + 1

            initial_sgn = torch.randn(batch_size, seq_len, num_initial, dtype=dtype)
            initial_log = torch.abs(
                torch.randn(batch_size, seq_len, num_initial, dtype=dtype)
            )
            ops_logits = torch.randn(batch_size, seq_len, dag_depth, 5, dtype=dtype)
            ops = torch.softmax(ops_logits, dim=-1)

            result_sgn, result_log = stack_based_execution_buffered(
                initial_sgn, initial_log, ops
            )

            assert result_sgn.dtype == dtype
            assert result_log.dtype == dtype

    # --------------------------------------------------------------------- #
    # Consolidated model integration (1 test)
    # --------------------------------------------------------------------- #
    def test_full_model_integration(self):
        """Test full model integration and gradient flow."""
        torch.manual_seed(42)

        # Test full model integration
        cfg = GPTConfig(
            vocab_size=50, block_size=8, n_layer=2, n_head=2, n_embd=32, dag_depth=2
        )
        model = GPT(cfg)
        model.train()

        # Create test batch
        batch_size = 2
        seq_len = 6
        x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits, loss = model(x, y)

        # Verify shapes
        assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
        assert loss.shape == ()
        assert loss.item() > 0

        # Backward pass
        loss.backward()

        # Test gradient flow in full model
        # Check that DAG components have gradients
        dag_params_with_grads = 0
        for name, param in model.named_parameters():
            if "dag" in name and param.requires_grad:
                if param.grad is not None:
                    dag_params_with_grads += 1
                    # Verify gradient is finite and reasonable
                    assert torch.isfinite(
                        param.grad
                    ).all(), f"Non-finite gradient in {name}"
                    assert (
                        param.grad.norm().item() < 100
                    ), f"Excessive gradient in {name}"

        # Should have some DAG parameters with gradients
        assert dag_params_with_grads > 0, "No DAG parameters received gradients"

        # Test that model can handle different sequence lengths
        for test_seq_len in [4, 8]:
            if test_seq_len <= cfg.block_size:
                x_test = torch.randint(0, cfg.vocab_size, (1, test_seq_len))
                model.zero_grad()
                with torch.no_grad():
                    logits_test, _ = model(x_test)
                assert logits_test.shape == (1, test_seq_len, cfg.vocab_size)

    # --------------------------------------------------------------------- #
    # Consolidated performance and stability (1 test)
    # --------------------------------------------------------------------- #
    def test_performance_and_stability(self):
        """Test memory efficiency, numerical stability, and performance."""
        torch.manual_seed(42)

        # Test numerical stability
        batch_size, seq_len, dag_depth = 2, 4, 3
        num_initial = dag_depth + 1

        # Create inputs with extreme values
        initial_sgn_extreme = torch.randn(batch_size, seq_len, num_initial) * 100
        initial_log_extreme = (
            torch.abs(torch.randn(batch_size, seq_len, num_initial)) * 10
        )
        ops_logits_extreme = torch.randn(batch_size, seq_len, dag_depth, 5) * 50
        ops_extreme = torch.softmax(ops_logits_extreme, dim=-1)

        # Both implementations should handle extreme values
        result_sgn_orig, result_log_orig = stack_based_execution_original(
            initial_sgn_extreme, initial_log_extreme, ops_extreme
        )
        result_sgn_buff, result_log_buff = stack_based_execution_buffered(
            initial_sgn_extreme, initial_log_extreme, ops_extreme
        )

        # Results should be finite and close
        assert (
            torch.isfinite(result_sgn_orig).all()
            and torch.isfinite(result_sgn_buff).all()
        )
        assert (
            torch.isfinite(result_log_orig).all()
            and torch.isfinite(result_log_buff).all()
        )
        assert torch.allclose(result_sgn_orig, result_sgn_buff, atol=1e-4)
        assert torch.allclose(result_log_orig, result_log_buff, atol=1e-4)

        # Test memory efficiency (basic test)
        # Buffered implementation should not use significantly more memory
        batch_size, seq_len, dag_depth = 4, 8, 4
        num_initial = dag_depth + 1

        initial_sgn = torch.randn(batch_size, seq_len, num_initial)
        initial_log = torch.abs(torch.randn(batch_size, seq_len, num_initial))
        ops_logits = torch.randn(batch_size, seq_len, dag_depth, 5)
        ops = torch.softmax(ops_logits, dim=-1)

        # Both implementations should complete without memory issues
        result_sgn_orig, result_log_orig = stack_based_execution_original(
            initial_sgn, initial_log, ops
        )
        result_sgn_buff, result_log_buff = stack_based_execution_buffered(
            initial_sgn, initial_log, ops
        )

        # Basic performance comparison - both should complete in reasonable time
        # and produce identical results
        assert torch.allclose(result_sgn_orig, result_sgn_buff, atol=1e-6)
        assert torch.allclose(result_log_orig, result_log_buff, atol=1e-6)

        # Test that buffered implementation doesn't degrade performance significantly
        # (This is a basic test - more detailed benchmarking would be done separately)
        import time

        # Small timing test
        start_time = time.time()
        for _ in range(5):
            stack_based_execution_buffered(initial_sgn, initial_log, ops)
        buffered_time = time.time() - start_time

        start_time = time.time()
        for _ in range(5):
            stack_based_execution_original(initial_sgn, initial_log, ops)
        original_time = time.time() - start_time

        # Buffered should not be more than 3x slower (generous tolerance)
        assert (
            buffered_time < original_time * 3
        ), "Buffered implementation significantly slower"


# --------------------------------------------------------------------- #
# Standalone performance test (outside class)
# --------------------------------------------------------------------- #
def test_performance_comparison():
    """Standalone performance comparison test."""
    torch.manual_seed(42)

    batch_size, seq_len, dag_depth = 4, 8, 4
    num_initial = dag_depth + 1

    initial_sgn = torch.randn(batch_size, seq_len, num_initial)
    initial_log = torch.abs(torch.randn(batch_size, seq_len, num_initial))
    ops_logits = torch.randn(batch_size, seq_len, dag_depth, 5)
    ops = torch.softmax(ops_logits, dim=-1)

    # Test that both implementations produce identical results
    result_sgn_orig, result_log_orig = stack_based_execution_original(
        initial_sgn, initial_log, ops
    )
    result_sgn_buff, result_log_buff = stack_based_execution_buffered(
        initial_sgn, initial_log, ops
    )

    assert torch.allclose(result_sgn_orig, result_sgn_buff, atol=1e-6)
    assert torch.allclose(result_log_orig, result_log_buff, atol=1e-6)

    # Basic performance check - both should complete quickly
    import time

    start_time = time.time()
    for _ in range(10):
        stack_based_execution_buffered(initial_sgn, initial_log, ops)
    total_time = time.time() - start_time

    # Should complete 10 iterations in reasonable time (< 1 second)
    assert total_time < 1.0, f"Performance test too slow: {total_time:.3f}s"
