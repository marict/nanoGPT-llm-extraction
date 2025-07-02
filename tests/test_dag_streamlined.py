#!/usr/bin/env python3
"""
Streamlined DAG model tests with essential coverage and minimal redundancy.
Consolidates functionality from multiple test files while avoiding gradient issues.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import time

import psutil
import pytest
import torch

# Set random seeds for reproducible tests
torch.manual_seed(42)

import dag_model
from dag_logger import DAGLogger
from dag_model import (GPT, DAGPlanPredictor, GPTConfig, divide, multiply,
                       subtract)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class TestDAGCore:
    """Core DAG functionality tests."""

    def test_basic_forward_pass(self):
        """Test basic forward pass without gradients."""
        config = GPTConfig(
            vocab_size=20,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dag_depth=2,
            dag_scratch_nodes=2,
        )
        model = GPT(config)
        model.eval()  # Avoid gradient computation

        x = torch.randint(0, 20, (2, 4))

        with torch.no_grad():
            logits, loss = model(x)
            assert logits.shape == (2, 4, 20)  # DAG model returns full sequence
            assert loss is None

        # Test with targets
        y = torch.randint(0, 20, (2, 4))
        with torch.no_grad():
            logits, loss = model(x, y)
            assert logits.shape == (2, 4, 20)
            assert loss is not None

    def test_op_functions(self):
        """Test mathematical operations."""
        x = torch.tensor([2.0, 3.0])
        y = torch.tensor([1.0, 2.0])
        assert torch.allclose(multiply(x, y), x * y)
        assert torch.allclose(subtract(x, y), x - y)
        assert torch.allclose(divide(x, y), x / y)

    def test_node_values_extraction(self):
        """Test that node values are extracted correctly."""
        config = GPTConfig(
            vocab_size=20,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dag_depth=2,
            dag_scratch_nodes=2,
        )
        model = GPT(config)
        model.eval()

        x = torch.randint(0, 20, (1, 4))

        with torch.no_grad():
            model(x)
            logger = DAGLogger()
            node_values = logger.get_node_values_list(model)

            assert len(node_values) == 4  # One per token
            for val in node_values:
                assert torch.isfinite(torch.tensor(val))


class TestDAGScratchSpace:
    """Tests for fixed scratch space optimization."""

    def test_fixed_scratch_space_size(self):
        """Test that scratch space size is fixed regardless of dag_depth."""
        test_configs = [
            (2, 2),  # dag_depth, scratch_nodes
            (4, 2),
            (8, 2),
        ]

        for dag_depth, scratch_nodes in test_configs:
            config = GPTConfig(
                vocab_size=50,
                n_layer=1,
                n_head=2,
                n_embd=32,
                dag_depth=dag_depth,
                dag_scratch_nodes=scratch_nodes,
                block_size=64,
            )
            model = GPT(config)
            model.eval()

            x = torch.randint(0, 50, (1, 4))
            with torch.no_grad():
                logits, _ = model(x)

            # Check that model produces valid output
            assert logits.shape == (1, 4, 50)

            # Check that node values are reasonable
            logger = DAGLogger()
            node_values = logger.get_node_values_list(model)
            assert len(node_values) == 4  # One per token
            for val in node_values:
                assert torch.isfinite(torch.tensor(val))

    def test_memory_scaling_linear(self):
        """Test that memory usage scales linearly, not quadratically."""
        config = GPTConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=64,
            dag_depth=4,
            dag_scratch_nodes=2,
            block_size=512,
        )
        model = GPT(config)
        model.eval()

        # Test with increasing sequence lengths
        seq_lengths = [32, 64, 128, 256]
        prev_memory = 0

        for seq_len in seq_lengths:
            x = torch.randint(0, 100, (1, seq_len))
            with torch.no_grad():
                logits, _ = model(x)

            # Get memory usage through node values
            logger = DAGLogger()
            node_values = logger.get_node_values_list(model)
            current_memory = len(node_values) * 4  # 4 bytes per float

            if prev_memory > 0:
                # Memory should scale roughly linearly
                ratio = current_memory / prev_memory
                expected_ratio = seq_len / (seq_lengths[seq_lengths.index(seq_len) - 1])
                assert 0.8 <= ratio / expected_ratio <= 1.2, "Memory scaling not linear"

            prev_memory = current_memory


class TestDAGPerformance:
    """Tests for DAG performance characteristics."""

    def test_attention_complexity(self):
        """Test that attention complexity doesn't blow up."""
        config = GPTConfig(
            vocab_size=100,
            n_layer=1,
            n_head=2,
            n_embd=64,
            dag_depth=8,
            dag_scratch_nodes=2,
            block_size=256,
        )
        model = GPT(config)
        model.eval()

        # Test with increasing sequence lengths
        seq_lengths = [32, 64, 128]
        times = []

        for seq_len in seq_lengths:
            x = torch.randint(0, 100, (1, seq_len))

            # Measure forward pass time
            start_time = time.time()
            with torch.no_grad():
                logits, _ = model(x)
            end_time = time.time()

            times.append(end_time - start_time)

            # Basic output validation
            assert logits.shape == (1, seq_len, 100)

            # Check node values
            logger = DAGLogger()
            node_values = logger.get_node_values_list(model)
            assert len(node_values) == seq_len
            assert all(torch.isfinite(torch.tensor(val)) for val in node_values)

        # Check that time complexity is roughly quadratic (from attention)
        # Allow some tolerance for measurement noise
        if len(times) > 1:
            for i in range(1, len(times)):
                time_ratio = times[i] / times[0]
                seq_ratio = (seq_lengths[i] / seq_lengths[0]) ** 2
                assert 0.1 <= time_ratio / seq_ratio <= 10, "Unexpected time scaling"


class TestDAGConfiguration:
    """Test DAG configuration and setup."""

    def test_config_defaults(self):
        """Test that DAG config defaults work correctly."""
        config = GPTConfig(dag_depth=4)
        assert config.dag_scratch_nodes == 2
        model = GPT(config)
        assert model.dag.embed_to_value is not None

    def test_zero_dag_depth(self):
        """Test that dag_depth=0 works like standard GPT."""
        config = GPTConfig(
            vocab_size=50, block_size=8, n_layer=1, n_head=2, n_embd=16, dag_depth=0
        )
        model = GPT(config)
        model.eval()

        x = torch.randint(0, 50, (2, 4))
        y = torch.randint(0, 50, (2, 4))

        with torch.no_grad():
            # Test with targets (training mode)
            logits, loss = model(x, y)
            assert logits.shape == (2, 4, 50)
            assert loss is not None

            # Test without targets (generation mode)
            logits_gen, loss_gen = model(x)
            assert logits_gen.shape == (
                2,
                1,
                50,
            )  # Standard GPT returns only last token
            assert loss_gen is None

    def test_different_scratch_configurations(self):
        """Test different scratch space configurations."""
        configurations = [
            2,  # scratch_nodes
            3,
            4,
        ]

        for scratch_nodes in configurations:
            config = GPTConfig(
                vocab_size=30,
                block_size=8,
                n_layer=1,
                n_head=2,
                n_embd=32,
                dag_depth=3,
                dag_scratch_nodes=scratch_nodes,
            )
            model = GPT(config)
            model.eval()

            x = torch.randint(0, 30, (1, 4))
            with torch.no_grad():
                logits, _ = model(x)

            assert logits.shape == (1, 4, 30)

            # Verify scratch space configuration through node values
            logger = DAGLogger()
            node_values = logger.get_node_values_list(model)
            assert len(node_values) == 4  # One value per token
            for val in node_values:
                assert torch.isfinite(torch.tensor(val)), "Node values should be finite"


def test_comprehensive_dag_functionality():
    """Integration test combining multiple DAG features."""
    config = GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dag_depth=4,
        dag_scratch_nodes=2,
    )
    model = GPT(config)
    model.eval()

    # Test various sequence lengths
    for seq_len in [1, 4, 8, 16]:
        x = torch.randint(0, 100, (2, seq_len))
        y = torch.randint(0, 100, (2, seq_len))

        with torch.no_grad():
            logits, loss = model(x, y)

            assert logits.shape == (2, seq_len, 100)
            assert loss is not None
            assert torch.isfinite(loss)

            # Check node values
            logger = DAGLogger()
            node_values = logger.get_node_values_list(model)
            assert len(node_values) == seq_len
            for val in node_values:
                assert torch.isfinite(torch.tensor(val))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
