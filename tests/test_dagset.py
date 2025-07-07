#!/usr/bin/env python
"""
test_dagset.py
Tests for the streaming DAG dataset functionality.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Add the data directory to the path so we can import the dagset module
sys.path.append(str(Path(__file__).parent.parent / "data" / "dagset"))

from streaming import (DAGDataLoader, DAGExample, StreamingDAGDataset,
                       create_dag_dataloaders, execute_dag_computation,
                       format_dag_as_text, generate_dag_dataset,
                       generate_random_dag_plan, generate_random_initial_value,
                       generate_single_dag_example)

# Import DAG operations for direct testing
sys.path.append(str(Path(__file__).parent.parent))
from dag_model import (LOG_LIM, add_log_space, divide_log_space,
                       identity_log_space, multiply_log_space,
                       subtract_log_space)


class TestStreamingDAGDataset(unittest.TestCase):
    """Test the streaming DAG dataset functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a fixed seed for reproducible tests
        np.random.seed(42)
        torch.manual_seed(42)

    def test_generate_random_initial_value(self):
        """Test random initial value generation."""
        # Test with default range
        sign, log_mag = generate_random_initial_value()
        self.assertIn(sign, [-1.0, 1.0])
        self.assertGreaterEqual(log_mag, 0.0)
        self.assertLessEqual(log_mag, LOG_LIM)

        # Test with custom range
        sign, log_mag = generate_random_initial_value((-5.0, 5.0))
        self.assertIn(sign, [-1.0, 1.0])
        self.assertGreaterEqual(log_mag, 0.0)
        self.assertLessEqual(log_mag, LOG_LIM)

    def test_generate_random_dag_plan(self):
        """Test DAG plan generation."""
        # Test depth 1
        operations = generate_random_dag_plan(depth=1, num_initial_values=1)
        self.assertEqual(len(operations), 1)
        operand1_idx, operand2_idx, operation_name = operations[0]
        self.assertEqual(operand1_idx, 0)  # Only one initial value
        self.assertEqual(operand2_idx, 0)
        self.assertIn(
            operation_name, ["add", "subtract", "multiply", "divide", "identity"]
        )

        # Test depth 3
        operations = generate_random_dag_plan(depth=3, num_initial_values=1)
        self.assertEqual(len(operations), 3)

        # Check that operand indices are valid
        for i, (op1_idx, op2_idx, op_name) in enumerate(operations):
            max_available = 1 + i  # Initial values + previous results
            self.assertGreaterEqual(op1_idx, 0)
            self.assertLess(op1_idx, max_available)
            self.assertGreaterEqual(op2_idx, 0)
            self.assertLess(op2_idx, max_available)
            self.assertIn(
                op_name, ["add", "subtract", "multiply", "divide", "identity"]
            )

    def test_execute_dag_computation(self):
        """Test DAG computation execution."""
        # Test simple addition
        initial_values = [(1.0, 1.0)]  # sign=1, log_mag=1 -> value ≈ e^1 ≈ 2.718
        operations = [(0, 0, "add")]

        result_values = execute_dag_computation(initial_values, operations)

        # Should have initial value + one result
        self.assertEqual(len(result_values), 2)

        # Check that the result is approximately correct
        sign0, log0 = result_values[0]
        sign1, log1 = result_values[1]

        # Initial value should be preserved
        self.assertAlmostEqual(sign0, 1.0, places=5)
        self.assertAlmostEqual(log0, 1.0, places=5)

        # Result should be approximately 2 * e^1
        expected_log = np.log(2 * np.exp(1.0))
        self.assertAlmostEqual(sign1, 1.0, places=5)
        self.assertAlmostEqual(log1, expected_log, places=3)

    def test_generate_single_dag_example(self):
        """Test single DAG example generation."""
        example = generate_single_dag_example(depth=2, num_initial_values=1)

        self.assertIsInstance(example, DAGExample)
        self.assertEqual(example.depth, 2)
        self.assertEqual(len(example.initial_values), 1)
        self.assertEqual(len(example.operations), 2)
        self.assertIsInstance(example.text, str)
        self.assertGreater(len(example.text), 0)

        # Check that text contains expected elements
        self.assertIn("DAG Computation", example.text)
        self.assertIn("v0 =", example.text)
        self.assertIn("Step 1:", example.text)
        self.assertIn("Step 2:", example.text)
        self.assertIn("Final result:", example.text)

    def test_generate_dag_dataset(self):
        """Test dataset generation."""
        # Generate a small dataset
        examples = generate_dag_dataset(
            num_examples=10, max_depth=3, min_depth=1, num_initial_values=1
        )

        self.assertEqual(len(examples), 10)

        # Check that all examples are valid
        for example in examples:
            self.assertIsInstance(example, DAGExample)
            self.assertGreaterEqual(example.depth, 1)
            self.assertLessEqual(example.depth, 3)
            self.assertEqual(len(example.initial_values), 1)
            self.assertEqual(len(example.operations), example.depth)
            self.assertIsInstance(example.text, str)
            self.assertGreater(len(example.text), 0)

    def test_streaming_dataset_basic(self):
        """Test basic streaming dataset functionality."""
        dataset = StreamingDAGDataset(max_depth=3, min_depth=1, seed=42)

        # Test batch generation
        tokens, text = dataset.generate_batch(batch_size=5)
        self.assertIsInstance(tokens, list)
        self.assertIsInstance(text, str)
        self.assertGreater(len(tokens), 0)
        self.assertGreater(len(text), 0)

        # Test specific token generation
        target_tokens = 1000
        tokens = dataset.generate_tokens(target_tokens)
        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), target_tokens)

    def test_streaming_dataset_reproducibility(self):
        """Test that streaming dataset is reproducible with same seed."""
        dataset1 = StreamingDAGDataset(max_depth=3, min_depth=1, seed=123)
        dataset2 = StreamingDAGDataset(max_depth=3, min_depth=1, seed=123)

        tokens1, text1 = dataset1.generate_batch(10)
        tokens2, text2 = dataset2.generate_batch(10)

        self.assertEqual(tokens1, tokens2)
        self.assertEqual(text1, text2)

    def test_streaming_dataset_train_val_split(self):
        """Test train/val split functionality."""
        dataset = StreamingDAGDataset(max_depth=3, min_depth=1, seed=42)

        train_tokens, val_tokens = dataset.get_train_val_split(
            train_examples=20, val_examples=10, split_seed=43
        )

        self.assertIsInstance(train_tokens, list)
        self.assertIsInstance(val_tokens, list)
        self.assertGreater(len(train_tokens), 0)
        self.assertGreater(len(val_tokens), 0)
        self.assertNotEqual(train_tokens, val_tokens)  # Should be different

    def test_dag_dataloader(self):
        """Test the DAG data loader."""
        dataset = StreamingDAGDataset(max_depth=3, min_depth=1, seed=42)

        dataloader = DAGDataLoader(
            dataset=dataset,
            batch_size=4,
            block_size=128,
            examples_per_batch=50,
        )

        # Test getting a few batches
        for i, (inputs, targets) in enumerate(dataloader):
            self.assertEqual(inputs.shape, (4, 128))
            self.assertEqual(targets.shape, (4, 128))
            self.assertEqual(inputs.dtype, torch.long)
            self.assertEqual(targets.dtype, torch.long)

            # Verify targets are shifted inputs
            self.assertTrue(torch.equal(inputs[:, 1:], targets[:, :-1]))

            if i >= 2:  # Just test a few batches
                break

    def test_create_dag_dataloaders(self):
        """Test the convenience function for creating data loaders."""
        train_loader, val_loader = create_dag_dataloaders(
            train_examples_per_batch=100,
            val_examples_per_batch=50,
            batch_size=4,
            block_size=64,
            max_depth=3,
            min_depth=1,
            train_seed=42,
            val_seed=43,
        )

        # Test train loader
        for i, (inputs, targets) in enumerate(train_loader):
            self.assertEqual(inputs.shape, (4, 64))
            self.assertEqual(targets.shape, (4, 64))
            if i >= 1:
                break

        # Test val loader
        for i, (inputs, targets) in enumerate(val_loader):
            self.assertEqual(inputs.shape, (4, 64))
            self.assertEqual(targets.shape, (4, 64))
            if i >= 1:
                break

    def test_dag_computation_consistency(self):
        """Test that DAG computation results are mathematically consistent."""
        # Create a specific example we can verify manually
        initial_values = [(1.0, np.log(2.0))]  # value = 2.0
        operations = [
            (0, 0, "add"),  # v1 = v0 + v0 = 2 + 2 = 4
            (1, 0, "multiply"),  # v2 = v1 * v0 = 4 * 2 = 8
            (2, 1, "divide"),  # v3 = v2 / v1 = 8 / 4 = 2
        ]

        result_values = execute_dag_computation(initial_values, operations)

        # Check intermediate results
        self.assertEqual(len(result_values), 4)  # initial + 3 operations

        # v0 = 2.0
        sign0, log0 = result_values[0]
        value0 = sign0 * np.exp(log0)
        self.assertAlmostEqual(value0, 2.0, places=5)

        # v1 = 4.0
        sign1, log1 = result_values[1]
        value1 = sign1 * np.exp(log1)
        self.assertAlmostEqual(value1, 4.0, places=5)

        # v2 = 8.0
        sign2, log2 = result_values[2]
        value2 = sign2 * np.exp(log2)
        self.assertAlmostEqual(value2, 8.0, places=5)

        # v3 = 2.0
        sign3, log3 = result_values[3]
        value3 = sign3 * np.exp(log3)
        self.assertAlmostEqual(value3, 2.0, places=5)

    def test_negative_values(self):
        """Test DAG computations with negative values."""
        # Test with negative initial value
        initial_values = [(-1.0, np.log(3.0))]  # value = -3.0
        operations = [(0, 0, "multiply")]  # v1 = v0 * v0 = (-3) * (-3) = 9

        result_values = execute_dag_computation(initial_values, operations)

        # Check result
        sign1, log1 = result_values[1]
        value1 = sign1 * np.exp(log1)
        self.assertAlmostEqual(value1, 9.0, places=5)

    def test_log_space_operations_integration(self):
        """Test integration with actual log-space operations."""
        # Test addition: 2 + 3 = 5
        sign1, log1 = 1.0, np.log(2.0)
        sign2, log2 = 1.0, np.log(3.0)

        result_sign, result_log = add_log_space(
            torch.tensor(sign1),
            torch.tensor(log1),
            torch.tensor(sign2),
            torch.tensor(log2),
            ignore_clip=True,
        )

        result_value = result_sign.item() * np.exp(result_log.item())
        self.assertAlmostEqual(result_value, 5.0, places=5)

    def test_streaming_infinite_generation(self):
        """Test that streaming can generate data indefinitely."""
        dataset = StreamingDAGDataset(max_depth=2, min_depth=1, seed=42)

        # Test streaming tokens
        token_stream = dataset.stream_tokens(batch_size=10)

        # Get several batches
        for i, tokens in enumerate(token_stream):
            self.assertIsInstance(tokens, list)
            self.assertGreater(len(tokens), 0)
            if i >= 3:  # Test first few batches
                break


if __name__ == "__main__":
    unittest.main()
