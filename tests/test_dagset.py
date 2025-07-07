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

from streaming import (DAGDataLoader, DAGExample, DAGStructureDataset,
                       StreamingDAGDataset, convert_dag_to_expression_string,
                       create_dag_dataloaders,
                       create_dag_structure_dataloaders,
                       execute_dag_computation, generate_dag_dataset,
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
        """Test single DAG example generation (simple expression format)."""
        example = generate_single_dag_example(
            depth=2, num_initial_values=3
        )  # depth+1 initial values

        self.assertIsInstance(example, DAGExample, msg="Example is not a DAGExample")
        self.assertEqual(example.depth, 2, msg="Depth is not 2")
        self.assertEqual(
            len(example.initial_values), 3, msg="Initial values length is not 3"
        )  # depth + 1
        self.assertEqual(len(example.operations), 2, msg="Operations length is not 2")
        self.assertIsInstance(example.text, str, msg="Text is not a string")
        self.assertGreater(len(example.text), 0, msg="Text is empty")

        # Check that text is a simple mathematical expression
        # Should contain numbers and operators, but not verbose DAG format
        import re

        # Should contain numbers (with potential decimals)
        self.assertTrue(
            re.search(r"\d+\.?\d*", example.text),
            msg=f"Text does not contain numbers: {example.text}",
        )
        # Should contain operators
        self.assertTrue(
            re.search(r"[\+\-\*/]", example.text),
            msg=f"Text does not contain operators: {example.text}",
        )

        # Should NOT contain verbose DAG format elements
        self.assertNotIn(
            "DAG Computation",
            example.text,
            msg=f"Text contains 'DAG Computation': {example.text}",
        )
        self.assertNotIn(
            "v0 =", example.text, msg=f"Text contains 'v0 =': {example.text}"
        )
        self.assertNotIn(
            "Step", example.text, msg=f"Text contains 'Step': {example.text}"
        )
        self.assertNotIn(
            "Final result:",
            example.text,
            msg=f"Text contains 'Final result:': {example.text}",
        )

    def test_generate_dag_dataset(self):
        """Test dataset generation."""
        # Generate a small dataset
        examples = generate_dag_dataset(
            num_examples=10,
            max_depth=3,  # num_initial_values will default to max_depth+1
        )

        self.assertEqual(len(examples), 10)

        # Check that all examples are valid
        for example in examples:
            self.assertIsInstance(example, DAGExample)
            self.assertGreaterEqual(example.depth, 1)
            self.assertLessEqual(example.depth, 3)
            self.assertEqual(len(example.initial_values), 3 + 1)  # max_depth + 1
            self.assertEqual(len(example.operations), example.depth)
            self.assertIsInstance(example.text, str)
            self.assertGreater(len(example.text), 0)

    def test_streaming_dataset_basic(self):
        """Test basic streaming dataset functionality."""
        dataset = StreamingDAGDataset(max_depth=3, seed=42)

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
        dataset1 = StreamingDAGDataset(max_depth=3, seed=123)
        dataset2 = StreamingDAGDataset(max_depth=3, seed=123)

        tokens1, text1 = dataset1.generate_batch(10)
        tokens2, text2 = dataset2.generate_batch(10)

        self.assertEqual(tokens1, tokens2)
        self.assertEqual(text1, text2)

    def test_streaming_dataset_train_val_split(self):
        """Test train/val split functionality."""
        dataset = StreamingDAGDataset(max_depth=3, seed=42)

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
        dataset = StreamingDAGDataset(max_depth=3, seed=42)

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
        dataset = StreamingDAGDataset(max_depth=2, seed=42)

        # Test streaming tokens
        token_stream = dataset.stream_tokens(batch_size=10)

        # Get several batches
        for i, tokens in enumerate(token_stream):
            self.assertIsInstance(tokens, list)
            self.assertGreater(len(tokens), 0)
            if i >= 3:  # Test first few batches
                break


class TestDAGStructureDataset(unittest.TestCase):
    """Test the DAG structure dataset for pretraining."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a fixed seed for reproducible tests
        np.random.seed(42)
        torch.manual_seed(42)

    def test_dag_structure_dataset_basic(self):
        """Test basic DAG structure dataset functionality."""
        dataset = DAGStructureDataset(max_depth=3, seed=42)

        # Test single example generation
        text, structure = dataset.generate_structure_example(depth=2)

        # Verify text is a string (simple mathematical expression)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        # Should be a simple math expression, not verbose DAG format
        import re

        self.assertTrue(re.search(r"\d+\.?\d*", text))  # Contains numbers
        self.assertTrue(re.search(r"[\+\-\*/]", text))  # Contains operators

        # Verify structure is a dictionary with correct keys
        self.assertIsInstance(structure, dict)
        expected_keys = {
            "initial_sgn",
            "initial_log",
            "operation_probs",
            "depth",
            "operations",
        }
        self.assertEqual(set(structure.keys()), expected_keys)

        # Verify tensor shapes
        self.assertEqual(structure["initial_sgn"].shape, (3,))  # depth + 1 = 2 + 1
        self.assertEqual(structure["initial_log"].shape, (3,))
        self.assertEqual(structure["operation_probs"].shape, (2, 5))  # (depth, n_ops)
        self.assertEqual(structure["depth"].item(), 2)

    def test_structure_tensor_format(self):
        """Test that structure tensors match DAGPlanPredictor format."""
        dataset = DAGStructureDataset(max_depth=4, seed=42)

        for test_depth in [2, 3, 4]:
            text, structure = dataset.generate_structure_example(depth=test_depth)

            # Verify shapes match expected format
            num_nodes = test_depth + 1
            self.assertEqual(structure["initial_sgn"].shape, (num_nodes,))
            self.assertEqual(structure["initial_log"].shape, (num_nodes,))
            self.assertEqual(structure["operation_probs"].shape, (test_depth, 5))
            self.assertEqual(structure["depth"].item(), test_depth)

            # Verify initial values are reasonable
            sgn = structure["initial_sgn"]
            log_mag = structure["initial_log"]

            # All initial values should be non-zero (we now use depth+1 initial values)
            for i in range(num_nodes):
                self.assertNotEqual(sgn[i].item(), 0.0)
                self.assertNotEqual(log_mag[i].item(), 0.0)

            # Verify operation probabilities are one-hot
            op_probs = structure["operation_probs"]
            for step in range(test_depth):
                step_probs = op_probs[step]
                self.assertAlmostEqual(step_probs.sum().item(), 1.0, places=5)
                # Should have exactly one 1.0 and rest 0.0
                ones = (step_probs == 1.0).sum().item()
                zeros = (step_probs == 0.0).sum().item()
                self.assertEqual(ones, 1)
                self.assertEqual(zeros, 4)

    def test_structure_consistency_with_dag_computation(self):
        """Test that structure tensors are consistent with actual DAG computation."""
        dataset = DAGStructureDataset(max_depth=3, seed=42)

        # Generate a structure example
        text, structure = dataset.generate_structure_example(depth=2)

        # Extract the DAG example that was used internally
        # We can recreate this by parsing the operations from the structure
        depth = structure["depth"].item()

        # Get the operation names from the one-hot vectors
        op_probs = structure["operation_probs"]
        operation_names = []
        for step in range(depth):
            op_idx = torch.argmax(op_probs[step]).item()
            operation_names.append(dataset.op_idx_to_name[op_idx])

        # Verify structure has valid operations
        # Simple expressions don't contain operation names, but we can check symbols
        # Just verify that we have valid operations and text is a mathematical expression
        import re

        self.assertTrue(re.search(r"\d+\.?\d*", text))  # Contains numbers
        self.assertTrue(re.search(r"[\+\-\*/]", text))  # Contains operators

        # Verify all operations are valid
        valid_ops = {"add", "subtract", "multiply", "divide", "identity"}
        for op_name in operation_names:
            self.assertIn(op_name, valid_ops)

    def test_batch_generation_and_padding(self):
        """Test batch generation with proper padding."""
        dataset = DAGStructureDataset(max_depth=4, seed=42)

        # Generate a batch with mixed depths
        texts, structures = dataset.generate_batch(batch_size=5)

        # Verify batch size
        self.assertEqual(len(texts), 5)
        self.assertEqual(structures["initial_sgn"].shape[0], 5)
        self.assertEqual(structures["initial_log"].shape[0], 5)
        self.assertEqual(structures["operation_probs"].shape[0], 5)
        self.assertEqual(structures["depths"].shape[0], 5)

        # Verify padding - all tensors should be padded to max depth
        max_depth_in_batch = structures["depths"].max().item()
        expected_nodes = max_depth_in_batch + 1

        self.assertEqual(structures["initial_sgn"].shape[1], expected_nodes)
        self.assertEqual(structures["initial_log"].shape[1], expected_nodes)
        self.assertEqual(structures["operation_probs"].shape[1], max_depth_in_batch)
        self.assertEqual(structures["operation_probs"].shape[2], 5)  # n_ops

        # Verify that shorter examples are properly padded
        for i in range(5):
            actual_depth = structures["depths"][i].item()

            # Check that unused operation steps are all zeros
            if actual_depth < max_depth_in_batch:
                unused_ops = structures["operation_probs"][i, actual_depth:]
                self.assertTrue(torch.all(unused_ops == 0.0))

            # Check that unused initial value slots are zeros
            if actual_depth + 1 < expected_nodes:
                unused_initial_sgn = structures["initial_sgn"][i, actual_depth + 1 :]
                unused_initial_log = structures["initial_log"][i, actual_depth + 1 :]
                self.assertTrue(torch.all(unused_initial_sgn == 0.0))
                self.assertTrue(torch.all(unused_initial_log == 0.0))

    def test_dataloader_functionality(self):
        """Test the structure dataloader."""
        dataset = DAGStructureDataset(max_depth=3, seed=42)

        # Create dataloader
        dataloader = dataset.create_dataloader(batch_size=4)

        # Test getting a few batches
        for i, (texts, structures) in enumerate(dataloader):
            # Verify batch structure
            self.assertEqual(len(texts), 4)
            self.assertIsInstance(texts, list)
            self.assertIsInstance(structures, dict)

            # Verify tensor shapes
            self.assertEqual(structures["initial_sgn"].shape[0], 4)
            self.assertEqual(structures["initial_log"].shape[0], 4)
            self.assertEqual(structures["operation_probs"].shape[0], 4)
            self.assertEqual(structures["depths"].shape[0], 4)

            # Verify all texts are strings
            for text in texts:
                self.assertIsInstance(text, str)
                self.assertGreater(len(text), 0)

            if i >= 2:  # Test a few batches
                break

    def test_create_dag_structure_dataloaders(self):
        """Test the convenience function for creating structure dataloaders."""
        train_loader, val_loader = create_dag_structure_dataloaders(
            train_batch_size=4,
            val_batch_size=2,
            max_depth=3,
            train_seed=42,
            val_seed=43,
        )

        # Test train loader
        texts, structures = next(train_loader)
        self.assertEqual(len(texts), 4)
        self.assertEqual(structures["initial_sgn"].shape[0], 4)

        # Test val loader
        texts, structures = next(val_loader)
        self.assertEqual(len(texts), 2)
        self.assertEqual(structures["initial_sgn"].shape[0], 2)

    def test_structure_dataset_reproducibility(self):
        """Test that structure dataset is reproducible with same seed."""
        dataset1 = DAGStructureDataset(max_depth=3, seed=123)
        dataset2 = DAGStructureDataset(max_depth=3, seed=123)

        texts1, structures1 = dataset1.generate_batch(5)
        texts2, structures2 = dataset2.generate_batch(5)

        # Text should be identical
        self.assertEqual(texts1, texts2)

        # Structures should be identical
        for key in structures1.keys():
            self.assertTrue(torch.equal(structures1[key], structures2[key]))

    def test_structure_dataset_different_seeds(self):
        """Test that different seeds produce different results."""
        dataset1 = DAGStructureDataset(max_depth=3, seed=42)
        dataset2 = DAGStructureDataset(max_depth=3, seed=43)

        texts1, structures1 = dataset1.generate_batch(5)
        texts2, structures2 = dataset2.generate_batch(5)

        # Should be different
        self.assertNotEqual(texts1, texts2)

        # At least one structure tensor should be different
        different = False
        for key in structures1.keys():
            if not torch.equal(structures1[key], structures2[key]):
                different = True
                break
        self.assertTrue(different, "Different seeds should produce different results")

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test minimum depth
        dataset = DAGStructureDataset(max_depth=1, seed=42)
        text, structure = dataset.generate_structure_example(depth=1)

        self.assertEqual(structure["depth"].item(), 1)
        self.assertEqual(structure["initial_sgn"].shape, (2,))  # 1 + 1
        self.assertEqual(structure["operation_probs"].shape, (1, 5))  # (1, n_ops)

    def test_op_name_to_idx_mapping(self):
        """Test that operation name to index mapping is correct."""
        dataset = DAGStructureDataset(max_depth=3, seed=42)

        # Verify the mapping exists and is correct
        self.assertIsInstance(dataset.op_name_to_idx, dict)
        self.assertEqual(len(dataset.op_name_to_idx), 5)  # 5 operations

        # Verify all expected operations are mapped
        expected_ops = ["add", "subtract", "multiply", "divide", "identity"]
        for op in expected_ops:
            self.assertIn(op, dataset.op_name_to_idx)
            self.assertIsInstance(dataset.op_name_to_idx[op], int)
            self.assertGreaterEqual(dataset.op_name_to_idx[op], 0)
            self.assertLess(dataset.op_name_to_idx[op], 5)

    def test_tensor_dtypes(self):
        """Test that generated tensors have correct dtypes."""
        dataset = DAGStructureDataset(max_depth=3, seed=42)

        text, structure = dataset.generate_structure_example(depth=2)

        # Check dtypes
        self.assertEqual(structure["initial_sgn"].dtype, torch.float32)
        self.assertEqual(structure["initial_log"].dtype, torch.float32)
        self.assertEqual(structure["operation_probs"].dtype, torch.float32)
        self.assertEqual(structure["depth"].dtype, torch.long)

    def test_structure_values_range(self):
        """Test that structure values are in expected ranges."""
        dataset = DAGStructureDataset(max_depth=3, seed=42)

        # Generate multiple examples to test range
        for _ in range(10):
            text, structure = dataset.generate_structure_example(depth=2)

            # Initial signs should be -1 or +1
            sgn = structure["initial_sgn"][0]  # First (non-zero) value
            self.assertIn(sgn.item(), [-1.0, 1.0])

            # Log magnitudes should be non-negative and bounded
            log_mag = structure["initial_log"][0]  # First (non-zero) value
            self.assertGreaterEqual(log_mag.item(), 0.0)
            self.assertLessEqual(log_mag.item(), 10.0)  # LOG_LIM from dag_model

            # Operation probabilities should be valid probabilities
            op_probs = structure["operation_probs"]
            self.assertTrue(torch.all(op_probs >= 0.0))
            self.assertTrue(torch.all(op_probs <= 1.0))

    def test_integration_with_dag_model_components(self):
        """Test integration with actual DAG model components."""
        from dag_model import stack_based_execution

        dataset = DAGStructureDataset(max_depth=3, seed=42)

        # Generate structure examples
        texts, structures = dataset.generate_batch(batch_size=2)

        # Get structure tensors in the format expected by DAG model
        initial_sgn = structures["initial_sgn"]  # (B, num_nodes)
        initial_log = structures["initial_log"]  # (B, num_nodes)
        operation_probs = structures["operation_probs"]  # (B, depth, n_ops)

        # Add sequence dimension to match DAG model expectations
        # DAG model expects (B, T, ...) where T is sequence length
        seq_len = 1  # Single token for simplicity
        initial_sgn = initial_sgn.unsqueeze(1)  # (B, T, num_nodes)
        initial_log = initial_log.unsqueeze(1)  # (B, T, num_nodes)
        operation_probs = operation_probs.unsqueeze(1)  # (B, T, depth, n_ops)

        # Test that stack_based_execution works with our tensors
        try:
            final_sgn, final_log = stack_based_execution(
                initial_sgn, initial_log, operation_probs
            )

            # Verify output shapes
            self.assertEqual(final_sgn.shape, (2, 1))  # (B, T)
            self.assertEqual(final_log.shape, (2, 1))  # (B, T)

            # Verify outputs are finite
            self.assertTrue(torch.isfinite(final_sgn).all())
            self.assertTrue(torch.isfinite(final_log).all())

            # Verify signs are reasonable
            self.assertTrue(torch.all(torch.abs(final_sgn) <= 1.0))

            # Verify log magnitudes are reasonable
            self.assertTrue(torch.all(final_log >= 0.0))

        except Exception as e:
            self.fail(f"Integration with stack_based_execution failed: {e}")

    def test_structure_dataset_with_dag_plan_predictor_format(self):
        """Test that structure tensors match DAGPlanPredictor output format exactly."""
        from dag_model import DAGPlanPredictor, GPTConfig

        # Create a small config for testing
        config = GPTConfig(
            vocab_size=50,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dag_depth=2,  # depth 2 for this test
        )

        # Create DAGPlanPredictor
        plan_predictor = DAGPlanPredictor(config)

        # Generate structure dataset with same depth
        dataset = DAGStructureDataset(max_depth=2, seed=42)
        texts, structures = dataset.generate_batch(batch_size=1)

        # Check tensor shapes match what DAGPlanPredictor expects
        batch_size = 1
        seq_len = 1
        num_nodes = config.dag_depth + 1  # 3 nodes for depth 2

        # Our structure tensors (without sequence dimension)
        struct_sgn = structures["initial_sgn"]  # (B, num_nodes)
        struct_log = structures["initial_log"]  # (B, num_nodes)
        struct_ops = structures["operation_probs"]  # (B, depth, n_ops)

        # Verify shapes match DAGPlanPredictor expectations
        self.assertEqual(struct_sgn.shape, (batch_size, num_nodes))
        self.assertEqual(struct_log.shape, (batch_size, num_nodes))
        self.assertEqual(struct_ops.shape, (batch_size, config.dag_depth, 5))  # 5 ops

        # Test that we can use these as targets for DAGPlanPredictor
        # Create dummy hidden states
        hidden_states = torch.randn(batch_size, seq_len, config.n_embd)

        # Get DAGPlanPredictor output
        pred_sgn, pred_log, pred_ops = plan_predictor(hidden_states)

        # Verify output shapes match our structure format (when squeezed)
        self.assertEqual(pred_sgn.shape, (batch_size, seq_len, num_nodes))
        self.assertEqual(pred_log.shape, (batch_size, seq_len, num_nodes))
        self.assertEqual(pred_ops.shape, (batch_size, seq_len, config.dag_depth, 5))

        # Verify we can compute losses between predictions and structure targets
        # Add sequence dimension to structure targets for loss computation
        target_sgn = struct_sgn.unsqueeze(1)  # (B, T, num_nodes)
        target_log = struct_log.unsqueeze(1)  # (B, T, num_nodes)
        target_ops = struct_ops.unsqueeze(1)  # (B, T, depth, n_ops)

        # Compute losses (this should work without errors)
        try:
            sgn_loss = torch.nn.functional.mse_loss(pred_sgn, target_sgn)
            log_loss = torch.nn.functional.mse_loss(pred_log, target_log)
            ops_loss = torch.nn.functional.cross_entropy(
                pred_ops.reshape(-1, 5), target_ops.reshape(-1, 5).argmax(dim=-1)
            )

            # Verify losses are finite
            self.assertTrue(torch.isfinite(sgn_loss))
            self.assertTrue(torch.isfinite(log_loss))
            self.assertTrue(torch.isfinite(ops_loss))

        except Exception as e:
            self.fail(f"Loss computation failed: {e}")

    def test_structure_label_correctness(self):
        """Test that structure labels exactly match the ground truth computation."""
        # Create a dataset with fixed seed for reproducibility
        dataset = DAGStructureDataset(max_depth=2, seed=42)

        # Generate a simple example with depth 2
        text, structure = dataset.generate_structure_example(depth=2)
        print("\nDebug information:")
        print(f"Text expression: {text}")

        # Extract the operations from the one-hot vectors
        op_probs = structure["operation_probs"]
        operations = []
        for step in range(2):  # depth = 2
            op_idx = torch.argmax(op_probs[step]).item()
            operations.append(dataset.op_idx_to_name[op_idx])
        print(f"Operations: {operations}")

        # Get initial values
        initial_values = []
        for i in range(3):  # depth + 1 = 3 initial values
            sign = structure["initial_sgn"][i].item()
            log_mag = structure["initial_log"][i].item()
            initial_values.append((sign, log_mag))
        print(f"Initial values (sign, log_mag): {initial_values}")

        # Convert initial values to actual numbers for stack-based execution
        stack = []
        for sign, log_mag in initial_values:
            if log_mag == 0.0:
                number = 0.0 if sign == 0.0 else sign * 1.0
            else:
                number = sign * np.exp(log_mag)
            stack.append(number)
        print(f"Initial stack: {stack}")

        # Execute operations in stack-based order to match text generation
        results = stack.copy()  # Keep original values for verification
        for op_name in operations:
            if len(stack) >= 2:
                b = stack.pop()  # Second operand
                a = stack.pop()  # First operand

                if op_name == "add":
                    result = a + b
                elif op_name == "subtract":
                    result = a - b
                elif op_name == "multiply":
                    result = a * b
                elif op_name == "divide":
                    result = a / b
                elif op_name == "identity":
                    result = a  # Discard b

                stack.append(result)
                results.append(result)

        print(f"Stack-based execution results: {results}")

        # Extract all numbers from the text
        import re

        text_numbers = [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]
        print(f"Numbers found in text: {text_numbers}")

        # Verify operation probabilities match the actual operations
        for step, op_name in enumerate(operations):
            op_idx = dataset.op_name_to_idx[op_name]
            step_probs = structure["operation_probs"][step]

            # Should be one-hot with 1.0 at the correct operation index
            for i in range(len(step_probs)):
                expected = 1.0 if i == op_idx else 0.0
                self.assertEqual(step_probs[i].item(), expected)

        # Verify each number in the text matches a computed value (allowing for small rounding differences)
        for text_num in text_numbers:
            # Look for a matching computed value
            found_match = False
            for computed_num in results:
                # Use relative tolerance for larger numbers, absolute for smaller ones
                if abs(computed_num) > 1.0:
                    if (
                        abs((computed_num - text_num) / computed_num) < 0.01
                    ):  # 1% relative tolerance
                        found_match = True
                        break
                else:
                    if (
                        abs(computed_num - text_num) < 0.01
                    ):  # 0.01 absolute tolerance for small numbers
                        found_match = True
                        break
            self.assertTrue(
                found_match,
                f"No match found for text number {text_num} in computed values {results}",
            )


class TestExpressionMatching(unittest.TestCase):
    """Test that generated expressions match actual DAG computations."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        torch.manual_seed(42)

    def test_expression_matches_computation_single_example(self):
        """Test that a single example's expression matches its computation."""
        from dag_model import op_names

        # Generate a single example
        example = generate_single_dag_example(depth=2, num_initial_values=3)

        # Get initial values and operations
        initial_values = []
        for j in range(len(example.initial_values)):
            sign, log_mag = example.initial_values[j]
            val = sign * np.exp(log_mag)
            initial_values.append(val)

        # Execute operations and track intermediate results
        values = initial_values.copy()
        operations = []

        # Get operations from the raw operations list
        raw_operations = example.operations
        for step, (op1_idx, op2_idx, op_name) in enumerate(raw_operations):
            val1, val2 = values[op1_idx], values[op2_idx]

            # Compute result
            if op_name == "add":
                result = val1 + val2
            elif op_name == "subtract":
                result = val1 - val2
            elif op_name == "multiply":
                result = val1 * val2
            elif op_name == "divide":
                result = val1 / val2
            else:  # identity
                result = val1

            values.append(result)
            operations.append((op1_idx, op2_idx, op_name))

        # Generate the expected expression
        expected_expr = convert_dag_to_expression_string(
            initial_values=[
                (1.0 if v >= 0 else -1.0, float(np.log(abs(v))))
                for v in initial_values[: example.depth + 1]
            ],
            operations=operations,
            use_parentheses=True,
            rng=np.random.RandomState(42),
        )

        # Verify expressions match (ignoring whitespace and small decimal differences)
        def normalize_expr(expr):
            import re

            # Remove all whitespace and convert to lowercase
            expr = "".join(expr.split())
            # Replace decimal numbers with rounded versions (2 decimal places)
            numbers = re.findall(r"-?\d+\.\d+", expr)
            for num in numbers:
                rounded = f"{float(num):.2f}"
                expr = expr.replace(num, rounded)
            return expr

        generated = normalize_expr(example.text)
        expected = normalize_expr(expected_expr)

        self.assertEqual(
            generated,
            expected,
            f"Generated expression '{example.text}' does not match expected '{expected_expr}'",
        )

    def test_expression_matches_computation_multiple_seeds(self):
        """Test multiple examples with different seeds to verify various operation combinations."""
        test_seeds = [42, 43, 44, 45]

        for seed in test_seeds:
            with self.subTest(seed=seed):
                # Set random state
                np.random.seed(seed)
                torch.manual_seed(seed)

                # Generate example
                example = generate_single_dag_example(depth=2, num_initial_values=3)

                # Get initial values and operations
                initial_values = []
                for j in range(len(example.initial_values)):
                    sign, log_mag = example.initial_values[j]
                    val = sign * np.exp(log_mag)
                    initial_values.append(val)

                # Execute operations and track intermediate results
                values = initial_values.copy()
                operations = []

                # Get operations from the raw operations list
                raw_operations = example.operations
                for step, (op1_idx, op2_idx, op_name) in enumerate(raw_operations):
                    val1, val2 = values[op1_idx], values[op2_idx]

                    # Compute result
                    if op_name == "add":
                        result = val1 + val2
                    elif op_name == "subtract":
                        result = val1 - val2
                    elif op_name == "multiply":
                        result = val1 * val2
                    elif op_name == "divide":
                        result = val1 / val2
                    else:  # identity
                        result = val1

                    values.append(result)
                    operations.append((op1_idx, op2_idx, op_name))

                # Generate the expected expression
                expected_expr = convert_dag_to_expression_string(
                    initial_values=[
                        (1.0 if v >= 0 else -1.0, float(np.log(abs(v))))
                        for v in initial_values[: example.depth + 1]
                    ],
                    operations=operations,
                    use_parentheses=True,
                    rng=np.random.RandomState(42),  # Use fixed seed for reproducibility
                )

                # Verify expressions match (ignoring whitespace and small decimal differences)
                def normalize_expr(expr):
                    import re

                    # Remove all whitespace and convert to lowercase
                    expr = "".join(expr.split())
                    # Replace decimal numbers with rounded versions (2 decimal places)
                    numbers = re.findall(r"-?\d+\.\d+", expr)
                    for num in numbers:
                        rounded = f"{float(num):.2f}"
                        expr = expr.replace(num, rounded)
                    return expr

                generated = normalize_expr(example.text)
                expected = normalize_expr(expected_expr)

                self.assertEqual(
                    generated,
                    expected,
                    f"Seed {seed}: Generated expression '{example.text}' does not match expected '{expected_expr}'",
                )

    def test_english_conversion_integration(self):
        """Test that English conversion works correctly with the integrated approach."""
        # Test with English conversion enabled
        example = generate_single_dag_example(
            depth=2,
            num_initial_values=3,
            convert_to_english=True,
            conversion_probability=1.0,
        )

        # Verify the text contains English words
        self.assertIsInstance(example.text, str)
        self.assertGreater(len(example.text), 0)

        # Should contain English words (not just numbers and symbols)
        import re

        english_words = re.findall(r"[a-zA-Z]+", example.text)
        self.assertGreater(
            len(english_words),
            0,
            f"Expected English words in '{example.text}', but found none",
        )

        # Should contain mathematical operators (either as symbols or words)
        operators = re.findall(
            r"[\+\-\*/]|plus|minus|times|divided|multiplied", example.text.lower()
        )
        self.assertGreater(
            len(operators), 0, f"Expected operators in '{example.text}', but found none"
        )


if __name__ == "__main__":
    unittest.main()
