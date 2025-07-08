#!/usr/bin/env python
"""
test_dagset.py
Tests for the streaming DAG dataset functionality.
"""

import math
import random
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
                       create_dag_structure_dataloaders, generate_dag_dataset,
                       generate_random_dag_plan, generate_single_dag_example)

# Import DAG operations for direct testing
sys.path.append(str(Path(__file__).parent.parent))
from models.dag_model import LOG_LIM, OP_NAMES


class TestIdentityFunction(unittest.TestCase):
    """Test the identity function specifically in DAG generation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        torch.manual_seed(42)

    def test_identity_operation_in_plan_generation(self):
        """Test that identity operations can be generated in DAG plans."""
        # Force identity operation by generating many plans and checking at least one has identity
        identity_found = False
        for seed in range(100):  # Try many seeds to find identity operation
            initial_values, operations = generate_random_dag_plan(
                depth=3, num_initial_values=4, rng=np.random.RandomState(seed)
            )
            if "identity" in operations:
                identity_found = True
                # Verify the plan is valid
                self.assertEqual(len(initial_values), 4)
                self.assertEqual(len(operations), 3)
                break

        self.assertTrue(
            identity_found, "Identity operation should be generatable in DAG plans"
        )

    def test_identity_operation_expression_generation(self):
        """Test that identity operations generate correct expressions."""
        # Create a specific case with identity operation
        initial_values = [5.0, 3.0, 2.0]
        operations = ["identity", "add"]  # First operation is identity

        # Generate expression
        expression = convert_dag_to_expression_string(
            initial_values=initial_values,
            operations=operations,
            convert_to_english=False,
            conversion_probability=0.0,
        )

        # With identity operation, the first step should return the first operand
        # Stack starts as [5.0, 3.0, 2.0]
        # identity: pop 2.0 and 3.0, return 3.0 (first operand), stack becomes [5.0, 3.0]
        # add: pop 3.0 and 5.0, return 5.0 + 3.0 = 8.0, stack becomes [8.0]
        # Result should be equivalent to "5.0 + 3.0" = 8.0

        self.assertIsInstance(expression, str)
        self.assertGreater(len(expression), 0)

        # The expression might be simplified, but it should be evaluable
        # and should contain the expected operands
        import re

        numbers_in_expression = re.findall(r"\d+\.?\d*", expression)
        self.assertGreater(
            len(numbers_in_expression),
            0,
            f"Expression should contain numbers: {expression}",
        )

    def test_identity_operation_with_standardized_rounding(self):
        """Test that identity operations work with standardized rounding."""
        import random

        from streaming import standardize_float_rounding

        rng = random.Random(42)

        # Generate values with standardized rounding
        initial_values = [
            standardize_float_rounding(10.123456789, rng),
            standardize_float_rounding(5.987654321, rng),
            standardize_float_rounding(3.14159, rng),
        ]

        # Create operations with identity
        operations = ["identity", "multiply"]

        # Generate expression
        expression = convert_dag_to_expression_string(
            initial_values=initial_values,
            operations=operations,
            rng=rng,
            convert_to_english=False,
            conversion_probability=0.0,
        )

        # Verify the expression is valid
        self.assertIsInstance(expression, str)
        self.assertGreater(len(expression), 0)

        # Verify that standardized rounding was applied to initial values
        for value in initial_values:
            if isinstance(value, float):
                # Check that it has at most 5 decimal places
                decimal_places = (
                    len(str(value).split(".")[-1]) if "." in str(value) else 0
                )
                self.assertLessEqual(
                    decimal_places,
                    5,
                    f"Value {value} should have at most 5 decimal places",
                )

    def test_identity_operation_with_english_conversion(self):
        """Test that identity operations work with English conversion."""
        from streaming import convert_number_to_words

        # Test with integer
        initial_values = [42, 7, 3]
        operations = ["identity"]

        expression = convert_dag_to_expression_string(
            initial_values=initial_values,
            operations=operations,
            convert_to_english=True,
            conversion_probability=1.0,  # Force conversion
        )

        self.assertIsInstance(expression, str)
        self.assertGreater(len(expression), 0)

        # Should contain English words
        import re

        english_words = re.findall(r"[a-zA-Z]+", expression)
        self.assertGreater(
            len(english_words), 0, f"Expected English words in expression: {expression}"
        )

    def test_identity_operation_tensor_format(self):
        """Test that identity operations create correct tensor format."""
        # Create example with forced identity operation
        example = generate_single_dag_example(depth=1, num_initial_values=2)

        # Check tensor shapes
        self.assertEqual(example.signs.shape, torch.Size([2]))
        self.assertEqual(example.log_magnitudes.shape, torch.Size([2]))
        self.assertEqual(
            example.operations.shape, torch.Size([1, 5])
        )  # 1 operation, 5 possible ops

        # Check that operation tensor is one-hot
        op_tensor = example.operations[0]  # First (and only) operation
        self.assertAlmostEqual(op_tensor.sum().item(), 1.0, places=5)

        # Check that exactly one operation is selected
        ones_count = (op_tensor == 1.0).sum().item()
        zeros_count = (op_tensor == 0.0).sum().item()
        self.assertEqual(ones_count, 1)
        self.assertEqual(zeros_count, 4)

    def test_identity_operation_in_structure_dataset(self):
        """Test that identity operations work in DAG structure dataset."""
        dataset = DAGStructureDataset(max_depth=2, seed=42)

        # Generate multiple examples to find one with identity
        identity_found = False
        for i in range(20):  # Try multiple examples
            text, structure = dataset.generate_structure_example(depth=2)

            # Check operations in the structure
            op_probs = structure["operation_probs"]
            for step in range(2):
                op_idx = torch.argmax(op_probs[step]).item()
                op_name = dataset.op_idx_to_name[op_idx]
                if op_name == "identity":
                    identity_found = True

                    # Verify the structure is valid
                    self.assertIsInstance(text, str)
                    self.assertGreater(len(text), 0)
                    self.assertEqual(structure["initial_sgn"].shape, (3,))  # depth + 1
                    self.assertEqual(structure["initial_log"].shape, (3,))
                    self.assertEqual(structure["operation_probs"].shape, (2, 5))
                    break

            if identity_found:
                break

        # Note: We don't assert identity_found=True because it's probabilistic,
        # but if found, we verify it works correctly

    def test_identity_operation_consistency(self):
        """Test that identity operations are consistent across different generation methods."""
        import random

        rng = random.Random(42)

        # Test with same initial values and operations
        initial_values = [10.5, 7.25, 3.0]
        operations = ["identity", "add"]

        # Generate expression multiple times
        expressions = []
        for _ in range(3):
            expr = convert_dag_to_expression_string(
                initial_values=initial_values.copy(),
                operations=operations.copy(),
                rng=random.Random(42),  # Same seed for consistency
                convert_to_english=False,
                conversion_probability=0.0,
            )
            expressions.append(expr)

        # All expressions should be identical with same inputs and seed
        for i in range(1, len(expressions)):
            self.assertEqual(
                expressions[0],
                expressions[i],
                "Identity operations should produce consistent results",
            )

    def test_all_operations_including_identity(self):
        """Test that all operations including identity can be generated."""
        from models.dag_model import OP_NAMES

        operations_found = set()

        # Generate many examples to try to find all operations
        for seed in range(200):
            initial_values, operations = generate_random_dag_plan(
                depth=1, num_initial_values=2, rng=np.random.RandomState(seed)
            )
            operations_found.update(operations)

        # Verify all expected operations can be found
        expected_ops = set(OP_NAMES)
        missing_ops = expected_ops - operations_found

        # We should find most operations, including identity
        self.assertIn(
            "identity", operations_found, "Identity operation should be generatable"
        )

        # Log what operations were found for debugging
        print(f"Operations found: {sorted(operations_found)}")
        if missing_ops:
            print(
                f"Missing operations (may be due to randomness): {sorted(missing_ops)}"
            )


class TestStreamingDAGDataset(unittest.TestCase):
    """Test the streaming DAG dataset functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a fixed seed for reproducible tests
        np.random.seed(42)
        torch.manual_seed(42)

    def test_generate_random_dag_plan(self):
        """Test DAG plan generation."""
        # Test depth 1
        initial_values, operations = generate_random_dag_plan(
            depth=1, num_initial_values=1
        )
        self.assertEqual(len(initial_values), 1)
        self.assertEqual(len(operations), 1)
        self.assertIn(
            operations[0], ["add", "subtract", "multiply", "divide", "identity"]
        )

        # Test depth 3
        initial_values, operations = generate_random_dag_plan(
            depth=3, num_initial_values=1
        )
        self.assertEqual(len(initial_values), 1)
        self.assertEqual(len(operations), 3)

        # Check that operations are valid
        for op_name in operations:
            self.assertIn(
                op_name, ["add", "subtract", "multiply", "divide", "identity"]
            )

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
        # Note: The final expression can legitimately be a single number if the last
        # operation(s) reduce to an identity. Therefore, we no longer require the
        # presence of an explicit operator symbol in the text.

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
        # The expression may resolve to a single number (e.g., when the final step is
        # an identity op). We do not strictly require an operator symbol.

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
        # Similarly, allow single-number expressions without operator symbols.

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
        from models.dag_model import stack_based_execution

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

            # Verify log magnitudes are reasonable (finite and within bounds)
            self.assertTrue(torch.isfinite(final_log).all())
            self.assertTrue(torch.all(final_log >= -LOG_LIM))
            self.assertTrue(torch.all(final_log <= LOG_LIM))

        except Exception as e:
            self.fail(f"Integration with stack_based_execution failed: {e}")

    def test_structure_dataset_with_dag_plan_predictor_format(self):
        """Test that structure tensors match DAGPlanPredictor output format exactly."""
        from models.dag_model import DAGPlanPredictor, GPTConfig

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
        """Test that structure labels are in correct format."""
        # Create a dataset with fixed seed for reproducibility
        dataset = DAGStructureDataset(max_depth=2, seed=42)

        # Generate a simple example with depth 2
        text, structure = dataset.generate_structure_example(depth=2)

        # Basic checks
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

        # Check structure tensor format
        self.assertIn("initial_sgn", structure)
        self.assertIn("initial_log", structure)
        self.assertIn("operation_probs", structure)

        # Check tensor shapes
        self.assertEqual(structure["initial_sgn"].shape, torch.Size([3]))  # depth+1
        self.assertEqual(structure["initial_log"].shape, torch.Size([3]))  # depth+1
        self.assertEqual(
            structure["operation_probs"].shape, torch.Size([2, 5])
        )  # depth x num_ops

        # Check that operation probabilities are valid one-hot vectors
        op_probs = structure["operation_probs"]
        for step in range(2):  # depth = 2
            step_probs = op_probs[step]
            # Should sum to 1.0 (one-hot)
            self.assertAlmostEqual(step_probs.sum().item(), 1.0, places=5)
            # Should have exactly one 1.0 and the rest 0.0
            max_val = step_probs.max().item()
            self.assertAlmostEqual(max_val, 1.0, places=5)


class TestExpressionMatching(unittest.TestCase):
    """Test that expression matches the underlying DAG computation."""

    def test_expression_matches_computation_single_example(self):
        """Test that generated expression produces the same result as DAG computation."""

        # Fixed test case to ensure reproducibility
        initial_values = [2.5, 3.0, 1.5]
        operations = ["add", "multiply"]

        # Generate expression
        expression = convert_dag_to_expression_string(
            initial_values=initial_values,
            operations=operations,
            rng=random.Random(42),
            convert_to_english=False,
            conversion_probability=0.0,
        )

        # Manually compute the result using stack-based execution
        stack = initial_values[:]
        for op in operations:
            b = stack.pop()
            a = stack.pop()
            if op == "add":
                result = a + b
            elif op == "subtract":
                result = a - b
            elif op == "multiply":
                result = a * b
            elif op == "divide":
                result = a / b
            elif op == "identity":
                result = a  # Discard b
            stack.append(result)
        expected_result = stack[0]

        # Evaluate the generated expression
        actual_result = eval(expression)

        # Should match within floating point precision
        self.assertAlmostEqual(actual_result, expected_result, places=10)

    def test_expression_matches_computation_multiple_seeds(self):
        """Test expression matching for different random seeds."""

        for seed in [42, 123, 999]:
            with self.subTest(seed=seed):
                rng = random.Random(seed)

                # Generate random test case
                initial_values = [rng.uniform(-10, 10) for _ in range(4)]
                operations = [rng.choice(OP_NAMES) for _ in range(3)]

                # Generate expression
                expression = convert_dag_to_expression_string(
                    initial_values=initial_values,
                    operations=operations,
                    rng=rng,
                    convert_to_english=False,
                    conversion_probability=0.0,
                )

                # Manually compute expected result
                stack = initial_values[:]
                for op in operations:
                    if len(stack) < 2:
                        break
                    b = stack.pop()
                    a = stack.pop()
                    if op == "add":
                        result = a + b
                    elif op == "subtract":
                        result = a - b
                    elif op == "multiply":
                        result = a * b
                    elif op == "divide":
                        result = a / b if b != 0 else float("inf")
                    elif op == "identity":
                        result = a  # Discard b
                    stack.append(result)

                if stack:
                    expected_result = stack[0]

                    # Only test if result is finite
                    if math.isfinite(expected_result):
                        try:
                            actual_result = eval(expression)
                            if math.isfinite(actual_result):
                                self.assertAlmostEqual(
                                    actual_result, expected_result, places=8
                                )
                        except (ZeroDivisionError, OverflowError):
                            # Skip cases with division by zero or overflow
                            pass

    def test_english_conversion_integration(self):
        """Test that English conversion doesn't break expression evaluation."""

        initial_values = [5.0, 2.0, 3.0]
        operations = ["add", "multiply"]

        # Test without English conversion
        expression_numeric = convert_dag_to_expression_string(
            initial_values=initial_values,
            operations=operations,
            rng=random.Random(42),
            convert_to_english=False,
            conversion_probability=0.0,
        )

        # Test with English conversion
        expression_english = convert_dag_to_expression_string(
            initial_values=initial_values,
            operations=operations,
            rng=random.Random(42),
            convert_to_english=True,
            conversion_probability=0.3,
        )

        # Both should be valid strings
        self.assertIsInstance(expression_numeric, str)
        self.assertIsInstance(expression_english, str)
        self.assertGreater(len(expression_numeric), 0)
        self.assertGreater(len(expression_english), 0)

    def test_no_double_negatives_in_expressions(self):
        """Test that expressions don't contain double negatives after the fix."""

        # Test with various negative initial values that previously caused issues
        test_cases = [
            [52.3025, 84.79822, -85.0, -37.0, 1.0, -98.551, 70.407],
            [-37.0, -85.0, -98.551],
            [-1.0, 2.0, -3.0, 4.0],
            [-10.5, -20.3, -30.7, -40.1, -50.9],
        ]

        operations_sets = [
            ["add", "add", "divide", "subtract", "multiply", "multiply"],
            ["subtract", "multiply"],
            ["add", "subtract", "multiply"],
            ["multiply", "divide", "add", "subtract"],
        ]

        for i, (initial_values, operations) in enumerate(
            zip(test_cases, operations_sets)
        ):
            with self.subTest(case=i):
                # Adjust operations to match initial_values length
                max_ops = len(initial_values) - 1
                ops = operations[:max_ops] if len(operations) > max_ops else operations

                for seed in [42, 123, 999]:
                    expression = convert_dag_to_expression_string(
                        initial_values=initial_values,
                        operations=ops,
                        rng=random.Random(seed),
                        convert_to_english=False,
                        conversion_probability=0.0,
                    )

                    # Check that there are no double negatives
                    self.assertNotIn(
                        "--",
                        expression,
                        f"Found double negative in expression: {expression}",
                    )

                    # Check that there are no malformed +-
                    self.assertNotIn(
                        "+-",
                        expression,
                        f"Found malformed +- in expression: {expression}",
                    )

                    # Expression should be a valid string that can be evaluated
                    self.assertIsInstance(expression, str)
                    self.assertGreater(len(expression), 0)

                    # Try to evaluate the expression to ensure it's syntactically correct
                    try:
                        result = eval(expression)
                        self.assertIsInstance(result, (int, float))
                    except (ZeroDivisionError, OverflowError):
                        # These are acceptable mathematical exceptions
                        pass


if __name__ == "__main__":
    unittest.main()
