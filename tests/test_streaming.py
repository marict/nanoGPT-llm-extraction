import random

import pytest
import sympy
import torch

from data.dagset.streaming import (DAGStructureDataset, _apply_sympy_op,
                                   convert_number_to_english,
                                   format_expression_string,
                                   generate_single_dag_example,
                                   number_to_string, plan_to_tensors)
from models.dag_model import OP_NAMES


# -----------------------------------------------------------------------------
# Simple number/formatting helpers
# -----------------------------------------------------------------------------
def test_convert_number_to_english_basic_cases():
    cases = [
        (0, "zero"),
        (5, "five"),
        (-5, "negative five"),
        (3.14, "three point one four"),
    ]
    for value, expected in cases:
        result = convert_number_to_english(value, max_decimal_places=2)
        assert result == expected


def test_number_to_string_integer_handling():
    rng = random.Random(42)
    # With probability 1.0 we should drop the ".0" suffix for integers
    assert number_to_string(10, rng, integer_no_decimal_probability=1.0) == "10"

    rng = random.Random(42)
    # With probability 0.0 we keep the ".0" suffix
    assert number_to_string(10, rng, integer_no_decimal_probability=0.0) == "10.0"


# -----------------------------------------------------------------------------
# Expression formatting & permutation helpers
# -----------------------------------------------------------------------------
def test_format_expression_string_conversion_and_identity():
    expr = "2+2"
    # No conversion → expression should simply gain spaces
    no_conv = format_expression_string(expr, english_conversion_probability=0.0, seed=1)
    assert no_conv == "2 + 2"

    # Full conversion → numbers become words, operator turns into an English token
    conv = format_expression_string(expr, english_conversion_probability=1.0, seed=1)
    assert "two" in conv  # numbers converted to words
    # At least one operator word should appear ("plus" or "added to")
    assert any(op in conv for op in ["plus", "added to"])


# -----------------------------------------------------------------------------
# SymPy operation helper
# -----------------------------------------------------------------------------
def test_apply_sympy_op_correctness():
    a, b = sympy.Integer(3), sympy.Integer(2)
    expected_values = {
        "add": 5,
        "subtract": 1,
        "multiply": 6,
        "divide": sympy.Rational(3, 2),
    }
    for op_name, expected in expected_values.items():
        expr = _apply_sympy_op(op_name, a, b)
        assert sympy.simplify(expr - expected) == 0

    # Identity should raise error rather than wrap the operand
    with pytest.raises(ValueError):
        _apply_sympy_op("identity", a, b)

    # Unknown operation should also raise
    with pytest.raises(ValueError):
        _apply_sympy_op("unknown", a, b)


# -----------------------------------------------------------------------------
# Tensor helpers & dataset generation
# -----------------------------------------------------------------------------


def test_float_to_digit_onehot_shape_and_properties():
    from data.dagset.streaming import \
        float_to_digit_onehot  # local import to avoid circular issues

    tensor = float_to_digit_onehot(12.34, max_digits=2, max_decimal_places=2)
    # Expected shape: D = max_digits + max_decimal_places = 4
    assert tensor.shape == (4, 10)
    # Each row should be a valid one-hot (sum equals 1)
    assert torch.allclose(tensor.sum(dim=1), torch.ones(4))


def test_plan_to_tensors_shapes():
    init_vals = [1.2, -3.4]
    operations = ["add"]
    structure_dict = plan_to_tensors(
        init_vals, operations, max_digits=2, max_decimal_places=2
    )

    # Check that we get the expected structure dict keys
    expected_keys = {
        "initial_sgn",
        "initial_log",
        "initial_digits",
        "operation_probs",
        "depth",
        "operations",
        "final_value_exec",
    }
    assert set(structure_dict.keys()) == expected_keys

    # Check tensor shapes - with depth=1 and 2 nodes, we get 2 nodes total
    assert structure_dict["initial_sgn"].shape == (2,)  # depth + 1 = 2 nodes
    assert structure_dict["initial_digits"].shape == (
        2,
        4,
        10,
    )  # (nodes, digits, classes)
    assert structure_dict["operation_probs"].shape == (1, len(OP_NAMES))  # (depth, ops)

    # Check signs match original values
    assert torch.equal(structure_dict["initial_sgn"][:2], torch.tensor([1.0, -1.0]))
    # Operation one-hot should have exactly one "1" at the correct index
    assert structure_dict["operation_probs"].sum() == 1
    assert structure_dict["operation_probs"][0, OP_NAMES.index("add")].item() == 1


def test_generate_single_dag_example_basic_properties():
    example = generate_single_dag_example(
        depth=3, seed=999, max_digits=2, max_decimal_places=2
    )
    assert example.depth == 3
    assert len(example.initial_values) == 4  # depth + 1
    # Operations tensor first dimension should equal depth
    assert example.operations.shape[0] == example.depth


def test_dag_structure_dataset_batch_shapes():
    dataset = DAGStructureDataset(
        max_depth=2, seed=321, max_digits=2, max_decimal_places=2
    )
    texts, batched_struct, _ = dataset.generate_batch(batch_size=2)

    # Basic sanity checks
    assert len(texts) == 2
    assert batched_struct["initial_sgn"].shape == (2, 3)  # batch, nodes(depth+1)
    assert batched_struct["operation_probs"].shape == (2, 2, len(OP_NAMES))


def test_batched_target_values_consistency():
    """Test that batched target values match the values from examples."""
    from data.dagset.streaming import create_dag_structure_dataloaders

    # Create a small test dataset
    train_loader, _ = create_dag_structure_dataloaders(
        train_batch_size=3,
        val_batch_size=3,
        max_depth=2,
        seed=42,
        max_digits=3,
        max_decimal_places=2,
    )

    # Get a batch
    texts, structures, examples = next(iter(train_loader))

    # Verify that batched target values match example values
    batched_initial_values = structures["target_initial_values"]
    batched_final_exec = structures["target_final_exec"]

    for i, example in enumerate(examples):
        # Check initial values
        num_values = min(len(example.initial_values), batched_initial_values.size(1))
        for j in range(num_values):
            expected = example.initial_values[j]
            actual = batched_initial_values[i, j].item()
            assert (
                abs(expected - actual) < 1e-5
            ), f"Initial value mismatch at [{i},{j}]: {expected} vs {actual}"

        # Check final execution value
        if example.final_value_exec is not None:
            expected_final = example.final_value_exec
            actual_final = batched_final_exec[i].item()
            assert (
                abs(expected_final - actual_final) < 1e-5
            ), f"Final exec value mismatch at [{i}]: {expected_final} vs {actual_final}"
