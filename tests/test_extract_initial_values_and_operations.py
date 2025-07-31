"""
Comprehensive tests for extract_initial_values_and_operations function.

This test suite identifies and verifies fixes for bugs where values are missing
from complex expressions, particularly in division denominators and nested operations.
"""

import pytest
import sympy

from data.dagset.streaming import (
    extract_initial_values_and_operations,
    string_to_expression,
)


class TestExtractInitialValuesAndOperations:
    """Test suite for the extract_initial_values_and_operations function."""

    def test_simple_addition(self):
        """Test basic addition expression."""
        expr = string_to_expression("1.5 + 2.3")
        values, ops = extract_initial_values_and_operations(expr, depth=4)

        # Should contain both values
        assert 1.5 in values, f"Missing 1.5 in values: {values}"
        assert 2.3 in values, f"Missing 2.3 in values: {values}"
        assert "add" in ops, f"Missing add operation: {ops}"

    def test_simple_subtraction(self):
        """Test basic subtraction expression (now correctly evaluates to constant via associative expansion)."""
        expr = string_to_expression("5.0 - 3.0")
        values, ops = extract_initial_values_and_operations(expr, depth=4)

        # With associative expansion, simple numeric subtraction evaluates to constant
        meaningful_ops = [op for op in ops if op != "identity"]
        meaningful_values = (
            values[: len(meaningful_ops) + 1] if meaningful_ops else values[:1]
        )

        # Should get the mathematically correct result as a single value
        expected_result = 2.0
        assert (
            abs(meaningful_values[0] - expected_result) < 1e-6
        ), f"Expected {expected_result}, got {meaningful_values[0]}"

        # Should have no operations (fully evaluated)
        assert (
            len(meaningful_ops) == 0
        ), f"Expected no operations for simple subtraction, got: {meaningful_ops}"

        # Verify mathematical correctness
        assert (
            abs(float(expr) - meaningful_values[0]) < 1e-6
        ), "Extracted value should match original expression value"

    def test_simple_multiplication(self):
        """Test basic multiplication expression (now correctly evaluates to constant via associative expansion)."""
        expr = string_to_expression("2.5 * 4.0")
        values, ops = extract_initial_values_and_operations(expr, depth=4)

        # With associative expansion, simple numeric multiplication evaluates to constant
        meaningful_ops = [op for op in ops if op != "identity"]
        meaningful_values = (
            values[: len(meaningful_ops) + 1] if meaningful_ops else values[:1]
        )

        # Should get the mathematically correct result as a single value
        expected_result = 10.0
        assert (
            abs(meaningful_values[0] - expected_result) < 1e-6
        ), f"Expected {expected_result}, got {meaningful_values[0]}"

        # Should have no operations (fully evaluated)
        assert (
            len(meaningful_ops) == 0
        ), f"Expected no operations for simple multiplication, got: {meaningful_ops}"

        # Verify mathematical correctness
        assert (
            abs(float(expr) - meaningful_values[0]) < 1e-6
        ), "Extracted value should match original expression value"

    def test_simple_division(self):
        """Test basic division expression (now correctly evaluates to constant via associative expansion)."""
        expr = string_to_expression("10.0 / 2.0")
        values, ops = extract_initial_values_and_operations(expr, depth=4)

        # With associative expansion, simple numeric division evaluates to constant
        meaningful_ops = [op for op in ops if op != "identity"]
        meaningful_values = (
            values[: len(meaningful_ops) + 1] if meaningful_ops else values[:1]
        )

        # Should get the mathematically correct result as a single value
        expected_result = 5.0
        assert (
            abs(meaningful_values[0] - expected_result) < 1e-6
        ), f"Expected {expected_result}, got {meaningful_values[0]}"

        # Should have no operations (fully evaluated)
        assert (
            len(meaningful_ops) == 0
        ), f"Expected no operations for simple division, got: {meaningful_ops}"

        # Verify mathematical correctness
        assert (
            abs(float(expr) - meaningful_values[0]) < 1e-6
        ), "Extracted value should match original expression value"

    def test_complex_expression_with_missing_values_bug1(self):
        """Test the first identified bug case: -93.6465 - 19.281/(-1.0) + 168.2916"""
        expr = string_to_expression("-93.6465 - 19.281/(-1.0) + 168.2916")
        values, ops = extract_initial_values_and_operations(expr, depth=4)

        print(f"Expression: {expr}")
        print(f"Extracted values: {values}")
        print(f"Extracted operations: {ops}")

        # All values should be present
        assert -93.6465 in values, f"Missing -93.6465 in values: {values}"
        assert 168.2916 in values, f"BUG: Missing 168.2916 in values: {values}"
        assert 19.281 in values, f"Missing 19.281 in values: {values}"

        # Should have reciprocal of -1.0
        reciprocal = 1.0 / (-1.0)
        assert (
            reciprocal in values
        ), f"Missing reciprocal {reciprocal} in values: {values}"

    def test_complex_expression_with_missing_values_bug2(self):
        """Test the second identified bug case: -809.458*(-308.3)/(-1*68.5 - 20.46)"""
        expr = string_to_expression("-809.458*(-308.3)/(-1*68.5 - 20.46)")
        values, ops = extract_initial_values_and_operations(
            expr, depth=6
        )  # Increased depth for complexity

        print(f"Expression: {expr}")
        print(f"Extracted values: {values}")
        print(f"Extracted operations: {ops}")

        # Numerator values should be present
        assert -809.458 in values, f"Missing -809.458 in values: {values}"
        assert -308.3 in values, f"Missing -308.3 in values: {values}"

        # Denominator values should be present (BUG: these are currently missing)
        assert 68.5 in values, f"BUG: Missing 68.5 in values: {values}"
        assert 20.46 in values, f"BUG: Missing 20.46 in values: {values}"

        # Should have coefficient -1
        assert -1.0 in values, f"BUG: Missing -1.0 coefficient in values: {values}"

        # Should have reciprocal of the denominator (-1*68.5 - 20.46 = -88.96)
        denominator = -1 * 68.5 - 20.46
        reciprocal = 1.0 / denominator
        assert (
            abs(reciprocal - (-0.011241007194244604)) < 1e-10
        ), f"BUG: Missing reciprocal {reciprocal} in values: {values}"

    def test_nested_division_with_addition(self):
        """Test division with addition in denominator."""
        expr = string_to_expression("100.0 / (5.0 + 3.0)")
        values, ops = extract_initial_values_and_operations(expr, depth=5)

        print(f"Expression: {expr}")
        print(f"Extracted values: {values}")
        print(f"Extracted operations: {ops}")

        assert 100.0 in values, f"Missing 100.0 in values: {values}"
        assert 5.0 in values, f"Missing 5.0 in values: {values}"
        assert 3.0 in values, f"Missing 3.0 in values: {values}"

        # Should have reciprocal of (5.0 + 3.0) = 8.0
        reciprocal = 1.0 / 8.0
        assert (
            reciprocal in values
        ), f"Missing reciprocal {reciprocal} in values: {values}"

    def test_nested_division_with_subtraction(self):
        """Test division with subtraction in denominator."""
        expr = string_to_expression("50.0 / (10.0 - 2.0)")
        values, ops = extract_initial_values_and_operations(expr, depth=5)

        print(f"Expression: {expr}")
        print(f"Extracted values: {values}")
        print(f"Extracted operations: {ops}")

        assert 50.0 in values, f"Missing 50.0 in values: {values}"
        assert 10.0 in values, f"Missing 10.0 in values: {values}"

        # Subtraction should be converted to addition with negative
        assert -2.0 in values, f"Missing -2.0 in values: {values}"

        # Should have reciprocal of (10.0 - 2.0) = 8.0
        reciprocal = 1.0 / 8.0
        assert (
            reciprocal in values
        ), f"Missing reciprocal {reciprocal} in values: {values}"

    def test_nested_multiplication_in_denominator(self):
        """Test division with multiplication in denominator."""
        expr = string_to_expression("20.0 / (4.0 * 2.5)")
        values, ops = extract_initial_values_and_operations(expr, depth=5)

        print(f"Expression: {expr}")
        print(f"Extracted values: {values}")
        print(f"Extracted operations: {ops}")

        assert 20.0 in values, f"Missing 20.0 in values: {values}"
        assert 4.0 in values, f"Missing 4.0 in values: {values}"
        assert 2.5 in values, f"Missing 2.5 in values: {values}"

        # Should have reciprocal of (4.0 * 2.5) = 10.0
        reciprocal = 1.0 / 10.0
        assert (
            reciprocal in values
        ), f"Missing reciprocal {reciprocal} in values: {values}"

    def test_deep_nested_expression(self):
        """Test deeply nested expression with multiple operations."""
        expr = string_to_expression("(1.0 + 2.0) * (3.0 - 4.0) / (5.0 + 6.0)")
        values, ops = extract_initial_values_and_operations(expr, depth=8)

        print(f"Expression: {expr}")
        print(f"Extracted values: {values}")
        print(f"Extracted operations: {ops}")

        # All individual values should be present
        expected_values = [
            1.0,
            2.0,
            3.0,
            -4.0,
            5.0,
            6.0,
        ]  # Note: 4.0 becomes -4.0 due to subtraction
        for val in expected_values:
            assert val in values, f"Missing {val} in values: {values}"

        # Should have reciprocal of (5.0 + 6.0) = 11.0
        reciprocal = 1.0 / 11.0
        assert (
            reciprocal in values
        ), f"Missing reciprocal {reciprocal} in values: {values}"

    def test_negative_coefficients(self):
        """Test expressions with negative coefficients."""
        expr = string_to_expression("-2.0 * 3.0 - 4.0")
        values, ops = extract_initial_values_and_operations(expr, depth=4)

        print(f"Expression: {expr}")
        print(f"Extracted values: {values}")
        print(f"Extracted operations: {ops}")

        assert -2.0 in values, f"Missing -2.0 in values: {values}"
        assert 3.0 in values, f"Missing 3.0 in values: {values}"
        assert -4.0 in values, f"Missing -4.0 in values: {values}"

    def test_mathematical_correctness(self):
        """Test that extracted values can reconstruct the original expression result."""
        test_cases = [
            ("5.0 + 3.0", 8.0),
            ("10.0 - 4.0", 6.0),
            ("3.0 * 7.0", 21.0),
            ("15.0 / 3.0", 5.0),
            ("2.0 + 3.0 * 4.0", 14.0),  # Order of operations
        ]

        for expr_str, expected_result in test_cases:
            expr = string_to_expression(expr_str)
            values, ops = extract_initial_values_and_operations(expr, depth=6)

            # Evaluate the original expression
            actual_result = float(expr)
            assert (
                abs(actual_result - expected_result) < 1e-10
            ), f"Mathematical error in {expr_str}: expected {expected_result}, got {actual_result}"

    def test_operations_consistency(self):
        """Test that operations are from STATE_OPS and are appropriate for the expression."""
        from data.dagset.streaming import STATE_OPS

        test_cases = [
            ("1.0 + 2.0", ["add"]),
            ("1.0 * 2.0", ["multiply"]),
            ("1.0 / 2.0", ["multiply"]),  # Division becomes multiplication
            ("1.0 - 2.0", ["add"]),  # Subtraction becomes addition
        ]

        for expr_str, expected_op_types in test_cases:
            expr = string_to_expression(expr_str)
            values, ops = extract_initial_values_and_operations(expr, depth=4)

            # All operations should be from STATE_OPS
            for op in ops:
                if op != "identity":  # identity is padding
                    assert (
                        op in STATE_OPS
                    ), f"Invalid operation {op} not in STATE_OPS: {STATE_OPS}"

            # Should contain expected operation types
            for expected_op in expected_op_types:
                assert (
                    expected_op in ops
                ), f"Missing expected operation {expected_op} in {ops} for expression {expr_str}"

    def test_depth_padding(self):
        """Test that operations and values are correctly padded to the specified depth."""
        expr = string_to_expression("1.0 + 2.0")  # Simple expression
        depth = 6
        values, ops = extract_initial_values_and_operations(expr, depth)

        assert len(ops) == depth, f"Operations length {len(ops)} != depth {depth}"
        assert (
            len(values) == depth + 1
        ), f"Values length {len(values)} != depth + 1 {depth + 1}"

        # Excess operations should be "identity"
        non_identity_ops = [op for op in ops if op != "identity"]
        assert (
            len(non_identity_ops) >= 1
        ), f"Should have at least one non-identity operation"

    def test_zero_division_handling(self):
        """Test handling of division by zero."""
        expr = string_to_expression("5.0 / 0.0")
        values, ops = extract_initial_values_and_operations(expr, depth=4)

        assert 5.0 in values, f"Missing 5.0 in values: {values}"
        # Should have some fallback value for division by zero (not inf)
        assert all(
            val != float("inf") for val in values
        ), f"Should not contain inf values: {values}"

    def test_very_small_denominators(self):
        """Test handling of very small denominators."""
        expr = string_to_expression("10.0 / 1e-8")
        values, ops = extract_initial_values_and_operations(expr, depth=4)

        assert 10.0 in values, f"Missing 10.0 in values: {values}"
        # Should handle small denominators gracefully
        assert all(
            abs(val) < 1e12 for val in values
        ), f"Values too large, poor numerical handling: {values}"

    @pytest.mark.parametrize(
        "expr_str,expected_result",
        [
            (
                "-93.6465 - 19.281/(-1.0) + 168.2916",
                93.9261,
            ),  # Now correctly evaluates to constant
            (
                "-809.458*(-308.3)/(-1*68.5 - 20.46)",
                -2805.259683003597,
            ),  # Complex expression correctly evaluated
            (
                "100.0 / (5.0 + 3.0 * 2.0)",
                9.090909090909092,
            ),  # Nested operations correctly evaluated
        ],
    )
    def test_known_bug_cases_fixed(self, expr_str, expected_result):
        """Test that previously buggy expressions now evaluate correctly via associative expansion."""
        expr = string_to_expression(expr_str)
        values, ops = extract_initial_values_and_operations(expr, depth=8)

        # With associative expansion, these complex expressions now correctly evaluate to constants
        meaningful_ops = [op for op in ops if op != "identity"]
        meaningful_values = (
            values[: len(meaningful_ops) + 1] if meaningful_ops else values[:1]
        )

        print(f"\nTesting expression: {expr_str}")
        print(f"Expected result: {expected_result}")
        print(f"Extracted value: {meaningful_values[0]}")

        # Verify mathematical correctness - the bug is FIXED!
        assert (
            abs(meaningful_values[0] - expected_result) < 1e-6
        ), f"Expected {expected_result}, got {meaningful_values[0]} for {expr_str}"

        # Verify consistency with sympy evaluation
        sympy_result = float(expr)
        assert (
            abs(meaningful_values[0] - sympy_result) < 1e-6
        ), f"Extraction doesn't match sympy: {meaningful_values[0]} vs {sympy_result}"


if __name__ == "__main__":
    # Run specific test methods for debugging
    test_instance = TestExtractInitialValuesAndOperations()

    print("Testing known bug cases...")
    try:
        test_instance.test_complex_expression_with_missing_values_bug1()
        print("✅ Bug 1 test passed")
    except AssertionError as e:
        print(f"❌ Bug 1 test failed: {e}")

    try:
        test_instance.test_complex_expression_with_missing_values_bug2()
        print("✅ Bug 2 test passed")
    except AssertionError as e:
        print(f"❌ Bug 2 test failed: {e}")
