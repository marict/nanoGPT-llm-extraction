"""
Test mathematical correctness of extract_initial_values_and_operations.

This test verifies that reconstructing a sympy expression from extracted values
and operations yields the same result as evaluating the original expression.
"""

import pytest
import sympy

from data.dagset.streaming import (
    extract_initial_values_and_operations,
    string_to_expression,
)


def apply_operation(op_name: str, second: float, top: float) -> float:
    """Apply a STATE_OPS operation to two values (same logic as _debug_apply_op)."""
    if op_name == "add":
        return second + top
    elif op_name == "multiply":
        return second * top
    elif op_name == "identity":
        return second
    else:
        raise ValueError(f"Unknown operation: {op_name}")


def execute_dag_stack(initial_values: list[float], operations: list[str]) -> float:
    """
    Execute DAG operations using stack-based execution that mimics the actual DAG model.

    CRITICAL: The DAG model uses RIGHT-TO-LEFT stack-based execution!

    How DAG execution works:
    1. Values are stored in a stack (left-to-right in array = bottom-to-top in stack)
    2. Operations are accessed RIGHT-TO-LEFT: ops[depth-1-step]
    3. Each step pops two values from stack top, applies operation, pushes result

    This means operations array should be ordered for right-to-left consumption:
    - For expression A+B*C, execution order should be: +, *, (right-to-left)
    - So operations array should be: [*, +] to give correct execution order
    """
    if not initial_values:
        return 0.0

    # Extract only meaningful operations (stop at first identity)
    meaningful_ops = []
    for op in operations:
        if op != "identity":
            meaningful_ops.append(op)
        else:
            break

    # Use only meaningful values (operations + 1)
    num_meaningful_values = len(meaningful_ops) + 1
    meaningful_values = initial_values[:num_meaningful_values]

    if not meaningful_values:
        return 0.0

    # Initialize stack with meaningful values only
    # Values are ordered left-to-right in array = bottom-to-top in stack
    stack = meaningful_values.copy()

    # CRITICAL: Apply operations RIGHT-TO-LEFT (like popping from operations stack)
    # DAG execution: ops[depth-1-step] means processing from right to left
    for op in reversed(meaningful_ops):
        if len(stack) < 2:
            break  # Can't apply operation if not enough values

        # Pop two values from stack (top and second-to-top)
        top = stack.pop()
        second = stack.pop()

        # Apply operation: second OP top
        result = apply_operation(op, second, top)

        # Push result back onto stack
        stack.append(result)

    # Return the final result (should be single value on stack)
    return stack[0] if stack else 0.0


class TestExtractionMathematicalCorrectness:
    """Test mathematical correctness of expression extraction and reconstruction."""

    @pytest.mark.parametrize(
        "expr_str,expected_result",
        [
            # Simple cases
            ("1.0", 1.0),
            ("5.0 + 3.0", 8.0),
            ("10.0 - 4.0", 6.0),
            ("3.0 * 7.0", 21.0),
            ("15.0 / 3.0", 5.0),
            # Multi-term expressions
            ("1.0 + 2.0 + 3.0", 6.0),
            ("10.0 - 3.0 - 2.0", 5.0),
            ("2.0 * 3.0 * 4.0", 24.0),
            # Mixed operations
            ("2.0 + 3.0 * 4.0", 14.0),
            ("(2.0 + 3.0) * 4.0", 20.0),
            ("10.0 / 2.0 + 5.0", 10.0),
            ("3.0 * (4.0 + 2.0)", 18.0),
            # The original bug cases
            ("1.5 + 2.5 + 3.5", 7.5),  # Multi-term addition that was broken
            # Division cases
            ("100.0 / (5.0 + 3.0)", 12.5),
            ("20.0 / (4.0 * 2.0)", 2.5),
            # Negative numbers
            ("-5.0 + 3.0", -2.0),
            ("5.0 - 8.0", -3.0),
            ("-2.0 * 3.0", -6.0),
            # Complex expressions
            ("(1.0 + 2.0) * (3.0 + 4.0)", 21.0),
            ("10.0 / (2.0 + 3.0) * 4.0", 8.0),
        ],
    )
    def test_mathematical_correctness(self, expr_str, expected_result):
        """Test that extracted values/operations reconstruct to the correct mathematical result."""

        # Parse expression and get expected result
        expr = string_to_expression(expr_str)
        sympy_result = float(expr)

        # Verify our expected result matches sympy
        assert (
            abs(sympy_result - expected_result) < 1e-6
        ), f"Test setup error: sympy gives {sympy_result}, expected {expected_result}"

        # Extract values and operations
        initial_values, operations = extract_initial_values_and_operations(
            expr, depth=8
        )

        # Reconstruct result using DAG execution
        reconstructed_result = execute_dag_stack(initial_values, operations)

        # Compare results
        error = abs(reconstructed_result - expected_result)
        assert error < 1e-6, (
            f"Mathematical mismatch for '{expr_str}':\n"
            f"  Expected: {expected_result}\n"
            f"  Sympy:    {sympy_result}\n"
            f"  Reconstructed: {reconstructed_result}\n"
            f"  Error:    {error}\n"
            f"  Values:   {initial_values}\n"
            f"  Operations: {operations}"
        )

        print(f"✅ '{expr_str}' → {reconstructed_result} (correct)")

    def test_original_bug_cases_mathematical_correctness(self):
        """Test the original bug cases for mathematical correctness."""

        test_cases = [
            # First bug case - was missing 168.2916
            ("-93.6465 - 19.281/(-1.0) + 168.2916", -93.6465 + 19.281 + 168.2916),
            # Second bug case - was missing denominator values
            (
                "-809.458*(-308.3)/(-1*68.5 - 20.46)",
                -809.458 * (-308.3) / (-1 * 68.5 - 20.46),
            ),
            # Simple multi-term addition that was broken
            ("1.0 + 2.0 + 3.0", 6.0),
            ("10.5 + 20.5 + 30.5", 61.5),
        ]

        for expr_str, expected in test_cases:
            expr = string_to_expression(expr_str)
            sympy_result = float(expr)

            # Extract and reconstruct
            initial_values, operations = extract_initial_values_and_operations(
                expr, depth=8
            )
            reconstructed = execute_dag_stack(initial_values, operations)

            # Check against both expected and sympy
            sympy_error = abs(reconstructed - sympy_result)
            expected_error = abs(reconstructed - expected)

            print(f"\nTesting: {expr_str}")
            print(f"  Sympy result: {sympy_result}")
            print(f"  Expected:     {expected}")
            print(f"  Reconstructed: {reconstructed}")
            print(f"  Values: {initial_values}")
            print(f"  Operations: {operations}")

            # Use sympy result as ground truth (more reliable than manual calculation)
            assert (
                sympy_error < 1e-3
            ), f"Failed for '{expr_str}': sympy={sympy_result}, reconstructed={reconstructed}, error={sympy_error}"

    def test_stack_execution_edge_cases(self):
        """Test edge cases in stack execution."""

        # Empty operations should return first value
        result = execute_dag_stack([5.0, 1.0, 1.0], ["identity", "identity"])
        assert result == 5.0

        # Single value with no operations
        result = execute_dag_stack([42.0], [])
        assert result == 42.0

        # More operations than values (should handle gracefully)
        result = execute_dag_stack([3.0, 2.0], ["add", "multiply", "add"])
        # Should apply what it can and return a result
        assert isinstance(result, float)

    def test_complex_nested_expressions(self):
        """Test complex expressions that might stress the extraction logic."""

        complex_cases = [
            "((1.0 + 2.0) * 3.0) / (4.0 - 1.0)",
            "(5.0 * 6.0) + (7.0 / 2.0)",
            "100.0 / (5.0 + 3.0 * 2.0)",  # This was a failing test case
            "(2.0 + 3.0) * (4.0 - 1.0) / (1.0 + 1.0)",
        ]

        for expr_str in complex_cases:
            expr = string_to_expression(expr_str)
            sympy_result = float(expr)

            # Extract and reconstruct
            initial_values, operations = extract_initial_values_and_operations(
                expr, depth=10
            )
            reconstructed = execute_dag_stack(initial_values, operations)

            error = abs(reconstructed - sympy_result)

            print(f"\nComplex case: {expr_str}")
            print(f"  Sympy: {sympy_result}")
            print(f"  Reconstructed: {reconstructed}")
            print(f"  Error: {error}")

            # Allow slightly higher tolerance for complex expressions
            assert (
                error < 1e-2 or error / abs(sympy_result) < 0.01
            ), f"Complex expression '{expr_str}' failed: sympy={sympy_result}, reconstructed={reconstructed}"

    def test_extraction_preserves_all_values(self):
        """Test that extraction doesn't lose important values."""

        # This expression should contain all these values
        expr_str = "1.5 + 2.5 + 3.5 + 4.5"
        expr = string_to_expression(expr_str)

        initial_values, operations = extract_initial_values_and_operations(
            expr, depth=6
        )

        # All the individual values should be present
        expected_values = [1.5, 2.5, 3.5, 4.5]
        for val in expected_values:
            assert (
                val in initial_values
            ), f"Missing value {val} in extracted values: {initial_values}"

        # Should have correct number of add operations (n-1 for n values)
        add_count = operations.count("add")
        assert (
            add_count == 3
        ), f"Expected 3 add operations for 4 values, got {add_count}: {operations}"


if __name__ == "__main__":
    # Run specific tests for debugging
    test_instance = TestExtractionMathematicalCorrectness()

    print("Testing mathematical correctness...")

    # Test a few key cases
    test_cases = [
        ("1.0 + 2.0 + 3.0", 6.0),
        ("-93.6465 - 19.281/(-1.0) + 168.2916", -93.6465 + 19.281 + 168.2916),
        ("100.0 / (5.0 + 3.0)", 12.5),
    ]

    for expr_str, expected in test_cases:
        try:
            test_instance.test_mathematical_correctness(expr_str, expected)
        except Exception as e:
            print(f"❌ Failed: {expr_str} - {e}")

    try:
        test_instance.test_original_bug_cases_mathematical_correctness()
        print("✅ Original bug cases test passed")
    except Exception as e:
        print(f"❌ Original bug cases failed: {e}")
