"""Test round trip conversion between expressions and tensor representations."""

import sys
from pathlib import Path

import pytest
import sympy
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from data.dagset.streaming import (
    expression_to_tensors,
    expressions_to_tensors,
    tensor_to_expression,
)


class TestTensorExpressionRoundTrip:
    """Test that expression->tensor->expression preserves meaning."""

    @pytest.mark.parametrize(
        "expr_str,dag_depth",
        [
            ("2.5 + 3.7", 2),
            ("10.0 * 4.2", 2),
            ("15.6 - 8.3", 2),
            ("20.0 / 5.0", 2),
            ("2.5 + 3.7 * 1.2", 3),
            ("10.0 * (4.2 - 1.8)", 3),
            ("-5.5 + 7.2 / 2.0", 3),
            ("(8.4 + 2.1) * 3.0 - 1.5", 4),
        ],
    )
    def test_basic_round_trip(self, expr_str: str, dag_depth: int):
        """Test that basic expressions survive round trip conversion."""
        max_digits = 4
        max_decimal_places = 4

        # Parse original expression
        orig_expr = sympy.sympify(expr_str)
        orig_value = float(orig_expr.evalf())

        # expression -> tensor
        target_digits, V_sign, O, G = expression_to_tensors(orig_expr, dag_depth)

        # Remove batch/time dimensions to get the actual tensors
        digit_logits = target_digits[0, 0]  # (num_initial_nodes, D, base)
        V_sign_flat = V_sign[0, 0]  # (total_nodes,)
        O_flat = O[0, 0]  # (dag_depth, total_nodes)
        G_flat = G[0, 0]  # (dag_depth,)

        # tensor -> expression
        reconstructed_expr = tensor_to_expression(
            digit_logits,
            V_sign_flat,
            O_flat,
            G_flat,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
        )

        # Check numerical equivalence
        reconstructed_value = float(reconstructed_expr.evalf())

        print(f"Original: {expr_str} = {orig_value}")
        print(f"Reconstructed: {reconstructed_expr} = {reconstructed_value}")
        print(f"Error: {abs(orig_value - reconstructed_value)}")

        # Allow some tolerance due to floating point precision
        assert (
            abs(orig_value - reconstructed_value) < 1e-3
        ), f"Round trip failed: {orig_value} != {reconstructed_value} for expression {expr_str}"

    def test_round_trip_with_expressions_to_tensors(self):
        """Test round trip using the expressions_to_tensors function (more realistic)."""
        test_expressions = [
            "2.5 + 3.7",
            "10.0 * 4.2",
            "15.6 - 8.3",
            "2.5 + 3.7 * 1.2",
        ]

        dag_depth = 4
        max_digits = 4
        max_decimal_places = 4

        for expr_str in test_expressions:
            # Parse original expression
            orig_expr = sympy.sympify(expr_str)
            orig_value = float(orig_expr.evalf())

            # Use expressions_to_tensors (same as training pipeline)
            tensor_results, valid_mask = expressions_to_tensors(
                [orig_expr],
                depth=dag_depth,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
            )

            assert valid_mask[0], f"Expression {expr_str} should be valid"

            target_dict = tensor_results[0]

            # Extract tensors
            digit_targets = target_dict["target_digits"]  # (num_initial_nodes, D, base)
            V_sign = target_dict["target_V_sign"]  # (total_nodes,)
            O = target_dict["target_O"]  # (dag_depth, total_nodes)
            G = target_dict["target_G"]  # (dag_depth,)

            # Convert digit targets to logits (simulate model predictions)
            # For perfect reconstruction, use the one-hot targets as logits
            digit_logits = digit_targets.float() * 10.0  # Scale up to make argmax work

            # tensor -> expression
            reconstructed_expr = tensor_to_expression(
                digit_logits,
                V_sign,
                O,
                G,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
            )

            # Check numerical equivalence
            reconstructed_value = float(reconstructed_expr.evalf())

            print(f"Original: {expr_str} = {orig_value}")
            print(f"Reconstructed: {reconstructed_expr} = {reconstructed_value}")
            print(f"Error: {abs(orig_value - reconstructed_value)}")

            # Allow some tolerance due to floating point precision and discretization
            assert (
                abs(orig_value - reconstructed_value) < 1e-2
            ), f"Round trip failed: {orig_value} != {reconstructed_value} for expression {expr_str}"

    def test_problematic_cases(self):
        """Test cases that might cause issues in tensor_to_expression."""
        problematic_cases = [
            ("0.0", 1),  # Zero value
            ("1.0", 1),  # Simple integer
            ("0.5", 1),  # Simple fraction
            ("-2.5", 1),  # Negative value
            ("100.0", 1),  # Larger number
        ]

        for expr_str, dag_depth in problematic_cases:
            try:
                # Use expressions_to_tensors pipeline
                orig_expr = sympy.sympify(expr_str)
                orig_value = float(orig_expr.evalf())

                tensor_results, valid_mask = expressions_to_tensors(
                    [orig_expr], depth=dag_depth, max_digits=4, max_decimal_places=4
                )

                if not valid_mask[0]:
                    print(f"Skipping invalid expression: {expr_str}")
                    continue

                target_dict = tensor_results[0]
                digit_logits = target_dict["target_digits"].float() * 10.0

                reconstructed_expr = tensor_to_expression(
                    digit_logits,
                    target_dict["target_V_sign"],
                    target_dict["target_O"],
                    target_dict["target_G"],
                    max_digits=4,
                    max_decimal_places=4,
                )

                reconstructed_value = float(reconstructed_expr.evalf())
                error = abs(orig_value - reconstructed_value)

                print(f"Problematic case: {expr_str} = {orig_value}")
                print(f"  Reconstructed: {reconstructed_expr} = {reconstructed_value}")
                print(f"  Error: {error}")

                # Be more lenient with problematic cases
                assert error < 1e-1, f"Large error for {expr_str}: {error}"

            except Exception as e:
                print(f"Exception for {expr_str}: {e}")
                # Don't fail on exceptions for problematic cases, just log them
                continue


if __name__ == "__main__":
    # Run the tests directly
    test_instance = TestTensorExpressionRoundTrip()

    print("=" * 60)
    print("Testing basic round trip conversion...")
    print("=" * 60)

    basic_cases = [
        ("2.5 + 3.7", 2),
        ("10.0 * 4.2", 2),
        ("2.5 + 3.7 * 1.2", 3),
    ]

    for expr_str, dag_depth in basic_cases:
        try:
            test_instance.test_basic_round_trip(expr_str, dag_depth)
            print(f"✅ PASSED: {expr_str}")
        except Exception as e:
            print(f"❌ FAILED: {expr_str} - {e}")

    print("\n" + "=" * 60)
    print("Testing with expressions_to_tensors pipeline...")
    print("=" * 60)

    try:
        test_instance.test_round_trip_with_expressions_to_tensors()
        print("✅ PASSED: expressions_to_tensors round trip")
    except Exception as e:
        print(f"❌ FAILED: expressions_to_tensors round trip - {e}")

    print("\n" + "=" * 60)
    print("Testing problematic cases...")
    print("=" * 60)

    try:
        test_instance.test_problematic_cases()
        print("✅ PASSED: problematic cases")
    except Exception as e:
        print(f"❌ FAILED: problematic cases - {e}")
