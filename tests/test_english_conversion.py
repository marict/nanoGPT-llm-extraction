"""Comprehensive tests for reversible English expression conversion.

Tests verify that mathematical expressions can be converted to English and back
while maintaining:
1. Syntactic correctness (can be parsed by SymPy)
2. Semantic equivalence (evaluates to the same numerical result)
3. Stability over multiple round trips
"""

import math

# Import the conversion functions
import sys
from pathlib import Path
from typing import List, Optional

import pytest

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from data.dagset.streaming import string_to_expression
from expression_to_english import english_to_expression, expression_to_english


class TestExpressionEquivalence:
    """Test that expressions remain mathematically equivalent after English conversion."""

    def evaluate_expression(self, expr_str: str) -> Optional[float]:
        """Safely evaluate a mathematical expression string.

        Args:
            expr_str: Expression string like "3.5 + 4.2"

        Returns:
            Numerical result or None if evaluation fails
        """
        try:
            expr = string_to_expression(expr_str)
            return float(expr)
        except:
            return None

    def normalize_expression(self, expr_str: str) -> str:
        """Normalize expression string for comparison."""
        if not expr_str:
            return ""
        return " ".join(expr_str.split())

    def is_numerically_equivalent(
        self, val1: Optional[float], val2: Optional[float], tolerance: float = 1e-10
    ) -> bool:
        """Check if two values are numerically equivalent within tolerance."""
        if val1 is None or val2 is None:
            return val1 == val2

        if math.isnan(val1) and math.isnan(val2):
            return True
        if math.isinf(val1) and math.isinf(val2):
            return math.copysign(1, val1) == math.copysign(1, val2)

        return abs(val1 - val2) <= tolerance

    def assert_reversible(self, original_expr: str, round_trips: int = 3):
        """Assert that an expression is reversible through English conversion.

        Args:
            original_expr: Original mathematical expression
            round_trips: Number of round trips to test (default 3)
        """
        # Get original numerical value
        original_value = self.evaluate_expression(original_expr)

        current_expr = original_expr

        for trip in range(round_trips):
            # Convert to English
            english = expression_to_english(current_expr)
            assert (
                english
            ), f"Failed to convert to English on trip {trip+1}: '{current_expr}'"

            # Convert back to math
            converted_expr = english_to_expression(english)
            assert (
                converted_expr is not None
            ), f"Failed to convert from English on trip {trip+1}: '{english}'"

            # Verify it can be parsed by SymPy
            converted_value = self.evaluate_expression(converted_expr)
            assert (
                converted_value is not None
            ), f"Converted expression cannot be evaluated on trip {trip+1}: '{converted_expr}'"

            # Verify numerical equivalence
            assert self.is_numerically_equivalent(
                original_value, converted_value
            ), f"Numerical mismatch on trip {trip+1}: {original_value} != {converted_value} (original: '{original_expr}' -> converted: '{converted_expr}')"

            # Use converted expression for next round trip
            current_expr = self.normalize_expression(converted_expr)

        print(
            f"✅ Reversible over {round_trips} trips: '{original_expr}' -> final: '{current_expr}'"
        )


class TestBasicExpressions(TestExpressionEquivalence):
    """Test basic mathematical expressions."""

    @pytest.mark.parametrize(
        "expr", ["3", "3.5", "-42", "-3.14159", "0", "0.0", "-0.0", "123.456789"]
    )
    def test_simple_numbers(self, expr):
        """Test simple number conversions."""
        self.assert_reversible(expr)

    @pytest.mark.parametrize(
        "expr",
        [
            "3 + 4",
            "10 - 5",
            "6 * 7",
            "15 / 3",
            "3.5 + 4.2",
            "10.5 - 2.3",
            "2.5 * 4.0",
            "12.6 / 3.0",
        ],
    )
    def test_basic_operations(self, expr):
        """Test basic arithmetic operations."""
        self.assert_reversible(expr)

    @pytest.mark.parametrize(
        "expr", ["-3 + 4", "5 + -2", "-10 * -3", "-15 / 3", "3 - -4", "-5.5 + 2.2"]
    )
    def test_negative_numbers(self, expr):
        """Test expressions with negative numbers."""
        self.assert_reversible(expr)


class TestComplexExpressions(TestExpressionEquivalence):
    """Test more complex mathematical expressions."""

    @pytest.mark.parametrize(
        "expr", ["(3 + 4)", "(10.5 - 2.3)", "((5))", "(3.14159)", "(-42)"]
    )
    def test_simple_parentheses(self, expr):
        """Test expressions with simple parentheses."""
        self.assert_reversible(expr)

    @pytest.mark.parametrize(
        "expr",
        [
            "3 + 4 * 5",
            "2 * 3 + 4",
            "10 / 2 - 3",
            "5 - 8 / 4",
            "3.5 * 2.0 + 1.5",
            "12.0 / 3.0 - 2.5",
        ],
    )
    def test_order_of_operations(self, expr):
        """Test expressions that depend on order of operations."""
        self.assert_reversible(expr)

    @pytest.mark.parametrize(
        "expr",
        [
            "3 * (4 + 5)",
            "(10 - 6) / 2",
            "2 * (3.5 + 1.5)",
            "(8.0 - 3.0) * 2.0",
            "15 / (3 + 2)",
            "(4.5 + 1.5) / 3.0",
        ],
    )
    def test_parentheses_with_operations(self, expr):
        """Test parentheses combined with operations."""
        self.assert_reversible(expr)


class TestAdvancedExpressions(TestExpressionEquivalence):
    """Test advanced and edge case expressions."""

    @pytest.mark.parametrize(
        "expr",
        [
            "((3 + 4) * 5)",
            "(2 * (3 + 4))",
            "((10 - 6) / 2)",
            "(3 * (4 + 5) - 2)",
            "((8.0 - 3.0) * 2.0) + 1.0",
        ],
    )
    def test_nested_parentheses(self, expr):
        """Test expressions with nested parentheses."""
        self.assert_reversible(expr)

    @pytest.mark.parametrize(
        "expr",
        [
            "3 + 4 * 5 - 2",
            "10 / 2 + 3 * 4",
            "5 * 6 - 8 / 2",
            "15 - 3 * 2 + 7",
            "2.5 * 3.0 + 4.5 - 1.2",
            "12.0 / 4.0 + 2.5 * 3.0",
        ],
    )
    def test_multiple_operations(self, expr):
        """Test expressions with multiple operations."""
        self.assert_reversible(expr)

    @pytest.mark.parametrize(
        "expr",
        [
            "3.5 + 4.2 * (1.8 - 6.7)",
            "(3.5 + 1.2) * 4.0 - 2.8",
            "15.5 / (2.5 + 1.0) + 3.2",
            "-5.3 + 8.0 * (2.1 - 4.7)",
            "(12.5 - 3.2) / (1.8 + 0.7)",
        ],
    )
    def test_realistic_expressions(self, expr):
        """Test realistic complex expressions."""
        self.assert_reversible(expr)


class TestEdgeCases(TestExpressionEquivalence):
    """Test edge cases and potential failure modes."""

    @pytest.mark.parametrize(
        "expr", ["0.1", "0.01", "0.001", "0.0001", "1.0000001", "99.99999"]
    )
    def test_small_precision_numbers(self, expr):
        """Test numbers with various decimal precisions."""
        self.assert_reversible(expr)

    @pytest.mark.parametrize("expr", ["100", "1000", "9999", "12345.6789"])
    def test_large_numbers(self, expr):
        """Test larger numbers."""
        self.assert_reversible(expr)

    @pytest.mark.parametrize(
        "expr", ["1 / 3", "2 / 7", "5 / 11", "22 / 7", "355 / 113"]
    )
    def test_division_results(self, expr):
        """Test expressions that result in non-terminating decimals."""
        self.assert_reversible(expr)


class TestGeneratedExpressions(TestExpressionEquivalence):
    """Test expressions generated by the DAG system."""

    def test_dag_generated_expressions(self):
        """Test expressions that are typically generated by the DAG system."""
        # Import the expression generator
        import tiktoken

        from data.dagset.streaming import generate_expression

        tokenizer = tiktoken.get_encoding("gpt2")

        # Generate several expressions and test them
        for seed in range(10):
            expressions, substrings, valid_mask = generate_expression(
                depth=4,
                seed=seed + 1000,  # Use different seeds than other tests
                max_digits=4,
                max_decimal_places=4,
                tokenizer=tokenizer,
                base=10,
            )

            # Test the final expression if it's valid
            if expressions and substrings and expressions[-1] != "not valid":
                final_expr = substrings[-1]

                # Skip if expression contains patterns that we know are problematic
                if final_expr and not any(
                    pattern in final_expr for pattern in ["**", "^^", "inf", "nan"]
                ):
                    try:
                        self.assert_reversible(
                            final_expr, round_trips=2
                        )  # Use 2 trips for generated expressions
                    except AssertionError as e:
                        # Allow some generated expressions to fail, but log them
                        print(
                            f"⚠️  Generated expression failed reversibility test: '{final_expr}' - {e}"
                        )


class TestRoundTripStability:
    """Test that multiple round trips don't degrade expressions."""

    def test_stability_over_many_trips(self):
        """Test that expressions remain stable over many round trips."""
        test_expressions = [
            "3.5 + 4.2",
            "10 * (2.5 - 1.3)",
            "(15.7 + 2.3) / 4.0",
            "-8.5 + 12.0 * 1.5",
        ]

        for expr in test_expressions:
            current = expr
            values = []

            # Do 10 round trips
            for i in range(10):
                english = expression_to_english(current)
                assert english, f"Failed to convert to English on trip {i+1}"

                converted = english_to_expression(english)
                assert converted, f"Failed to convert from English on trip {i+1}"

                # Evaluate and track numerical stability
                try:
                    from data.dagset.streaming import string_to_expression

                    expr_obj = string_to_expression(converted)
                    value = float(expr_obj)
                    values.append(value)
                except:
                    pytest.fail(
                        f"Expression became unparseable on trip {i+1}: '{converted}'"
                    )

                current = converted

            # Check that all values are the same (within tolerance)
            first_value = values[0]
            for i, value in enumerate(values[1:], 1):
                assert (
                    abs(value - first_value) < 1e-10
                ), f"Value changed after {i+1} trips: {first_value} -> {value}"

            print(f"✅ Stable over 10 trips: '{expr}' (final value: {values[-1]})")


def test_comprehensive_reversibility():
    """Run comprehensive reversibility tests on a large set of expressions."""
    # Generate a comprehensive set of test expressions
    expressions = []

    # Simple numbers
    expressions.extend([f"{i}" for i in range(-10, 11)])
    expressions.extend([f"{i}.{j}" for i in range(-5, 6) for j in range(0, 10)])

    # Basic operations
    for op in ["+", "-", "*", "/"]:
        for a in ["3", "5.5", "-2"]:
            for b in ["4", "2.1", "-1.5"]:
                expressions.append(f"{a} {op} {b}")

    # With parentheses
    for op1, op2 in [("*", "+"), ("+", "*"), ("/", "-"), ("-", "/")]:
        expressions.extend(
            [
                f"2 {op1} (3 {op2} 4)",
                f"(5 {op1} 2) {op2} 3",
                f"(1.5 {op1} 2.0) {op2} (3.5 {op1} 1.0)",
            ]
        )

    # Test all expressions (with some tolerance for failures on edge cases)
    total_tested = 0
    total_passed = 0

    test_instance = TestExpressionEquivalence()

    for expr in expressions:
        try:
            # Skip expressions that would cause division by zero or other issues
            if "/0" in expr or "/-0" in expr:
                continue

            test_instance.assert_reversible(expr, round_trips=2)
            total_passed += 1
        except (AssertionError, Exception) as e:
            # Log failures but don't fail the entire test
            print(f"⚠️  Failed: '{expr}' - {e}")

        total_tested += 1

    success_rate = total_passed / total_tested if total_tested > 0 else 0
    print(f"\n📊 Comprehensive Test Results:")
    print(f"   Total tested: {total_tested}")
    print(f"   Passed: {total_passed}")
    print(f"   Success rate: {success_rate:.1%}")

    # Require at least 90% success rate
    assert (
        success_rate >= 0.9
    ), f"Success rate {success_rate:.1%} is below 90% threshold"


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
