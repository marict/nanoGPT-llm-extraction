import re

import pytest
import sympy

from data.dagset.streaming import plan_to_string_expression


class TestExpressionTransformations:
    """Validate new probabilistic expression transformations added to plan_to_string_expression."""

    def test_expression_simplification_probability(self):
        """Expressions should simplify when probability is set to 1.0."""
        initial_values = [2.0, 3.0]
        operations = ["add"]

        # No simplification expected
        expr_no_simp = plan_to_string_expression(
            initial_values=initial_values,
            operations=operations,
            seed=123,
            english_conversion_probability=0.0,
            expression_simplification_probability=0.0,
        )

        # Force simplification
        expr_simp = plan_to_string_expression(
            initial_values=initial_values,
            operations=operations,
            seed=123,
            english_conversion_probability=0.0,
            expression_simplification_probability=1.0,
        )

        # Expressions should differ – simplified version should no longer contain the '+' operator
        assert (
            expr_no_simp != expr_simp
        ), "Simplification probability not applied as expected"
        assert "+" not in expr_simp, "Simplified expression still contains '+' operator"

        # Both expressions should be mathematically equivalent
        assert sympy.sympify(expr_no_simp) == sympy.sympify(expr_simp)

    def test_expression_permutation_probability(self):
        """Operand order should be permuted when probability is set to 1.0."""
        initial_values = [1.0, 2.0, 3.0]
        operations = ["add", "add"]  # Results in a + b + c pattern

        expr_no_perm = plan_to_string_expression(
            initial_values=initial_values,
            operations=operations,
            seed=456,
            english_conversion_probability=0.0,
            expression_permutation_probability=0.0,
        )

        # Generate multiple permuted expressions with varying seeds to ensure that
        # permutation produces at least some variability while preserving numeric content.
        permuted_variants = set()
        for s in range(460, 470):
            permuted_variants.add(
                plan_to_string_expression(
                    initial_values=initial_values,
                    operations=operations,
                    seed=s,
                    english_conversion_probability=0.0,
                    expression_permutation_probability=1.0,
                )
            )

        # At least two distinct permutations should be produced across different seeds.
        assert (
            len(permuted_variants) > 1
        ), "Permutation did not create variability across seeds"

        # Ensure that numeric literals remain the same across permutations.
        for variant in permuted_variants:
            nums_variant = sorted(re.findall(r"-?\d+\.?\d*", variant))
            nums_baseline = sorted(re.findall(r"-?\d+\.?\d*", expr_no_perm))
            assert (
                nums_variant == nums_baseline
            ), "Permuted expression changed numeric values"
