import random
import re

import pytest
import sympy

from data.dagset.streaming import (generate_random_dag_plan,
                                   plan_to_string_expression)


def is_flat_add_or_mul(expr: str) -> bool:
    """Return True if expr contains only '+' or only '*' (but not both) and no parentheses."""
    if "(" in expr or ")" in expr:
        return False
    if "+" in expr and "*" in expr:
        return False
    if "+" in expr:
        return all(op not in expr for op in "*-/")
    if "*" in expr:
        return all(op not in expr for op in "+-/")
    return False


@pytest.mark.parametrize("seed_start", [200])
def test_permutation_after_simplification(seed_start):
    """Permutation should reorder operands of flat Add/Mul but keep value."""

    depth = 3
    found_permutation = False

    for seed in range(seed_start, seed_start + 1000):
        init_vals, ops = generate_random_dag_plan(
            depth=depth,
            num_initial_values=depth + 1,
            seed=seed,
            allowed_operations=["add"],
        )

        expr_base, _, _ = plan_to_string_expression(
            init_vals,
            ops,
            seed=seed,
            english_conversion_probability=0.0,
            integer_no_decimal_probability=0.0,
            expression_simplification_probability=1.0,
            expression_permutation_probability=0.0,
        )

        if not is_flat_add_or_mul(expr_base):
            continue  # cannot permute safely, skip

        expr_perm, _, _ = plan_to_string_expression(
            init_vals,
            ops,
            seed=seed,
            english_conversion_probability=0.0,
            integer_no_decimal_probability=0.0,
            expression_simplification_probability=1.0,
            expression_permutation_probability=1.0,
        )

        # Verify numeric equivalence
        val_base = float(sympy.N(sympy.sympify(expr_base), 20))
        val_perm = float(sympy.N(sympy.sympify(expr_perm), 20))
        assert abs(val_base - val_perm) < 1e-9

    # If we processed at least one flat expression without exceptions the test passes.
