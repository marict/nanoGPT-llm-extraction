import random

import pytest
import sympy

from data.dagset.streaming import (generate_random_dag_plan,
                                   plan_to_string_expression)


@pytest.mark.parametrize("seed", [0, 17, 42, 123, 999])
def test_simplification_matches_sympy(seed):
    """`plan_to_string_expression` with simplification prob 1.0 must be numerically
    equivalent to `sympy.simplify` applied directly to the unsimplified text.
    """

    depth = 4
    init_vals, ops = generate_random_dag_plan(
        depth=depth, num_initial_values=depth + 1, seed=seed
    )

    # Original unsimplified expression
    expr_raw = plan_to_string_expression(
        init_vals,
        ops,
        seed=seed,
        english_conversion_probability=0.0,
        integer_no_decimal_probability=0.0,
        expression_simplification_probability=0.0,
        expression_permutation_probability=0.0,
    )

    # Our implementation's simplified form
    expr_simplified = plan_to_string_expression(
        init_vals,
        ops,
        seed=seed,
        english_conversion_probability=0.0,
        integer_no_decimal_probability=0.0,
        expression_simplification_probability=1.0,
        expression_permutation_probability=0.0,
    )

    # Canonical SymPy simplification
    expr_sympy_simplified = str(sympy.simplify(sympy.sympify(expr_raw)))

    # All simplified forms should be numerically equal
    val_impl = float(sympy.N(sympy.sympify(expr_simplified), 25))
    val_ref = float(sympy.N(sympy.sympify(expr_sympy_simplified), 25))

    assert (
        abs(val_impl - val_ref) < 1e-9
    ), f"Seed {seed} simplification mismatch: impl={val_impl}, sympy={val_ref}"
