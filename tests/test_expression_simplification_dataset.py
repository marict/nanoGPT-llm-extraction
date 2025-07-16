import re

import pytest
import sympy

from data.dagset.streaming import DAGStructureDataset


@pytest.mark.parametrize("depth", [1])
def test_dataset_expression_simplification(depth):
    """Dataset should simplify additive expression when probability is 1.0."""
    # Only one add operation so simplification yields single numeric literal.

    ds_no_simp = DAGStructureDataset(
        max_depth=depth,
        seed=0,
        english_conversion_probability=0.0,
        integer_no_decimal_probability=0.0,
        expression_permutation_probability=0.0,
        expression_simplification_probability=0.0,
        allowed_operations=["add"],
    )
    expr_no, _ = ds_no_simp.generate_structure_example(depth=depth, seed=123)

    ds_simp = DAGStructureDataset(
        max_depth=depth,
        seed=0,
        english_conversion_probability=0.0,
        integer_no_decimal_probability=0.0,
        expression_permutation_probability=0.0,
        expression_simplification_probability=1.0,
        allowed_operations=["add"],
    )
    expr_simp, _ = ds_simp.generate_structure_example(depth=depth, seed=123)

    # Should simplify (remove '+')
    assert expr_no != expr_simp, "Simplification probability did not affect expression"
    assert "+" not in expr_simp

    # Mathematical equivalence should hold
    assert sympy.sympify(expr_no) == sympy.sympify(expr_simp)
