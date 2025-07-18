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

    # Simplification should remove the explicit '+' operator from the text.  If
    # the unsimplified variant already lacks a '+', it means the plan collapsed
    # to an identity (e.g. right operand was zero).  In that situation the two
    # strings will legitimately be equal – the critical property is that the
    # simplified form never *introduces* a '+'.

    if "+" in expr_no:
        # When an addition operator is present, simplification must change the
        # expression string.
        assert (
            expr_no != expr_simp
        ), "Simplification probability did not affect expression"

    # In all cases the simplified expression itself should not contain a '+'.
    assert "+" not in expr_simp

    # Mathematical equivalence should hold
    assert sympy.sympify(expr_no) == sympy.sympify(expr_simp)
