import pytest
import torch

from data.dagset.streaming import generate_single_dag_example


@pytest.mark.parametrize(
    "depth,seed", [(3, 42), (3, 43), (4, 321), (5, 123), (8, 9999)]
)
def test_generate_single_dag_example_determinism(depth: int, seed: int):
    """Ensure that repeated calls with identical arguments yield identical DAGExample objects."""
    kwargs = dict(
        depth=depth,
        seed=seed,
        english_conversion_probability=0.25,
        integer_no_decimal_probability=0.5,
        expression_expansion_probability=0.1,
        expression_simplification_probability=0.1,
        max_digits=4,
        max_decimal_places=6,
        allowed_operations=None,
    )

    example1 = generate_single_dag_example(**kwargs)
    example2 = generate_single_dag_example(**kwargs)

    # Check that text is identical
    assert example1.text == example2.text

    # Check that initial values are identical
    assert example1.initial_values == example2.initial_values

    # Check that operations are identical
    assert example1.operations_named == example2.operations_named

    # Check that sympy expressions are identical
    assert example1.expr == example2.expr
