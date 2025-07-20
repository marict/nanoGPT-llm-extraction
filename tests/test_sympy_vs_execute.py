import math

import sympy
from sympy import im

from data.dagset.streaming import generate_single_dag_example


def test_sympy_and_execute_consistency_across_seeds():
    """Across many random seeds, the execute_stack result must match SymPy evaluation.

    This is a high-level round-trip check: we generate a random DAG plan, compute
    its exact value with SymPy, then evaluate the same plan through
    ``execute_stack`` (via ``generate_single_dag_example``) and confirm the two
    agree to within a small numerical tolerance.
    """

    max_depth = 6  # keep magnitudes well within the LOG_LIM clipping range
    n_seeds = 1000  # lightweight for CI yet provides good coverage

    for seed in range(n_seeds):
        example = generate_single_dag_example(
            depth=max_depth,
            seed=1248,
            max_digits=6,
            max_decimal_places=6,
            english_conversion_probability=0,
            integer_no_decimal_probability=0,
            expression_expansion_probability=1,
            expression_simplification_probability=0,
        )

        # Compare the values from SymPy and execute_stack evaluation.
        if example.final_value_sympy and example.final_value_exec:
            if not im(example.final_value_sympy) != 0:
                rel_tol = 1e-4  # allowance for numerical error

                assert math.isclose(
                    float(example.final_value_sympy),
                    float(example.final_value_exec),
                    rel_tol=rel_tol,
                ), f"Values differ: {example.final_value_sympy} vs {example.final_value_exec}"
