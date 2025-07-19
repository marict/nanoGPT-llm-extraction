import math

from data.dagset.streaming import generate_single_dag_example


def test_sympy_and_execute_consistency_across_seeds():
    """Across many random seeds, the execute_stack result must match SymPy evaluation.

    This is a high-level round-trip check: we generate a random DAG plan, compute
    its exact value with SymPy, then evaluate the same plan through
    ``execute_stack`` (via ``generate_single_dag_example``) and confirm the two
    agree to within a small numerical tolerance.
    """

    max_depth = 4  # keep magnitudes well within the LOG_LIM clipping range
    n_seeds = 1000  # lightweight for CI yet provides good coverage

    for seed in range(n_seeds):
        example = generate_single_dag_example(
            depth=max_depth,
            seed=44,
            max_digits=6,
            max_decimal_places=6,
            english_conversion_probability=0,
            integer_no_decimal_probability=0,
            expression_expansion_probability=0,
            expression_simplification_probability=0,
        )

        import pdb

        pdb.set_trace()
        sym_val = float(example.final_value_sympy)
        exec_val = example.final_value_exec

        # Both values should be finite real numbers.
        assert math.isfinite(sym_val)
        assert math.isfinite(exec_val)

        # They should match closely (exact equality aside from fp rounding).
        assert math.isclose(
            exec_val, sym_val, rel_tol=1e-4, abs_tol=1e-4
        ), f"Seed {seed}: SymPy {sym_val} vs execute_stack {exec_val} differ too much"
