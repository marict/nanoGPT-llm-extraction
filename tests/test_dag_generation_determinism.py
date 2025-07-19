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

    # Collect all differences instead of failing fast on the first one.
    diffs = []

    # Simple scalar / list comparisons
    if example1.text != example2.text:
        diffs.append("text differs")
    if example1.initial_values != example2.initial_values:
        diffs.append("initial_values differ")
    if example1.operations_named != example2.operations_named:
        diffs.append("operations_named differ")
    if example1.seed != example2.seed:
        diffs.append("seed differs")
    if example1.did_expand != example2.did_expand:
        diffs.append("did_expand flag differs")
    if example1.did_simplify != example2.did_simplify:
        diffs.append("did_simplify flag differs")
    if example1.allowed_operations != example2.allowed_operations:
        diffs.append("allowed_operations differ")

    # Tensor comparisons
    if not torch.equal(example1.signs, example2.signs):
        diffs.append("signs tensor differs")
    if not torch.equal(example1.digits, example2.digits):
        diffs.append("digits tensor differs")
    if not torch.equal(example1.operations, example2.operations):
        diffs.append("operations tensor differs")

    # Numerical value comparisons (approximate)
    if not (example1.final_value_sympy == pytest.approx(example2.final_value_sympy)):
        diffs.append("final_value_sympy differs")
    if not (example1.final_value_exec == pytest.approx(example2.final_value_exec)):
        diffs.append("final_value_exec differs")

    # If any differences were found, fail the test once with a comprehensive message.
    if diffs:
        pytest.fail("DAGExample mismatch: \n- " + "\n- ".join(diffs))
