from data.dagset.streaming import generate_single_dag_example


def test_permute_changes_text():
    """Permutation and simplification probabilities of 1.0 should alter the text output.

    We construct a deterministic flat additive expression via the private
    override hooks so that permutation shuffles operand order.
    """
    depth = 3
    initial_values = [1.0, 2.0, 3.0, 4.0]  # 4 operands → depth 3 plan
    operations = ["add", "multiply", "add"]

    # Baseline – no permutation or simplification
    ex_base = generate_single_dag_example(
        depth=depth,
        seed=0,
        english_conversion_probability=0.0,
        integer_no_decimal_probability=0.0,
        expression_permutation_probability=0.0,
        expression_simplification_probability=0.0,
        _operations_override=operations,
        _initial_values_override=initial_values,
    )

    # Permutation-only variant – should reorder operands without collapsing them.
    ex_perm = generate_single_dag_example(
        depth=depth,
        seed=0,
        english_conversion_probability=0.0,
        integer_no_decimal_probability=0.0,
        expression_permutation_probability=1.0,
        expression_simplification_probability=0.0,
        _operations_override=operations,
        _initial_values_override=initial_values,
    )

    # Textual differences expected
    assert ex_perm.text != ex_base.text, "Permutation did not reorder operands"
