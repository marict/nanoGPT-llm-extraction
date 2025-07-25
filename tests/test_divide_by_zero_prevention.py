from data.dagset.streaming import generate_single_dag_example


def test_divide_by_zero_prevention():
    """Test that the DAG generation prevents divide by zero operations."""
    # Generate an example with only divide operations
    example = generate_single_dag_example(
        depth=3,  # Use multiple operations to increase chance of divide
        seed=42,
        allowed_operations=[
            "divide",
            "identity",
        ],  # Force only division operations plus required identity
    )

    # For each division operation, verify the corresponding value is not zero
    for i, op in enumerate(example.operations_named):
        if op == "divide":
            divisor_idx = i + 1
            assert (
                example.structure_dict["target_initial_values"][divisor_idx] != 0.0
            ), f"Found zero divisor at index {divisor_idx}"

    # Verify the final value exists and is finite
    final_value_exec = example.structure_dict["target_final_exec"]
    assert final_value_exec is not None
    assert example.final_value_sympy is not None
    assert not any(
        map(
            lambda x: x == float("inf") or x == float("-inf"),
            [final_value_exec, example.final_value_sympy],
        )
    )


def test_divide_by_zero_prevention_random():
    """Test that random DAG generation never produces divide by zero operations."""
    # Generate multiple examples with random seeds to ensure no divide by zero occurs
    for seed in range(10):
        example = generate_single_dag_example(
            depth=3,  # Use a reasonable depth
            seed=seed,
            allowed_operations=[
                "divide",
                "identity",
            ],  # Force only division operations plus required identity
        )

        # For each division operation, check that the corresponding value is not zero
        for i, op in enumerate(example.operations_named):
            if op == "divide":
                divisor_idx = i + 1
                assert (
                    example.structure_dict["target_initial_values"][divisor_idx] != 0.0
                ), f"Found zero divisor at index {divisor_idx} with seed {seed}"

        # Verify all final values are finite
        final_value_exec = example.structure_dict["target_final_exec"]
        assert final_value_exec is not None
        assert example.final_value_sympy is not None
        assert not any(
            map(
                lambda x: x == float("inf") or x == float("-inf"),
                [final_value_exec, example.final_value_sympy],
            )
        )
