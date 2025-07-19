import math

from data.dagset.streaming import plan_to_tensors, tensors_to_plan


def test_tensors_to_plan_round_trip():
    """Test that converting a plan to tensors and back preserves the values."""
    # Test parameters
    max_digits = 4
    max_decimal_places = 6

    # Create a test plan with various numbers and operations
    initial_values = [
        1.234,  # Simple positive decimal
        -5.678,  # Simple negative decimal
        10.0,  # Integer with decimal
        -0.001,  # Small negative number
        123.456,  # Larger number
    ]
    operations = ["add", "subtract", "multiply", "divide"]

    # Convert plan to tensors
    signs, digits, ops = plan_to_tensors(
        initial_values=initial_values,
        operations=operations,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
    )

    # Convert tensors back to plan
    recovered_values, recovered_ops = tensors_to_plan(
        signs=signs,
        digits=digits,
        operations=ops,
        max_digits=max_digits,
    )

    # Check that operations match exactly
    assert (
        recovered_ops == operations
    ), f"Operations mismatch: {recovered_ops} != {operations}"

    # Check that values match within floating point precision
    assert len(recovered_values) == len(initial_values)
    for orig, rec in zip(initial_values, recovered_values):
        assert math.isclose(orig, rec, rel_tol=1e-6), f"Value mismatch: {orig} != {rec}"
