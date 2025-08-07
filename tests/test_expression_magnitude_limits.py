import pytest
import sympy
import torch

from data.dagset.streaming import expression_to_tensors, float_to_digit_onehot


def test_expression_to_tensors_handles_large_magnitudes():
    """Test that expression_to_tensors handles large magnitude values gracefully."""

    # Test with a large magnitude value that exceeds the digit representation limits
    large_expr = sympy.sympify("-907.0")

    # This should not raise an exception, but should handle the large value appropriately
    try:
        target_digits, target_V_sign, target_O, target_G = expression_to_tensors(
            large_expr, dag_depth=2, max_digits=2, max_decimal_places=1
        )
        # If it succeeds, verify the shapes are correct
        # D = max_digits + max_decimal_places = 2 + 1 = 3
        assert target_digits.shape == (
            1,
            1,
            3,
            3,
            10,
        )  # (batch, time, num_initial_nodes, D, base)
        assert target_V_sign.shape == (1, 1, 5)  # (batch, time, total_nodes)
        assert target_O.shape == (1, 1, 2, 5)  # (batch, time, dag_depth, total_nodes)
        assert target_G.shape == (1, 1, 2)  # (batch, time, dag_depth)
    except ValueError as e:
        # If it fails, the error should be informative
        assert "magnitude" in str(e).lower() or "exceeds" in str(e).lower()


def test_float_to_digit_onehot_handles_large_values():
    """Test that float_to_digit_onehot handles large values appropriately."""

    # Test with a value that exceeds the representation limits
    large_value = 907.0
    max_digits = 2
    max_decimal_places = 1
    base = 10

    # Calculate the maximum representable value
    max_value = (base**max_digits - 1) / (base**max_decimal_places)

    # This should raise a ValueError with a clear message
    with pytest.raises(ValueError) as exc_info:
        float_to_digit_onehot(large_value, max_digits, max_decimal_places, base)

    error_msg = str(exc_info.value)
    assert "magnitude" in error_msg.lower()
    assert "exceeds" in error_msg.lower()
    assert str(max_value) in error_msg


def test_expression_to_tensors_with_reasonable_limits():
    """Test that expression_to_tensors works with reasonable magnitude limits."""

    # Test with values that should fit within the representation limits
    reasonable_expr = sympy.sympify("45.3 + 23.7")

    target_digits, target_V_sign, target_O, target_G = expression_to_tensors(
        reasonable_expr, dag_depth=2, max_digits=2, max_decimal_places=1
    )

    # Verify the shapes are correct
    # D = max_digits + max_decimal_places = 2 + 1 = 3
    assert target_digits.shape == (1, 1, 3, 3, 10)
    assert target_V_sign.shape == (1, 1, 5)
    assert target_O.shape == (1, 1, 2, 5)
    assert target_G.shape == (1, 1, 2)


if __name__ == "__main__":
    pytest.main([__file__])
