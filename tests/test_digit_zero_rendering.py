import pytest
import torch

from predictor_utils import digits_to_magnitude


def _string_to_digit_onehot(s: str) -> torch.Tensor:
    """Convert a digit string (without sign or decimal point) to one-hot tensor.

    Args:
        s: String containing only digits 0-9, length defines D (max_digits + max_decimal_places).

    Returns:
        Tensor of shape (D, 10) where each row is one-hot digit encoding.
    """
    D = len(s)
    one_hot = torch.zeros(D, 10)
    for i, ch in enumerate(s):
        idx = int(ch)
        one_hot[i, idx] = 1.0
    return one_hot


@pytest.mark.parametrize(
    "digit_str,max_digits,max_decimal_places,expected",
    [
        ("00124500", 4, 4, 12.45),  # 0012.4500 -> 12.45
        ("00005000", 4, 4, 0.5),  # 0000.5000 -> 0.5
        ("01000000", 4, 4, 100.0),  # 0100.0000 -> 100.0
    ],
)
def test_leading_trailing_zero_rendering(
    digit_str, max_digits, max_decimal_places, expected
):
    """Extra leading or trailing zeros should not alter the numeric magnitude after conversion."""
    one_hot = _string_to_digit_onehot(digit_str)

    # Compute magnitude using utility under test
    mag = digits_to_magnitude(one_hot, max_digits, max_decimal_places, base=10)

    assert pytest.approx(mag.item(), rel=1e-6) == expected
