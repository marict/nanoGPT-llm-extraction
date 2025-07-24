import pytest
import torch

from predictor_utils import _compute_digit_loss


@pytest.mark.parametrize("device_type", ["cpu"])
def test_digit_loss_raises_on_all_zero_target(device_type):
    """_compute_digit_loss should raise if any target row is all zeros."""
    # Minimal tensor sizes
    B, T, N, D, base = 1, 1, 1, 3, 10

    # Random logits for the prediction branch
    pred_digit_logits = torch.randn(B, T, N, D, base)

    # Construct target with an invalid all-zero row (last digit position)
    target_digits = torch.zeros(B, T, N, D, base)
    # Make the first two digit slots valid one-hot vectors (digit 1 and 2)
    target_digits[0, 0, 0, 0, 1] = 1  # first digit → class 1
    target_digits[0, 0, 0, 1, 2] = 1  # second digit → class 2
    # Third digit slot intentionally remains all zeros → should trigger error

    with pytest.raises(ValueError, match="all zeros"):
        _compute_digit_loss(pred_digit_logits, target_digits, device_type)


@pytest.mark.parametrize("device_type", ["cpu"])
def test_digit_loss_valid_one_hot(device_type):
    """_compute_digit_loss should run without error on proper one-hot targets."""
    B, T, N, D, base = 2, 1, 1, 4, 10

    pred_digit_logits = torch.randn(B, T, N, D, base)

    # Build valid one-hot targets: alternate classes 0-3
    target_digits = torch.zeros(B, T, N, D, base)
    for d in range(D):
        target_digits[:, 0, 0, d, d % base] = 1.0

    loss = _compute_digit_loss(pred_digit_logits, target_digits, device_type)
    assert loss.ndim == 0  # scalar tensor
    assert torch.isfinite(loss) and loss > 0
