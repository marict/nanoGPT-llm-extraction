"""Test that digit display logic in print_detailed_validation_sample works correctly."""

import pytest
import torch

from evaluate import _extract_initial_value


class MockConfig:
    """Mock config for testing."""

    def __init__(self, max_digits=4, max_decimal_places=4):
        self.max_digits = max_digits
        self.max_decimal_places = max_decimal_places


def create_digit_logits_from_digits(digits, base=10):
    """Create digit logits tensor from a list of digit values.

    Args:
        digits: List of integers representing digit values
        base: Base of the number system (default 10)

    Returns:
        Tensor of shape (len(digits), base) with high logits for the specified digits
    """
    D = len(digits)
    logits = torch.zeros(D, base)
    for i, digit in enumerate(digits):
        # Set high logit for the target digit, low for others
        logits[i, digit] = 10.0  # High logit
        logits[i, :digit] = -10.0  # Low logits for digits before
        logits[i, digit + 1 :] = -10.0  # Low logits for digits after
    return logits


def create_onehot_digit_tensor(digits, base=10):
    """Create one-hot digit tensor from a list of digit values."""
    D = len(digits)
    onehot = torch.zeros(D, base)
    for i, digit in enumerate(digits):
        onehot[i, digit] = 1.0
    return onehot


@pytest.fixture
def test_config():
    """Create test config."""
    return MockConfig(max_digits=4, max_decimal_places=4)


def test_extract_initial_value_predicted_digits(test_config):
    """Test that _extract_initial_value correctly converts predicted digit logits to float values."""

    # Define test cases with expected displayed values
    test_cases = [
        {
            "digits": [1, 2, 3, 4, 5, 6, 7, 8],  # 1234.5678
            "sign": 1.0,
            "expected_value": 1234.5678,
            "description": "Normal positive number",
        },
        {
            "digits": [0, 0, 4, 2, 7, 3, 0, 0],  # 0042.7300 -> 42.73
            "sign": 1.0,
            "expected_value": 42.73,
            "description": "Leading zeros should be removed",
        },
        {
            "digits": [9, 8, 7, 6, 1, 2, 3, 4],  # 9876.1234
            "sign": -1.0,
            "expected_value": -9876.1234,
            "description": "Negative number",
        },
        {
            "digits": [0, 0, 0, 1, 0, 0, 0, 0],  # 0001.0000 -> 1.0
            "sign": 1.0,
            "expected_value": 1.0,
            "description": "Single digit with leading zeros",
        },
        {
            "digits": [0, 0, 0, 0, 0, 0, 0, 1],  # 0000.0001 -> 0.0001
            "sign": 1.0,
            "expected_value": 0.0001,
            "description": "Small decimal number",
        },
    ]

    # Test each case
    for case in test_cases:
        digits = case["digits"]
        sign = case["sign"]
        expected_value = case["expected_value"]
        description = case["description"]

        # Create digit logits from the digit pattern
        digit_logits = create_digit_logits_from_digits(digits)

        # Call _extract_initial_value with predicted data (is_target=False)
        predicted_value = _extract_initial_value(
            digit_data=digit_logits, sign=sign, cfg=test_config, is_target=False
        )

        # Verify the predicted value matches expected
        assert abs(predicted_value - expected_value) < 1e-6, (
            f"Predicted value mismatch for {description}:\n"
            f"  Expected: {expected_value}\n"
            f"  Got: {predicted_value}\n"
            f"  Digits: {digits}\n"
            f"  Sign: {sign}"
        )


def test_extract_initial_value_edge_cases(test_config):
    """Test edge cases for digit extraction."""

    # Test cases that might cause issues
    edge_cases = [
        {
            "digits": [0, 0, 0, 0, 0, 0, 0, 0],  # All zeros -> 0.0
            "sign": 1.0,
            "expected_value": 0.0,
            "description": "All zeros",
        },
        {
            "digits": [9, 9, 9, 9, 9, 9, 9, 9],  # 9999.9999
            "sign": 1.0,
            "expected_value": 9999.9999,
            "description": "All nines",
        },
        {
            "digits": [0, 0, 0, 0, 1, 0, 0, 0],  # 0000.1000 -> 0.1
            "sign": -1.0,
            "expected_value": -0.1,
            "description": "Negative decimal",
        },
    ]

    # Test each edge case
    for case in edge_cases:
        digits = case["digits"]
        sign = case["sign"]
        expected_value = case["expected_value"]
        description = case["description"]

        # Create digit logits from the digit pattern
        digit_logits = create_digit_logits_from_digits(digits)

        # Call _extract_initial_value with predicted data (is_target=False)
        predicted_value = _extract_initial_value(
            digit_data=digit_logits, sign=sign, cfg=test_config, is_target=False
        )

        # Verify the predicted value matches expected
        assert abs(predicted_value - expected_value) < 1e-6, (
            f"Predicted value mismatch for edge case {description}:\n"
            f"  Expected: {expected_value}\n"
            f"  Got: {predicted_value}\n"
            f"  Digits: {digits}\n"
            f"  Sign: {sign}"
        )


def test_extract_initial_value_different_config():
    """Test with different max_digits and max_decimal_places."""

    # Test with smaller precision
    small_config = MockConfig(max_digits=2, max_decimal_places=2)

    test_cases = [
        {
            "digits": [4, 2, 7, 3],  # 42.73
            "sign": 1.0,
            "expected_value": 42.73,
            "description": "Small config number",
        },
        {
            "digits": [0, 5, 0, 1],  # 05.01 -> 5.01
            "sign": -1.0,
            "expected_value": -5.01,
            "description": "Small config with leading zero",
        },
    ]

    # Test each case with the smaller config
    for case in test_cases:
        digits = case["digits"]
        sign = case["sign"]
        expected_value = case["expected_value"]
        description = case["description"]

        # Create digit logits from the digit pattern
        digit_logits = create_digit_logits_from_digits(digits)

        # Call _extract_initial_value with predicted data (is_target=False)
        predicted_value = _extract_initial_value(
            digit_data=digit_logits, sign=sign, cfg=small_config, is_target=False
        )

        # Verify the predicted value matches expected
        assert abs(predicted_value - expected_value) < 1e-6, (
            f"Predicted value mismatch for small config {description}:\n"
            f"  Expected: {expected_value}\n"
            f"  Got: {predicted_value}\n"
            f"  Digits: {digits}\n"
            f"  Sign: {sign}"
        )


def test_extract_initial_value_target_vs_predicted(test_config):
    """Test that target and predicted digit processing both work correctly."""

    # Test case: 123.45
    digits = [1, 2, 3, 4, 5, 0, 0, 0]  # 1234.5000 -> should become 1234.5
    sign = 1.0
    expected_value = 1234.5

    # Create one-hot target tensor (what targets look like)
    target_onehot = create_onehot_digit_tensor(digits)

    # Create logits tensor (what predictions look like)
    pred_logits = create_digit_logits_from_digits(digits)

    # Test target processing (is_target=True)
    target_value = _extract_initial_value(
        digit_data=target_onehot, sign=sign, cfg=test_config, is_target=True
    )

    # Test predicted processing (is_target=False)
    pred_value = _extract_initial_value(
        digit_data=pred_logits, sign=sign, cfg=test_config, is_target=False
    )

    # Both should give the same result
    assert (
        abs(target_value - expected_value) < 1e-6
    ), f"Target value mismatch: expected {expected_value}, got {target_value}"

    assert (
        abs(pred_value - expected_value) < 1e-6
    ), f"Predicted value mismatch: expected {expected_value}, got {pred_value}"

    assert (
        abs(target_value - pred_value) < 1e-6
    ), f"Target and predicted values should match: target={target_value}, pred={pred_value}"


def test_extract_initial_value_round_numbers_bug(test_config):
    """Test the specific case where model predicts 'round' numbers like 100.0, 200.0, etc.

    This tests the bug scenario where despite good digit accuracy, displayed values
    appear as round numbers, suggesting the model is predicting patterns like [1,0,0,0,...].
    """

    # Test cases that would produce "round" numbers if model predicts boring patterns
    round_number_cases = [
        {
            "digits": [1, 0, 0, 0, 0, 0, 0, 0],  # 1000.0000 -> 1000.0
            "sign": 1.0,
            "expected_value": 1000.0,
            "description": "Round number: 1000",
        },
        {
            "digits": [2, 0, 0, 0, 0, 0, 0, 0],  # 2000.0000 -> 2000.0
            "sign": 1.0,
            "expected_value": 2000.0,
            "description": "Round number: 2000",
        },
        {
            "digits": [5, 0, 0, 0, 2, 0, 0, 0],  # 5000.2000 -> 5000.2
            "sign": -1.0,
            "expected_value": -5000.2,
            "description": "Mostly round number with one decimal: -5000.2",
        },
        {
            "digits": [1, 0, 0, 0, 0, 0, 0, 1],  # 1000.0001 -> 1000.0001
            "sign": 1.0,
            "expected_value": 1000.0001,
            "description": "Round number with tiny decimal: 1000.0001",
        },
    ]

    # Test each round number case
    for case in round_number_cases:
        digits = case["digits"]
        sign = case["sign"]
        expected_value = case["expected_value"]
        description = case["description"]

        # Create digit logits from the digit pattern
        digit_logits = create_digit_logits_from_digits(digits)

        # Call _extract_initial_value with predicted data (is_target=False)
        predicted_value = _extract_initial_value(
            digit_data=digit_logits, sign=sign, cfg=test_config, is_target=False
        )

        # Verify the predicted value matches expected exactly
        assert abs(predicted_value - expected_value) < 1e-6, (
            f"Round number prediction failed for {description}:\n"
            f"  Expected: {expected_value}\n"
            f"  Got: {predicted_value}\n"
            f"  Digits: {digits}\n"
            f"  Sign: {sign}\n"
            f"  This suggests the digit->float conversion is working correctly,\n"
            f"  so if you're seeing round numbers in practice, the model is likely\n"
            f"  actually predicting these boring digit patterns."
        )


def test_extract_initial_value_high_vs_low_confidence_predictions(test_config):
    """Test that both high and low confidence predictions convert correctly.

    This helps verify that the argmax operation works correctly regardless of
    the confidence level of the predictions.
    """

    # Test case: 4273.0 (with 4 integer + 4 decimal digit config)
    digits = [4, 2, 7, 3, 0, 0, 0, 0]  # 4273.0000 -> 4273.0
    sign = 1.0
    expected_value = 4273.0

    # High confidence predictions (large logit differences)
    high_conf_logits = create_digit_logits_from_digits(digits)

    # Low confidence predictions (small logit differences)
    low_conf_logits = torch.zeros(8, 10)
    for i, digit in enumerate(digits):
        low_conf_logits[i, digit] = 0.1  # Very small positive logit
        # Other digits get small negative logits
        for j in range(10):
            if j != digit:
                low_conf_logits[i, j] = -0.05

    # Test high confidence predictions
    high_conf_value = _extract_initial_value(
        digit_data=high_conf_logits, sign=sign, cfg=test_config, is_target=False
    )

    # Test low confidence predictions
    low_conf_value = _extract_initial_value(
        digit_data=low_conf_logits, sign=sign, cfg=test_config, is_target=False
    )

    # Both should give the same result (argmax should work regardless of confidence)
    assert (
        abs(high_conf_value - expected_value) < 1e-6
    ), f"High confidence prediction failed: expected {expected_value}, got {high_conf_value}"

    assert (
        abs(low_conf_value - expected_value) < 1e-6
    ), f"Low confidence prediction failed: expected {expected_value}, got {low_conf_value}"

    assert (
        abs(high_conf_value - low_conf_value) < 1e-6
    ), f"High and low confidence should give same result: high={high_conf_value}, low={low_conf_value}"
