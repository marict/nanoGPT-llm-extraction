#!/usr/bin/env python
"""
Test the improved add_log_space function for near-cancellation accuracy.
"""

import math

import pytest
import torch

from models.dag_model import add_log_space


def test_specific_failing_case():
    """Test the exact case that was failing: 2780.0566 + (-2784.0) ≈ -3.943"""

    # The specific values from your failing case
    val_x = 2780.0566200000003
    val_y = -2784.0
    expected = val_x + val_y  # ≈ -3.9433799999997

    # Convert to sign/log representation
    sx = torch.tensor([1.0], dtype=torch.float64)
    lx = torch.tensor([math.log10(abs(val_x))], dtype=torch.float64)
    sy = torch.tensor([-1.0], dtype=torch.float64)
    ly = torch.tensor([math.log10(abs(val_y))], dtype=torch.float64)

    print(f"Input values: {val_x} + ({val_y}) = {expected}")
    print(f"Log magnitudes: lx={lx.item():.6f}, ly={ly.item():.6f}")
    print(f"Log difference: {abs(lx.item() - ly.item()):.8f}")

    # Test the improved add_log_space function
    result_sgn, result_log = add_log_space(sx, lx, sy, ly, ignore_clip=True)

    # Convert back to linear value
    result_value = result_sgn.item() * (10 ** result_log.item())

    print(f"Result: {result_value}")
    print(f"Expected: {expected}")
    print(f"Error: {abs(result_value - expected)}")
    print(f"Relative error: {abs(result_value - expected) / abs(expected):.8f}")

    # Assert the result is much more accurate than before
    # Previously got -6.403..., now should get close to -3.943...
    assert (
        abs(result_value - expected) < 0.1
    ), f"Result {result_value} not close to expected {expected}"
    assert (
        abs(result_value - expected) / abs(expected) < 0.01
    ), f"Relative error too large: {abs(result_value - expected) / abs(expected)}"


def test_near_cancellation_cases():
    """Test various near-cancellation scenarios."""

    test_cases = [
        # (val_x, val_y, description)
        (1000.0, -999.9, "Small difference"),
        (2780.0566, -2784.0, "Original failing case"),
        (1.0001, -1.0, "Very small positive result"),
        (100.123, -100.120, "Tiny difference"),
        (1e6, -999999.0, "Large numbers, small difference"),
    ]

    for val_x, val_y, description in test_cases:
        expected = val_x + val_y

        # Skip if result is too close to zero (log space struggles with very tiny numbers)
        if abs(expected) < 1e-10:
            continue

        # Convert to sign/log representation
        sx = torch.tensor([1.0 if val_x >= 0 else -1.0], dtype=torch.float64)
        lx = torch.tensor([math.log10(abs(val_x))], dtype=torch.float64)
        sy = torch.tensor([1.0 if val_y >= 0 else -1.0], dtype=torch.float64)
        ly = torch.tensor([math.log10(abs(val_y))], dtype=torch.float64)

        # Test add_log_space
        result_sgn, result_log = add_log_space(sx, lx, sy, ly, ignore_clip=True)
        result_value = result_sgn.item() * (10 ** result_log.item())

        rel_error = (
            abs(result_value - expected) / abs(expected)
            if expected != 0
            else abs(result_value)
        )

        print(f"{description}: {val_x} + ({val_y}) = {expected}")
        print(f"  Got: {result_value}, Rel error: {rel_error:.8f}")

        # Allow for some numerical error, but should be much better than before
        assert (
            rel_error < 0.05
        ), f"Case '{description}': Relative error {rel_error} too large"


def test_normal_cases_still_work():
    """Ensure normal (non-near-cancellation) cases still work correctly."""

    test_cases = [
        (100.0, 50.0),  # Different magnitudes, same sign
        (100.0, -50.0),  # Different magnitudes, opposite sign
        (1e5, 1e3),  # Very different magnitudes
        (-200.0, -300.0),  # Both negative
        (1.5, 2.7),  # Small numbers
    ]

    for val_x, val_y in test_cases:
        expected = val_x + val_y

        # Convert to sign/log representation
        sx = torch.tensor([1.0 if val_x >= 0 else -1.0], dtype=torch.float64)
        lx = torch.tensor([math.log10(abs(val_x))], dtype=torch.float64)
        sy = torch.tensor([1.0 if val_y >= 0 else -1.0], dtype=torch.float64)
        ly = torch.tensor([math.log10(abs(val_y))], dtype=torch.float64)

        # Test add_log_space
        result_sgn, result_log = add_log_space(sx, lx, sy, ly, ignore_clip=True)
        result_value = result_sgn.item() * (10 ** result_log.item())

        rel_error = (
            abs(result_value - expected) / abs(expected)
            if expected != 0
            else abs(result_value)
        )

        # Normal cases should have very small error
        assert (
            rel_error < 1e-10
        ), f"Normal case {val_x} + {val_y}: error {rel_error} too large"


def test_batch_near_cancellation():
    """Test that the vectorized version works correctly with batches containing near-cancellation."""

    # Create a batch with mix of normal and near-cancellation cases
    vals_x = [100.0, 2780.0566, 1000.0, 50.0]
    vals_y = [-50.0, -2784.0, -999.9, 25.0]
    expected = [vx + vy for vx, vy in zip(vals_x, vals_y)]

    # Convert to tensors
    sx = torch.tensor([1.0 if vx >= 0 else -1.0 for vx in vals_x], dtype=torch.float64)
    lx = torch.tensor([math.log10(abs(vx)) for vx in vals_x], dtype=torch.float64)
    sy = torch.tensor([1.0 if vy >= 0 else -1.0 for vy in vals_y], dtype=torch.float64)
    ly = torch.tensor([math.log10(abs(vy)) for vy in vals_y], dtype=torch.float64)

    # Test batch operation
    result_sgn, result_log = add_log_space(sx, lx, sy, ly, ignore_clip=True)
    result_values = (
        result_sgn * torch.pow(torch.tensor(10.0, dtype=torch.float64), result_log)
    ).tolist()

    for i, (result, expect) in enumerate(zip(result_values, expected)):
        rel_error = abs(result - expect) / abs(expect) if expect != 0 else abs(result)
        print(
            f"Batch[{i}]: {vals_x[i]} + {vals_y[i]} = {expect}, got {result}, rel_error={rel_error:.8f}"
        )
        assert rel_error < 0.05, f"Batch element {i}: error {rel_error} too large"


if __name__ == "__main__":
    test_specific_failing_case()
    test_near_cancellation_cases()
    test_normal_cases_still_work()
    test_batch_near_cancellation()
    print("All tests passed!")
