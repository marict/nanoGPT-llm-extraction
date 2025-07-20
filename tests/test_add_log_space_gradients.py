#!/usr/bin/env python
"""
Test gradient flow through the improved add_log_space function.
"""

import math

import torch

from models.dag_model import add_log_space


def test_gradient_flow_normal_case():
    """Test gradient flow through normal (non-near-cancellation) cases."""

    # Create tensors that require gradients
    lx = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)  # log10(100)
    ly = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)  # log10(10)
    sx = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    sy = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)

    # Forward pass: 100 + (-10) = 90
    result_sgn, result_log = add_log_space(sx, lx, sy, ly, ignore_clip=True)

    # Convert to scalar loss
    result_value = result_sgn * torch.pow(
        torch.tensor(10.0, dtype=torch.float64), result_log
    )
    loss = result_value.sum()

    print(f"Normal case: result_value = {result_value.item()}")
    print(f"Expected: ~90")

    # Backward pass
    loss.backward()

    # Check that gradients exist and are non-zero
    assert lx.grad is not None, "lx should have gradients"
    assert ly.grad is not None, "ly should have gradients"
    assert sx.grad is not None, "sx should have gradients"
    assert sy.grad is not None, "sy should have gradients"

    print(f"Gradients: lx.grad={lx.grad.item():.6f}, ly.grad={ly.grad.item():.6f}")
    print(f"           sx.grad={sx.grad.item():.6f}, sy.grad={sy.grad.item():.6f}")

    # Gradients should be non-zero (unless there's a mathematical reason for them to be zero)
    assert abs(lx.grad.item()) > 1e-10, f"lx gradient too small: {lx.grad.item()}"
    assert abs(sx.grad.item()) > 1e-10, f"sx gradient too small: {sx.grad.item()}"

    print("✓ Normal case: Gradients flow correctly")


def test_gradient_flow_near_cancellation():
    """Test gradient flow through near-cancellation (linear fallback) path."""

    # Create near-cancellation case: values that trigger linear arithmetic
    val_x = 2780.0566
    val_y = -2784.0

    lx = torch.tensor([math.log10(abs(val_x))], dtype=torch.float64, requires_grad=True)
    ly = torch.tensor([math.log10(abs(val_y))], dtype=torch.float64, requires_grad=True)
    sx = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    sy = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)

    print(f"Near-cancellation case: {val_x} + {val_y}")
    print(f"Log difference: {abs(lx.item() - ly.item()):.8f}")

    # Forward pass
    result_sgn, result_log = add_log_space(sx, lx, sy, ly, ignore_clip=True)

    # Convert to scalar loss
    result_value = result_sgn * torch.pow(
        torch.tensor(10.0, dtype=torch.float64), result_log
    )
    loss = result_value.sum()

    print(f"Result: {result_value.item()}")
    print(f"Expected: ~{val_x + val_y}")

    # Backward pass
    loss.backward()

    # Check that gradients exist and are finite
    assert lx.grad is not None, "lx should have gradients"
    assert ly.grad is not None, "ly should have gradients"
    assert sx.grad is not None, "sx should have gradients"
    assert sy.grad is not None, "sy should have gradients"

    assert torch.isfinite(lx.grad), f"lx gradient not finite: {lx.grad.item()}"
    assert torch.isfinite(ly.grad), f"ly gradient not finite: {ly.grad.item()}"
    assert torch.isfinite(sx.grad), f"sx gradient not finite: {sx.grad.item()}"
    assert torch.isfinite(sy.grad), f"sy gradient not finite: {sy.grad.item()}"

    print(f"Gradients: lx.grad={lx.grad.item():.6f}, ly.grad={ly.grad.item():.6f}")
    print(f"           sx.grad={sx.grad.item():.6f}, sy.grad={sy.grad.item():.6f}")

    # For near-cancellation, gradients might be large due to sensitivity
    # But they should be finite and meaningful
    assert abs(lx.grad.item()) > 1e-15, f"lx gradient too small: {lx.grad.item()}"
    assert abs(sx.grad.item()) > 1e-15, f"sx gradient too small: {sx.grad.item()}"

    print("✓ Near-cancellation case: Gradients flow correctly")


def test_gradient_flow_batch():
    """Test gradient flow through batch operations mixing normal and near-cancellation."""

    # Mix of normal and near-cancellation cases
    lx = torch.tensor(
        [2.0, math.log10(2780.0566)], dtype=torch.float64, requires_grad=True
    )
    ly = torch.tensor(
        [1.0, math.log10(2784.0)], dtype=torch.float64, requires_grad=True
    )
    sx = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
    sy = torch.tensor([-1.0, -1.0], dtype=torch.float64, requires_grad=True)

    # Forward pass
    result_sgn, result_log = add_log_space(sx, lx, sy, ly, ignore_clip=True)

    # Convert to scalar loss (sum of both results)
    result_values = result_sgn * torch.pow(
        torch.tensor(10.0, dtype=torch.float64), result_log
    )
    loss = result_values.sum()

    print(f"Batch results: {result_values.tolist()}")

    # Backward pass
    loss.backward()

    # Check all gradients
    for i, (name, grad) in enumerate(
        [("lx", lx.grad), ("ly", ly.grad), ("sx", sx.grad), ("sy", sy.grad)]
    ):
        assert grad is not None, f"{name} should have gradients"
        assert torch.isfinite(grad).all(), f"{name} gradients not finite: {grad}"
        print(f"{name}.grad = {grad.tolist()}")

    print("✓ Batch case: Gradients flow correctly")


def test_gradient_numerical_stability():
    """Test that gradients don't explode or vanish in edge cases."""

    # Test various edge cases
    test_cases = [
        ("Very close values", 1000.0001, -1000.0),
        ("Perfect cancellation", 100.0, -100.0),
        ("Small numbers", 0.001, -0.0009),
    ]

    for description, val_x, val_y in test_cases:
        if abs(val_x + val_y) < 1e-12:  # Skip true perfect cancellation
            continue

        lx = torch.tensor(
            [math.log10(abs(val_x))], dtype=torch.float64, requires_grad=True
        )
        ly = torch.tensor(
            [math.log10(abs(val_y))], dtype=torch.float64, requires_grad=True
        )
        sx = torch.tensor(
            [1.0 if val_x >= 0 else -1.0], dtype=torch.float64, requires_grad=True
        )
        sy = torch.tensor(
            [1.0 if val_y >= 0 else -1.0], dtype=torch.float64, requires_grad=True
        )

        try:
            result_sgn, result_log = add_log_space(sx, lx, sy, ly, ignore_clip=True)
            result_value = result_sgn * torch.pow(
                torch.tensor(10.0, dtype=torch.float64), result_log
            )
            loss = result_value.sum()
            loss.backward()

            # Check gradient magnitudes are reasonable
            grad_magnitudes = [
                abs(g.item())
                for g in [lx.grad, ly.grad, sx.grad, sy.grad]
                if g is not None
            ]
            max_grad = max(grad_magnitudes)

            print(f"{description}: max gradient magnitude = {max_grad:.2e}")
            assert max_grad < 1e10, f"Gradient too large for {description}: {max_grad}"
            assert max_grad > 1e-15, f"Gradient too small for {description}: {max_grad}"

        except Exception as e:
            print(f"Error in {description}: {e}")
            raise

    print("✓ Gradient numerical stability: OK")


if __name__ == "__main__":
    test_gradient_flow_normal_case()
    print()
    test_gradient_flow_near_cancellation()
    print()
    test_gradient_flow_batch()
    print()
    test_gradient_numerical_stability()
    print("\n✅ All gradient tests passed!")
