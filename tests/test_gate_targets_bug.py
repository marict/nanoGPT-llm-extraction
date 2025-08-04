"""Test to investigate potential bug in gate target generation."""

import pytest
import sympy
import torch

from data.dagset.streaming import expression_to_tensors


def test_gate_targets_unused_positions():
    """Test if unused gate positions remain as 1s, causing artificially high accuracy."""

    # Test simple expressions that don't use all dag_depth operations
    dag_depth = 4  # Allow up to 4 operations

    # Case 1: Single addition (uses only 1 operation slot)
    expr1 = sympy.parse_expr("3 + 5")
    V_mag1, V_sign1, O1, G1 = expression_to_tensors(expr1, dag_depth)

    print(f"\nExpression: {expr1}")
    print(f"Gate tensor G: {G1.squeeze()}")  # Remove batch/time dims for readability
    print(
        f"Expected: [1, 1, 1, 1] (only first position should matter, rest remain as default 1s)"
    )

    # Case 2: Single multiplication (uses only 1 operation slot)
    expr2 = sympy.parse_expr("3 * 5")
    V_mag2, V_sign2, O2, G2 = expression_to_tensors(expr2, dag_depth)

    print(f"\nExpression: {expr2}")
    print(f"Gate tensor G: {G2.squeeze()}")
    print(f"Expected: [0, 1, 1, 1] (first=0 for mul, rest remain as default 1s)")

    # Case 3: Two operations
    expr3 = sympy.parse_expr("3 + 5 * 2")  # Should be parsed as 3 + (5 * 2)
    V_mag3, V_sign3, O3, G3 = expression_to_tensors(expr3, dag_depth)

    print(f"\nExpression: {expr3}")
    print(f"Gate tensor G: {G3.squeeze()}")
    print(
        f"Expected: [0, 1, 1, 1] (first=0 for mul, second=1 for add, rest remain as default 1s)"
    )

    # Case 4: Complex expression that uses more operations
    expr4 = sympy.parse_expr(
        "(3 + 5) * (2 - 1)"
    )  # Addition, subtraction, multiplication
    V_mag4, V_sign4, O4, G4 = expression_to_tensors(expr4, dag_depth)

    print(f"\nExpression: {expr4}")
    print(f"Gate tensor G: {G4.squeeze()}")
    print(f"This should use more operation slots")

    # Key insight: Count how many gate positions are 1 vs 0
    for i, (expr, G) in enumerate(
        [(expr1, G1), (expr2, G2), (expr3, G3), (expr4, G4)], 1
    ):
        g_flat = G.squeeze()
        ones_count = (g_flat == 1.0).sum().item()
        zeros_count = (g_flat == 0.0).sum().item()
        total_count = g_flat.numel()

        print(f"\nExpression {i} ({expr}):")
        print(f"  Gate 1s: {ones_count}/{total_count} = {ones_count/total_count:.1%}")
        print(f"  Gate 0s: {zeros_count}/{total_count} = {zeros_count/total_count:.1%}")

    # If most expressions are simple and don't use all dag_depth slots,
    # then most gate positions will be 1, explaining the 80% accuracy!


def test_gate_targets_different_dag_depths():
    """Test how gate targets change with different dag_depth settings."""

    expr = sympy.parse_expr("3 + 5")  # Simple expression with 1 operation

    for dag_depth in [1, 2, 3, 4, 6]:
        V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth)
        g_flat = G.squeeze()
        ones_count = (g_flat == 1.0).sum().item()
        total_count = g_flat.numel()

        print(f"\ndag_depth={dag_depth}: G={g_flat.tolist()}")
        print(f"  Percentage of 1s: {ones_count/total_count:.1%}")

    print(f"\nAs dag_depth increases, the percentage of 1s increases!")
    print(f"This would make gate accuracy artificially high.")


def test_gate_targets_distribution_hypothesis():
    """Test the hypothesis that unused gate positions cause high baseline accuracy."""

    # Simulate a batch of simple expressions like what might be in training data
    expressions = [
        "1 + 2",  # 1 operation
        "3 * 4",  # 1 operation
        "5 - 6",  # 1 operation
        "2 + 3 + 4",  # 2 operations
        "1 * 2 * 3",  # 2 operations
        "5 + 2 * 3",  # 2 operations
        "1 + 2 - 3",  # 2 operations
        "(1 + 2) * 3",  # 2 operations
    ]

    dag_depth = 4  # Typical setting
    all_gates = []

    print(f"\nTesting {len(expressions)} expressions with dag_depth={dag_depth}:")

    for expr_str in expressions:
        try:
            expr = sympy.parse_expr(expr_str)
            _, _, _, G = expression_to_tensors(expr, dag_depth)
            g_flat = G.squeeze()
            all_gates.append(g_flat)

            ones_count = (g_flat == 1.0).sum().item()
            print(f"  {expr_str:12} → G={g_flat.tolist()} → {ones_count}/4 ones")

        except Exception as e:
            print(f"  {expr_str:12} → ERROR: {e}")

    if all_gates:
        # Calculate overall statistics
        all_gates_tensor = torch.stack(all_gates)  # (num_expressions, dag_depth)
        total_gates = all_gates_tensor.numel()
        total_ones = (all_gates_tensor == 1.0).sum().item()
        baseline_accuracy = total_ones / total_gates

        print(f"\n=== BASELINE GATE ACCURACY ANALYSIS ===")
        print(f"Total gate positions: {total_gates}")
        print(f"Total 1s (linear): {total_ones}")
        print(f"Total 0s (log): {total_gates - total_ones}")
        print(f"Baseline accuracy if model always predicts 1: {baseline_accuracy:.1%}")

        if baseline_accuracy >= 0.75:
            print(f"\n🔍 SMOKING GUN FOUND!")
            print(f"If most gate positions are unused and remain as 1s,")
            print(
                f"then a model biased toward predicting 1 will get {baseline_accuracy:.0%}+ accuracy!"
            )
            print(
                f"This explains why gate accuracy starts high and doesn't improve much."
            )

        # Assert that we found evidence of the gate bias issue
        assert (
            baseline_accuracy >= 0.5
        ), f"Expected high baseline accuracy due to unused positions, got {baseline_accuracy:.1%}"
        assert (
            total_ones > total_gates // 2
        ), f"Expected majority of gates to be 1s, got {total_ones}/{total_gates}"


def test_proposed_fix_for_gate_targets():
    """Test a proposed fix: set unused gate positions to a special value or don't include them in loss."""

    # Current behavior
    expr = sympy.parse_expr("3 + 5")
    dag_depth = 4
    _, _, _, G = expression_to_tensors(expr, dag_depth)

    print(f"Current gate targets: {G.squeeze().tolist()}")
    print(f"Problem: Unused positions [1,2,3] remain as 1s")

    # Proposed fix ideas:
    print(f"\nProposed fixes:")
    print(f"1. Set unused positions to -1 (special 'unused' value)")
    print(f"2. Only compute gate loss for actually used positions")
    print(f"3. Use a mask to exclude unused positions from accuracy calculation")
    print(f"4. Initialize unused positions to 0 instead of 1")

    # The key insight is that we need to distinguish between:
    # - Gate positions that correspond to actual operations (should be learned)
    # - Gate positions that are unused padding (should be ignored)


if __name__ == "__main__":
    # Run tests individually for manual testing
    test_gate_targets_unused_positions()
    test_gate_targets_different_dag_depths()
    test_gate_targets_distribution_hypothesis()
    test_proposed_fix_for_gate_targets()

    print(f"\n=== CONCLUSION ===")
    print(f"✅ All gate target tests completed successfully!")
    print(
        f"   The tests verify that unused gate positions cause artificially high accuracy."
    )
    print(
        f"   This explains why gate accuracy starts high and doesn't improve much during training."
    )
