"""Test gate targets using actual generated expressions from the data pipeline."""

import pytest
import tiktoken
import torch

from data.dagset.generate_expression import generate_expression
from data.dagset.streaming import expression_to_tensors


def test_real_generated_expressions_gate_targets():
    """Test gate targets using expressions from the actual data generation pipeline."""

    print("=== Testing Gate Targets with Real Generated Expressions ===")

    # Use the same parameters as training
    dag_depth = 4
    max_digits = 4
    max_decimal_places = 4
    tokenizer = tiktoken.get_encoding("cl100k_base")

    all_gates = []
    valid_expressions = []

    # Generate several expressions and check their gate targets
    for seed in range(10):
        try:
            expressions, substrings, valid_mask = generate_expression(
                depth=dag_depth,
                seed=seed,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
                tokenizer=tokenizer,
            )

            for i, (expr, is_valid) in enumerate(zip(expressions, valid_mask)):
                if is_valid and hasattr(
                    expr, "args"
                ):  # Skip string/invalid expressions
                    try:
                        V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth)
                        g_flat = G.squeeze()
                        all_gates.append(g_flat)
                        valid_expressions.append(str(expr))

                        ones_count = (g_flat == 1.0).sum().item()
                        zeros_count = (g_flat == 0.0).sum().item()

                        print(
                            f"Seed {seed:2d}, Expr {i}: {str(expr):20} → G={g_flat.tolist()} ({ones_count} ones, {zeros_count} zeros)"
                        )

                    except Exception as e:
                        print(f"Seed {seed:2d}, Expr {i}: {str(expr):20} → ERROR: {e}")

        except Exception as e:
            print(f"Seed {seed:2d}: Generation failed with {e}")

    if all_gates:
        # Calculate overall statistics
        all_gates_tensor = torch.stack(all_gates)
        total_gates = all_gates_tensor.numel()
        total_ones = (all_gates_tensor == 1.0).sum().item()
        total_zeros = (all_gates_tensor == 0.0).sum().item()
        baseline_accuracy = total_ones / total_gates

        print(f"\n=== REAL DATA GATE ANALYSIS ===")
        print(f"Valid expressions analyzed: {len(all_gates)}")
        print(f"Total gate positions: {total_gates}")
        print(f"Total 1s (linear): {total_ones} ({total_ones/total_gates:.1%})")
        print(f"Total 0s (log): {total_zeros} ({total_zeros/total_gates:.1%})")
        print(f"Baseline accuracy if model always predicts 1: {baseline_accuracy:.1%}")

        # Analyze gate patterns
        gate_patterns = {}
        for gates in all_gates:
            pattern = tuple(gates.tolist())
            gate_patterns[pattern] = gate_patterns.get(pattern, 0) + 1

        print(f"\n=== GATE PATTERNS ===")
        for pattern, count in sorted(
            gate_patterns.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = count / len(all_gates) * 100
            print(f"Pattern {list(pattern)}: {count:2d} times ({percentage:4.1f}%)")

        if baseline_accuracy >= 0.75:
            print(f"\n🚨 CONFIRMED: Gate accuracy bug in real data!")
            print(f"   {baseline_accuracy:.0%} of gate targets are 1s")
            print(f"   Model can achieve high accuracy by just predicting 1s")
        else:
            print(f"\n✅ Real data looks more balanced than test expressions")

        # Assert that we found evidence of the gate bias issue
        assert (
            baseline_accuracy >= 0.5
        ), f"Expected high baseline accuracy due to gate bias, got {baseline_accuracy:.1%}"
        assert len(gate_patterns) > 0, "Expected to find gate patterns in real data"

        # Verify the most common pattern is all 1s (indicating bias)
        most_common_pattern = max(gate_patterns.items(), key=lambda x: x[1])
        assert most_common_pattern[0] == (
            1.0,
            1.0,
            1.0,
            1.0,
        ), f"Expected most common pattern to be all 1s, got {most_common_pattern[0]}"

    else:
        # If no valid expressions found, this is a test setup issue
        pytest.fail("No valid expressions found - test setup problem")


def test_gate_vs_operation_count():
    """Test if the number of actual operations matches the number of meaningful gate positions."""

    print("\n=== Testing Operation Count vs Gate Usage ===")

    dag_depth = 4
    tokenizer = tiktoken.get_encoding("cl100k_base")

    for seed in range(5):
        expressions, _, valid_mask = generate_expression(
            depth=dag_depth,
            seed=seed,
            max_digits=4,
            max_decimal_places=4,
            tokenizer=tokenizer,
        )

        for i, (expr, is_valid) in enumerate(zip(expressions, valid_mask)):
            if is_valid and hasattr(expr, "args"):
                # Count actual operations in the expression
                def count_operations(node):
                    if hasattr(node, "args") and len(node.args) > 1:
                        return 1 + sum(count_operations(arg) for arg in node.args)
                    return 0

                op_count = count_operations(expr)

                V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth)
                g_flat = G.squeeze()

                # Find meaningful gate positions (non-default values or actually used)
                # This is tricky because we need to know which positions are actually used

                print(f"Expression: {expr}")
                print(f"  Counted operations: {op_count}")
                print(f"  Gate tensor: {g_flat.tolist()}")
                print(
                    f"  Expected: First {op_count} positions should vary, rest might be default 1s"
                )
                print()


if __name__ == "__main__":
    baseline, patterns = test_real_generated_expressions_gate_targets()
    test_gate_vs_operation_count()

    print(f"\n=== FINAL DIAGNOSIS ===")
    if baseline >= 0.8:
        print(f"🔴 BUG CONFIRMED: {baseline:.0%} of real gate targets are 1s!")
        print(f"   This explains the 80% gate accuracy from training start.")
        print(f"   The model achieves high accuracy by learning to predict mostly 1s.")
        print(
            f"   Gate accuracy doesn't improve because the targets are wrong/meaningless."
        )
    elif baseline >= 0.6:
        print(f"🟡 LIKELY BUG: {baseline:.0%} of gate targets are 1s")
        print(f"   This could explain elevated gate accuracy")
    else:
        print(f"🟢 Gate distribution looks reasonable: {baseline:.0%} ones")
        print(f"   The 80% accuracy issue might be elsewhere")
