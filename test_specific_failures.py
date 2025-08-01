#!/usr/bin/env python3
"""
Targeted tests for specific failure patterns in tensor conversion.
"""

import sympy
import torch

from scratch.newdag import NewDAGExecutor, expression_to_tensors


def test_nested_operations():
    """Test expressions with nested operations that create intermediate Pow nodes."""

    test_cases = [
        # Nested divisions - create Pow nodes that must be available as operands
        ("8 / (2 * 3)", 1.3333333333333333),  # 8 / 6
        ("12 / (4 / 2)", 6.0),  # 12 / 2
        ("20 / (10 / (5 / 2))", 5.0),  # 20 / 4
        # Nested with mixed operations
        ("10 - 6 / (2 * 3)", 9.0),  # 10 - 1
        ("5 + 8 / (4 + 4)", 5.0 + 1.0),  # 5 + 1
        # Multiple levels of nesting
        ("24 / ((2 * 3) * (1 + 1))", 2.0),  # 24 / 12
        ("100 / (10 * (2 + 3))", 2.0),  # 100 / 50
    ]

    print("=== Testing Nested Operations ===")
    all_passed = True

    for expr_str, expected in test_cases:
        try:
            expr = sympy.sympify(expr_str, evaluate=False)

            print(f"\nTesting: {expr_str}")
            print(f"SymPy: {expr}")
            print(f"Expected: {expected}")

            # Show the postorder traversal to understand structure
            from sympy import postorder_traversal

            print("Traversal:")
            for i, node in enumerate(postorder_traversal(expr)):
                node_type = type(node).__name__
                print(f"  {i}: {node} ({node_type})")

            # Convert and execute
            V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=15)
            executor = NewDAGExecutor(dag_depth=15)
            result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=False)

            error = abs(result.item() - expected)
            match = error < 1e-6

            print(f"DAG result: {result.item()}")
            print(f"Error: {error}")
            print(f"✅ PASS" if match else f"❌ FAIL")

            if not match:
                all_passed = False

        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            all_passed = False

    return all_passed


def test_multi_argument_operations():
    """Test multi-argument operations that should work with our new approach."""

    test_cases = [
        # Multi-argument additions
        ("1 + 2 + 3 + 4", 10.0),
        ("5 + 10 + 15", 30.0),
        # Multi-argument multiplications
        ("2 * 3 * 4 * 5", 120.0),
        ("1.5 * 2 * 4", 12.0),
        # Multi-argument subtractions
        ("20 - 5 - 3 - 2", 10.0),
        ("100 - 25 - 25", 50.0),
        # Multi-argument divisions
        ("120 / 2 / 3 / 4", 5.0),
        ("64 / 4 / 2", 8.0),
        # Mixed multi-argument
        ("2 * 3 + 4 * 5", 26.0),
        ("20 / 2 - 5", 5.0),
    ]

    print("\n=== Testing Multi-Argument Operations ===")
    all_passed = True

    for expr_str, expected in test_cases:
        try:
            expr = sympy.sympify(expr_str, evaluate=False)

            print(f"\nTesting: {expr_str}")
            print(f"SymPy: {expr} with {len(expr.args)} args")

            V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=15)
            executor = NewDAGExecutor(dag_depth=15)
            result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=False)

            error = abs(result.item() - expected)
            match = error < 1e-6

            print(f"DAG result: {result.item()}")
            print(f"✅ PASS" if match else f"❌ FAIL")

            if not match:
                all_passed = False

        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            all_passed = False

    return all_passed


def test_sign_and_negation_issues():
    """Test expressions that have sign/negation issues."""

    test_cases = [
        # Basic negations
        ("-(2 * 3)", -6.0),
        ("-5 + 3", -2.0),
        ("10 * (-2)", -20.0),
        # Multiple negations
        ("(-2) * (-3)", 6.0),
        ("-(-5)", 5.0),
        # Negations in complex expressions
        ("-2 * 3 + 4", -2.0),  # -6 + 4
        ("5 - (-3) * 2", 11.0),  # 5 - (-6) = 5 + 6
        # From actual failing cases (simplified)
        ("-25 + 8 * (-4) * (-3)", 71.0),  # -25 + 96
    ]

    print("\n=== Testing Sign and Negation Issues ===")
    all_passed = True

    for expr_str, expected in test_cases:
        try:
            expr = sympy.sympify(expr_str, evaluate=False)

            print(f"\nTesting: {expr_str}")
            print(f"SymPy: {expr}")

            V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=15)
            executor = NewDAGExecutor(dag_depth=15)
            result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=False)

            error = abs(result.item() - expected)
            match = error < 1e-6

            print(f"Expected: {expected}")
            print(f"DAG result: {result.item()}")
            print(f"✅ PASS" if match else f"❌ FAIL")

            if not match:
                all_passed = False

        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            all_passed = False

    return all_passed


def main():
    """Run all targeted failure tests."""

    print("🎯 TARGETED FAILURE TESTS")
    print("=" * 50)

    test1_pass = test_multi_argument_operations()
    test2_pass = test_nested_operations()
    test3_pass = test_sign_and_negation_issues()

    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"Multi-argument operations: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Nested operations: {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"Sign/negation issues: {'✅ PASS' if test3_pass else '❌ FAIL'}")

    overall_pass = test1_pass and test2_pass and test3_pass
    print(
        f"\n🎯 OVERALL: {'✅ ALL TESTS PASSED!' if overall_pass else '❌ Some tests failed'}"
    )

    return overall_pass


if __name__ == "__main__":
    main()
