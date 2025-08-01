"""
Comprehensive test for DAG execution that validates against SymPy evaluation.

This test:
1. Generates expressions using generate_expression
2. Combines and flattens results until we have about 1000 expressions
3. Removes invalid expressions
4. For each expression, compares:
   - DAG execution result (expression_to_tensors -> DAGExecutor)
   - Direct SymPy evaluation
"""

import pytest
import sympy
import torch
import tqdm
from tiktoken import get_encoding

from data.dagset.generate_expression import generate_expression
from data.dagset.streaming import expression_to_tensors
from models.dag_model import DAGExecutor


def test_dag_executor_simple():
    """
    Simple test with a few basic expressions to verify DAG executor works correctly.
    This test ensures no major regressions before running the comprehensive test.
    """
    from tiktoken import get_encoding

    tokenizer = get_encoding("gpt2")
    depth = 4
    executor = DAGExecutor(dag_depth=depth)

    # Test with some simple expressions
    simple_expressions = [
        sympy.parse_expr("2.5 + 3.7"),  # Simple addition
        sympy.parse_expr("4.0 * 1.5"),  # Simple multiplication
        sympy.parse_expr("10.0 - 3.0"),  # Simple subtraction
        sympy.parse_expr("8.0 / 2.0"),  # Simple division
    ]

    failures = []
    for expr in simple_expressions:
        try:
            # Convert expression to tensors
            V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=depth)

            # Execute DAG
            dag_result = executor.forward(V_mag, V_sign, O, G)
            dag_value = dag_result[0, 0].item()

            # Compare with SymPy
            sympy_value = float(expr.evalf())

            error = abs(sympy_value - dag_value)
            relative_error = error / max(abs(sympy_value), 1e-8)

            print(f"Expression: {expr}")
            print(f"  SymPy: {sympy_value}")
            print(f"  DAG:   {dag_value}")
            print(f"  Error: {error:.6f}")

            if error > 1e-3 and relative_error > 1e-3:
                failures.append(
                    {
                        "expr": str(expr),
                        "sympy_value": sympy_value,
                        "dag_value": dag_value,
                        "error": error,
                    }
                )

        except Exception as e:
            failures.append({"expr": str(expr), "error": f"Exception: {e}"})
            print(f"Expression: {expr} - Exception: {e}")

    if failures:
        pytest.fail(
            f"Simple DAG executor test failed on {len(failures)} expressions: {failures}"
        )

    print("✅ Simple DAG executor test passed!")


def test_dag_executor_comprehensive():
    """
    Comprehensive test that validates DAG execution against SymPy evaluation.
    Tests the production DAGExecutor to ensure no regressions from the old implementation.
    """

    # Test parameters
    depth = 6
    max_digits = 6
    max_decimal_places = 6
    tokenizer = get_encoding("gpt2")

    print(f"Target parameters:")
    print(f"  Depth: {depth}")
    print(f"  Max digits: {max_digits}")
    print(f"  Max decimal places: {max_decimal_places}")
    print()

    # Step 1: Generate expressions until we have enough
    print("Step 1: Generating expressions...")
    all_expressions = set()
    num_expressions = 1000

    for i in tqdm.tqdm(
        range(num_expressions), total=num_expressions, desc="Generating expressions"
    ):
        expressions, _, _ = generate_expression(
            depth=depth,
            seed=i,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            tokenizer=tokenizer,
        )
        all_expressions.update(expressions)

    print(f"Generated {len(all_expressions)} expressions")
    all_expressions = [expr for expr in all_expressions if expr != "not valid"]
    print(f"After filtering, {len(all_expressions)} expressions remain")

    # Order expressions by str length for consistent testing
    all_expressions.sort(key=lambda x: len(str(x)), reverse=True)

    print(f"\nStep 2: Testing {len(all_expressions)} expressions...")
    print("Comparing DAG execution vs SymPy evaluation...")

    # Use the production DAGExecutor from dag_model.py
    executor = DAGExecutor(dag_depth=depth)
    passed = 0
    failed = 0
    tolerance = 1e-3
    failures = []

    for i, expr in tqdm.tqdm(
        enumerate(all_expressions),
        total=len(all_expressions),
        desc="Testing expressions",
    ):
        try:
            # Get target state from expression_to_tensors (from streaming.py)
            V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=depth)

            # Get final result from DAGExecutor
            dag_result = executor.forward(V_mag, V_sign, O, G)
            dag_value = dag_result[0, 0].item()

            # Compare with direct SymPy evaluation
            sympy_value = float(expr)

            # Check if results match within tolerance
            error = abs(sympy_value - dag_value)
            relative_error = error / max(abs(sympy_value), 1e-8)

            if error < tolerance or relative_error < tolerance:
                passed += 1
            else:
                failed += 1
                failure_info = {
                    "expr": str(expr),
                    "sympy_value": sympy_value,
                    "dag_value": dag_value,
                    "error": error,
                    "relative_error": relative_error,
                }
                failures.append(failure_info)

                if failed <= 10:  # Only show first 10 failures to avoid spam
                    print(f"  FAIL #{failed}: expr={expr}")
                    print(f"    SymPy: {sympy_value}")
                    print(f"    DAG:   {dag_value}")
                    print(f"    Error: {error:.6f} (relative: {relative_error:.6f})")

        except Exception as e:
            failed += 1
            failure_info = {
                "expr": str(expr),
                "error": f"Exception: {e}",
                "sympy_value": None,
                "dag_value": None,
                "relative_error": None,
            }
            failures.append(failure_info)

            if failed <= 10:
                print(f"  FAIL #{failed}: expr={expr}")
                print(f"    Exception: {e}")

    # Step 3: Report results and assert success
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total expressions tested: {len(all_expressions)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed / len(all_expressions) * 100:.2f}%")

    if failed > 0:
        print(
            f"\n❌ TEST FAILED: {failed} out of {len(all_expressions)} expressions failed validation"
        )
        if failed > 10:
            print(
                f"   (Showing only first 10 failures above, {failed - 10} more failures occurred)"
            )

        # Provide detailed failure information for debugging
        pytest.fail(
            f"DAG executor failed on {failed}/{len(all_expressions)} expressions. "
            f"First failure: {failures[0] if failures else 'N/A'}"
        )
    else:
        print(
            f"\n✅ TEST PASSED: All {len(all_expressions)} expressions passed validation!"
        )

    # Assert success for pytest
    assert failed == 0, f"DAG executor failed on {failed} expressions"


if __name__ == "__main__":
    # Run comprehensive test directly
    test_dag_executor_comprehensive()
