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
import tqdm
from tiktoken import get_encoding

from data.dagset.generate_expression import generate_expression
from data.dagset.streaming import expressions_to_tensors
from models.dag_model import DAGExecutor


def test_specific_expression_fix():
    """
    Test the specific expression that was failing due to the sharpening bug.
    """
    depth = 4
    executor = DAGExecutor(dag_depth=depth, max_digits=8, max_decimal_places=4)

    expression_str = "1.0*(-45.1399*7019.3999 - 5.2659)"
    expr = sympy.parse_expr(expression_str)

    # Convert expression to tensors
    target_tensors, valid_mask = expressions_to_tensors(
        [expr], depth=depth, max_digits=8, max_decimal_places=4
    )
    assert valid_mask[0], f"Expression {expr} should be valid"

    target = target_tensors[0]
    # Add batch/time dimensions for DAGExecutor
    digit_logits = target["target_digits"].unsqueeze(0).unsqueeze(0)
    V_sign = target["target_V_sign"].unsqueeze(0).unsqueeze(0)
    O = target["target_O"].unsqueeze(0).unsqueeze(0)
    G = target["target_G"].unsqueeze(0).unsqueeze(0)

    # Execute DAG
    dag_result = executor.forward(digit_logits, V_sign, O, G)
    dag_value = dag_result[0, 0].item()

    expected_value = float(expr.evalf())
    assert (
        abs(dag_value - expected_value) < 1e-2
    ), f"Expected {expected_value}, but got {dag_value}"


def test_dag_executor_simple():
    """
    Simple test with a few basic expressions to verify DAG executor works correctly.
    This test ensures no major regressions before running the comprehensive test.
    """
    depth = 4
    executor = DAGExecutor(dag_depth=depth, max_digits=4, max_decimal_places=4)

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
            target_tensors, valid_mask = expressions_to_tensors(
                [expr], depth=depth, max_digits=4, max_decimal_places=4
            )
            if not valid_mask[0]:
                raise ValueError(f"Expression {expr} is not valid")

            target = target_tensors[0]
            # Add batch/time dimensions for DAGExecutor
            digit_logits = target["target_digits"].unsqueeze(0).unsqueeze(0)
            V_sign = target["target_V_sign"].unsqueeze(0).unsqueeze(0)
            O = target["target_O"].unsqueeze(0).unsqueeze(0)
            G = target["target_G"].unsqueeze(0).unsqueeze(0)

            # Execute DAG
            dag_result = executor.forward(digit_logits, V_sign, O, G)
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
    depth = 4
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
    all_expressions_set = set()  # Use set for deduplication
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
        all_expressions_set.update(expressions)

    print(f"Generated {len(all_expressions_set)} expressions")
    # Convert to sorted list for deterministic ordering
    all_expressions = [expr for expr in all_expressions_set if expr != "not valid"]
    print(f"After filtering, {len(all_expressions)} expressions remain")

    # Order expressions by length (descending) then alphabetically for fully deterministic order
    all_expressions.sort(key=lambda x: (-len(str(x)), str(x)))

    print(f"\nStep 2: Testing {len(all_expressions)} expressions...")
    print("Comparing DAG execution vs SymPy evaluation...")

    # Use the production DAGExecutor from dag_model.py
    executor = DAGExecutor(
        dag_depth=depth, max_digits=max_digits, max_decimal_places=max_decimal_places
    )
    passed = 0
    failed = 0
    tolerance = 1e-2  # Increased tolerance for digit prediction system approximation
    failures = []

    for i, expr in tqdm.tqdm(
        enumerate(all_expressions),
        total=len(all_expressions),
        desc="Testing expressions",
    ):
        try:
            # Get target state directly in digit tensor format
            target_tensors, valid_mask = expressions_to_tensors(
                [expr],
                depth=depth,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
            )
            if not valid_mask[0]:
                raise ValueError(f"Expression {expr} is not valid")

            target = target_tensors[0]
            # Add batch/time dimensions for DAGExecutor
            digit_logits = target["target_digits"].unsqueeze(0).unsqueeze(0)
            V_sign = target["target_V_sign"].unsqueeze(0).unsqueeze(0)
            O = target["target_O"].unsqueeze(0).unsqueeze(0)
            G = target["target_G"].unsqueeze(0).unsqueeze(0)

            # Get final result from DAGExecutor
            dag_result = executor.forward(digit_logits, V_sign, O, G)
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
                "index": i,
                "expr": str(expr),
                "error": f"Exception: {e}",
                "sympy_value": None,
                "dag_value": None,
                "relative_error": None,
            }
            failures.append(failure_info)
            print(failure_info)

    # Step 3: Report results and assert success
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total expressions tested: {len(all_expressions)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    success_rate = passed / len(all_expressions) * 100
    print(f"Success rate: {success_rate:.2f}%")

    # We should always pass this test.
    min_success_rate = 100

    if success_rate < min_success_rate:
        print(
            f"\n❌ TEST FAILED: {failed} out of {len(all_expressions)} expressions failed validation"
        )
        print(
            f"   Success rate {success_rate:.2f}% is below minimum required {min_success_rate}%"
        )
        if failed > 10:
            print(
                f"   (Showing only first 10 failures above, {failed - 10} more failures occurred)"
            )

        # Provide detailed failure information for debugging
        pytest.fail(
            f"DAG executor success rate {success_rate:.2f}% below minimum {min_success_rate}%. "
            f"Failed on {failed}/{len(all_expressions)} expressions. "
            f"First failure: {failures[0] if failures else 'N/A'}"
        )
    else:
        print(
            f"\n✅ TEST PASSED: Success rate {success_rate:.2f}% meets minimum {min_success_rate}%"
        )
        if failed > 0:
            print(
                f"   Note: {failed} expressions failed but this is within acceptable tolerance for digit prediction system"
            )
        else:
            print(
                f"   Perfect: All {len(all_expressions)} expressions passed validation!"
            )


if __name__ == "__main__":
    # Run comprehensive test directly
    test_dag_executor_comprehensive()
