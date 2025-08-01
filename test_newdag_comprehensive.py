#!/usr/bin/env python
"""
Comprehensive test for newdag.py that validates DAG execution against SymPy evaluation.

This test:
1. Generates expressions using generate_expression with depth=15, max_digits=6, max_decimal_places=6
2. Combines and flattens results until we have about 2000 expressions
3. Removes invalid expressions
4. For each expression, compares:
   - DAG execution result (expression_to_tensors -> execute_with_plan)
   - Direct SymPy evaluation
"""

import sys
from pathlib import Path

import tqdm

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "data"))
sys.path.append(str(Path(__file__).parent / "scratch"))

import sympy
from tiktoken import get_encoding

from data.dagset.generate_expression import generate_expression
from scratch.newdag import NewDAGExecutor, expression_to_tensors


def test_newdag_comprehensive():
    """
    Comprehensive test that validates DAG execution against SymPy evaluation.
    """

    # Test parameters
    depth = 15
    max_digits = 6
    max_decimal_places = 6
    base = 10
    tokenizer = get_encoding("gpt2")

    print(f"Target parameters:")
    print(f"  Depth: {depth}")
    print(f"  Max digits: {max_digits}")
    print(f"  Max decimal places: {max_decimal_places}")
    print(f"  Base: {base}")
    print()

    # Step 1: Generate expressions until we have enough
    print("Step 1: Generating expressions...")
    all_expressions = set()
    num_expressions = 100

    for i in tqdm.tqdm(
        range(num_expressions), total=num_expressions, desc="Generating expressions"
    ):
        expressions, _, _ = generate_expression(
            depth=depth,
            seed=i + 54325432543,
            max_digits=max_digits,
            tokenizer=tokenizer,
            max_decimal_places=max_decimal_places,
        )
        all_expressions.update(expressions)

    print(f"Generated {len(all_expressions)} expressions")
    all_expressions = [expr for expr in all_expressions if expr != "not valid"]
    print(f"After filtering, {len(all_expressions)} expressions remain")

    # Filter out overly complex expressions (longer than 60 characters)
    all_expressions = [expr for expr in all_expressions if len(str(expr)) <= 60]
    print(f"After complexity filtering, {len(all_expressions)} expressions remain")

    # Order expressions by str length
    all_expressions.sort(key=lambda x: len(str(x)), reverse=True)

    print(f"\nStep 3: Testing {len(all_expressions)} expressions...")
    print("Comparing DAG execution vs SymPy evaluation...")

    executor = NewDAGExecutor(dag_depth=depth)
    passed = 0
    failed = 0
    tolerance = 1e-3

    for i, expr in tqdm.tqdm(
        enumerate(all_expressions),
        total=len(all_expressions),
        desc="Testing expressions",
    ):
        # Get target state from expression_to_tensors
        V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=depth)

        # Get final result from execute_with_plan
        dag_result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=False)

        dag_value = dag_result[0, 0].item()
        sympy_value = float(expr)

        # Check if results match
        error = abs(sympy_value - dag_value)
        relative_error = error / max(abs(sympy_value), 1e-8)

        if error < tolerance or relative_error < tolerance:
            passed += 1
        else:
            failed += 1
            if failed <= 10:  # Only show first 10 failures to avoid spam
                print(f"  FAIL #{failed}: expr={expr}")
                print(f"    SymPy: {sympy_value}")
                print(f"    DAG:   {dag_value}")
                print(f"    Error: {error:.6f} (relative: {relative_error:.6f})")

    # Step 4: Report results
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
        return False
    else:
        print(
            f"\n✅ TEST PASSED: All {len(all_expressions)} expressions passed validation!"
        )
        return True


if __name__ == "__main__":
    # Run comprehensive test
    comprehensive_test_success = test_newdag_comprehensive()
    sys.exit(0 if comprehensive_test_success else 1)
