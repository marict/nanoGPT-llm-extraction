#!/usr/bin/env python
"""
Comprehensive test for newdag.py that validates DAG execution against SymPy evaluation.

This test:
1. Generates expressions using streaming.py with depth=6, max_digits=6, max_decimal_places=6
2. Combines and flattens results until we have about 1000 expressions
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
from scratch.newdag import NewDAGExecutor, expression_to_tensors, purge_negative_ones


def test_purge_negative_ones():
    """
    Test the purge_negative_ones function that simplifies -1 * x to -x.
    """
    print("=== Testing purge_negative_ones function ===")

    # Test case 1: Simple -1 * x
    x = sympy.Symbol("x")
    expr1 = sympy.Float(-1.0) * x
    result1 = purge_negative_ones(expr1)
    expected1 = -x
    assert result1.equals(expected1), f"Expected {expected1}, got {result1}"
    print(f"✓ Test 1 passed: {expr1} -> {result1}")

    # Test case 2: x * -1 (reverse order)
    expr2 = x * sympy.Float(-1.0)
    result2 = purge_negative_ones(expr2)
    expected2 = -x
    assert result2.equals(expected2), f"Expected {expected2}, got {result2}"
    print(f"✓ Test 2 passed: {expr2} -> {result2}")

    # Test case 3: Complex expression with -1 multiplication
    y = sympy.Symbol("y")
    expr3 = (sympy.Float(-1.0) * x) + (y * sympy.Float(-1.0))
    result3 = purge_negative_ones(expr3)
    expected3 = -x + (-y)  # This should evaluate to -x - y
    # For comparison, let's check if they're mathematically equivalent
    diff = (result3 - expected3).simplify()
    assert diff == 0, f"Expected equivalent to {expected3}, got {result3}"
    print(f"✓ Test 3 passed: {expr3} -> {result3}")

    # Test case 4: Expression without -1 multiplication (should be unchanged)
    expr4 = x + y * 2
    result4 = purge_negative_ones(expr4)
    assert result4.equals(
        expr4
    ), f"Expression should be unchanged: {expr4} vs {result4}"
    print(f"✓ Test 4 passed: {expr4} remains unchanged")

    # Test case 5: Nested expression with multiple -1 multiplications
    z = sympy.Symbol("z")
    expr5 = (sympy.Float(-1.0) * x) * (sympy.Float(-1.0) * y) + z
    result5 = purge_negative_ones(expr5)
    # After purging: (-x) * (-y) + z = x*y + z
    expected5 = x * y + z
    # Check mathematical equivalence
    diff = (result5 - expected5).simplify()
    assert diff == 0, f"Expected equivalent to {expected5}, got {result5}"
    print(f"✓ Test 5 passed: {expr5} -> {result5}")

    print("All purge_negative_ones tests passed! ✅\n")
    return True


def test_newdag_comprehensive():
    """
    Comprehensive test that validates DAG execution against SymPy evaluation.
    """
    print("=== Testing newdag.py with 1000 expressions ===")

    # Test parameters
    depth = 6
    max_digits = 6
    max_decimal_places = 6
    base = 10
    tokenizer = get_encoding("gpt2")
    target_count = 1000

    print(f"Target parameters:")
    print(f"  Depth: {depth}")
    print(f"  Max digits: {max_digits}")
    print(f"  Max decimal places: {max_decimal_places}")
    print(f"  Base: {base}")
    print(f"  Target expression count: {target_count}")
    print()

    # Step 1: Generate expressions until we have enough
    print("Step 1: Generating expressions...")
    all_expressions = set()
    seed = 42

    num_expressions = 100

    for i in tqdm.tqdm(
        range(num_expressions), total=num_expressions, desc="Generating expressions"
    ):
        expressions, _, _ = generate_expression(
            depth=depth,
            seed=i,
            max_digits=max_digits,
            tokenizer=tokenizer,
            max_decimal_places=max_decimal_places,
        )
        all_expressions.update(expressions)

    print(f"Generated {len(all_expressions)} expressions")
    all_expressions = [expr for expr in all_expressions if expr != "not valid"]
    print(f"After filtering, {len(all_expressions)} expressions remain")

    # Order expressions by str length
    all_expressions.sort(key=lambda x: len(str(x)), reverse=True)

    # Print top 10 expressions
    for expr in all_expressions[:10]:
        print(f"Expression: {expr}")
        print(f"Length: {len(str(expr))}")
        print()
    # Print bottom 10 expressions
    for expr in all_expressions[-10:]:
        print(f"Expression: {expr}")
        print(f"Length: {len(str(expr))}")
        print()

    print(f"\nStep 3: Testing {target_count} expressions...")
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
        # a. Get target state from expression_to_tensors
        V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=depth)

        # b. Get final result from execute_with_plan
        dag_result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=False)
        dag_value = dag_result[0, 0].item()

        # c. Get final result from SymPy evaluation
        sympy_value = float(expr)

        # d. Check if results match
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
    print(f"Total expressions tested: {target_count}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed / target_count * 100:.2f}%")

    if failed > 0:
        print(
            f"\n❌ TEST FAILED: {failed} out of {target_count} expressions failed validation"
        )
        if failed > 10:
            print(
                f"   (Showing only first 10 failures above, {failed - 10} more failures occurred)"
            )
        return False
    else:
        print(f"\n✅ TEST PASSED: All {target_count} expressions passed validation!")
        return True


if __name__ == "__main__":
    # Run both tests
    purge_test_success = test_purge_negative_ones()
    comprehensive_test_success = test_newdag_comprehensive()

    overall_success = purge_test_success and comprehensive_test_success
    sys.exit(0 if overall_success else 1)
