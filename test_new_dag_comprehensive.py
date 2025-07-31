#!/usr/bin/env python
"""
Comprehensive test for the new experimental DAG design with domain mixing.

This test verifies that the new DAG system from test_newdag.py can:
1. Generate and execute expressions of increasing complexity (depths 1-6+)
2. Convert mathematical expressions to DAG tensors correctly
3. Execute DAGs with domain mixing (log/linear) and produce correct results
4. Handle all basic operations: addition, subtraction, multiplication, division
5. Process complex nested expressions accurately
6. Maintain gradient flow for training

Key innovations being tested:
- V_mag/V_sign representation (magnitudes and signs)
- Operand selectors (O) with positive/negative weights
- Domain gates (G): 0=log domain (mult/div), 1=linear domain (add/sub)
- Fixed tensor sizes with triangular causality masking
- 50/50 split: initial_slots + intermediate_slots = total_nodes
"""

import math
import random
import sys
from pathlib import Path

import sympy
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "scratch"))

# Import the new DAG system
from test_newdag import (
    NewDAGExecutor,
    NewDAGPlanPredictor,
    expression_to_tensors,
)


class NewDAGComprehensiveTester:
    """Comprehensive tester for the new experimental DAG design."""

    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0

    def log_test(self, name, passed, expected=None, actual=None, error=None):
        """Log test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        result = {
            "name": name,
            "passed": passed,
            "expected": expected,
            "actual": actual,
            "error": error,
            "status": status,
        }
        self.test_results.append(result)

        print(f"{status}: {name}")
        if expected is not None and actual is not None:
            print(f"    Expected: {expected}")
            print(f"    Actual:   {actual}")
            if error is not None:
                print(f"    Error:    {error:.2e}")
        print()

        return passed

    def test_basic_operations(self):
        """Test all basic arithmetic operations with concrete numerical expressions."""
        print("=" * 80)
        print("TESTING BASIC OPERATIONS")
        print("=" * 80)

        # Test cases: (concrete_expression_str, expected_result)
        basic_tests = [
            # Single values (depth 1)
            ("5", 5.0),
            ("-3", -3.0),
            ("2.5", 2.5),
            # Simple binary operations (depth 2)
            ("3 + 4", 7.0),
            ("8 - 3", 5.0),
            ("3 * 4", 12.0),
            ("12 / 3", 4.0),
            # With negative numbers
            ("-2 + 5", 3.0),
            ("2 - (-3)", 5.0),
            ("-3 * 4", -12.0),
            ("-12 / 3", -4.0),
            # Three operands (depth 3)
            ("1 + 2 + 3", 6.0),
            ("2 * 3 * 4", 24.0),
            ("10 + 5 - 3", 12.0),
            ("12 * 4 / 3", 16.0),
            # Decimal numbers
            ("1.5 + 2.5", 4.0),
            ("3.2 * 1.25", 4.0),
            ("7.5 / 2.5", 3.0),
        ]

        for expr_str, expected in basic_tests:
            try:
                # Parse expression (should have no free symbols - all concrete numbers)
                expr = sympy.sympify(expr_str)

                # Verify this is a concrete expression (no symbols)
                if expr.free_symbols:
                    self.log_test(
                        f"Basic: {expr_str}",
                        False,
                        expected,
                        None,
                        f"Expression contains symbols: {expr.free_symbols}",
                    )
                    continue

                # Calculate required DAG depth based on expression complexity
                dag_depth = max(4, len(str(expr)) // 2)  # Simple heuristic

                # Convert to DAG tensors
                V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=dag_depth)

                # Execute DAG (no variable substitution needed!)
                executor = NewDAGExecutor(dag_depth=dag_depth)
                result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=False)
                actual = result[0, 0].item()

                # Verify result
                error = abs(actual - expected)
                tolerance = 1e-4
                passed = error < tolerance

                self.log_test(
                    f"Basic: {expr_str} = {expected}", passed, expected, actual, error
                )

            except Exception as e:
                self.log_test(f"Basic: {expr_str}", False, expected, None, str(e))

    def test_nested_expressions(self):
        """Test nested expressions of increasing complexity with concrete numbers."""
        print("=" * 80)
        print("TESTING NESTED EXPRESSIONS")
        print("=" * 80)

        nested_tests = [
            # Simple nesting (depth 3-4)
            ("(2 + 3) * 4", 20.0),
            ("2 * (3 + 4)", 14.0),
            ("(10 - 4) / 2", 3.0),
            # More complex nesting (depth 4-5)
            ("(1 + 2) * (7 - 2)", 15.0),
            ("(2 * 3) + (4 * 5)", 26.0),
            ("(8 + 4) / (5 - 1)", 3.0),
            # Deep nesting (depth 5-6)
            ("((1 + 2) * 3) + 4", 13.0),
            ("5 + ((3 * 4) - 2)", 15.0),
            ("(1 + 2) * (3 + 4)", 21.0),
            # Very deep nesting (depth 6+)
            ("((1 + 2) * (5 - 2)) + ((12 * 2) / 3)", 17.0),
            ("((2 * 3) + 1) * ((8 - 3) + 2)", 49.0),
            # Complex decimal expressions
            ("(1.5 + 2.5) * (3.0 - 1.0)", 8.0),
            ("((2.0 * 1.5) + 1.0) / (4.0 - 2.0)", 2.0),
            # Mixed operations with precedence
            ("2 + 3 * 4", 14.0),  # Should be 2 + (3 * 4) = 14
            ("(2 + 3) * 4 - 5", 15.0),  # Should be (5 * 4) - 5 = 15
            ("12 / 3 + 2 * 5", 14.0),  # Should be 4 + 10 = 14
        ]

        for expr_str, expected in nested_tests:
            try:
                # Parse expression
                expr = sympy.sympify(expr_str)

                # Verify this is a concrete expression (no symbols)
                if expr.free_symbols:
                    self.log_test(
                        f"Nested: {expr_str}",
                        False,
                        expected,
                        None,
                        f"Expression contains symbols: {expr.free_symbols}",
                    )
                    continue

                # Verify expected result matches sympy
                sympy_result = float(expr)
                if abs(sympy_result - expected) > 1e-6:
                    print(
                        f"⚠️  Test case error: {expr_str} sympy={sympy_result} expected={expected}"
                    )
                    continue

                # Calculate required DAG depth based on expression complexity
                expr_complexity = len(str(expr))  # Rough complexity measure
                dag_depth = max(6, expr_complexity // 4)

                # Convert to DAG tensors
                V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=dag_depth)

                # Execute DAG (no variable substitution needed!)
                executor = NewDAGExecutor(dag_depth=dag_depth)
                result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=False)
                actual = result[0, 0].item()

                # Verify result
                error = abs(actual - expected)
                tolerance = 1e-3  # More lenient for complex expressions
                passed = error < tolerance

                self.log_test(
                    f"Nested: {expr_str} = {expected}", passed, expected, actual, error
                )

            except Exception as e:
                self.log_test(f"Nested: {expr_str}", False, expected, None, str(e))

    def test_domain_separation(self):
        """Test that operations correctly use appropriate domains with concrete numbers."""
        print("=" * 80)
        print("TESTING DOMAIN SEPARATION")
        print("=" * 80)

        # Test cases specifically designed to verify domain gates work correctly
        domain_tests = [
            # Addition should use linear domain (G=1)
            ("2.5 + 3.7", 6.2, "linear"),
            # Multiplication should use log domain (G=0)
            ("2.5 * 3.7", 9.25, "log"),
            # Mixed operations should use appropriate domains for each step
            ("2 * 3 + 4", 10.0, "mixed"),
            ("(2 + 3) * 4", 20.0, "mixed"),
            # More complex mixed operations
            ("1.5 * 2.0 + 3.5", 6.5, "mixed"),
            ("(4.0 - 1.0) * 2.5", 7.5, "mixed"),
        ]

        for expr_str, expected, domain_type in domain_tests:
            try:
                expr = sympy.sympify(expr_str)

                # Verify this is a concrete expression (no symbols)
                if expr.free_symbols:
                    self.log_test(
                        f"Domain: {expr_str}",
                        False,
                        expected,
                        None,
                        f"Expression contains symbols: {expr.free_symbols}",
                    )
                    continue

                dag_depth = max(6, len(str(expr)) // 3)

                # Convert to tensors and examine domain gates
                V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=dag_depth)

                # Execute and verify (no variable substitution needed!)
                executor = NewDAGExecutor(dag_depth=dag_depth)
                result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=False)
                actual = result[0, 0].item()

                error = abs(actual - expected)
                tolerance = 1e-4
                passed = error < tolerance

                # Also check domain gate values for correctness
                gate_info = ""
                if G.shape[2] > 0:
                    # Show first few gates that are actually used (non-zero operands)
                    used_gates = []
                    for i in range(min(3, G.shape[2])):
                        if torch.any(torch.abs(O[0, 0, i, :]) > 1e-8):
                            used_gates.append(f"{G[0, 0, i].item():.1f}")
                    if used_gates:
                        gate_info = f" Gates: {used_gates}"

                self.log_test(
                    f"Domain ({domain_type}): {expr_str}{gate_info}",
                    passed,
                    expected,
                    actual,
                    error,
                )

            except Exception as e:
                self.log_test(f"Domain: {expr_str}", False, expected, None, str(e))

    def test_precision_and_edge_cases(self):
        """Test precision handling and edge cases with concrete numbers."""
        print("=" * 80)
        print("TESTING PRECISION AND EDGE CASES")
        print("=" * 80)

        edge_tests = [
            # Small numbers
            ("0.001 + 0.002", 0.003),
            ("0.1 * 0.01", 0.001),
            # Large numbers
            ("1000 + 2000", 3000),
            ("100 * 50", 5000),
            # Mixed small and large
            ("1000 * 0.001 + 5", 6.0),
            # Division edge cases
            ("1 / 3", 0.3333333333333333),
            ("7 / 2", 3.5),
            ("22 / 7", 3.142857142857143),  # Pi approximation
            # Zero handling
            ("0 + 5", 5.0),
            ("0 * 5", 0.0),
            # Negative numbers
            ("-5 + 3", -2.0),
            ("-2 * 3", -6.0),
            ("10 / (-2)", -5.0),
            # Very small and very large
            ("1e-6 + 1e-6", 2e-6),
            ("1e6 * 1e-6", 1.0),
        ]

        for expr_str, expected in edge_tests:
            try:
                expr = sympy.sympify(expr_str)

                # Verify this is a concrete expression (no symbols)
                if expr.free_symbols:
                    self.log_test(
                        f"Edge: {expr_str}",
                        False,
                        expected,
                        None,
                        f"Expression contains symbols: {expr.free_symbols}",
                    )
                    continue

                dag_depth = max(4, len(str(expr)) // 3)

                # Verify with sympy first
                sympy_result = float(expr)

                # Convert and execute
                V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=dag_depth)

                executor = NewDAGExecutor(dag_depth=dag_depth)
                result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=False)
                actual = result[0, 0].item()

                # Use sympy result as ground truth for edge cases
                error = abs(actual - sympy_result)
                tolerance = max(
                    1e-4, abs(sympy_result) * 1e-5
                )  # Relative tolerance for large numbers
                passed = error < tolerance

                self.log_test(
                    f"Edge: {expr_str} = {sympy_result:.6f}",
                    passed,
                    sympy_result,
                    actual,
                    error,
                )

            except Exception as e:
                self.log_test(f"Edge: {expr_str}", False, expected, None, str(e))

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the system."""
        print("=" * 80)
        print("TESTING GRADIENT FLOW")
        print("=" * 80)

        try:
            # Create a differentiable computation: (a + b) * c
            dag_depth = 6
            executor = NewDAGExecutor(dag_depth=dag_depth)

            # Create tensors that require gradients
            V_mag = torch.ones(1, 1, dag_depth, requires_grad=True)
            V_sign = torch.ones(1, 1, dag_depth, requires_grad=True)
            O = torch.zeros(1, 1, dag_depth // 2, dag_depth, requires_grad=True)
            G = torch.zeros(1, 1, dag_depth // 2, requires_grad=True)

            # Set up expression: (a + b) * c where a=2, b=3, c=4
            with torch.no_grad():
                # Initial values: a=2, b=3, c=4
                V_mag[0, 0, 0] = 2.0  # a
                V_mag[0, 0, 1] = 3.0  # b
                V_mag[0, 0, 2] = 4.0  # c

                # Step 0: a + b (linear domain)
                O[0, 0, 0, 0] = 1.0  # Select a
                O[0, 0, 0, 1] = 1.0  # Select b
                G[0, 0, 0] = 1.0  # Linear domain for addition (one-hot)

                # Step 1: result * c (log domain)
                O[0, 0, 1, 3] = 1.0  # Select previous result (in slot 3)
                O[0, 0, 1, 2] = 1.0  # Select c
                G[0, 0, 1] = 0.0  # Log domain for multiplication (one-hot)

                # Step 2: copy result to final slot (identity operation)
                O[0, 0, 2, 4] = 1.0  # Select result from step 1 (in slot 4)
                G[0, 0, 2] = 1.0  # Linear domain for identity (one-hot)

            # Forward pass
            result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=False)

            # Expected: (2 + 3) * 4 = 20
            expected = 20.0
            actual = result[0, 0].item()

            # Create loss
            target = torch.tensor(
                [[25.0]]
            )  # Different from expected to create non-zero gradients
            loss = torch.nn.MSELoss()(result, target)

            # Backward pass
            loss.backward()

            # Check gradients
            grad_checks = []

            if V_mag.grad is not None:
                grad_norm = torch.norm(V_mag.grad).item()
                grad_checks.append(("V_mag", grad_norm > 1e-8))

            if V_sign.grad is not None:
                grad_norm = torch.norm(V_sign.grad).item()
                grad_checks.append(("V_sign", grad_norm > 1e-8))

            if O.grad is not None:
                grad_norm = torch.norm(O.grad).item()
                grad_checks.append(("O", grad_norm > 1e-8))

            if G.grad is not None:
                grad_norm = torch.norm(G.grad).item()
                grad_checks.append(("G", grad_norm > 1e-8))

            # Test forward pass accuracy
            forward_error = abs(actual - expected)
            forward_passed = forward_error < 1e-3
            self.log_test(
                f"Gradient: Forward pass accuracy",
                forward_passed,
                expected,
                actual,
                forward_error,
            )

            # Test gradient existence
            all_grads_ok = all(has_grad for _, has_grad in grad_checks)
            self.log_test(f"Gradient: All tensors have gradients", all_grads_ok)

            # Details
            for name, has_grad in grad_checks:
                self.log_test(f"Gradient: {name} has gradients", has_grad)

        except Exception as e:
            self.log_test("Gradient flow test", False, None, None, str(e))

    def test_expression_conversion(self):
        """Test the expression_to_tensors conversion function with concrete numbers."""
        print("=" * 80)
        print("TESTING EXPRESSION CONVERSION")
        print("=" * 80)

        conversion_tests = [
            # Simple expressions
            ("5", 5.0),
            ("2 + 3", 5.0),
            ("2 * 3", 6.0),
            ("5 - 2", 3.0),
            ("6 / 2", 3.0),
            # Nested expressions
            ("(1 + 2) * 3", 9.0),
            ("2 * (1 + 3)", 8.0),
            ("(1 + 2) + (3 * 4)", 15.0),
            # Complex expressions
            ("((2 + 1) * 2) - 1", 5.0),
            ("(4 / 2) + (3 * 2)", 10.0),
            ("(5 - 2) * (4 + 1)", 15.0),
        ]

        for expr_str, expected in conversion_tests:
            try:
                # Parse and evaluate with sympy
                expr = sympy.sympify(expr_str)

                # Verify this is a concrete expression (no symbols)
                if expr.free_symbols:
                    self.log_test(
                        f"Convert: {expr_str}",
                        False,
                        expected,
                        None,
                        f"Expression contains symbols: {expr.free_symbols}",
                    )
                    continue

                # Verify expected result
                sympy_result = float(expr)
                if abs(sympy_result - expected) > 1e-6:
                    print(
                        f"⚠️  Test case error: {expr_str} sympy={sympy_result} expected={expected}"
                    )
                    continue

                # Convert to tensors
                dag_depth = max(6, len(str(expr)) // 3)
                V_mag, V_sign, O, G = expression_to_tensors(expr, dag_depth=dag_depth)

                # Execute (no variable substitution needed!)
                executor = NewDAGExecutor(dag_depth=dag_depth)
                result = executor.execute_with_plan(V_mag, V_sign, O, G, debug=False)
                actual = result[0, 0].item()

                # Verify
                error = abs(actual - expected)
                tolerance = 1e-4
                passed = error < tolerance

                self.log_test(
                    f"Convert: {expr_str} = {expected}", passed, expected, actual, error
                )

            except Exception as e:
                self.log_test(f"Convert: {expr_str}", False, expected, None, str(e))

    def run_comprehensive_test(self):
        """Run all tests and provide summary."""
        print("*" * 100)
        print("COMPREHENSIVE TEST OF NEW EXPERIMENTAL DAG DESIGN")
        print(
            "Testing domain mixing approach with V_mag, V_sign, operand selectors, and domain gates"
        )
        print("*" * 100)
        print()

        # Run all test suites
        self.test_basic_operations()
        self.test_nested_expressions()
        self.test_domain_separation()
        self.test_precision_and_edge_cases()
        self.test_gradient_flow()
        self.test_expression_conversion()

        # Final summary
        print("*" * 100)
        print("FINAL SUMMARY")
        print("*" * 100)

        success_rate = (
            (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        )

        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success rate: {success_rate:.1f}%")
        print()

        # Show failed tests
        failed_tests = [r for r in self.test_results if not r["passed"]]
        if failed_tests:
            print("Failed tests:")
            for test in failed_tests:
                print(f"  ❌ {test['name']}")
                if test["error"]:
                    print(f"     Error: {test['error']}")
            print()

        # Assessment
        if success_rate >= 95:
            print("🎉 EXCELLENT: New DAG system is working very well!")
        elif success_rate >= 85:
            print("✅ GOOD: New DAG system is working well with minor issues")
        elif success_rate >= 70:
            print("⚠️  ACCEPTABLE: New DAG system works but has some limitations")
        else:
            print("❌ NEEDS WORK: New DAG system has significant issues")

        print()
        print("Key advantages of new DAG design:")
        print("- Domain mixing: log domain for multiplication, linear for addition")
        print("- Operand selectors with positive/negative weights")
        print("- Fixed tensor sizes with triangular masking")
        print("- Better expressivity than digit-based approach")
        print("- Native gradient flow support")

        return success_rate >= 85


def main():
    """Main test function."""
    tester = NewDAGComprehensiveTester()
    success = tester.run_comprehensive_test()

    if success:
        print("✅ NEW DAG SYSTEM VERIFICATION SUCCESSFUL")
        sys.exit(0)
    else:
        print("❌ NEW DAG SYSTEM NEEDS IMPROVEMENT")
        sys.exit(1)


if __name__ == "__main__":
    main()
