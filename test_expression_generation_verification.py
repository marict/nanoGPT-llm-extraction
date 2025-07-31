#!/usr/bin/env python
"""
Comprehensive test to verify expression generation and DAG conversion for depths 1-6.

This test verifies that:
1. Expressions can be generated at each depth level
2. All intermediate DAGs can be converted to tensors
3. DAG execution produces the same result as sympy evaluation
4. No errors are hidden during the process
"""

import math
import random
import sys
from pathlib import Path

import sympy
import torch
from tiktoken import get_encoding

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import from the project
from data.dagset.streaming import (
    expressions_to_tensors,
    generate_expression,
    plan_to_tensors,
    string_to_expression,
)
from models.dag_model import execute_stack


class ExpressionGenerationVerifier:
    """Comprehensive verification of expression generation and DAG execution."""

    def __init__(self, max_digits=4, max_decimal_places=6, base=10):
        self.max_digits = max_digits
        self.max_decimal_places = max_decimal_places
        self.base = base
        self.tokenizer = get_encoding("gpt2")

    def verify_single_expression(
        self, expressions, substrings, valid_mask, depth, seed
    ):
        """Verify a single generated expression and all its intermediate DAGs."""
        print(f"\n{'='*60}")
        print(f"TESTING EXPRESSION (depth={depth}, seed={seed})")
        print(f"{'='*60}")

        results = []
        errors = []

        # Get the final expression for comparison
        final_expr = expressions[-1] if expressions else None
        final_substring = substrings[-1] if substrings else ""

        print(f"Final expression: {final_expr}")
        print(f"Final substring: '{final_substring}'")
        print(f"Number of intermediate steps: {len(expressions)}")

        if final_expr == "not valid":
            print(
                "⚠️  Final expression is invalid, but we'll still test intermediate steps"
            )
        else:
            # Evaluate the final expression with sympy
            try:
                sympy_result = float(final_expr)
                print(f"SymPy evaluation: {sympy_result}")
            except Exception as e:
                print(f"❌ Failed to evaluate final expression with SymPy: {e}")
                sympy_result = None

        # Convert all expressions to tensors
        try:
            tensor_results, tensor_valid_mask = expressions_to_tensors(
                expressions,
                depth=depth,
                max_digits=self.max_digits,
                max_decimal_places=self.max_decimal_places,
                base=self.base,
            )
            print(
                f"✅ Successfully converted {len(tensor_results)} expressions to tensors"
            )
        except Exception as e:
            error_msg = f"❌ Failed to convert expressions to tensors: {e}"
            print(error_msg)
            errors.append(error_msg)
            return results, errors

        # Test each intermediate DAG
        for i, (expr, substring, is_valid, tensor_dict, tensor_is_valid) in enumerate(
            zip(expressions, substrings, valid_mask, tensor_results, tensor_valid_mask)
        ):
            print(f"\n--- Step {i+1}/{len(expressions)} ---")
            print(f"Expression: {expr}")
            print(f"Substring: '{substring}'")
            print(f"Valid (generation): {is_valid}")
            print(f"Valid (tensor): {tensor_is_valid}")

            step_result = {
                "step": i + 1,
                "expression": expr,
                "substring": substring,
                "valid_generation": is_valid,
                "valid_tensor": tensor_is_valid,
                "sympy_result": None,
                "dag_result": None,
                "error": None,
                "match": False,
            }

            if expr == "not valid":
                print("⚠️  Expression marked as 'not valid', testing zero DAG")
                step_result["sympy_result"] = 0.0  # Zero DAG should give 0
            else:
                # Evaluate with sympy
                try:
                    sympy_result = float(expr)
                    step_result["sympy_result"] = sympy_result
                    print(f"SymPy result: {sympy_result}")
                except Exception as e:
                    error_msg = f"Failed to evaluate with SymPy: {e}"
                    print(f"❌ {error_msg}")
                    step_result["error"] = error_msg
                    errors.append(f"Step {i+1}: {error_msg}")
                    results.append(step_result)
                    continue

            # Execute the DAG
            try:
                # Prepare tensors for execution
                initial_sgn = (
                    tensor_dict["target_initial_sgn"].unsqueeze(0).unsqueeze(0)
                )  # (1, 1, N)
                digit_probs = (
                    tensor_dict["target_initial_digits"].unsqueeze(0).unsqueeze(0)
                )  # (1, 1, N, D, base)
                operation_probs = (
                    tensor_dict["target_operation_probs"].unsqueeze(0).unsqueeze(0)
                )  # (1, 1, depth, n_ops)

                print(f"Tensor shapes:")
                print(f"  initial_sgn: {initial_sgn.shape}")
                print(f"  digit_probs: {digit_probs.shape}")
                print(f"  operation_probs: {operation_probs.shape}")

                # Execute the stack
                final_sgn, final_log = execute_stack(
                    initial_sgn,
                    digit_probs,
                    operation_probs,
                    max_digits=self.max_digits,
                    max_decimal_places=self.max_decimal_places,
                    base=self.base,
                    ignore_clip=False,
                )

                dag_result = float(final_sgn.squeeze() * torch.exp(final_log.squeeze()))
                step_result["dag_result"] = dag_result
                print(f"DAG result: {dag_result}")

                # Check if results match
                if step_result["sympy_result"] is not None:
                    error = abs(dag_result - step_result["sympy_result"])
                    tolerance = 1e-3  # Allow for numerical precision differences

                    if error < tolerance:
                        step_result["match"] = True
                        print(f"✅ Results match (error: {error:.2e})")
                    else:
                        error_msg = f"Results don't match (error: {error:.2e}, tolerance: {tolerance})"
                        print(f"❌ {error_msg}")
                        step_result["error"] = error_msg
                        errors.append(f"Step {i+1}: {error_msg}")

            except Exception as e:
                error_msg = f"Failed to execute DAG: {e}"
                print(f"❌ {error_msg}")
                step_result["error"] = error_msg
                errors.append(f"Step {i+1}: {error_msg}")

            results.append(step_result)

        return results, errors

    def test_depth(self, depth, num_samples=5):
        """Test expression generation for a specific depth."""
        print(f"\n{'#'*80}")
        print(f"TESTING DEPTH {depth}")
        print(f"{'#'*80}")

        all_results = []
        all_errors = []

        for sample in range(num_samples):
            seed = 1000 + depth * 100 + sample
            print(f"\n--- Sample {sample+1}/{num_samples} (seed={seed}) ---")

            try:
                # Generate expression
                expressions, substrings, valid_mask = generate_expression(
                    depth=depth,
                    seed=seed,
                    max_digits=self.max_digits,
                    max_decimal_places=self.max_decimal_places,
                    tokenizer=self.tokenizer,
                    base=self.base,
                )

                print(f"Generated {len(expressions)} intermediate expressions")

                # Verify the expression
                results, errors = self.verify_single_expression(
                    expressions, substrings, valid_mask, depth, seed
                )

                all_results.extend(results)
                all_errors.extend(errors)

                # Summary for this sample
                valid_count = sum(1 for r in results if r["match"])
                total_count = len(results)
                print(
                    f"\nSample summary: {valid_count}/{total_count} steps verified successfully"
                )

            except Exception as e:
                error_msg = f"Failed to generate/test expression for depth {depth}, sample {sample+1}: {e}"
                print(f"❌ {error_msg}")
                all_errors.append(error_msg)

        # Summary for this depth
        valid_count = sum(1 for r in all_results if r["match"])
        total_count = len(all_results)
        error_count = len(all_errors)

        print(f"\n{'='*50}")
        print(f"DEPTH {depth} SUMMARY")
        print(f"{'='*50}")
        print(f"Total steps tested: {total_count}")
        print(f"Steps verified successfully: {valid_count}")
        print(f"Steps with errors: {error_count}")
        print(
            f"Success rate: {valid_count/total_count*100:.1f}%"
            if total_count > 0
            else "No tests completed"
        )

        if all_errors:
            print(f"\nErrors encountered:")
            for i, error in enumerate(all_errors[:10], 1):  # Show first 10 errors
                print(f"  {i}. {error}")
            if len(all_errors) > 10:
                print(f"  ... and {len(all_errors) - 10} more errors")

        return all_results, all_errors

    def run_full_test(self, max_depth=6, samples_per_depth=3):
        """Run the full test suite for depths 1 to max_depth."""
        print(f"{'*'*100}")
        print(f"COMPREHENSIVE EXPRESSION GENERATION VERIFICATION")
        print(f"Testing depths 1-{max_depth} with {samples_per_depth} samples each")
        print(
            f"Max digits: {self.max_digits}, Max decimal places: {self.max_decimal_places}, Base: {self.base}"
        )
        print(f"{'*'*100}")

        all_results = []
        all_errors = []
        depth_summaries = []

        for depth in range(1, max_depth + 1):
            try:
                results, errors = self.test_depth(depth, samples_per_depth)
                all_results.extend(results)
                all_errors.extend(errors)

                # Calculate depth summary
                valid_count = sum(1 for r in results if r["match"])
                total_count = len(results)
                error_count = len(errors)

                depth_summaries.append(
                    {
                        "depth": depth,
                        "total": total_count,
                        "valid": valid_count,
                        "errors": error_count,
                        "success_rate": (
                            valid_count / total_count * 100 if total_count > 0 else 0
                        ),
                    }
                )

            except Exception as e:
                error_msg = f"Failed to test depth {depth}: {e}"
                print(f"❌ {error_msg}")
                all_errors.append(error_msg)

        # Final summary
        print(f"\n{'*'*100}")
        print(f"FINAL SUMMARY")
        print(f"{'*'*100}")

        total_valid = sum(1 for r in all_results if r["match"])
        total_tested = len(all_results)
        total_errors = len(all_errors)

        print(f"Overall statistics:")
        print(f"  Total steps tested: {total_tested}")
        print(f"  Steps verified successfully: {total_valid}")
        print(f"  Steps with errors: {total_errors}")
        print(
            f"  Overall success rate: {total_valid/total_tested*100:.1f}%"
            if total_tested > 0
            else "No tests completed"
        )

        print(f"\nPer-depth breakdown:")
        print(f"{'Depth':<6} {'Total':<8} {'Valid':<8} {'Errors':<8} {'Success%':<10}")
        print(f"{'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for summary in depth_summaries:
            print(
                f"{summary['depth']:<6} {summary['total']:<8} {summary['valid']:<8} {summary['errors']:<8} {summary['success_rate']:<10.1f}"
            )

        if all_errors:
            print(f"\nAll errors encountered ({len(all_errors)} total):")
            for i, error in enumerate(all_errors, 1):
                print(f"  {i}. {error}")
        else:
            print(f"\n🎉 No errors encountered!")

        print(f"\n{'*'*100}")

        return all_results, all_errors, depth_summaries


def main():
    """Main function to run the verification."""
    print("Starting comprehensive expression generation verification...")

    # Create verifier
    verifier = ExpressionGenerationVerifier(max_digits=4, max_decimal_places=6, base=10)

    # Run the full test
    results, errors, summaries = verifier.run_full_test(
        max_depth=6, samples_per_depth=3
    )

    # Exit with error code if there were errors
    if errors:
        print(f"\n❌ Test completed with {len(errors)} errors")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
