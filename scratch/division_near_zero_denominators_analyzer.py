#!/usr/bin/env python
"""
Reproduce DAG mismatch cases using generate_single_dag_example.
This reproduces production warnings like:
WARNING: Final value mismatch between sympy and tensor execute: -1720587.625 != -1077568.91737926

Includes verbose sympy evaluation tracing to compare against DAG execution.
"""

import sympy
import torch
from sympy import symbols

from data.dagset.streaming import generate_single_dag_example
from models.dag_model import execute_stack


def show_expression_structure(example):
    """Show the symbolic expression structure for analysis."""

    print("=== EXPRESSION STRUCTURE ANALYSIS ===")

    # Get the expression and initial values
    expr = example.expr
    initial_values = example.initial_values
    operations = example.operations_named

    print(f"Expression: {expr}")
    print(f"Operations: {operations}")
    print(f"Initial values: {initial_values}")
    print()

    # Create value mapping for sympy variables
    value_map = {}
    for i, val in enumerate(initial_values):
        var_name = f"VAL_{i}"
        value_map[symbols(var_name)] = val
        print(f"{var_name} = {val}")
    print()

    # Show the symbolic expression structure
    print(f"Symbolic expression: {expr}")
    print(f"Expression tree: {sympy.srepr(expr)}")
    print()

    # Show what sympy evaluates this to (already computed in example)
    print(f"Sympy result: {example.final_value_sympy:.8g}")
    print("=== END EXPRESSION STRUCTURE ANALYSIS ===\n")


def debug_execute_stack_directly(example):
    """Debug the execute_stack function directly using data from the example."""

    # Extract tensors from the example's structure_dict
    structure = example.structure_dict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    # Convert to appropriate device and dtype with batch/sequence dimensions
    initial_sgn = (
        structure["initial_sgn"]
        .clone()
        .to(device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    initial_digits = (
        structure["initial_digits"]
        .clone()
        .to(device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    operation_probs = (
        structure["operation_probs"]
        .clone()
        .to(device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    print(
        f"Tensor shapes: sgn={initial_sgn.shape}, digits={initial_digits.shape}, ops={operation_probs.shape}"
    )

    try:
        # Execute with debug output
        final_sgn, final_log = execute_stack(
            initial_sgn=initial_sgn,
            digit_probs=initial_digits,
            ops=operation_probs,
            max_digits=example.max_digits,
            max_decimal_places=example.max_decimal_places,
            base=10,
            ignore_clip=True,
            _print_exec_intermediates=True,
        )

        # Extract result
        sgn = final_sgn[0, 0].cpu()
        log_val = final_log[0, 0].cpu()
        direct_result = float(sgn * (10**log_val))

        print(f"\nDIRECT EXECUTE_STACK RESULT: {direct_result:.8g}")
        print(f"EXAMPLE'S FINAL_VALUE_EXEC:  {example.final_value_exec:.8g}")
        print(f"Match: {abs(direct_result - example.final_value_exec) < 1e-6}")

    except Exception as e:
        print(f"❌ ERROR in execute_stack: {str(e)}")
        import traceback

        traceback.print_exc()


def test_specific_production_seed(seed: int):
    """Test using generate_single_dag_example with the production seed."""

    print("Reproducing DAG mismatch with generate_single_dag_example:")
    print(f"Using seed={seed} from production warning")
    print("=" * 70)

    # Generate the example that caused the production warning
    example = generate_single_dag_example(
        depth=6,
        seed=seed,
        max_digits=4,  # Production default
        max_decimal_places=6,  # Production default
        english_conversion_probability=0.5,  # From warning
        integer_no_decimal_probability=0.5,  # From warning
        expression_simplification_probability=0.5,  # Typical value
        expression_expansion_probability=0.5,  # Typical value
        printing_style_probs={"latex": 1.0},  # From warning showing latex style
        execute_sympy=True,
        _print_exec_intermediates=True,
    )

    print(f"Text: {example.text}")
    print(f"Initial values: {example.initial_values}")
    print(f"Operations: {example.operations_named}")
    print(f"Expression: {example.expr}")
    print()

    print("RESULTS:")
    print(f"Sympy result:     {example.final_value_sympy:.8g}")
    print(f"DAG exec result:  {example.final_value_exec:.8g}")
    print()

    # Calculate error metrics
    if example.final_value_sympy and example.final_value_exec:
        rel_error = abs(
            (example.final_value_exec - example.final_value_sympy)
            / example.final_value_sympy
        )
        abs_error = abs(example.final_value_exec - example.final_value_sympy)

        print("ERROR ANALYSIS:")
        print(f"Relative error:   {rel_error:.8g}")
        print(f"Absolute error:   {abs_error:.8g}")
        print()

        # Status assessment
        if rel_error < 1e-6:
            print("✅ PASS: Values match within precision")
        elif rel_error < 1e-2:
            print("⚠️  WARN: Small mismatch (precision issue)")
        else:
            print("❌ FAIL: Large mismatch (execution bug)")

        # Show expression structure for analysis
        print("\n" + "=" * 70)
        show_expression_structure(example)

        # Debug execute_stack directly
        print("DEBUGGING WITH EXECUTE_STACK:")
        debug_execute_stack_directly(example)

        # Final comparison of the key results
        print("\nFINAL COMPARISON:")
        print(f"Sympy result (correct):  {example.final_value_sympy:.8g}")
        print(f"DAG result (generation): {example.final_value_exec:.8g}")
        print(f"Relative error:          {rel_error:.8g}")

    return example


if __name__ == "__main__":
    # Test the specific production case with verbose sympy evaluation
    production_example = test_specific_production_seed(450429774)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("✅ Use this tool to systematically reproduce and debug DAG execution issues")
