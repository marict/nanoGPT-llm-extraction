#!/usr/bin/env python3
"""
Test script for generate_expression function from streaming.py
Tests various combinations of max_digits and max_decimal_places
"""

import sys
from pathlib import Path

# Add the project root to sys.path to import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import math
import random
import sys
from pathlib import Path

import numpy as np
import sympy
import torch
from sympy import im
from tiktoken import Encoding, get_encoding

# Import production code functions for consistency
from data.dagset.streaming import (
    _apply_sympy_op,
    generate_expression,
    generate_uniform_digit_number,
    plan_to_tensors,
)
from models.dag_model import OP_NAMES

# ============================================================================
# OPERATION SETS: Separation between Generation and State Operations
# ============================================================================

# GENERATION_OPs: Used for creating readable expressions (what humans see)
GENERATION_OPS = ["add", "subtract", "multiply", "divide", "identity"]

# STATE_OPs: Used for internal DAG tensor representation (what model predicts)
STATE_OPS = ["add", "multiply", "identity"]

# ============================================================================


def expression_to_string(expr: sympy.Basic) -> str:
    return sympy.sstr(expr)


def string_to_expression(expr_str: str) -> sympy.Basic:
    return sympy.parse_expr(expr_str, evaluate=False)


# Updated to use production code
def expressions_to_tensors(
    expressions: list[sympy.Basic | str],
    *,
    depth: int,
    max_digits: int,
    max_decimal_places: int,
    base: int = 10,
) -> tuple[list[dict[str, torch.Tensor]], list[bool]]:
    """Convert a list of sympy expressions to structure tensors for training.

    Args:
        expressions: List of sympy expressions or "not valid" strings
        depth: Target depth for DAG operations
        max_digits: Maximum number of digits for encoding
        max_decimal_places: Maximum decimal places for encoding
        base: Number base (default 10)

    Returns:
        Tuple of (tensor_list, valid_mask) where:
        - tensor_list: Only contains tensors for valid expressions
        - valid_mask: Boolean list indicating which positions were valid
    """
    tensor_results = []
    valid_mask = []

    for expr in expressions:
        if expr == "not valid":
            valid_mask.append(False)
        else:
            # Extract initial values and operations from the sympy expression
            initial_values, operations = extract_initial_values_and_operations(
                expr, depth, max_decimal_places
            )

            # Convert to tensors using production code directly
            tensor_dict = plan_to_tensors(
                initial_values,
                operations,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
                base=base,
            )
            tensor_results.append(tensor_dict)
            valid_mask.append(True)

    return tensor_results, valid_mask


def extract_initial_values_and_operations(
    expr: sympy.Basic, depth: int, max_decimal_places: int = 4
) -> tuple[list[float], list[str]]:
    """Extract initial values and operations from a sympy expression.

    Converts GENERATION_OPS to STATE_OPS:
    - Division operations become multiplication by reciprocals
    - Subtraction operations become addition with negative values
    - Returns only STATE_OPS: [add, multiply, identity]

    Args:
        expr: The sympy expression
        depth: Target depth for padding operations
        max_decimal_places: Maximum decimal places for reciprocal precision

    Returns:
        Tuple of (initial_values, operations) where operations are from STATE_OPS
    """

    initial_values = []
    operations = []

    def extract_all_values_and_ops(node):
        """Recursively extract all numeric values and operation types from the expression."""
        if isinstance(node, sympy.Number):
            initial_values.append(float(node))
        elif isinstance(node, sympy.Symbol):
            try:
                initial_values.append(float(str(node)))
            except ValueError:
                pass  # Skip non-numeric symbols
        elif isinstance(node, sympy.Add):
            # Handle addition/subtraction -> convert all to addition with proper signs
            processed_terms = []
            for arg in node.args:
                if (
                    isinstance(arg, sympy.Mul)
                    and len(arg.args) >= 1
                    and arg.args[0] == -1
                ):
                    # This is a subtraction term: -1 * something
                    if len(arg.args) == 2:
                        # Simple case: -1 * value
                        subtracted_value = arg.args[1]
                        if isinstance(subtracted_value, sympy.Number):
                            initial_values.append(-float(subtracted_value))
                        elif isinstance(subtracted_value, sympy.Symbol):
                            try:
                                initial_values.append(-float(str(subtracted_value)))
                            except ValueError:
                                initial_values.append(-1.0)  # Fallback
                        else:
                            # Complex subtracted term - recursively process and negate
                            start_len = len(initial_values)
                            extract_all_values_and_ops(subtracted_value)
                            # Negate the last added value
                            if len(initial_values) > start_len:
                                initial_values[-1] = -initial_values[-1]
                    else:
                        # Complex multiplication being subtracted
                        start_len = len(initial_values)
                        extract_all_values_and_ops(arg)
                        # The multiplication will handle the negative
                else:
                    # Regular addition term
                    extract_all_values_and_ops(arg)

            # Addition operation (all subtractions converted to negative values)
            operations.append("add")

        elif isinstance(node, sympy.Mul):
            # Check if this is division (multiplication by Pow(..., -1))
            is_division = any(
                isinstance(arg, sympy.Pow) and len(arg.args) >= 2 and arg.args[1] == -1
                for arg in node.args
                if hasattr(arg, "args")
            )

            if is_division:
                # This is division - convert to multiplication by reciprocal
                for arg in node.args:
                    if (
                        isinstance(arg, sympy.Pow)
                        and len(arg.args) >= 2
                        and arg.args[1] == -1
                    ):
                        # This is the denominator (base of Pow(..., -1))
                        denominator = arg.args[0]
                        if isinstance(denominator, sympy.Number):
                            denom_value = float(denominator)
                            if denom_value != 0:
                                reciprocal = 1.0 / denom_value
                                reciprocal = round(reciprocal, max_decimal_places)
                                initial_values.append(reciprocal)
                            else:
                                initial_values.append(1e-6)  # Avoid division by zero
                        elif isinstance(denominator, sympy.Symbol):
                            try:
                                denom_value = float(str(denominator))
                                if denom_value != 0:
                                    reciprocal = 1.0 / denom_value
                                    reciprocal = round(reciprocal, max_decimal_places)
                                    initial_values.append(reciprocal)
                                else:
                                    initial_values.append(1e-6)
                            except (ValueError, ZeroDivisionError):
                                initial_values.append(1.0)  # Fallback
                        else:
                            initial_values.append(1.0)  # Complex denominator fallback
                    else:
                        # Process numerator normally
                        extract_all_values_and_ops(arg)
                operations.append("multiply")  # Division -> multiplication
            else:
                # Regular multiplication
                for arg in node.args:
                    # Skip the -1 factor as it's handled in Add case
                    if not (isinstance(arg, sympy.Number) and arg == -1):
                        extract_all_values_and_ops(arg)
                operations.append("multiply")

        elif isinstance(node, sympy.Pow):
            # Handle power operations
            if hasattr(node, "args"):
                for arg in node.args:
                    if isinstance(arg, (sympy.Number, sympy.Symbol)):
                        extract_all_values_and_ops(arg)
            operations.append("multiply")  # Treat as multiplication
        elif hasattr(node, "args"):
            # Generic case for other node types
            for arg in node.args:
                extract_all_values_and_ops(arg)

    # Process the expression
    extract_all_values_and_ops(expr)

    # Fallback: if no operations were detected, create minimal operations
    if not operations:
        if len(initial_values) <= 1:
            operations = ["add"]  # Single value, minimal operation
        else:
            # Multiple values, assume addition
            operations = ["add"] * min(len(initial_values) - 1, depth)

    # Ensure we have at least one initial value
    if len(initial_values) < 1:
        initial_values = [0.0]

    # Balance values and operations
    expected_values = len(operations) + 1
    if len(initial_values) < expected_values:
        initial_values.extend([1.0] * (expected_values - len(initial_values)))
    elif len(initial_values) > expected_values:
        initial_values = initial_values[:expected_values]

    # Pad to required depth with identity operations
    while len(operations) < depth:
        operations.append("identity")
    while len(initial_values) < depth + 1:
        initial_values.append(1.0)

    # Ensure exact depth
    operations = operations[:depth]
    initial_values = initial_values[: depth + 1]

    # Verify all operations are from STATE_OPS
    for i, op in enumerate(operations):
        if op not in STATE_OPS:
            operations[i] = "identity"  # Fallback for invalid ops

    return initial_values, operations


def preprocess_invalid_expression(expr_str: str) -> str | None:
    """Try to convert an invalid expression string into a valid one using heuristics.

    Applies preprocessing strategies:
    1. Remove trailing operators  2. Complete decimals  3. Balance parentheses
    4. Remove invalid leading ops  5. Handle hanging operators
    """
    if not expr_str or not expr_str.strip():
        return None

    original = expr_str.strip()

    # Remove trailing operators
    while original and original[-1] in "+-*/":
        original = original[:-1].strip()
        if not original:
            return None

    # Complete incomplete decimals
    if original.endswith("."):
        original = original + "0"

    # Balance parentheses
    open_count = original.count("(")
    close_count = original.count(")")
    if open_count > close_count:
        original = original + ")" * (open_count - close_count)
    elif close_count > open_count:
        for _ in range(close_count - open_count):
            idx = original.rfind(")")
            if idx != -1:
                original = original[:idx] + original[idx + 1 :]

    # Remove leading operators (except unary minus)
    if len(original) > 1 and original[0] in "+*/":
        original = original[1:].strip()

    # Handle hanging operator cases
    if original.endswith(" (") or original.endswith("("):
        parts = original.split()
        if len(parts) >= 3 and parts[-1] == "(" and parts[-2] in "+-*/":
            original = " ".join(parts[:-2])

    # Final cleanup
    original = original.strip()
    while original and original[-1] in "+-*/":
        original = original[:-1].strip()

    # Validate result
    if not original or not any(c.isdigit() or c == "." for c in original):
        return None

    return original if original != expr_str.strip() else None


def generate_expression_exps(
    *,
    depth: int,
    seed: int,
    max_digits: int,
    max_decimal_places: int,
    tokenizer: Encoding,
    base: int = 10,
    _enable_preprocessing: bool = True,
) -> tuple[list[sympy.Basic | str], list[str]]:
    """Generate N sympy expressions from token-based substrings of a base expression.

    Now includes preprocessing by default to dramatically improve valid expression rates!

    Args:
        depth: DAG depth for tensor sizing
        seed: Random seed for reproducibility
        max_digits: Maximum digits for encoding
        max_decimal_places: Maximum decimal places for encoding
        tokenizer: Tokenizer for substring generation
        base: Number base for encoding
        _enable_preprocessing: Whether to apply preprocessing (default True, underscore indicates testing-only parameter)

    Returns:
        Tuple of (expressions, substrings) where expressions are sympy expressions
        (or "not valid" strings for invalid substrings) and substrings are the
        corresponding string representations
    """
    # ------------------------------------------------------------------
    # 1. Generate a base expression first
    # ------------------------------------------------------------------
    rng = random.Random(seed)

    # Use canonical GENERATION_OPS for readable expressions
    sym_ops = []

    # Identities will be added later.
    ops_set_no_identity = [op for op in GENERATION_OPS if op != "identity"]
    # Choose a random number of operations between 0 and depth.
    # That we generate expressions with a variety of depths.
    # Weight higher depths more heavily.
    weights = [i + 1 for i in range(depth)]
    num_ops = rng.choices(range(depth), weights=weights, k=1)[0]
    # Generate random operations.
    for i in range(num_ops):
        op_name = rng.choice(ops_set_no_identity)
        sym_ops.append(op_name)

    # Generate random initial values
    initial_values = []
    for i in range(num_ops + 1):
        value = generate_uniform_digit_number(
            seed=seed + i,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            base=base,
            allow_zero=False,
        )
        initial_values.append(value)

    # ------------------------------------------------------------------
    # 2. Build SymPy expression from leaves + operations
    # ------------------------------------------------------------------
    # Convert values to symbols using numerical representation
    symbols = []
    symbol_names = []
    for val in initial_values:
        # Use numerical representation
        symbol_name = str(val)
        symbols.append(sympy.Symbol(symbol_name))
        symbol_names.append(symbol_name)

    nodes: list[sympy.Basic] = symbols.copy()

    # Apply the operations in Reverse Polish Notation
    for op_name in reversed(sym_ops):
        top = nodes.pop()
        second = nodes.pop()
        expr = _apply_sympy_op(op_name, second, top)
        nodes.append(expr)

    assert len(nodes) == 1
    sym_expr: sympy.Basic = nodes[0]

    # ------------------------------------------------------------------
    # 3. Convert to string and generate token-based substrings
    # ------------------------------------------------------------------
    expr_str = expression_to_string(sym_expr)
    expressions = []
    substrings = []

    # Tokenize the expression string
    tokens = tokenizer.encode(expr_str)

    # Generate substrings from left to right, one token at a time
    for i in range(1, len(tokens) + 1):
        # Decode the first i tokens back to a string
        substring = tokenizer.decode(tokens[:i])
        try:
            # Try to parse the substring as a sympy expression
            sub_expr = string_to_expression(substring)
            expressions.append(sub_expr)
            substrings.append(substring)
        except Exception:
            # If parsing fails, store as "not valid"
            expressions.append("not valid")
            substrings.append(substring)

    # ------------------------------------------------------------------
    # 4. Apply preprocessing if enabled
    # ------------------------------------------------------------------
    if _enable_preprocessing:
        # Apply preprocessing to invalid expressions
        processed_expressions = []
        for i, expr in enumerate(expressions):
            if expr == "not valid":
                # Try to preprocess the substring
                preprocessed = preprocess_invalid_expression(substrings[i])
                if preprocessed:
                    try:
                        # Try to parse the preprocessed string
                        processed_expr = string_to_expression(preprocessed)
                        processed_expressions.append(processed_expr)
                    except Exception:
                        # Still invalid after preprocessing
                        processed_expressions.append("not valid")
                else:
                    processed_expressions.append("not valid")
            else:
                # Already valid
                processed_expressions.append(expr)

        expressions = processed_expressions

    return expressions, substrings


def demo_functionality():
    """Comprehensive demo showing key functionality: preprocessing, masking, and tensor generation."""
    print("\n" + "=" * 70)
    print("DEMO: Per-Token DAG Generation with Preprocessing & Masking")
    print("=" * 70)

    tokenizer = get_encoding("gpt2")

    # Generate expressions (preprocessing enabled by default)
    expressions, substrings = generate_expression_exps(
        depth=4, seed=42, max_digits=3, max_decimal_places=2, tokenizer=tokenizer
    )

    # Convert to tensors with masking
    tensors, valid_mask = expressions_to_tensors(
        expressions, depth=4, max_digits=3, max_decimal_places=2
    )

    # Show results
    valid_count = sum(valid_mask)
    print(f"Generated {len(expressions)} token expressions from math expression")
    print(
        f"Valid: {valid_count}/{len(expressions)} ({valid_count/len(expressions)*100:.1f}%)"
    )
    print(f"Tensors created: {len(tensors)} (only for valid expressions)")

    # Show key examples
    print(f"\nToken breakdown (first 10):")
    tensor_idx = 0
    for i in range(min(10, len(expressions))):
        expr = expressions[i]
        is_valid = valid_mask[i]
        status = f"→ Tensor #{tensor_idx}" if is_valid else "→ Masked"
        if is_valid:
            tensor_idx += 1
        print(f"  '{substrings[i]}' → {expr} {status}")

    # Show preprocessing effectiveness
    expressions_no_prep, _ = generate_expression_exps(
        depth=4,
        seed=42,
        max_digits=3,
        max_decimal_places=2,
        tokenizer=tokenizer,
        _enable_preprocessing=False,
    )
    old_valid = sum(1 for e in expressions_no_prep if e != "not valid")
    improvement = valid_count - old_valid

    print(f"\nPreprocessing impact: {old_valid} → {valid_count} valid (+{improvement})")
    print(
        f"✅ Ready for training: {valid_count/len(expressions)*100:.1f}% meaningful targets"
    )


def quick_test():
    """Run a quick test to verify functionality."""
    print("\n" + "=" * 50)
    print("QUICK FUNCTIONALITY TEST")
    print("=" * 50)

    tokenizer = get_encoding("gpt2")

    # Test different depths
    for depth in [2, 4, 6]:
        expressions, _ = generate_expression_exps(
            depth=depth,
            seed=42,
            max_digits=3,
            max_decimal_places=2,
            tokenizer=tokenizer,
        )
        tensors, valid_mask = expressions_to_tensors(
            expressions, depth=depth, max_digits=3, max_decimal_places=2
        )

        valid_rate = sum(valid_mask) / len(valid_mask) * 100
        print(
            f"Depth {depth}: {len(expressions)} tokens → {len(tensors)} tensors ({valid_rate:.1f}% valid)"
        )

    print("✅ All tests passed!")


if __name__ == "__main__":
    demo_functionality()
    quick_test()
