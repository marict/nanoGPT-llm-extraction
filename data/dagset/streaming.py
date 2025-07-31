#!/usr/bin/env python
"""
streaming.py
On-the-fly DAG dataset generation for training.
"""

import logging
import math
import random
import sys
import warnings
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import sympy
import torch
from num2words import num2words
from sympy import im
from tiktoken import get_encoding

from models.dag_model import execute_stack

from .preprocess_invalid_expression import preprocess_invalid_expression

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.dag_model import (
    LOG_LIM,
    OP_NAMES,
    compute_multi_value_statistics,
    compute_single_value_statistics,
)

# ============================================================================
# GENERATION SYSTEM: Separation between Generation and State Operations
# ============================================================================

# GENERATION_OPs: Used for creating readable expressions (what humans see)
GENERATION_OPS = ["add", "subtract", "multiply", "divide", "identity"]

# STATE_OPs: Used for internal DAG tensor representation (what model predicts)
STATE_OPS = ["add", "multiply", "identity"]

# ============================================================================

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #


@dataclass
class DAGExample:
    """Base class for DAG computation examples with essential shared attributes."""

    text: str
    structure_dict: dict[str, torch.Tensor]
    depth: int
    max_digits: int
    max_decimal_places: int
    base: int
    seed: int

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}(seed={self.seed}, text={self.text}, depth={self.depth}, base={self.base})"


@dataclass
class DAGTrainExample(DAGExample):
    """Lightweight DAG example for training with minimal attributes to reduce memory overhead."""


@dataclass
class DAGValExample(DAGExample):
    """Full DAG example for validation with all attributes for logging and debugging.

    Note: This represents N different expressions (per-token), where the fields below
    correspond to the final/complete expression in the sequence.
    """

    full_operations_named: list[str]  # Operations for the final complete expression
    full_operations: list[str]  # Operations for the final complete expression
    final_value_sympy: float | None = None
    full_expr: sympy.Basic | None = None  # The final complete expression

    def __str__(self):
        signs_shape = self.structure_dict["target_initial_sgn"].shape
        digits_shape = self.structure_dict["target_initial_digits"].shape
        operations_shape = self.structure_dict["target_operation_probs"].shape
        final_value_exec = self.structure_dict["target_final_exec"]
        return f"DAGValExample(seed={self.seed}, text={self.text}, depth={self.depth}, signs={signs_shape}, digits={digits_shape}, operations={operations_shape}, full_operations_named={self.full_operations_named}, final_value_sympy={self.final_value_sympy}, final_value_exec={final_value_exec}, full_expr={self.full_expr}, base={self.base})"


# ============================================================================
# EXTRACTION SYSTEM: Convert GENERATION_OPS to STATE_OPS
# ============================================================================


def extract_initial_values_and_operations(
    expr: sympy.Basic, depth: int, max_decimal_places: int = 4
) -> tuple[list[float], list[str]]:
    """Extract initial values and operations from a sympy expression.

    KEY INNOVATION: Uses sympy.expand() to convert nested expressions into associative
    forms that are compatible with the DAG's two-list representation. This eliminates
    the fundamental tree structure problem by flattening expressions like:
    - (a + b) * c  →  a*c + b*c  (flat addition)
    - (a + b) * (c + d)  →  a*c + a*d + b*c + b*d  (flat addition)

    This approach leverages mathematical associativity of add/multiply operations
    to ensure any nested expression can be represented as flat operations chains,
    solving the right-to-left execution order issues with complex expressions.

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

    # STEP 1: Expand the expression to eliminate nested structure
    # This converts (a+b)*c → a*c + b*c, making it associative
    #
    # CRITICAL: This expansion ONLY affects tensor extraction, NEVER the text representation!
    # The model sees original text like "(2+3)*4" while tensors get mathematically correct
    # flattened representation. This separation is essential for training quality.
    expanded_expr = sympy.expand(expr)

    # STEP 2: Force evaluation only for problematic complex denominators
    # Only evaluate if expression contains division that expansion couldn't handle
    if not isinstance(
        expanded_expr, (sympy.Number, sympy.Float, sympy.Integer)
    ) and any(
        isinstance(arg, sympy.Pow) and len(arg.args) >= 2 and arg.args[1] == -1
        for arg in expanded_expr.args
        if hasattr(expanded_expr, "args")
    ):
        try:
            # Force evaluation only for division cases that need it
            evaluated = float(expanded_expr)
            expanded_expr = sympy.Float(evaluated)
        except (TypeError, ValueError):
            # Keep symbolic form if evaluation fails
            pass

    initial_values = []
    operations = []

    def add_numeric_value(value):
        """Helper to safely add a numeric value."""
        try:
            initial_values.append(
                float(value) if isinstance(value, str) else float(value)
            )
        except (ValueError, TypeError):
            pass

    def add_reciprocal(denominator):
        """Helper to add reciprocal of a value."""
        try:
            denom_val = float(denominator)
            if denom_val != 0:
                # Use higher precision for reciprocals to avoid mathematical errors
                reciprocal = 1.0 / denom_val
                initial_values.append(
                    reciprocal
                )  # Don't round reciprocals aggressively
            else:
                initial_values.append(1e-6)
        except:
            initial_values.append(1.0)

    def extract_all_values_and_ops(node):
        """Recursively extract numeric values and operations from expressions."""
        if isinstance(node, (sympy.Number, sympy.Symbol)):
            add_numeric_value(node)

        elif isinstance(node, sympy.Add):
            # Process all addition/subtraction terms
            for arg in node.args:
                if (
                    isinstance(arg, sympy.Mul)
                    and len(arg.args) >= 1
                    and arg.args[0] == -1
                ):
                    # Subtraction: -1 * something
                    if len(arg.args) == 2:
                        add_numeric_value(-float(arg.args[1]))
                    else:
                        extract_all_values_and_ops(arg)
                else:
                    extract_all_values_and_ops(arg)

            # Add n-1 operations for n terms (correct operations count)
            num_terms = len(node.args)
            if num_terms > 1:
                operations.extend(["add"] * (num_terms - 1))

        elif isinstance(node, sympy.Mul):
            # Check for division (has Pow(..., -1))
            division_args = [
                a
                for a in node.args
                if isinstance(a, sympy.Pow) and len(a.args) >= 2 and a.args[1] == -1
            ]

            if division_args:
                # Process division: extract numerator and denominator
                numerator_args = [arg for arg in node.args if arg not in division_args]

                # Extract numerator values
                for arg in numerator_args:
                    extract_all_values_and_ops(arg)

                # Extract denominator values
                for arg in division_args:
                    denominator = arg.args[0]
                    # For complex denominators, extract all values and operations
                    if hasattr(denominator, "args") and len(denominator.args) > 1:
                        extract_all_values_and_ops(denominator)
                    else:
                        # Simple denominator - add reciprocal directly
                        add_reciprocal(denominator)

                # Add operations for numerator (if multiple terms)
                if len(numerator_args) > 1:
                    operations.extend(["multiply"] * (len(numerator_args) - 1))

                # Add operation for division itself
                operations.append("multiply")
            else:
                # Regular multiplication - skip -1 factors (handled by Add)
                non_neg_one_args = [
                    arg
                    for arg in node.args
                    if not (isinstance(arg, sympy.Number) and arg == -1)
                ]

                for arg in non_neg_one_args:
                    extract_all_values_and_ops(arg)

                # Add n-1 multiply operations for n terms (correct operations count)
                num_terms = len(non_neg_one_args)
                if num_terms > 1:
                    operations.extend(["multiply"] * (num_terms - 1))

        elif hasattr(node, "args"):
            # Generic recursion for other types (Pow, etc.)
            for arg in node.args:
                extract_all_values_and_ops(arg)
            if isinstance(node, sympy.Pow):
                operations.append("multiply")

    # Process the expanded expression (now associative!)
    extract_all_values_and_ops(expanded_expr)

    # CRITICAL: Check if expansion created more operations than max depth allows
    meaningful_ops = [op for op in operations if op != "identity"]
    if len(meaningful_ops) > depth:
        raise ValueError(
            f"Expansion depth exceeded maximum allowed depth in extract_initial_values_and_operations:\n"
            f"  Original expression: {expr}\n"
            f"  Expanded expression: {expanded_expr}\n"
            f"  Meaningful operations: {len(meaningful_ops)} (requires depth {len(meaningful_ops)})\n"
            f"  Maximum allowed depth: {depth}\n"
            f"  Operations: {meaningful_ops}\n"
            f"  \n"
            f"  This expression is too complex for the current DAG depth limit.\n"
            f"  Either increase the dataset depth limit or simplify the expression.\n"
            f"  DO NOT silently truncate - this would cause incorrect mathematical results."
        )

    # Fallback: if no operations were detected, create minimal operations
    if not operations:
        if len(initial_values) <= 1:
            # Single value requires one identity operation for DAG compatibility
            operations = ["identity"]
        else:
            # Multiple values with no detected operations is an error - fail completely
            raise ValueError(
                f"No operations detected for multi-value expression: {expr}\n"
                f"Values: {initial_values}\n"
                f"This indicates an extraction bug that must be fixed."
            )

    # Ensure we have at least one initial value
    if len(initial_values) < 1:
        initial_values = [0.0]

    # Balance values and operations - NO TRUNCATION, FAIL COMPLETELY IF MISMATCH
    expected_values = len(operations) + 1
    if len(initial_values) < expected_values:
        initial_values.extend([1.0] * (expected_values - len(initial_values)))
    elif len(initial_values) > expected_values:
        # NEVER truncate - this causes silent data loss and bugs
        # Instead, fail completely so we can identify and fix the root cause
        meaningful_ops = [op for op in operations if op != "identity"]
        meaningful_values = [v for v in initial_values if v != 1.0]
        raise ValueError(
            f"Value/operation count mismatch in extract_initial_values_and_operations:\n"
            f"  Expression: {expr}\n"
            f"  Expected {expected_values} values (operations + 1)\n"
            f"  Got {len(initial_values)} values: {initial_values}\n"
            f"  Operations ({len(operations)}): {operations}\n"
            f"  Meaningful operations ({len(meaningful_ops)}): {meaningful_ops}\n"
            f"  Meaningful values ({len(meaningful_values)}): {meaningful_values}\n"
            f"  This indicates a bug in the extraction logic that must be fixed."
        )

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


def string_to_expression(expr_str: str) -> sympy.Basic:
    """Convert a string to a sympy expression."""
    return sympy.parse_expr(expr_str, evaluate=False)


def expression_to_string(expr: sympy.Basic) -> str:
    """Convert a sympy expression to string."""
    return str(expr)


def convert_number_to_english(number: float, max_decimal_places: int = 6) -> str:
    """Convert *number* to its English word equivalent using *num2words*.

    The value is first rounded (half-up) to *max_decimal_places* decimal digits to
    avoid extremely long fractional strings, then converted.  Negatives are
    rendered with the "negative" prefix to preserve the previous output style.
    """
    # Quantise using Decimal to avoid floating-point surprises (e.g. 0.1+0.2)
    quantised = Decimal(str(number)).quantize(
        Decimal(10) ** -max_decimal_places, rounding=ROUND_HALF_UP
    )

    words = num2words(abs(quantised))
    return f"negative {words}" if quantised < 0 else words


def expressions_to_tensors(
    expressions: list[sympy.Basic | str],
    *,
    depth: int,
    max_digits: int,
    max_decimal_places: int,
    base: int = 10,
) -> tuple[list[dict[str, torch.Tensor]], list[bool]]:
    """Convert a list of sympy expressions to structure tensors using STATE_OPS.

    This converts GENERATION_OPS to STATE_OPS:
    - Division → multiplication by reciprocals
    - Subtraction → addition with negative values
    - Returns tensors using only STATE_OPS: [add, multiply, identity]
    - Invalid expressions get "zero DAG" representations

    Args:
        expressions: List of sympy expressions or "not valid" strings
        depth: Target depth for DAG operations
        max_digits: Maximum number of digits for encoding
        max_decimal_places: Maximum decimal places for encoding
        base: Number base (default 10)

    Returns:
        Tuple of (tensor_list, valid_mask) where:
        - tensor_list: Contains T tensors (one per token position, including zero DAGs)
        - valid_mask: Boolean list indicating which positions were valid
    """
    tensor_results = []
    valid_mask = []

    for expr in expressions:
        if expr == "not valid":
            # Create zero DAG for invalid token position
            zero_initial_values = [0.0] * (depth + 1)
            zero_operations = ["identity"] * depth

            zero_tensor_dict = plan_to_tensors(
                zero_initial_values,
                zero_operations,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
                base=base,
            )
            # Override final execution to be exactly zero
            zero_tensor_dict["target_final_exec"] = 0.0

            tensor_results.append(zero_tensor_dict)
            valid_mask.append(False)
        else:
            # Extract initial values and operations from the sympy expression
            # This converts GENERATION_OPS to STATE_OPS
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


def generate_uniform_digit_number(
    seed: int | None = None,
    max_digits: int = 4,
    max_decimal_places: int = 6,
    base: int = 10,
    allow_zero: bool = True,
    integer_no_decimal_probability: float = 0.0,
) -> float | int:
    """Generate a random number within digit constraints for the specified base.

    A random number with controlled digit constraints, as float or int.
    For a given base, generates numbers using digits 0 to (base-1).

    Args:
        seed: Random seed for reproducibility
        max_digits: Maximum number of integer digits in the specified base
        max_decimal_places: Maximum number of decimal places in the specified base
        base: Number base (2-36 supported)
        allow_zero: Whether to allow zero values
        integer_no_decimal_probability: Probability of returning integer instead of float

    Returns:
        A number appropriate for the specified base
    """
    if base < 2 or base > 36:
        raise ValueError(f"Base must be between 2 and 36, got {base}")

    rng = random.Random(seed or random.randint(0, 10000))

    # Determine if we're generating a positive or negative number
    sign = 1 if rng.random() < 0.5 else -1

    # 1. Generate a random number of integer digits
    num_int_digits = rng.randint(1, max_digits)

    # 2. Generate random integer digits (first digit can't be 0 for multi-digit integers)
    if num_int_digits > 1:
        first_digit = rng.randint(1, base - 1)
        other_digits = [rng.randint(0, base - 1) for _ in range(num_int_digits - 1)]
        int_digits = [first_digit] + other_digits
    else:
        # For single digit, if we don't allow zero, force non-zero
        if not allow_zero:
            int_digits = [rng.randint(1, base - 1)]
        else:
            int_digits = [rng.randint(0, base - 1)]

    # 3. Generate a random number of decimal places
    num_decimal_places = rng.randint(0, max_decimal_places)

    # 4. Generate random decimal digits for the specified base
    decimal_digits = [rng.randint(0, base - 1) for _ in range(num_decimal_places)]

    # 5. Combine integer and decimal parts using the specified base
    int_part = sum(d * (base**i) for i, d in enumerate(reversed(int_digits)))

    # If int_part is 0, and we have decimal places, make sure at least one decimal digit is non-zero
    # to avoid generating a complete zero when not allowed
    if int_part == 0 and not allow_zero and num_decimal_places > 0:
        if all(d == 0 for d in decimal_digits):
            # Replace a random decimal digit with non-zero
            idx = rng.randint(0, len(decimal_digits) - 1)
            decimal_digits[idx] = rng.randint(1, base - 1)

    # Check if the number is a whole number (no decimal places)
    is_whole_number = num_decimal_places == 0

    # Calculate the actual result as a float initially using the specified base
    if num_decimal_places > 0:
        decimal_part = sum(d * (base ** -(i + 1)) for i, d in enumerate(decimal_digits))
        result = int_part + decimal_part
    else:
        result = float(int_part)

    # Apply the sign
    result *= sign

    # Convert to integer if it's a whole number and the random probability is met
    if is_whole_number and rng.random() < integer_no_decimal_probability:
        return int(result)

    return result


def _apply_sympy_op(op_name: str, second: sympy.Basic, top: sympy.Basic) -> sympy.Basic:
    """Return a SymPy expression representing *a <op> b* without evaluation."""
    if op_name == "add":
        return sympy.Add(second, top, evaluate=False)
    if op_name == "subtract":
        return sympy.Add(second, -top, evaluate=False)
    if op_name == "multiply":
        return sympy.Mul(second, top, evaluate=False)
    if op_name == "divide":
        return sympy.Mul(second, sympy.Pow(top, -1, evaluate=False), evaluate=False)
    if op_name == "identity":
        # For symbol creation in expression generation, we return the second operand
        return second

    raise ValueError(f"Unknown operation: {op_name}")


def generate_expression(
    *,
    depth: int,
    seed: int,
    max_digits: int,
    max_decimal_places: int,
    tokenizer,  # tiktoken.Encoding
    base: int = 10,
    _enable_preprocessing: bool = True,
) -> tuple[list[sympy.Basic | str], list[str], list[bool]]:
    """Generate expressions using GENERATION_OPS with per-token approach.

    This implements the system that:
    1. Uses GENERATION_OPS for readable expressions (division/subtraction visible)
    2. Generates per-token substrings for training
    3. Applies preprocessing to convert invalid expressions to valid ones
    4. Returns masking information for invalid tokens

    Args:
        depth: Target depth for DAG operations
        seed: Random seed for reproducibility
        max_digits: Maximum number of digits for encoding
        max_decimal_places: Maximum decimal places for encoding
        tokenizer: tiktoken tokenizer for creating per-token substrings
        base: Number base (default 10)
        _enable_preprocessing: Whether to apply preprocessing (default True, underscore indicates testing-only parameter)

    Returns:
        Tuple of (expressions, substrings, valid_mask) where:
        - expressions: List of sympy expressions or "not valid" strings
        - substrings: List of corresponding string representations
        - valid_mask: Boolean list indicating which expressions are valid
    """

    # ------------------------------------------------------------------
    # 1. Generate a base expression using GENERATION_OPS
    # ------------------------------------------------------------------
    rng = random.Random(seed)

    # Use canonical GENERATION_OPS
    sym_ops = []

    # Identities will be added later.
    ops_set_no_identity = [op for op in GENERATION_OPS if op != "identity"]
    # Choose a random number of operations between 0 and depth.
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
            integer_no_decimal_probability=0.0,
        )
        initial_values.append(value)

    # ------------------------------------------------------------------
    # 2. Build SymPy expression from leaves + operations
    # ------------------------------------------------------------------
    # Convert values to symbols
    symbols = []
    symbol_names = []
    for val in initial_values:
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
    base_expr: sympy.Basic = nodes[0]

    # ------------------------------------------------------------------
    # 3. Generate per-token substrings and create expressions
    # ------------------------------------------------------------------
    base_text = str(base_expr)
    tokens = tokenizer.encode(base_text)

    expressions = []
    substrings = []

    # Generate all possible substring expressions from tokenization
    for i in range(len(tokens)):
        substring_tokens = tokens[: i + 1]
        substring = tokenizer.decode(substring_tokens)
        substrings.append(substring)

        # Try to parse as expression
        try:
            expr = string_to_expression(substring.strip())
            # Verify it's a valid mathematical expression
            float(expr)  # This will raise if not evaluable
            expressions.append(expr)
        except:
            expressions.append("not valid")

    # ------------------------------------------------------------------
    # 4. Apply preprocessing if enabled
    # ------------------------------------------------------------------
    if _enable_preprocessing:
        processed_expressions = []
        for i, expr in enumerate(expressions):
            if expr == "not valid":
                preprocessed = preprocess_invalid_expression(substrings[i])
                if preprocessed:
                    try:
                        processed_expr = string_to_expression(preprocessed)
                        # Verify it's evaluable
                        float(processed_expr)
                        processed_expressions.append(processed_expr)
                    except Exception:
                        processed_expressions.append("not valid")
                else:
                    processed_expressions.append("not valid")
            else:
                processed_expressions.append(expr)
        expressions = processed_expressions

    # ------------------------------------------------------------------
    # 5. Create validity mask
    # ------------------------------------------------------------------
    valid_mask = [expr != "not valid" for expr in expressions]

    return expressions, substrings, valid_mask


def float_to_digit_onehot(
    value: float, max_digits: int, max_decimal_places: int, base: int = 10
) -> torch.Tensor:
    """Convert a float into a one-hot tensor of shape (D, base) where D = max_digits + max_decimal_places.

    The number is first clipped to the largest representable value to avoid overflow. The integer part is
    left-padded with zeros and the fractional part is right-padded with zeros so that their total length
    equals *max_digits + max_decimal_places*.

    Args:
        value: The float value to convert
        max_digits: Maximum number of integer digits
        max_decimal_places: Maximum number of decimal places
        base: Number base for digit representation (2-36 supported)
    """
    if base < 2 or base > 36:
        raise ValueError(f"Base must be between 2 and 36, got {base}")

    # Clip magnitude so that it fits in the available digits
    limit = base**max_digits - base ** (-max_decimal_places)
    abs_val = min(abs(value), limit)

    if base == 10:
        # For base 10, use string formatting to avoid floating point precision issues
        # Format with enough decimal places and then extract digits
        format_str = f"{{:.{max_decimal_places}f}}"
        value_str = format_str.format(abs_val)

        # Split into integer and decimal parts
        if "." in value_str:
            int_part_str, frac_part_str = value_str.split(".")
        else:
            int_part_str = value_str
            frac_part_str = ""

        # Pad integer part to max_digits (left pad with zeros)
        int_part_str = int_part_str.zfill(max_digits)[-max_digits:]

        # Pad fractional part to max_decimal_places (right pad with zeros)
        frac_part_str = (frac_part_str + "0" * max_decimal_places)[:max_decimal_places]

        # Combine and convert to digit list
        all_digits_str = int_part_str + frac_part_str
        all_digits = [int(d) for d in all_digits_str]
    else:
        # For non-base-10, use the original method but with rounding to mitigate precision issues
        # Convert to the target base
        # First convert integer part
        int_part = int(abs_val)
        frac_part = abs_val - int_part

        # Convert integer part to target base
        if int_part == 0:
            int_digits = [0]
        else:
            int_digits = []
            temp = int_part
            while temp > 0:
                int_digits.append(temp % base)
                temp = temp // base
            int_digits.reverse()  # Most significant digit first

        # Pad or truncate integer part to exactly max_digits
        if len(int_digits) > max_digits:
            int_digits = int_digits[-max_digits:]  # Keep least significant digits
        else:
            int_digits = [0] * (
                max_digits - len(int_digits)
            ) + int_digits  # Pad with zeros

        # Convert fractional part to target base with rounding
        frac_digits = []
        temp_frac = frac_part
        for _ in range(max_decimal_places):
            temp_frac *= base
            # Add small epsilon and round to mitigate floating point precision issues
            digit = (
                round(temp_frac + 1e-10) if temp_frac < base - 0.5 else int(temp_frac)
            )
            digit = min(max(digit, 0), base - 1)  # Clamp to valid range
            frac_digits.append(digit)
            temp_frac -= digit

        # Combine integer and fractional digits
        all_digits = int_digits + frac_digits

    D = max_digits + max_decimal_places
    assert len(all_digits) == D, f"Expected {D} digits, got {len(all_digits)}"

    # Create one-hot encoding
    one_hot = torch.zeros(D, base)
    for i, digit in enumerate(all_digits):
        # Clamp digit to valid range (should not be needed but safety check)
        digit = min(max(digit, 0), base - 1)
        one_hot[i, digit] = 1.0

    return one_hot


# Updated to include digit arguments
def plan_to_tensors(
    initial_values: list[float],
    operations: list[str],
    *,
    max_digits: int,
    max_decimal_places: int,
    base: int = 10,
) -> dict[str, torch.Tensor]:
    """Convert a DAG plan to structure tensors for training."""

    depth = len(operations)
    num_scratch_nodes = depth + 1

    if len(initial_values) != num_scratch_nodes:
        raise ValueError(
            f"Initial values count mismatch: got {len(initial_values)} values but expected {num_scratch_nodes} "
            f"for depth {depth}. This indicates a bug in expression generation."
        )
    if len(operations) != depth:
        raise ValueError(
            f"Operations count mismatch: got {len(operations)} operations but expected {depth} "
            f"for depth {depth}. This indicates a bug in expression generation."
        )

    # Convert initial values to signs and digit one-hots
    signs = torch.tensor([1.0 if v >= 0.0 else -1.0 for v in initial_values])
    digits_onehots: list[torch.Tensor] = []
    for v in initial_values:
        one_hot = float_to_digit_onehot(v, max_digits, max_decimal_places, base)
        digits_onehots.append(one_hot)

    # (num_nodes, D, base)
    digits_tensor = torch.stack(digits_onehots, dim=0)

    # Convert operations to one-hot encoded tensors (length already validated)
    # Use STATE_OPS for the simplified 3-operation system
    operation_to_index = {op: i for i, op in enumerate(STATE_OPS)}
    operations_one_hot = torch.zeros(depth, len(STATE_OPS))

    for i, op in enumerate(operations):
        op_idx = operation_to_index[op]
        operations_one_hot[i, op_idx] = 1.0

    # Execute the DAG to get the final value
    with torch.no_grad():
        # Use double precision during reference execution to minimize numerical errors
        sign_tensor = signs.view(1, 1, -1).to(torch.float64)
        digit_probs = digits_tensor.unsqueeze(0).unsqueeze(0).to(torch.float64)
        op_probs = operations_one_hot.unsqueeze(0).unsqueeze(0).to(torch.float64)

        final_sgn, final_log, intermediate_values = execute_stack(
            sign_tensor,
            digit_probs,
            op_probs,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            base=base,
            ignore_clip=True,
            return_intermediates=True,
        )

        final_value_exec = (final_sgn * torch.exp(final_log)).item()

    initial_log = torch.zeros(num_scratch_nodes)

    # Compute log magnitudes (natural log)
    for i in range(num_scratch_nodes):
        v = abs(initial_values[i]) if initial_values[i] != 0 else 1e-6
        log_val = math.log(v)
        # Clip to LOG_LIM for safety
        log_val = max(min(log_val, LOG_LIM), -LOG_LIM)
        initial_log[i] = log_val

    # Create target tensors - since we validated dimensions, just convert directly
    target_initial_values = torch.tensor(
        initial_values[:num_scratch_nodes], dtype=torch.float32
    )

    # Compute statistics for auxiliary prediction
    # Intermediate values already collected during execute_stack call above

    # Compute statistics (let errors surface to identify unstable stats)
    target_initial_stats = torch.tensor(
        compute_multi_value_statistics(initial_values), dtype=torch.float32
    )

    target_intermediate_stats = (
        torch.tensor(
            compute_multi_value_statistics(intermediate_values), dtype=torch.float32
        )
        if intermediate_values
        else torch.zeros(15, dtype=torch.float32)
    )

    target_final_stats = torch.tensor(
        compute_single_value_statistics(final_value_exec), dtype=torch.float32
    )

    return {
        "target_initial_sgn": signs,
        "target_initial_log": initial_log,
        "target_initial_digits": digits_tensor,
        "target_operation_probs": operations_one_hot,
        "target_final_exec": final_value_exec,
        "target_initial_values": target_initial_values,
        # Statistics targets - same for all tokens since they relate to the same expression
        "target_initial_stats": target_initial_stats,
        "target_intermediate_stats": target_intermediate_stats,
        "target_final_stats": target_final_stats,
    }


def tensors_to_plan(
    signs: torch.Tensor,
    digits: torch.Tensor,
    operations: torch.Tensor,
    *,
    max_digits: int,
) -> tuple[list[float], list[str]]:
    """Convert tensors back to a DAG plan.

    Args:
        signs: Tensor of signs (1.0 for positive, -1.0 for negative)
        digits: One-hot encoded digits tensor (num_nodes, D, 10)
        operations: One-hot encoded operations tensor (depth, num_ops)
        max_digits: Maximum number of integer digits
        max_decimal_places: Maximum number of decimal places

    Returns:
        Tuple of (initial_values, operations)
    """
    # Convert digit one-hots back to values
    initial_values = []
    for i in range(digits.shape[0]):
        # Get the digit indices from one-hot encoding
        digit_indices = torch.argmax(digits[i], dim=1)

        # Convert to string representation
        digit_str = "".join(str(d.item()) for d in digit_indices)

        # Split into integer and decimal parts
        int_part = digit_str[:max_digits]
        dec_part = digit_str[max_digits:]

        # Remove leading zeros from integer part, but keep at least one digit
        int_part = str(int(int_part))

        # Combine parts and convert to float
        value = float(f"{int_part}.{dec_part}")

        # Apply sign
        value *= signs[i].item()
        initial_values.append(value)

    # Convert operation one-hots back to operation names
    op_indices = torch.argmax(operations, dim=1)
    operation_list = [OP_NAMES[idx.item()] for idx in op_indices]

    return initial_values, operation_list


def create_dag_structure_dataloaders(
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    max_depth: int = 8,
    seed: int = 42,
    max_digits: int = 4,
    max_decimal_places: int = 6,
    base: int = 10,
    block_size: int = 8,
) -> Tuple[Iterator, Iterator]:
    """Create per-token train/val DAG structure dataloaders.

    Uses the streaming.py per-token system directly.

    Returns:
        Tuple of (train_loader, val_loader) that yield:
        - texts: List[str]
        - target_tensors: List[Dict[str, torch.Tensor]]
        - valid_masks: torch.Tensor (B, T)
    """

    def generate_batch(batch_size: int, training_seed: int):
        """Generate a batch using the per-token format."""
        from tiktoken import get_encoding

        tokenizer = get_encoding("gpt2")

        all_texts = []
        all_target_tensors = []
        all_valid_masks = []

        for i in range(batch_size):
            # Generate expressions
            expressions, substrings, valid_mask_list = generate_expression(
                depth=max_depth,
                seed=training_seed + i,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
                tokenizer=tokenizer,
            )

            # Convert to tensors
            target_tensors, tensor_valid_mask = expressions_to_tensors(
                expressions,
                depth=max_depth,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
            )

            # Get main expression (last one) as string
            last_expr = expressions[-1] if expressions else ""
            if last_expr == "not valid":
                text = (
                    substrings[-1] if substrings else ""
                )  # Use raw substring for invalid expressions
            else:
                text = str(last_expr)  # Convert sympy expression to string

            all_texts.append(text)
            all_target_tensors.append(
                target_tensors
            )  # Keep as list of lists (batch structure)
            all_valid_masks.append(tensor_valid_mask)

        # Pad valid masks to block_size (not just max within batch)
        batched_valid_masks = []
        for mask in all_valid_masks:
            # Truncate or pad to block_size to match model expectations
            if len(mask) > block_size:
                padded = mask[:block_size]  # Truncate if too long
            else:
                padded = mask + [False] * (block_size - len(mask))  # Pad with False
            batched_valid_masks.append(padded)

        valid_masks_tensor = torch.tensor(batched_valid_masks, dtype=torch.bool)

        # Pad target tensors to block_size as well
        padded_target_tensors = []
        for target_tensors in all_target_tensors:
            # Create zero DAG for padding positions
            zero_initial_values = [0.0] * (max_depth + 1)
            zero_operations = ["identity"] * max_depth
            zero_tensor_dict = plan_to_tensors(
                zero_initial_values,
                zero_operations,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
                base=base,
            )
            zero_tensor_dict["target_final_exec"] = 0.0

            # Truncate or pad to block_size
            if len(target_tensors) > block_size:
                padded_tensors = target_tensors[:block_size]  # Truncate if too long
            else:
                padded_tensors = target_tensors + [zero_tensor_dict] * (
                    block_size - len(target_tensors)
                )
            padded_target_tensors.append(padded_tensors)

        return all_texts, padded_target_tensors, valid_masks_tensor

    def train_loader():
        counter = 0
        while True:
            texts, target_tensors, valid_masks = generate_batch(
                train_batch_size, seed + counter
            )
            counter += train_batch_size
            yield texts, target_tensors, valid_masks

    def val_loader():
        counter = 0
        while True:
            texts, target_tensors, valid_masks = generate_batch(
                val_batch_size, seed + counter + 10000
            )
            counter += val_batch_size
            yield texts, target_tensors, valid_masks

    return train_loader(), val_loader()
