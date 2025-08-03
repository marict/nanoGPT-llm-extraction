"""
Expression generation functionality for DAG dataset creation.

This module contains functions for generating mathematical expressions
using SymPy with per-token tokenization approach.
"""

import random

import sympy

from .preprocess_invalid_expression import preprocess_invalid_expression

# GENERATION_OPs: Used for creating readable expressions (what humans see)
GENERATION_OPS = ["add", "subtract", "multiply", "divide", "identity"]


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

    rng = random.Random(seed if seed is not None else random.randint(0, 10000))

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


def string_to_expression(expr_str: str) -> sympy.Basic:
    """
    Parse a string into a SymPy expression.

    Args:
        expr_str: Mathematical expression as string

    Returns:
        SymPy expression

    Raises:
        Various exceptions if parsing fails
    """
    # Remove whitespace and handle edge cases
    expr_str = expr_str.strip()
    if not expr_str:
        raise ValueError("Empty expression string")

    # Use SymPy's parsing with safety checks
    try:
        # Use sympify with evaluate=False to preserve structure
        expr = sympy.sympify(expr_str, evaluate=False)
        return expr
    except Exception as e:
        raise ValueError(f"Failed to parse expression '{expr_str}': {e}")


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
        max_digits: Maximum number of digits for number generation
        max_decimal_places: Maximum decimal places for number generation
        tokenizer: tiktoken tokenizer for creating per-token substrings
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
            base=10,  # Always use base 10
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
