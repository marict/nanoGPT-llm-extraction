#!/usr/bin/env python
"""
Expression formatting utilities for converting arithmetic expressions to English.
"""
import logging
import random
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List

import sympy
from num2words import num2words


def number_to_string(
    num: float | int,
    rng: random.Random,
    integer_no_decimal_probability: float = 0.0,
) -> str:
    """Return a compact human-readable string for *num*."""

    # Handle integer case with optional probability to drop ".0"
    if float(num).is_integer():
        if rng.random() < integer_no_decimal_probability:
            return f"{int(num)}"
        return f"{int(num)}.0"

    # For non-integers fall back to a general format that trims needless zeros
    # while keeping enough precision to round-trip typical binary64 floats.
    return format(num, ".15g")


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


def format_expression_string(
    expression: sympy.Basic | None,
    english_conversion_probability: float = 0.0,
    seed: int = 42,
    printing_style_probs: Dict[str, float] | None = None,
) -> tuple[str, str]:
    """Enhanced expression formatting with probabilistic style selection.

    Args:
        expression: SymPy expression to format (can be None when train=True)
        english_conversion_probability: Probability of converting operations to English
        seed: Random seed for consistency
        printing_style_probs: Style probability distribution

    Returns:
        Formatted expression string
        Style used for rendering
    """

    # Validate input type
    if expression is not None and not isinstance(expression, sympy.Basic):
        raise TypeError(
            f"expression must be a SymPy expression, got {type(expression)}"
        )

    # Set default style probabilities
    if printing_style_probs is None or not printing_style_probs:
        printing_style_probs = {"sstr": 1.0}

    # Sample printing style
    rng = random.Random(seed)
    style = sample_printing_style(printing_style_probs, rng)

    # Render expression in selected style
    expr_str, style = render_expr(expression, style)

    # Apply English conversion to operations based on the style
    if english_conversion_probability > 0:
        # Only convert operations for sstr and latex styles
        if style in ["sstr", "latex"]:
            # Add proper spacing around operators
            expr_str = add_operator_spacing(expr_str)
            expr_str = substitute_operations(
                expr_str, english_conversion_probability, seed
            )

    return expr_str, style


# --------------------------------------------------------------------------- #
# Multi-Style Expression Rendering System
# --------------------------------------------------------------------------- #

# Supported printing styles
SUPPORTED_STYLES = {"sstr", "pretty", "ascii", "latex", "repr"}


def validate_printing_style_probs(printing_style_probs: Dict[str, float]) -> None:
    """Validate printing style probability dictionary.

    Args:
        printing_style_probs: Dictionary mapping style names to probabilities

    Raises:
        ValueError: If probabilities don't sum to 1.0 or contain unsupported styles
    """
    if not printing_style_probs:
        raise ValueError("printing_style_probs cannot be empty")

    # Check for unsupported styles
    unsupported = set(printing_style_probs.keys()) - SUPPORTED_STYLES
    if unsupported:
        raise ValueError(
            f"Unsupported printing styles: {unsupported}. "
            f"Supported styles: {SUPPORTED_STYLES}"
        )

    # Check probabilities sum to 1.0 (with small tolerance for floating point)
    total_prob = sum(printing_style_probs.values())
    if abs(total_prob - 1.0) > 1e-6:
        raise ValueError(
            f"Printing style probabilities must sum to 1.0, got {total_prob}"
        )


def sample_printing_style(
    printing_style_probs: Dict[str, float], rng: random.Random
) -> str:
    """Sample a printing style based on the provided probabilities.

    Args:
        printing_style_probs: Dictionary mapping style names to probabilities
        rng: Random number generator for consistent sampling

    Returns:
        Selected printing style name
    """
    validate_printing_style_probs(printing_style_probs)

    # Use random choice with weights
    styles = list(printing_style_probs.keys())
    weights = list(printing_style_probs.values())

    return rng.choices(styles, weights=weights)[0]


def render_expr(expr: sympy.Basic, style: str) -> tuple[str, str]:
    """Render a SymPy expression using the specified printing style.

    Args:
        expr: SymPy expression to render
        style: Printing style ("sstr", "pretty", "ascii", "latex")

    Returns:
        String representation of the expression
        Style used for rendering since it might change during a fallback.

    Raises:
        ValueError: If style is not supported
    """
    if style not in SUPPORTED_STYLES:
        raise ValueError(f"Unsupported style: {style}. Supported: {SUPPORTED_STYLES}")

    try:
        if style == "sstr":
            return sympy.sstr(expr), style
        elif style == "pretty":
            return sympy.pretty(expr), style
        elif style == "ascii":
            return sympy.pretty(expr, use_unicode=False), style
        elif style == "latex":
            return sympy.latex(expr), style
        elif style == "repr":
            return repr(expr), style
        else:
            raise ValueError(f"Unhandled style: {style}")
    except Exception as e:
        # SymPy pretty printing can sometimes fail with internal bugs
        # Fall back to the most reliable string representation
        logging.warning(
            f"SymPy rendering failed for style '{style}' with error: {e}. Falling back to 'sstr' style."
        )
        return sympy.sstr(expr), "sstr"


def add_operator_spacing(expression: str) -> str:
    """Add proper spacing around operators while preserving line structure.

    Args:
        expression: Expression string to add spacing to

    Returns:
        Expression string with properly spaced operators
    """
    result = expression.strip()
    lines = result.split("\n")
    processed_lines = []

    for line in lines:
        # Add spaces around operators in each line individually
        line = re.sub(r"(\d)\s*([+*/^=()])\s*", r"\1 \2 ", line)
        line = re.sub(r"([+*/^=()])\s*(\d)", r"\1 \2", line)
        line = re.sub(r"(\d|\))\s*-\s*", r"\1 - ", line)  # Binary minus
        line = re.sub(r"(^|[+*/^=(])\s*-\s*(\d)", r"\1- \2", line)  # Unary minus
        line = re.sub(r" +", " ", line).strip()
        processed_lines.append(line)

    # Reconstruct the multi-line result
    return "\n".join(processed_lines)


def substitute_operations(
    expression: str, english_probability: float, seed: int
) -> str:
    """Substitute mathematical operations with English equivalents.

    Args:
        expression: Expression string with operations to convert
        english_probability: Probability of converting each operator
        seed: Random seed for consistency

    Returns:
        Expression with operations potentially converted to English
    """
    rng = random.Random(seed + 54321)  # Different seed offset than operands
    result = expression

    # Process line by line to maintain structure
    lines = result.split("\n")
    processed_lines = []

    for line in lines:
        current_line = line

        # Step 1: Convert visual fractions to English
        if rng.random() < english_probability:
            # This step is handled separately since fractions span multiple lines
            pass  # Fraction handling is now in substitute_operands

        # Step 2: Convert non-minus operators to English
        if rng.random() < english_probability:
            # Handle multi-character operators first
            if "**" in current_line and rng.random() < english_probability:
                current_line = current_line.replace("**", " to_the_power_of ")

            # Handle single operators
            operators = {
                "+": ["plus", "added to"],
                "*": ["times", "multiplied by"],
                "/": ["divided by", "over"],
                "^": ["to the power of", "raised to"],
            }

            for op, options in operators.items():
                if op in current_line and rng.random() < english_probability:
                    english_op = rng.choice(options)
                    current_line = current_line.replace(op, f" {english_op} ")

        # Step 3: Handle minus signs systematically
        if rng.random() < english_probability:
            # First, identify and convert binary minus (subtraction)
            # Pattern: number/paren followed by minus followed by space and number (not another minus)
            minus_options = ["minus", "subtract", "less"]
            english_minus = rng.choice(minus_options)
            current_line = re.sub(
                r"(\d|\))\s*-\s+(\d)", rf"\1 {english_minus} \2", current_line
            )

        # Clean up line-level formatting
        current_line = re.sub(
            r"to_the_power_of", "to the power of", current_line
        )  # Fix temp replacement
        current_line = re.sub(r"\s+", " ", current_line).strip()
        processed_lines.append(current_line)

    return "\n".join(processed_lines)
