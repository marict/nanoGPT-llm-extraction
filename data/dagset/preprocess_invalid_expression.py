"""
Preprocessing utilities for converting invalid expression substrings into valid ones.

This module contains heuristics to fix common issues in partial expression strings,
such as incomplete decimals, unbalanced parentheses, and trailing operators.
"""


def preprocess_invalid_expression(expr_str: str) -> str | None:
    """Try to convert an invalid expression string into a valid one using heuristics.

    Applies preprocessing strategies:
    1. Complete decimals  2. Balance parentheses  3. Remove trailing operators
    4. Remove invalid leading ops  5. Handle hanging operators
    """
    if not expr_str or not expr_str.strip():
        return None

    original = expr_str.strip()

    # Complete decimals first
    if original.endswith("."):
        original = original + "0"

    # Balance parentheses BEFORE removing operators to preserve operators inside parens
    open_count = original.count("(")
    close_count = original.count(")")
    if open_count > close_count:
        original = original + ")" * (open_count - close_count)
    elif close_count > open_count:
        # Remove excess closing parens from the right
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

    # Remove trailing operators (but only OUTSIDE of balanced parentheses)
    # This is more sophisticated - we only remove trailing ops if they're not inside parens
    while original and original[-1] in "+-*/":
        # Check if this operator is inside balanced parentheses
        paren_depth = 0
        inside_parens = False
        for i, char in enumerate(original):
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            # If we're at the last character and inside parens, don't remove
            if i == len(original) - 1 and paren_depth > 0:
                inside_parens = True
                break

        if inside_parens:
            break  # Don't remove operators inside parentheses

        original = original[:-1].strip()
        if not original:
            return None

    # Final cleanup
    original = original.strip()

    # Complete decimals AGAIN after all other processing (in case trailing ops were removed)
    if original.endswith("."):
        original = original + "0"

    # Validate result - be more lenient about parentheses-only expressions
    if not original:
        return None

    # Allow expressions with just parentheses and basic content
    has_content = any(c.isdigit() or c == "." for c in original)
    if not has_content and not (original.count("(") > 0 and original.count(")") > 0):
        return None

    return original if original != expr_str.strip() else None
