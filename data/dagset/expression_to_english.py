"""Reversible conversion between mathematical expressions and English descriptions.

This module provides bidirectional conversion between mathematical expressions
and their English equivalents, enabling models to learn from natural language
descriptions of mathematical operations.

Example:
    "3.5 + 4.2 * (1.8 - 6.7)" <-> "three point five plus four point two times open parenthesis one point eight minus six point seven close parenthesis"
"""

import re
from typing import Optional

# Number word mappings
DIGIT_TO_WORD = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

WORD_TO_DIGIT = {v: k for k, v in DIGIT_TO_WORD.items()}

# Operation mappings
OP_TO_WORD = {
    "+": "plus",
    "-": "minus",
    "*": "times",
    "/": "divided by",
    "(": "open parenthesis",
    ")": "close parenthesis",
    ".": "point",
}

WORD_TO_OP = {v: k for k, v in OP_TO_WORD.items()}

# Special handling for "divided by" which is two words
WORD_TO_OP["divided"] = "/"  # Will need special handling for partial matches


def number_to_english(number_str: str) -> str:
    """Convert a number string to English words.

    Args:
        number_str: Number like "3.5", "-42.17", "0"

    Returns:
        English description like "three point five", "minus four two point one seven"
    """
    if not number_str:
        return ""

    result = []
    i = 0

    # Handle negative sign
    if number_str.startswith("-"):
        result.append("minus")
        i = 1

    # Process each character individually for precise reversibility
    while i < len(number_str):
        char = number_str[i]
        if char in DIGIT_TO_WORD:
            result.append(DIGIT_TO_WORD[char])
        elif char == ".":
            result.append("point")
        i += 1

    return " ".join(result)


def english_to_number(english_str: str) -> Optional[str]:
    """Convert English number words back to number string.

    Args:
        english_str: English like "three point five" or "minus four two"

    Returns:
        Number string like "3.5" or "-42", or None if conversion fails
    """
    if not english_str.strip():
        return None

    words = english_str.strip().split()
    result = []

    i = 0
    # Handle leading minus
    if i < len(words) and words[i] == "minus":
        result.append("-")
        i += 1

    # Convert remaining words - each digit word maps to one digit
    while i < len(words):
        word = words[i]
        if word in WORD_TO_DIGIT:
            result.append(WORD_TO_DIGIT[word])
        elif word == "point":
            result.append(".")
        else:
            # Unknown word - this might be an incomplete number
            return None
        i += 1

    number_str = "".join(result)

    # Allow incomplete numbers for partial parsing
    if not number_str or number_str in ["-"]:
        return None

    # Allow incomplete decimals like "3." or "-42."
    if number_str.endswith("."):
        return number_str

    # Check if it's a valid number format for complete numbers
    try:
        float(number_str)
        return number_str
    except ValueError:
        return None


def expression_to_english(expr_str: str) -> str:
    """Convert mathematical expression to English description.

    Args:
        expr_str: Expression like "3.5 + 4.2 * (1.8 - 6.7)"

    Returns:
        English description like "three point five plus four point two times..."
    """
    if not expr_str.strip():
        return ""

    # Tokenize the expression into numbers, operators, and whitespace
    tokens = re.findall(r"-?\d+\.?\d*|\+|\-|\*|\/|\(|\)|\s+", expr_str)

    english_parts = []

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Check if it's a number (including negative numbers)
        if re.match(r"-?\d+\.?\d*$", token):
            english_parts.append(number_to_english(token))
        # Check if it's an operator
        elif token in OP_TO_WORD:
            english_parts.append(OP_TO_WORD[token])
        else:
            # Unknown token - skip for now
            continue

    return " ".join(english_parts)


def english_to_expression(english_str: str) -> Optional[str]:
    """Convert English description back to mathematical expression.

    Args:
        english_str: English like "three point five plus four point two"

    Returns:
        Expression string like "3.5 + 4.2", or None if conversion fails
    """
    if not english_str.strip():
        return None

    words = english_str.strip().split()
    if not words:
        return None

    result_parts = []
    i = 0

    while i < len(words):
        word = words[i]

        # Check if this word starts a number
        if word in {
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        } or (
            word == "minus"
            and i + 1 < len(words)
            and words[i + 1]
            in {
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            }
        ):
            # Special handling for consecutive "minus" words (double negative)
            if word == "minus" and i + 1 < len(words) and words[i + 1] == "minus":
                # Handle "minus minus" as "- -"
                result_parts.append("-")
                i += 1
                continue

            # Try to parse a number starting at this position
            number_words = []
            j = i

            # Collect consecutive number-related words
            while j < len(words):
                if words[j] in {
                    "minus",
                    "zero",
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    "seven",
                    "eight",
                    "nine",
                    "point",
                }:
                    # Only include 'minus' if it's at the start (negative sign)
                    if words[j] == "minus" and j > i:
                        # This is 'minus' after other digits - likely an operator
                        break
                    number_words.append(words[j])
                    j += 1
                else:
                    break

            if number_words:
                # Try to convert collected words to a number
                number_str = english_to_number(" ".join(number_words))
                if number_str is not None:
                    result_parts.append(number_str)
                    i = j
                    continue
                else:
                    return None

        # Handle two-word operations first
        if word == "divided" and i + 1 < len(words) and words[i + 1] == "by":
            result_parts.append("/")
            i += 2
            continue

        # Handle parentheses (two words)
        if word == "open" and i + 1 < len(words) and words[i + 1] == "parenthesis":
            result_parts.append("(")
            i += 2
            continue

        if word == "close" and i + 1 < len(words) and words[i + 1] == "parenthesis":
            result_parts.append(")")
            i += 2
            continue

        # Handle incomplete two-word operations (for per-token parsing)
        if word == "divided":
            # Incomplete "divided by" - return None for now
            return None

        if word in ["open", "close"]:
            # Incomplete parenthesis - return None for now
            return None

        # Handle single-word operations (including standalone 'minus')
        if word in WORD_TO_OP:
            result_parts.append(WORD_TO_OP[word])
            i += 1
            continue

        # Unknown word - might be incomplete conversion
        return None

    if not result_parts:
        return None

    # Join with spaces for readability
    expression = " ".join(result_parts)

    # Allow expressions ending with operators (important for per-token training)
    # Only reject expressions that are JUST operators
    if expression.strip() in ["+", "-", "*", "/", "(", "."]:
        return None

    return expression


def normalize_expression(expr: str) -> str:
    """Normalize expression spacing for comparison."""
    if not expr:
        return ""
    # Remove extra spaces and normalize
    return " ".join(expr.split())


def test_conversions():
    """Test the conversion functions with various examples."""
    test_cases = [
        "3.5",
        "-42.17",
        "3.5 + 4.2",
        "3.5 + 4.2 * (1.8 - 6.7)",
        "-5.3 + 85.0*(3.34 - 5145.0)",
        "42 / 7",
        "((3.5 + 1.2) * 4.0) - 2.8",
    ]

    print("🧪 Testing Expression ↔ English Conversion")
    print("=" * 60)

    for expr in test_cases:
        english = expression_to_english(expr)
        back_to_expr = english_to_expression(english)

        # Test reversibility with normalized comparison
        original_norm = normalize_expression(expr)
        back_norm = normalize_expression(back_to_expr) if back_to_expr else None
        roundtrip_ok = back_norm == original_norm

        print(f"Original:  {expr}")
        print(f"English:   {english}")
        print(f"Back:      {back_to_expr}")
        print(f"✅ Reversible: {roundtrip_ok}")
        if not roundtrip_ok and back_to_expr:
            print(f"   Original normalized: '{original_norm}'")
            print(f"   Back normalized:     '{back_norm}'")
        print()

    # Test partial conversions (for per-token processing)
    print("🔄 Testing Partial Conversions (for per-token use)")
    print("=" * 60)

    partial_cases = [
        "three",
        "three point",
        "three point five",
        "three point five plus",
        "three point five plus four",
        "three point five plus four point two",
        "three point five plus four point two times",
        "open parenthesis three point five",
        "open parenthesis three point five close parenthesis",
    ]

    for partial in partial_cases:
        result = english_to_expression(partial)
        print(f"'{partial}' → '{result}'")


if __name__ == "__main__":
    test_conversions()
