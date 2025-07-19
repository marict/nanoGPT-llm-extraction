#!/usr/bin/env python
"""
Expression formatting utilities for converting arithmetic expressions to English.
"""

import io
import random
import re
import tokenize
from decimal import ROUND_HALF_UP, Decimal
from typing import List, Tuple, Union

import sympy
from num2words import num2words


def number_to_string(
    num: float,
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


class ExpressionFormatter:
    """Handles conversion of arithmetic expressions to English with proper negative number handling."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.symbol_to_english = {
            "+": ["plus", "added to"],
            "-": ["minus", "subtract", "less"],
            "*": ["times", "multiplied by"],
            "/": ["divided by", "over", "divide by"],
        }

    def _tokenize_expression(self, expression: str) -> List[Tuple[int, str]]:
        """Tokenize the expression into (token_type, token_string) pairs."""
        tokens = []
        for tok_type, tok_str, *_ in tokenize.generate_tokens(
            io.StringIO(expression).readline
        ):
            if tok_type == tokenize.ENDMARKER:
                break
            tokens.append((tok_type, tok_str))
        return tokens

    def _is_unary_minus(self, tokens: List[Tuple[int, str]], index: int) -> bool:
        """Determine if the minus at the given index is unary (negative) or binary (subtraction)."""
        if tokens[index][1] != "-" or index + 1 >= len(tokens):
            return False

        # Must be followed by a number and either at start or after opening paren/operator
        return tokens[index + 1][0] == tokenize.NUMBER and (
            index == 0 or tokens[index - 1][1] in "(-+*/"
        )

    def _convert_number(
        self, token_str: str, should_convert: bool, max_decimal_places: int
    ) -> str:
        """Convert a number token to English if requested."""
        if not should_convert:
            return token_str

        try:
            val = int(token_str)
        except ValueError:
            val = float(token_str)
        return convert_number_to_english(val, max_decimal_places)

    def _process_negative_number(
        self, number_token: str, english_probability: float, max_decimal_places: int
    ) -> List[str]:
        """Process a negative number, returning converted tokens."""
        convert_minus = self.rng.random() < english_probability
        convert_number = self.rng.random() < english_probability

        if convert_minus and convert_number:
            # Convert to "negative [number_in_english]"
            val = float(number_token) if "." in number_token else int(number_token)
            english_number = convert_number_to_english(val, max_decimal_places)
            return [f"negative {english_number}"]
        elif convert_minus:
            # Convert minus to English, keep number as-is
            return [self.rng.choice(self.symbol_to_english["-"]), number_token]
        elif convert_number:
            # Keep minus as-is, convert number to English
            return ["-", self._convert_number(number_token, True, max_decimal_places)]
        else:
            # Keep both as-is
            return ["-", number_token]

    def format_expression(
        self,
        expression: str,
        english_conversion_probability: float = 0.0,
        max_decimal_places: int = 6,
    ) -> str:
        """Convert arithmetic expression to English with proper negative number handling."""
        tokens = self._tokenize_expression(expression)
        converted = []
        i = 0

        while i < len(tokens):
            tok_type, tok_str = tokens[i]

            # Handle unary minus (negative number)
            if tok_type == tokenize.OP and self._is_unary_minus(tokens, i):
                result_tokens = self._process_negative_number(
                    tokens[i + 1][1], english_conversion_probability, max_decimal_places
                )
                converted.extend(result_tokens)
                i += 2
            # Handle regular number
            elif tok_type == tokenize.NUMBER:
                converted.append(
                    self._convert_number(
                        tok_str,
                        self.rng.random() < english_conversion_probability,
                        max_decimal_places,
                    )
                )
                i += 1
            # Handle operator
            elif tok_type == tokenize.OP and tok_str in self.symbol_to_english:
                if self.rng.random() < english_conversion_probability:
                    converted.append(self.rng.choice(self.symbol_to_english[tok_str]))
                else:
                    converted.append(tok_str)
                i += 1
            # Everything else (identifiers, whitespace, etc.)
            else:
                converted.append(tok_str)
                i += 1

        return " ".join(converted).strip()


def format_expression_string(
    expression: Union[str, sympy.Basic],
    english_conversion_probability: float = 0.0,
    seed: int = 42,
    max_decimal_places: int = 6,
    initial_values: list[float] | None = None,
    integer_no_decimal_probability: float = 0.0,
) -> str:
    """Converts an arithmetic expression into English.

    The expression can be either:
    1. A string like "-1.2 + (5 / 3)" which will be converted to "negative one point two plus five divided by three"
    2. A SymPy expression with placeholders (VAL_0, VAL_1, etc.) that will be replaced with values from initial_values

    Args:
        expression: String or SymPy expression to format
        english_conversion_probability: Probability of converting numbers/operators to English words
        seed: Random seed for consistent conversions
        max_decimal_places: Maximum decimal places to show in formatted numbers
        initial_values: List of values to substitute for VAL_X placeholders in SymPy expressions
        integer_no_decimal_probability: Probability of dropping ".0" from integer values
    """
    # If we have a SymPy expression, convert it to string and handle placeholders
    if isinstance(expression, sympy.Basic):
        if initial_values is None:
            raise ValueError("initial_values must be provided for SymPy expressions")

        expr_str = sympy.sstr(expression)

        # Replace placeholders with formatted numbers
        rng_int = random.Random(seed + 12345)
        for idx, val in enumerate(initial_values):
            placeholder = f"VAL_{idx}"
            formatted_abs = number_to_string(
                abs(val),
                rng=rng_int,
                integer_no_decimal_probability=integer_no_decimal_probability,
            )
            replacement = formatted_abs if val >= 0 else f"-{formatted_abs}"
            expr_str = expr_str.replace(placeholder, replacement)

        # Collapse spaces that might appear inside numeric literals
        expr_str = re.sub(r"\s+", "", expr_str)
        expression = expr_str

    formatter = ExpressionFormatter(seed)
    return formatter.format_expression(
        expression, english_conversion_probability, max_decimal_places
    )
