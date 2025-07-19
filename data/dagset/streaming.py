#!/usr/bin/env python
"""
streaming.py
On-the-fly DAG dataset generation for training.
"""

import io
import logging
import math
import random
import re
import sys
import tokenize
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import sympy
import torch
from num2words import num2words
from sympy import im
from sympy.printing.str import StrPrinter
from tiktoken import get_encoding

from models.dag_model import execute_stack


class VerboseMulPrinter(StrPrinter):
    def _print_Mul(self, expr):
        return " * ".join(self._print(arg) for arg in expr.args)


def verbose_str(expr):
    return VerboseMulPrinter().doprint(expr)


# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.dag_model import LOG_LIM, OP_NAMES


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


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #


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


def format_expression_string(
    expression: str,
    english_conversion_probability: float = 0.0,
    seed: int = 42,
    max_decimal_places: int = 6,
) -> str:
    """Pretty-print *expression* with optional English replacements.

    The built-in *tokenize* module is used for lexing, which eliminates the
    bespoke scanner and correctly handles numbers, parentheses and operator
    tokens (including disambiguating unary minus).
    """

    rng = random.Random(seed)

    # Operator → english synonyms
    symbol_to_english = {
        "+": ["plus", "added to"],
        "-": ["minus", "subtract", "less"],
        "*": ["times", "multiplied by", "times"],
        "/": ["divided by", "over", "divide by"],
    }

    converted: list[str] = []

    for tok_type, tok_str, *_ in tokenize.generate_tokens(
        io.StringIO(expression).readline
    ):
        if tok_type == tokenize.ENDMARKER:
            break

        # ------------------------------------------------------------------
        # 1. Numeric literals
        # ------------------------------------------------------------------
        if tok_type == tokenize.NUMBER:
            if rng.random() < english_conversion_probability:
                # Convert to int where possible for nicer wording ("three" vs "three point zero")
                try:
                    val = int(tok_str)
                except ValueError:
                    val = float(tok_str)
                converted.append(convert_number_to_english(val, max_decimal_places))
            else:
                converted.append(tok_str)
            continue

        # ------------------------------------------------------------------
        # 2. Operators (including parentheses)
        # ------------------------------------------------------------------
        if tok_type == tokenize.OP:
            if (
                tok_str in symbol_to_english
                and rng.random() < english_conversion_probability
            ):
                converted.append(rng.choice(symbol_to_english[tok_str]))
            else:
                converted.append(tok_str)
            continue

        # ------------------------------------------------------------------
        # 3. Everything else (identifiers, whitespace, etc.)
        # ------------------------------------------------------------------
        converted.append(tok_str)

    return " ".join(converted).strip()


@dataclass
class DAGExample:
    """Lightweight container for a DAG computation example."""

    text: str
    depth: int
    initial_values: list[float]  # for logging
    signs: torch.Tensor  # (D+1)
    digits: torch.Tensor  # (D+1, digits_total, 10)
    # log magnitudes are now computed on-the-fly during structure tensor creation
    operations: torch.Tensor  # (D, num_ops)
    operations_named: list[str]  # (D, num_ops)
    seed: int
    did_expand: bool
    did_simplify: bool
    final_value_sympy: float | None = None  # exact symbolic evaluation
    final_value_exec: float | None = None  # value from execute_stack
    allowed_operations: list[str] | None = None
    expr: sympy.Basic | None = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"DAGExample(seed={self.seed}, text={self.text}, depth={self.depth}, initial_values={self.initial_values}, signs={self.signs.shape}, digits={self.digits.shape}, operations={self.operations.shape}, operations_named={self.operations_named}, did_expand={self.did_expand}, did_simplify={self.did_simplify}, final_value_sympy={self.final_value_sympy}, final_value_exec={self.final_value_exec}, allowed_operations={self.allowed_operations}, expr={self.expr})"


def generate_uniform_digit_number(
    seed: int = None,
    max_digits: int = 4,
    max_decimal_places: int = 6,
    allow_zero: bool = True,
) -> float:
    """Generate a number with uniform distribution over digit count combinations."""
    rng = random.Random(seed)
    num_integer_digits = rng.randint(0, max_digits)

    # 0 decimal places = integer
    num_decimal_places = rng.randint(0, max_decimal_places)

    # If zero is not allowed, and the number is 0.0, generate a random number with 1 digit, either integer or decimal
    if not allow_zero and (num_integer_digits == 0 and num_decimal_places == 0):
        if rng.random() < 0.5:
            num_integer_digits = 1
        else:
            num_decimal_places = 1

    # Generate integer part with the chosen number of digits
    if num_integer_digits == 0:
        # 0-digit: just 0 (for numbers like 0.003)
        integer_part = 0
    elif num_integer_digits == 1:
        # 1-digit: 1 to 9 (0 is handled by num_integer_digits=0)
        integer_part = rng.randint(1, 9)
    else:
        # n-digit: from 10^(n-1) to 10^n - 1
        min_val = 10 ** (num_integer_digits - 1)
        max_val = 10**num_integer_digits - 1
        integer_part = rng.randint(min_val, max_val)

    # Generate decimal part with the chosen number of decimal places
    if num_decimal_places == 0:
        # Integer (no decimal part)
        decimal_part = 0.0
    else:
        # Generate exactly num_decimal_places digits after decimal
        min_decimal = 1
        if num_decimal_places > 1:
            min_decimal = 10 ** (num_decimal_places - 1)
        max_decimal = 10**num_decimal_places - 1

        # Handle case where min_decimal can be > max_decimal if max_decimal is small
        if min_decimal > max_decimal:
            decimal_digits = max_decimal
        else:
            decimal_digits = rng.randint(min_decimal, max_decimal)

        decimal_part = decimal_digits / (10**num_decimal_places)

    # Combine integer and decimal parts
    magnitude = integer_part + decimal_part

    sign = rng.choice([-1, 1])
    value = sign * magnitude

    return value


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
        raise ValueError(
            "Identity operation not supported, it should be padded onto the end of the operations list after generation."
        )
    raise ValueError(f"Unsupported op_name: {op_name}")


def _generate_expression(
    *,
    depth: int,
    seed: int,
    max_digits: int,
    max_decimal_places: int,
    allowed_operations: list[str] | None,
    expression_simplification_probability: float,
    expression_expansion_probability: float,
    override_initial_values: list[float] | None = None,
    override_operations: list[str] | None = None,
) -> tuple[sympy.Basic, list[float], list[str], bool, bool]:
    """Create a random SymPy expression of exactly *depth* operations.

    Returns (expr, initial_values, operations, did_simplify, did_expand).
    """
    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 1. Determine leaf values & operations (override vs random)
    # ------------------------------------------------------------------
    if (override_initial_values is None) ^ (override_operations is None):
        raise ValueError(
            "Both override_initial_values and override_operations must be provided together (or neither)."
        )

    if override_initial_values is not None:
        # Use caller-supplied plan verbatim
        if len(override_operations) != depth:
            raise ValueError("Length of override_operations must equal depth")
        if len(override_initial_values) != depth + 1:
            raise ValueError("override_initial_values must have depth+1 elements")

        initial_values = list(override_initial_values)
        sym_ops = list(override_operations)
    else:
        expression_size = rng.randint(1, depth + 1)

        # Default to all operations if none are provided
        if allowed_operations is None:
            op_pool = [op for op in OP_NAMES]
        else:
            op_pool = [op for op in allowed_operations]

        assert (
            "identity" in op_pool
        ), "Identity operation must be in the allowed operations pool"

        non_identity_op_pool = [op for op in op_pool if op != "identity"]

        sym_ops = rng.choices(non_identity_op_pool, k=expression_size - 1)

        initial_values = [
            generate_uniform_digit_number(
                seed=seed * 7919 + i,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
            )
            for i in range(len(sym_ops) + 1)
        ]

        # Ensure we never divide by zero under the current stack evaluation semantics.
        for i, op in enumerate(sym_ops):
            if op == "divide":
                denom_index = i + 1
                if initial_values[denom_index] == 0.0:
                    initial_values[denom_index] = generate_uniform_digit_number(
                        seed=seed * 7919 + i,
                        max_digits=max_digits,
                        max_decimal_places=max_decimal_places,
                        allow_zero=False,
                    )

    # ------------------------------------------------------------------
    # 2. Build SymPy expression from leaves + operations
    # ------------------------------------------------------------------
    symbols = [sympy.Symbol(f"VAL_{i}") for i in range(len(initial_values))]

    nodes: list[sympy.Basic] = symbols.copy()
    ops_map: dict[sympy.Basic, str] = {}

    # Apply the operations in Reverse Polish Notation
    for op_name in reversed(sym_ops):
        top = nodes.pop()
        second = nodes.pop()
        expr = _apply_sympy_op(op_name, second, top)
        nodes.append(expr)
        ops_map[expr] = op_name

    assert len(nodes) == 1
    sym_expr: sympy.Basic = nodes[0]

    # Execute the sympy expression and store the final value for validation purposes.
    value_map = {symbols[i]: initial_values[i] for i in range(len(initial_values))}
    final_value = sympy.N(sym_expr.subs(value_map))

    # Check if result is complex (nonzero imaginary part)
    if im(final_value) != 0:
        logging.warning(
            f"Generated complex expression, sym_expr: {sym_expr}, initial_values: {initial_values}, seed: {seed}, regenerating..."
        )
        return None, None, None, None, None, None

    # We swap some operations with identity to play better with the DAG model, but this is not compatible with sympy.
    operations = [op for op in sym_ops]
    for i, op in enumerate(operations):
        if op == "multiply":
            # If second value is 0, then we can discard the top and replace with 0
            if initial_values[i] == 0.0:
                operations[i] = "identity"

        if op == "divide":
            num_index = i
            if initial_values[num_index] == 0.0:
                operations[num_index] = "identity"

    # Pad the rest of the operations with identity
    operations.extend(["identity"] * (depth - len(operations)))
    # Pad the rest of initial values with 1.0
    initial_values.extend([1.0] * (depth + 1 - len(initial_values)))

    # ------------------------------------------------------------------
    # 3. Optional simplify / expand for prettified expression rendering.
    # ------------------------------------------------------------------
    did_simplify = False
    did_expand = False
    if rng.random() < expression_simplification_probability:
        sym_expr = sympy.simplify(sym_expr)
        did_simplify = True

    if rng.random() < expression_expansion_probability:
        sym_expr = sympy.expand(sym_expr)
        did_expand = True

    return sym_expr, initial_values, operations, final_value, did_simplify, did_expand


def _expression_to_text(
    *,
    expr: sympy.Basic,
    initial_values: list[float],
    seed: int,
    english_conversion_probability: float,
    integer_no_decimal_probability: float,
    max_decimal_places: int,
) -> str:
    """Convert *expr* into a pretty-printed, optionally English-augmented string."""
    expr_str = sympy.sstr(expr)

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

    # Final English / spacing formatting
    return format_expression_string(
        expr_str,
        english_conversion_probability,
        seed,
        max_decimal_places,
    )


def float_to_digit_onehot(
    value: float, max_digits: int, max_decimal_places: int
) -> torch.Tensor:
    """Convert a float into a one-hot tensor of shape (D, 10) where D = max_digits + max_decimal_places.

    The number is first clipped to the largest representable value to avoid overflow.  The integer part is
    left-padded with zeros and the fractional part is right-padded with zeros so that their total length
    equals *max_digits + max_decimal_places*.
    """

    # Clip magnitude so that it fits in the available digits
    limit = 10**max_digits - 10 ** (-max_decimal_places)
    abs_val = min(abs(value), limit)

    # Build fixed-width string representation without sign and without decimal point
    # We first format with the desired number of decimal places, then pad the integer
    # part with leading zeros to exactly *max_digits* characters. Python's format
    # specification does not support fixed-width integer padding when combined
    # with a fractional component, so we handle the padding manually.

    s = f"{abs_val:.{max_decimal_places}f}"  # e.g. 1.2300 (variable int width)
    int_part, frac_part = s.split(".")

    # Left-pad the integer part so that it always has *max_digits* characters.
    # This guarantees the total digit count equals D = max_digits + max_decimal_places.
    if len(int_part) > max_digits:
        # This should not happen thanks to the earlier clipping, but guard anyway.
        int_part = int_part[-max_digits:]
    else:
        int_part = int_part.zfill(max_digits)

    # Ensure the fractional part has exactly *max_decimal_places* digits
    if len(frac_part) < max_decimal_places:
        frac_part = frac_part.ljust(max_decimal_places, "0")
    elif len(frac_part) > max_decimal_places:
        frac_part = frac_part[:max_decimal_places]

    s_digits = int_part + frac_part

    D = max_digits + max_decimal_places
    assert len(s_digits) == D, f"Expected {D} digits, got {len(s_digits)}"

    one_hot = torch.zeros(D, 10)
    for i, ch in enumerate(s_digits):
        one_hot[i, int(ch)] = 1.0
    return one_hot


# Updated to include digit arguments
def plan_to_tensors(
    initial_values: list[float],
    operations: list[str],
    *,
    max_digits: int,
    max_decimal_places: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a DAG plan to tensors for training.

    Args:
        initial_values: List of initial values
        operations: List of operations

    Returns:
        Tuple of (signs, digits_tensor, operations_one_hot)
    """
    # Convert initial values to signs and digit one-hots
    signs = torch.tensor([1.0 if v >= 0.0 else -1.0 for v in initial_values])

    digits_onehots: list[torch.Tensor] = []
    for v in initial_values:
        one_hot = float_to_digit_onehot(v, max_digits, max_decimal_places)
        digits_onehots.append(one_hot)
    # (num_nodes, D, 10)
    digits_tensor = torch.stack(digits_onehots, dim=0)

    # Convert operations to one-hot encoded tensors
    operation_to_index = {op: i for i, op in enumerate(OP_NAMES)}
    depth = len(operations)
    operations_one_hot = torch.zeros(depth, len(OP_NAMES))

    for i, op in enumerate(operations):
        op_idx = operation_to_index[op]
        operations_one_hot[i, op_idx] = 1.0

    return signs, digits_tensor, operations_one_hot


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


def generate_single_dag_example(
    depth: int,
    num_initial_values: int = None,
    seed: int = 42,
    english_conversion_probability: float = 0.0,
    integer_no_decimal_probability: float = 0.0,
    expression_expansion_probability: float = 0.0,
    expression_simplification_probability: float = 0.0,
    max_digits: int = 4,
    max_decimal_places: int = 6,
    allowed_operations: list[str] | None = None,
    # Test-only overrides – callers should provide **both** or **neither**
    _operations_override: list[str] | None = None,
    _initial_values_override: list[float] | None = None,
) -> DAGExample:
    """Generate a single DAG computation example as a simple math expression."""
    # Determine number of initial values to match DAG predictor expectations
    if num_initial_values is None:
        # For DAG with depth n, we need n+1 initial values
        num_initial_values = depth + 1

    # Some expressions are invalid (ex. /0), so we retry a few times.
    valid = False
    max_retries = 10  # Prevent infinite loops
    attempt = 0
    while not valid and attempt < max_retries:
        attempt += 1
        (
            sym_expr,
            initial_values,
            operations,
            final_value_sympy,
            did_simplify,
            did_expand,
        ) = _generate_expression(
            depth=depth,
            seed=seed + attempt,  # Use different seed each time
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            allowed_operations=allowed_operations,
            expression_simplification_probability=expression_simplification_probability,
            expression_expansion_probability=expression_expansion_probability,
            override_initial_values=_initial_values_override,
            override_operations=_operations_override,
        )
        if final_value_sympy is not None:
            valid = True

    if not valid:
        raise ValueError(
            f"Failed to generate valid expression after {max_retries} attempts, seed: {seed}"
        )

    text = _expression_to_text(
        expr=sym_expr,
        initial_values=initial_values,
        seed=seed,
        english_conversion_probability=english_conversion_probability,
        integer_no_decimal_probability=integer_no_decimal_probability,
        max_decimal_places=max_decimal_places,
    )
    signs, digits_tensor, operations_tensor = plan_to_tensors(
        initial_values=initial_values,
        operations=operations,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
    )

    with torch.no_grad():
        # Use double precision during reference execution to minimize numerical errors
        sign_tensor = signs.view(1, 1, -1).to(torch.float64)
        digit_probs = digits_tensor.unsqueeze(0).unsqueeze(0).to(torch.float64)
        op_probs = operations_tensor.unsqueeze(0).unsqueeze(0).to(torch.float64)

        final_sgn, final_log = execute_stack(
            sign_tensor,
            digit_probs,
            op_probs,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            ignore_clip=True,
        )

        final_value_exec = (
            final_sgn * torch.pow(torch.tensor(10.0, dtype=final_log.dtype), final_log)
        ).item()

    example = DAGExample(
        text=text,
        depth=depth,
        initial_values=initial_values,
        signs=signs,
        digits=digits_tensor,
        operations=operations_tensor,
        operations_named=operations,
        seed=seed,
        did_expand=did_expand,
        did_simplify=did_simplify,
        final_value_sympy=final_value_sympy,
        final_value_exec=final_value_exec,
        allowed_operations=allowed_operations,
        expr=sym_expr,
    )

    if not math.isclose(
        example.final_value_exec,
        example.final_value_sympy,
        abs_tol=1e-3,
        rel_tol=1e-3,
    ):
        logging.warning(
            f"\n\n-------------------WARNING: Final value mismatch between sympy and tensor execute: {example.final_value_exec} != {example.final_value_sympy}, \nexample: {example}\n\n-------------------"
        )

    return example


class DAGStructureDataset:
    """
    Dataset for pretraining DAG predictor on structure prediction.
    Maps text descriptions to DAG structure tensors.
    """

    def __init__(
        self,
        max_depth: int = 8,
        num_initial_values: int = None,
        seed: int = 42,
        tokenizer: str = "gpt2",
        max_seq_length: int = 512,
        english_conversion_probability: float = 0.0,
        integer_no_decimal_probability: float = 0.0,
        expression_simplification_probability: float = 0.0,
        expression_expansion_probability: float = 0.0,
        max_digits: int = 4,
        max_decimal_places: int = 6,
        allowed_operations: list[str] | None = None,
    ):
        """Initialize the DAG structure dataset."""
        self.max_depth = max_depth
        # Set num_initial_values to match DAG predictor expectations
        self.num_initial_values = (
            num_initial_values if num_initial_values is not None else max_depth + 1
        )
        self.seed = seed
        self.num_generated = 0
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.english_conversion_probability = english_conversion_probability
        self.integer_no_decimal_probability = integer_no_decimal_probability
        self.expression_simplification_probability = (
            expression_simplification_probability
        )
        self.expression_expansion_probability = expression_expansion_probability
        self.max_digits = max_digits
        self.max_decimal_places = max_decimal_places
        # If a subset of operations is provided, validate and store mapping.
        if allowed_operations is not None:
            invalid_ops = [op for op in allowed_operations if op not in OP_NAMES]
            if invalid_ops:
                raise ValueError(
                    f"Invalid operations provided: {invalid_ops}. Available operations: {OP_NAMES}"
                )
            self.allowed_operations = list(allowed_operations)
        else:
            self.allowed_operations = None  # None implies full OP_NAMES

        # Pre-compute column indices mapping from allowed subset to global OP_NAMES
        if self.allowed_operations is None:
            self.allowed_op_indices = list(range(len(OP_NAMES)))
        else:
            self.allowed_op_indices = [
                OP_NAMES.index(op) for op in self.allowed_operations
            ]

        # Initialize tokenizer
        self.enc = get_encoding(tokenizer)

        # Operation name to index mapping
        self.op_name_to_idx = {name: i for i, name in enumerate(OP_NAMES)}
        self.op_idx_to_name = {i: name for i, name in enumerate(OP_NAMES)}

    def generate_structure_example(
        self,
        depth: int,
        seed: int = 42,
    ) -> Tuple[
        str, Dict[str, torch.Tensor] | Tuple[str, Dict[str, torch.Tensor], "DAGExample"]
    ]:
        """Generate a single (text, structure) pair.

        Args:
            depth: DAG depth for this example

        Returns:
            Tuple of (text_description, structure_tensors)
        """
        # Generate a random state using the data loader seed
        # Generate the DAG example (structure-only, no execution needed)
        example = generate_single_dag_example(
            depth=depth,
            num_initial_values=self.num_initial_values,
            seed=seed + self.num_generated,
            english_conversion_probability=self.english_conversion_probability,
            integer_no_decimal_probability=self.integer_no_decimal_probability,
            expression_simplification_probability=self.expression_simplification_probability,
            expression_expansion_probability=self.expression_expansion_probability,
            max_digits=self.max_digits,
            max_decimal_places=self.max_decimal_places,
            allowed_operations=self.allowed_operations,
        )

        self.num_generated += 1
        # Extract text
        text = example.text

        # Create structure tensors
        structure = self._create_structure_tensors(example)

        return text, structure, example

    def _create_structure_tensors(self, example: DAGExample) -> Dict[str, torch.Tensor]:
        """Convert DAG example to structure tensors matching DAGPlanPredictor format.

        Args:
            example: DAG computation example

        Returns:
            Dictionary with structure tensors
        """
        depth = example.depth
        num_scratch_nodes = depth + 1  # Following DAG model convention

        # Create initial values tensors
        initial_sgn = torch.zeros(num_scratch_nodes)
        # digits tensor shape (num_nodes, D, 10)
        digits_template = torch.zeros_like(example.digits)
        # log-magnitude tensor (consistent with historical interface)
        initial_log = torch.zeros(num_scratch_nodes)

        # Fill tensors from the example up to the available nodes
        signs_tensor = example.signs
        digits_tensor = example.digits
        copy_len = min(len(signs_tensor), num_scratch_nodes)

        initial_sgn[:copy_len] = signs_tensor[:copy_len]
        digits_template[:copy_len] = digits_tensor[:copy_len]

        # Compute log magnitudes (natural log) on-the-fly
        for i in range(copy_len):
            v = (
                abs(example.initial_values[i])
                if example.initial_values[i] != 0
                else 1e-6
            )
            log_val = math.log(v)
            # Clip to LOG_LIM for safety
            log_val = max(min(log_val, LOG_LIM), -LOG_LIM)
            initial_log[i] = log_val

        # Pad or trim the example's operation tensor so that it always has
        # *depth* rows.  This guarantees consistent tensor shapes downstream
        # even when an early identity truncates the plan length.

        op_len = example.operations.shape[0]
        operation_probs = torch.zeros(depth, len(OP_NAMES))
        rows_to_copy = min(op_len, depth)
        operation_probs[:rows_to_copy] = example.operations[:rows_to_copy]

        # For rows beyond the generated length, explicitly mark them as identity
        identity_idx = OP_NAMES.index("identity")
        if rows_to_copy < depth:
            operation_probs[rows_to_copy:, identity_idx] = 1.0

        if self.allowed_operations is not None:
            disallowed_idx = [
                i for i in range(len(OP_NAMES)) if i not in self.allowed_op_indices
            ]
            if disallowed_idx:
                operation_probs[:, disallowed_idx] = 0.0

        return {
            "initial_sgn": initial_sgn,
            "initial_log": initial_log,
            "initial_digits": digits_template,
            "operation_probs": operation_probs,
            "depth": torch.tensor(depth, dtype=torch.long),
            "operations": example.operations,  # raw list (for debugging)
        }

    def generate_batch(
        self, batch_size: int, seed: int = 42
    ) -> Tuple[List[str], Dict[str, torch.Tensor], List["DAGExample"]]:
        """Generate a batch of structure examples.

        Args:
            batch_size: Number of examples to generate

        Returns:
            Tuple of (text_list, batched_structure_tensors)
        """
        texts: list[str] = []
        structures: list[Dict[str, torch.Tensor]] = []
        examples: list[DAGExample] = []

        for i in range(batch_size):
            # Use fixed depth - all examples in dataset should have the same depth
            # The identity function allows us to handle cases with effective depth < max_depth naturally
            depth = self.max_depth

            # Generate example
            text, structure, example = self.generate_structure_example(
                depth, seed=seed + i
            )
            texts.append(text)
            structures.append(structure)
            examples.append(example)

        # Batch the structure tensors
        batched_structure = self._batch_structures(structures)

        return texts, batched_structure, examples

    def _batch_structures(
        self, structures: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Batch structure tensors with proper padding.

        Args:
            structures: List of structure dictionaries

        Returns:
            Batched structure tensors
        """
        batch_size = len(structures)
        max_depth = max(s["depth"].item() for s in structures)
        max_nodes = max_depth + 1

        # Initialize batched tensors
        batched_initial_sgn = torch.zeros(batch_size, max_nodes)
        batched_initial_log = torch.zeros(batch_size, max_nodes)
        D_total = self.max_digits + self.max_decimal_places
        batched_initial_digits = torch.zeros(batch_size, max_nodes, D_total, 10)
        batched_operation_probs = torch.zeros(batch_size, max_depth, len(OP_NAMES))
        batched_depths = torch.zeros(batch_size, dtype=torch.long)

        # Fill batched tensors
        for i, structure in enumerate(structures):
            depth = structure["depth"].item()
            nodes = depth + 1

            # Copy initial values
            batched_initial_sgn[i, :nodes] = structure["initial_sgn"][:nodes]
            batched_initial_log[i, :nodes] = structure["initial_log"][:nodes]
            batched_initial_digits[i, :nodes] = structure["initial_digits"][:nodes]

            # Copy operation probabilities
            batched_operation_probs[i, :depth] = structure["operation_probs"][:depth]

            # Store depth
            batched_depths[i] = depth

        return {
            "initial_sgn": batched_initial_sgn,
            "initial_log": batched_initial_log,
            "initial_digits": batched_initial_digits,
            "operation_probs": batched_operation_probs,
            "depths": batched_depths,
        }

    def create_dataloader(
        self, batch_size: int = 32, seed: int = 42
    ) -> Iterator[Tuple[List[str], Dict[str, torch.Tensor], List["DAGExample"]]]:
        """Create an infinite dataloader for structure examples.

        Args:
            batch_size: Batch size

        Yields:
            Batches of (texts, structure_tensors)
        """
        i = 0
        while True:
            texts, structures, examples = self.generate_batch(batch_size, seed=seed + i)
            i += 1
            yield texts, structures, examples


def create_dag_structure_dataloaders(
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    max_depth: int = 8,
    seed: int = 42,
    english_conversion_probability: float = 0.0,
    integer_no_decimal_probability: float = 0.0,
    expression_simplification_probability: float = 0.0,
    expression_expansion_probability: float = 0.0,
    max_digits: int = 4,  # Maximum number of integer digits for uniform digit distribution
    max_decimal_places: int = 6,  # Auto-derived from max_digits for uniform string distribution
    allowed_operations: list[str] | None = None,
) -> Tuple[Iterator, Iterator]:
    """Create train/val DAG structure dataloaders for predictor training.

    Args:
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        max_depth: DAG depth (all examples will have this depth)
        seed: Seed for training data
        english_conversion_probability: Probability of converting to English (0.0 to 1.0)
        integer_no_decimal_probability: Probability of converting to integers (0.0 to 1.0)
        max_digits: Maximum number of integer digits (1-4 means 1-digit to 4-digit integers)
        max_decimal_places: Maximum decimal places.

    Returns:
        Tuple of (train_loader, val_loader) iterators
    """
    # Use configurable english conversion rate

    # Create datasets with fixed depth
    train_dataset = DAGStructureDataset(
        max_depth=max_depth,
        seed=seed,
        english_conversion_probability=english_conversion_probability,
        integer_no_decimal_probability=integer_no_decimal_probability,
        expression_simplification_probability=expression_simplification_probability,
        expression_expansion_probability=expression_expansion_probability,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        allowed_operations=allowed_operations,
    )

    val_dataset = DAGStructureDataset(
        max_depth=max_depth,
        seed=seed,
        english_conversion_probability=english_conversion_probability,
        integer_no_decimal_probability=integer_no_decimal_probability,
        expression_simplification_probability=expression_simplification_probability,
        expression_expansion_probability=expression_expansion_probability,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        allowed_operations=allowed_operations,
    )

    # Create dataloaders
    train_loader = train_dataset.create_dataloader(train_batch_size, seed=seed)
    val_loader = val_dataset.create_dataloader(val_batch_size, seed=seed)

    return train_loader, val_loader


if __name__ == "__main__":
    # Simple example usage
    print("DAG Streaming Dataset Example")
    print("=" * 40)

    # Test the new padding approach
    print("\nTesting pad_dag_plan:")
    initial_values = [5.5, 3.2, 1.8, 4.1, 7.3, 2.9, 6.4]
    operations = ["add", "multiply", "identity", "subtract", "add", "divide"]

    print(f"Original plan:")
    print(f"Initial values: {initial_values}")
    print(f"Operations: {operations}")

    # Test the structure dataset (used in train_predictor.py)
    print("\nTesting DAGStructureDataset with consistent stack-based processing:")
    print("(Both initial values and operations processed right-to-left)")
    structure_dataset = DAGStructureDataset(
        max_depth=3, seed=42, max_digits=3
    )  # max_decimal_places auto-derived as 2
    texts, structures, seeds = structure_dataset.generate_batch(2)
    print(f"Generated {len(texts)} structure examples")
    print(f"Structure keys: {list(structures.keys())}")
    print(f"Initial signs shape: {structures['initial_sgn'].shape}")
    print(f"Operation probs shape: {structures['operation_probs'].shape}")
    print(f"Seeds: {seeds}")

    print("\n✅ Example completed successfully!")
    print("✅ Operations now processed consistently as stack (right-to-left)")
