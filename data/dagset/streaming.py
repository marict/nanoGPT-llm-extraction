#!/usr/bin/env python
"""
streaming.py
On-the-fly DAG dataset generation for training.
"""

import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import sympy
import torch
from num2words import num2words
from tiktoken import get_encoding

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.dag_model import LOG_LIM, OP_NAMES


def convert_number_to_english(number: float, max_decimal_places: int = 6) -> str:
    """Convert a number to its English word equivalent.

    Args:
        number: The number to convert (int or float)

    Returns:
        String representation (either words or original format)
    """

    # Handle zero
    if number == 0 or (isinstance(number, float) and abs(number) < 0.00001):
        return "zero"

    # Handle negative numbers
    is_negative = number < 0
    abs_number = abs(number)

    # Handle integers
    if isinstance(abs_number, int) or (
        isinstance(abs_number, float) and abs_number.is_integer()
    ):
        result = num2words(int(abs_number))
        return f"negative {result}" if is_negative else result

    # Handle floats with decimals
    if isinstance(abs_number, float):
        # Convert to string to get the decimal representation
        number_str = str(abs_number)

        if "." in number_str:
            parts = number_str.split(".")
            integer_part = int(parts[0]) if parts[0] else 0
            decimal_part = parts[1]

            # Build the result
            result = num2words(integer_part) if integer_part != 0 else "zero"
            result += " point"

            # Convert each decimal digit to words (limit to 5 decimal places)
            for digit in decimal_part[:max_decimal_places]:
                result += " " + num2words(int(digit))

            return f"negative {result}" if is_negative else result

    return str(number)


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
    """Format an expression string with optional english words and spaces.

    Args:
        expression: Expression string
        english_conversion_probability: Probability of converting each individual token
        rng: Random number generator

    Returns:
        Formatted expression string with optional english words and spaces
    """
    rng = random.Random(seed)

    # Symbol to English word mappings
    symbol_to_english = {
        "+": ["plus", "added to"],
        "-": ["minus", "subtract", "less"],
        "*": ["times", "multiplied by", "times"],
        "/": ["divided by", "over", "divide by"],
    }

    # Split the expression into tokens (numbers, operators, parentheses)
    # Use a more sophisticated approach to handle negative numbers vs minus operators
    tokens = []
    i = 0
    while i < len(expression):
        if expression[i] == "-":
            # Check if this is a negative number or a minus operator
            # It's a negative number if it's at the start or after (, +, -, *, /
            # We need to look backwards past spaces to find the actual operator
            is_negative_number = False
            if i == 0:
                is_negative_number = True
            else:
                # Look backwards past spaces to find the last non-space character
                k = i - 1
                while k >= 0 and expression[k] == " ":
                    k -= 1
                if k >= 0 and expression[k] in "(+-*/":
                    is_negative_number = True

            if is_negative_number:
                # Look ahead to see if there's a number after the minus
                j = i + 1
                while j < len(expression) and expression[j] in "0123456789.":
                    j += 1
                if j > i + 1:  # Found digits after minus
                    tokens.append(expression[i:j])
                    i = j
                    continue
            # Otherwise, it's a minus operator
            tokens.append("-")
            i += 1
        elif expression[i] in "0123456789":
            # Regular positive number
            j = i
            while j < len(expression) and expression[j] in "0123456789.":
                j += 1
            tokens.append(expression[i:j])
            i = j
        elif expression[i] in "+*/()":
            tokens.append(expression[i])
            i += 1
        elif expression[i] == " ":
            i += 1  # Skip spaces
        else:
            i += 1  # Skip unknown characters

    # Clean up tokens by removing empty strings
    tokens = [t for t in tokens if t.strip()]

    converted_tokens = []
    for token in tokens:
        if token in symbol_to_english:
            # Randomly convert operator to English based on probability
            if rng.random() < english_conversion_probability:
                converted_tokens.append(rng.choice(symbol_to_english[token]))
            else:
                converted_tokens.append(token)
        elif token in ["(", ")"]:
            # Keep parentheses as symbols - don't convert to English
            converted_tokens.append(token)
        elif re.match(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", token):
            # Randomly convert number to words based on probability
            # Handle both positive and negative numbers, int and float types
            try:
                if "." in token:
                    number = float(token)
                else:
                    number = int(token)
            except ValueError:
                number = float(token)

            if rng.random() < english_conversion_probability:
                converted_tokens.append(
                    convert_number_to_english(number, max_decimal_places)
                )
            else:
                converted_tokens.append(token)
        else:
            # Keep token as is
            converted_tokens.append(token)

    return " ".join(converted_tokens)


# --------------------------------------------------------------------------- #
# Recursive permutation helper
# --------------------------------------------------------------------------- #


def permute_expression(expr: sympy.Basic, rng: random.Random) -> sympy.Basic:
    """Recursively shuffle operands of every commutative Add/Mul node.

    * Every subtree is visited depth-first; children are permuted **after** they
      themselves have been processed, guaranteeing maximal variability.
    * Only nodes with two or more operands are shuffled.
    * Non-commutative operations (Sub, Div, Pow, etc.) are left untouched.
    """

    # Leaf node – nothing to permute
    if not isinstance(expr, (sympy.Add, sympy.Mul)):
        return expr

    # Recurse into children first
    permuted_children = [permute_expression(arg, rng) for arg in expr.args]

    # Shuffle operands if there are at least two
    if len(permuted_children) > 1:
        rng.shuffle(permuted_children)

    if isinstance(expr, sympy.Add):
        return sympy.Add(*permuted_children, evaluate=False)
    else:  # sympy.Mul
        return sympy.Mul(*permuted_children, evaluate=False)


def permute_additive_flat_expression(expr: str, rng: random.Random) -> str:
    """Return a permuted additive expression string.

    Preconditions (caller MUST guarantee):
    1. *expr* contains only the '+' binary operator.
    2. Expression is *flat* (no parentheses) and therefore commutative.

    The function shuffles the numeric literals (with their leading sign) using
    *rng* and reconstructs a canonical ``" + "``-joined string.  If either
    pre-condition is violated an exception is raised.
    """

    if any(op in expr for op in "*/-"):
        raise ValueError(
            "permute_additive_flat_expression requires '+'-only expression without '-' operator"
        )

    if "(" in expr or ")" in expr:
        raise ValueError("Expression must be flat (no parentheses)")

    # Extract numeric tokens including scientific notation.
    tokens = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", expr)
    if len(tokens) < 2:
        raise ValueError("Need at least two additive operands to permute")

    rng.shuffle(tokens)
    return " + ".join(tokens)


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
    seed: int
    did_expand: bool
    did_simplify: bool


def generate_uniform_digit_number(
    seed: int = None, max_digits: int = 4, max_decimal_places: int = 6
) -> float:
    """Generate a number with uniform distribution over digit count combinations."""
    rng = random.Random(seed)
    num_integer_digits = rng.randint(0, max_digits)

    # 0 decimal places = integer
    num_decimal_places = rng.randint(0, max_decimal_places)

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


# NEW EXPRESSION-FIRST HELPERS --------------------------------------------------


def _apply_sympy_op(op_name: str, a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    """Return a SymPy expression representing *a <op> b* without evaluation."""
    if op_name == "add":
        return sympy.Add(a, b, evaluate=False)
    if op_name == "subtract":
        return sympy.Add(a, -b, evaluate=False)
    if op_name == "multiply":
        return sympy.Mul(a, b, evaluate=False)
    if op_name == "divide":
        return sympy.Mul(a, sympy.Pow(b, -1, evaluate=False), evaluate=False)
    if op_name == "identity":
        # Represent identity as a 2-argument dummy function so traversal
        # records the op while the mathematical value equals *a*.
        IdentityFunc = sympy.Function("IDENTITY")
        return IdentityFunc(a, b)
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
        operations = list(override_operations)
    else:
        num_leaves = depth + 1
        initial_values = [
            generate_uniform_digit_number(
                seed=seed * 7919 + i,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
            )
            for i in range(num_leaves)
        ]

        # Choose operations under identity cutoff logic
        if allowed_operations is None:
            op_pool = [op for op in OP_NAMES if op != "identity"]
        else:
            invalid = [op for op in allowed_operations if op not in OP_NAMES]
            if invalid:
                raise ValueError(
                    f"Invalid operations provided: {invalid}. Available operations: {OP_NAMES}"
                )
            op_pool = [op for op in allowed_operations if op != "identity"]
            if not op_pool:
                op_pool = [op for op in OP_NAMES if op != "identity"]

        cutoff_idx = rng.randint(0, depth)
        operations = [
            ("identity" if idx >= cutoff_idx else rng.choice(op_pool))
            for idx in range(depth)
        ]

    # ------------------------------------------------------------------
    # 2. Build SymPy expression from leaves + operations
    # ------------------------------------------------------------------
    symbols = [sympy.Symbol(f"VAL_{i}") for i in range(len(initial_values))]
    placeholder_to_value = dict(zip(symbols, initial_values))

    nodes: list[sympy.Basic] = symbols.copy()
    ops_map: dict[sympy.Basic, str] = {}

    for op_name in reversed(operations):  # process stackwise (right-to-left)
        b = nodes.pop()
        a = nodes.pop()
        expr = _apply_sympy_op(op_name, a, b)
        nodes.append(expr)
        ops_map[expr] = op_name

    assert len(nodes) == 1
    sym_expr: sympy.Basic = nodes[0]

    # ------------------------------------------------------------------
    # 4. Deterministic post-order traversal to derive plan BEFORE any
    #    algebraic transforms so that we always traverse a strictly *binary*
    #    tree (the structure produced by the stack-based op application).  Any
    #    subsequent simplify/expand calls may merge operands into n-ary nodes
    #    but those changes must *not* affect the recorded plan.
    # ------------------------------------------------------------------
    derived_ops: list[str] = []
    derived_initials: list[float] = []
    symbol_to_index: dict[sympy.Symbol, int] = {}

    def _traverse(node: sympy.Basic) -> int:
        if isinstance(node, sympy.Symbol):
            if node not in symbol_to_index:
                idx = len(symbol_to_index)
                symbol_to_index[node] = idx
                derived_initials.append(placeholder_to_value[node])
            return symbol_to_index[node]
        # With traversal now happening before any algebraic transforms, every
        # internal node is guaranteed to have **exactly two** operands.
        if len(node.args) != 2:
            raise ValueError("Non-binary node encountered before simplify/expand phase")

        left, right = node.args
        _traverse(left)
        _traverse(right)

        op_name = ops_map.get(node, "identity")
        derived_ops.append(op_name)

        # Return index of the left child so that parent nodes refer to it.
        return symbol_to_index[left]

    _traverse(sym_expr)

    assert len(derived_ops) == depth, "Traversal yielded wrong op count"
    assert len(derived_initials) == depth + 1, "Incorrect initial value count"

    # ------------------------------------------------------------------
    # 5. Optional simplify / expand for prettified expression rendering.
    #    These *must* happen *after* plan derivation so they don't interfere
    #    with the binary traversal assumptions above.
    # ------------------------------------------------------------------
    did_simplify = False
    did_expand = False
    if rng.random() < expression_simplification_probability:
        sym_expr = sympy.simplify(sym_expr)
        did_simplify = True

    if rng.random() < expression_expansion_probability:
        sym_expr = sympy.expand(sym_expr)
        did_expand = True

    return sym_expr, derived_initials, derived_ops, did_simplify, did_expand


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
    # sympy.pretty produces multi-line box drawings which are hard to post-process.
    # sstr yields a compact single-line representation respecting arg order.
    expr_str = sympy.sstr(expr, order="lex")

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

    # ------------------------------------------------------------------ #
    # Plan generation – either use caller-supplied overrides (for tests) or
    # fall back to stochastic generation.
    # ------------------------------------------------------------------ #

    # Always call _generate_expression_plan with overrides parameters
    sym_expr, initial_values, operations, did_simplify, did_expand = (
        _generate_expression(
            depth=depth,
            seed=seed,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            allowed_operations=allowed_operations,
            expression_simplification_probability=expression_simplification_probability,
            expression_expansion_probability=expression_expansion_probability,
            override_initial_values=(
                _initial_values_override
                if _initial_values_override is not None
                else None
            ),
            override_operations=(
                _operations_override if _operations_override is not None else None
            ),
        )
    )

    expression = _expression_to_text(
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

    return DAGExample(
        text=expression,
        depth=depth,
        initial_values=initial_values,
        signs=signs,
        digits=digits_tensor,
        operations=operations_tensor,
        seed=seed,
        did_expand=did_expand,
        did_simplify=did_simplify,
    )


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
