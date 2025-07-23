#!/usr/bin/env python
"""
streaming.py
On-the-fly DAG dataset generation for training.
"""

import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import sympy
import torch
from sympy import im
from tiktoken import get_encoding

from data.dagset.expression_to_string import convert_number_to_english
from models.dag_model import execute_stack

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.dag_model import LOG_LIM, OP_NAMES

from .expression_to_string import format_expression_string

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #


@dataclass
class DAGExample:
    """Lightweight container for a DAG computation example."""

    text: str
    depth: int
    max_digits: int
    max_decimal_places: int
    english_conversion_probability: float
    integer_no_decimal_probability: float
    printing_style: str
    initial_values: list[float]  # for logging
    signs: torch.Tensor  # (D+1)
    digits: torch.Tensor  # (D+1, digits_total, base)
    base: int
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
    structure_dict: dict[str, torch.Tensor] | None = None  # structure format tensors

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"DAGExample(seed={self.seed}, text={self.text}, depth={self.depth}, initial_values={self.initial_values}, signs={self.signs.shape}, digits={self.digits.shape}, operations={self.operations.shape}, operations_named={self.operations_named}, did_expand={self.did_expand}, did_simplify={self.did_simplify}, final_value_sympy={self.final_value_sympy}, final_value_exec={self.final_value_exec}, allowed_operations={self.allowed_operations}, expr={self.expr}, english_conversion_probability={self.english_conversion_probability}, integer_no_decimal_probability={self.integer_no_decimal_probability}, printing_style={self.printing_style}, base={self.base})"


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
    base: int = 10,
    allowed_operations: list[str] | None,
    expression_simplification_probability: float,
    expression_expansion_probability: float,
    english_conversion_probability: float = 0.0,
    integer_no_decimal_probability: float = 0.0,
    override_initial_values: list[float] | None = None,
    override_operations: list[str] | None = None,
    execute_sympy: bool = True,
) -> tuple[sympy.Basic, list[float], list[str], float | None, bool, bool, bool]:
    """Generate a sympy expression.

    Returns:
        Tuple of (sympy_expr, initial_values, operations, final_value, did_simplify, did_expand, is_complex)
    """
    # ------------------------------------------------------------------
    # 1. Generate random initial values and operations
    # ------------------------------------------------------------------
    rng = random.Random(seed)

    # Use provided operations or generate random ones
    ops_set = allowed_operations or OP_NAMES
    sym_ops = []

    if override_operations is not None:
        sym_ops = list(override_operations)
    else:
        # Identities will be added later.
        ops_set_no_identity = [op for op in ops_set if op != "identity"]
        # Choose a random number of operations between 0 and depth.
        # That we generate expressions with a variety of depths.
        # Weight higher depths more heavily.
        weights = [i + 1 for i in range(depth)]
        num_ops = rng.choices(range(depth), weights=weights, k=1)[0]
        # Generate random operations.
        for i in range(num_ops):
            op_name = rng.choice(ops_set_no_identity)
            sym_ops.append(op_name)

    # Use provided initial values or generate random ones
    initial_values = []
    if override_initial_values is not None:
        initial_values = list(override_initial_values)
    else:
        # Generate random values.
        for i in range(num_ops + 1):
            value = generate_uniform_digit_number(
                seed=seed + i,
                max_digits=max_digits,
                max_decimal_places=max_decimal_places,
                base=base,
                allow_zero=False,
                integer_no_decimal_probability=integer_no_decimal_probability,
            )
            initial_values.append(value)

    # ------------------------------------------------------------------
    # 2. Build SymPy expression from leaves + operations
    # ------------------------------------------------------------------
    # Convert values to symbols (possibly with English names based on probability)
    symbols = []
    symbol_names = []
    for val in initial_values:
        # Determine if this value should be converted to English
        should_convert = rng.random() < english_conversion_probability

        if should_convert:
            symbol_name = convert_number_to_english(val)
        else:
            # Use numerical representation
            symbol_name = str(val)

        symbols.append(sympy.Symbol(symbol_name))
        symbol_names.append(symbol_name)

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

    # For logging purposes, create a version of the sym_expr with the initial values replaced with VAL_0, VAL_1, etc.
    sym_expr_with_vals = sym_expr.subs(
        [
            (sympy.Symbol(name), sympy.Symbol(f"VAL_{i}"))
            for i, name in enumerate(symbol_names)
        ]
    )

    final_value = None
    if execute_sympy:
        # Execute the sympy expression and store the final value for validation purposes.
        value_map = {symbols[i]: initial_values[i] for i in range(len(initial_values))}
        final_value = sympy.N(sym_expr.subs(value_map))

        if im(final_value) != 0:
            final_value = np.inf

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

    return (
        sym_expr,
        sym_expr_with_vals,
        initial_values,
        operations,
        final_value,
        did_simplify,
        did_expand,
    )


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
        int_digits = [0] * (max_digits - len(int_digits)) + int_digits  # Pad with zeros

    # Convert fractional part to target base
    frac_digits = []
    temp_frac = frac_part
    for _ in range(max_decimal_places):
        temp_frac *= base
        digit = int(temp_frac)
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
    depth: int | None = None,
    allowed_operations: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a DAG plan to structure tensors for training."""
    # Convert initial values to signs and digit one-hots
    signs = torch.tensor([1.0 if v >= 0.0 else -1.0 for v in initial_values])

    digits_onehots: list[torch.Tensor] = []
    for v in initial_values:
        one_hot = float_to_digit_onehot(v, max_digits, max_decimal_places, base)
        digits_onehots.append(one_hot)
    # (num_nodes, D, base)
    digits_tensor = torch.stack(digits_onehots, dim=0)

    # Convert operations to one-hot encoded tensors
    operation_to_index = {op: i for i, op in enumerate(OP_NAMES)}
    op_depth = len(operations)
    operations_one_hot = torch.zeros(op_depth, len(OP_NAMES))

    for i, op in enumerate(operations):
        op_idx = operation_to_index[op]
        operations_one_hot[i, op_idx] = 1.0

    # Execute the DAG to get the final value
    with torch.no_grad():
        # Use double precision during reference execution to minimize numerical errors
        sign_tensor = signs.view(1, 1, -1).to(torch.float64)
        digit_probs = digits_tensor.unsqueeze(0).unsqueeze(0).to(torch.float64)
        op_probs = operations_one_hot.unsqueeze(0).unsqueeze(0).to(torch.float64)

        final_sgn, final_log = execute_stack(
            sign_tensor,
            digit_probs,
            op_probs,
            max_digits=max_digits,
            max_decimal_places=max_decimal_places,
            base=base,
            ignore_clip=True,
        )

        final_value_exec = (final_sgn * torch.exp(final_log)).item()

    # Return structure dict format for predictor training
    if depth is None:
        depth = len(operations)

    num_scratch_nodes = depth + 1  # Following DAG model convention

    # Create padded initial values tensors
    initial_sgn = torch.zeros(num_scratch_nodes)
    initial_log = torch.zeros(num_scratch_nodes)
    digits_template = torch.zeros(
        num_scratch_nodes, max_digits + max_decimal_places, base
    )

    # Fill tensors from the computed values up to the available nodes
    copy_len = min(len(signs), num_scratch_nodes)
    initial_sgn[:copy_len] = signs[:copy_len]
    digits_template[:copy_len] = digits_tensor[:copy_len]

    # Compute log magnitudes (natural log) on-the-fly
    for i in range(copy_len):
        v = abs(initial_values[i]) if initial_values[i] != 0 else 1e-6
        log_val = math.log(v)
        # Clip to LOG_LIM for safety
        log_val = max(min(log_val, LOG_LIM), -LOG_LIM)
        initial_log[i] = log_val

    # Pad or trim the operation tensor so that it always has *depth* rows
    operation_probs = torch.zeros(depth, len(OP_NAMES))
    rows_to_copy = min(op_depth, depth)
    operation_probs[:rows_to_copy] = operations_one_hot[:rows_to_copy]

    # For rows beyond the generated length, explicitly mark them as identity
    identity_idx = OP_NAMES.index("identity")
    if rows_to_copy < depth:
        operation_probs[rows_to_copy:, identity_idx] = 1.0

    # Filter allowed operations if specified
    if allowed_operations is not None:
        allowed_op_indices = [OP_NAMES.index(op) for op in allowed_operations]
        disallowed_idx = [
            i for i in range(len(OP_NAMES)) if i not in allowed_op_indices
        ]
        if disallowed_idx:
            operation_probs[:, disallowed_idx] = 0.0

    return {
        "initial_sgn": initial_sgn,
        "initial_log": initial_log,
        "initial_digits": digits_template,
        "operation_probs": operation_probs,
        "depth": torch.tensor(depth, dtype=torch.long),
        "operations": operations_one_hot,  # keep original for debugging
        "final_value_exec": torch.tensor(final_value_exec, dtype=torch.float32),
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


def generate_single_dag_example(
    depth: int,
    seed: int = 42,
    english_conversion_probability: float = 0.0,
    integer_no_decimal_probability: float = 0.0,
    expression_expansion_probability: float = 0.0,
    expression_simplification_probability: float = 0.0,
    max_digits: int = 4,
    max_decimal_places: int = 6,
    base: int = 10,
    allowed_operations: list[str] | None = None,
    execute_sympy: bool = True,
    printing_style_probs: dict[str, float] | None = None,
    # Test-only overrides – callers should provide **both** or **neither**
    _operations_override: list[str] | None = None,
    _initial_values_override: list[float] | None = None,
) -> DAGExample:
    """Generate a single DAG computation example as a simple math expression."""

    (
        sym_expr,
        sym_expr_with_vals,
        initial_values,
        operations,
        final_value_sympy,
        did_simplify,
        did_expand,
    ) = generate_expression(
        depth=depth,
        seed=seed,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        base=base,
        allowed_operations=allowed_operations,
        expression_simplification_probability=expression_simplification_probability,
        expression_expansion_probability=expression_expansion_probability,
        english_conversion_probability=english_conversion_probability,
        integer_no_decimal_probability=integer_no_decimal_probability,
        override_initial_values=_initial_values_override,
        override_operations=_operations_override,
        execute_sympy=execute_sympy,
    )

    # Now we render the expression - no need for English conversion of operands
    # since they're already converted if needed
    text, printing_style = format_expression_string(
        expression=sym_expr,
        seed=seed,
        english_conversion_probability=english_conversion_probability,
        printing_style_probs=printing_style_probs,
    )
    # Use plan_to_tensors to get the structure dict directly
    structure_dict = plan_to_tensors(
        initial_values=initial_values,
        operations=operations,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        base=base,
        depth=depth,
        allowed_operations=allowed_operations,
    )

    example = DAGExample(
        text=text,
        depth=depth,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        base=base,
        english_conversion_probability=english_conversion_probability,
        integer_no_decimal_probability=integer_no_decimal_probability,
        printing_style=printing_style,
        initial_values=initial_values,
        structure_dict=structure_dict,
        signs=structure_dict["initial_sgn"],
        digits=structure_dict["initial_digits"],
        operations=structure_dict["operation_probs"],
        final_value_exec=structure_dict["final_value_exec"].item(),
        final_value_sympy=final_value_sympy,
        operations_named=operations,
        seed=seed,
        did_expand=did_expand,
        did_simplify=did_simplify,
        allowed_operations=allowed_operations,
        expr=sym_expr_with_vals,
    )

    if (
        final_value_sympy != np.inf
        and execute_sympy
        and not math.isclose(
            example.final_value_exec,
            example.final_value_sympy,
            abs_tol=1e-2,
            rel_tol=1e-2,
        )
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
        seed: int = 42,
        tokenizer: str = "gpt2",
        max_seq_length: int = 512,
        english_conversion_probability: float = 0.0,
        integer_no_decimal_probability: float = 0.0,
        expression_simplification_probability: float = 0.0,
        expression_expansion_probability: float = 0.0,
        max_digits: int = 4,
        max_decimal_places: int = 6,
        base: int = 10,
        allowed_operations: list[str] | None = None,
        printing_style_probs: dict[str, float] | None = None,
    ):
        """Initialize the DAG structure dataset."""
        self.max_depth = max_depth
        # For DAG with depth n, we need n+1 initial values
        self.num_initial_values = max_depth + 1
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
        self.printing_style_probs = printing_style_probs
        self.max_digits = max_digits
        self.max_decimal_places = max_decimal_places
        self.base = base
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

    def generate_batch(
        self, batch_size: int, seed: int = 42, execute_sympy: bool = True
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

            try:
                # Generate example directly with structure tensors
                example = generate_single_dag_example(
                    depth=depth,
                    seed=seed + self.num_generated + i,
                    english_conversion_probability=self.english_conversion_probability,
                    integer_no_decimal_probability=self.integer_no_decimal_probability,
                    expression_simplification_probability=self.expression_simplification_probability,
                    expression_expansion_probability=self.expression_expansion_probability,
                    max_digits=self.max_digits,
                    max_decimal_places=self.max_decimal_places,
                    base=self.base,
                    allowed_operations=self.allowed_operations,
                    execute_sympy=execute_sympy,
                    printing_style_probs=self.printing_style_probs,
                )
            except Exception:
                logging.error(
                    f"Error generating Example with seed: {seed + self.num_generated + i}"
                )
                raise

            texts.append(example.text)
            structures.append(example.structure_dict)
            examples.append(example)

        # Update generation counter
        self.num_generated += batch_size

        # Batch the structure tensors
        batched_structure = self._batch_structures(structures, examples)

        return texts, batched_structure, examples

    def _batch_structures(
        self, structures: List[Dict[str, torch.Tensor]], examples: List["DAGExample"]
    ) -> Dict[str, torch.Tensor]:
        """Batch structure tensors with proper padding.

        Args:
            structures: List of structure dictionaries
            examples: List of DAGExample objects for extracting target values

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
        # Use the configured base instead of hardcoding base 10
        batched_initial_digits = torch.zeros(batch_size, max_nodes, D_total, self.base)
        batched_operation_probs = torch.zeros(batch_size, max_depth, len(OP_NAMES))
        batched_depths = torch.zeros(batch_size, dtype=torch.long)

        # Initialize target tensors for the new loss terms
        batched_target_initial_values = torch.zeros(batch_size, max_nodes)
        batched_target_final_exec = torch.zeros(batch_size)

        # Fill batched tensors
        for i, (structure, example) in enumerate(zip(structures, examples)):
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

            # Extract target initial values from example
            num_values = min(len(example.initial_values), max_nodes)
            batched_target_initial_values[i, :num_values] = torch.tensor(
                example.initial_values[:num_values], dtype=torch.float32
            )

            # Extract target final execution value from example
            if example.final_value_exec is not None:
                batched_target_final_exec[i] = torch.tensor(
                    example.final_value_exec, dtype=torch.float32
                )

        return {
            "initial_sgn": batched_initial_sgn,
            "initial_log": batched_initial_log,
            "initial_digits": batched_initial_digits,
            "operation_probs": batched_operation_probs,
            "depths": batched_depths,
            "target_initial_values": batched_target_initial_values,
            "target_final_exec": batched_target_final_exec,
        }

    def create_dataloader(
        self, batch_size: int = 32, seed: int = 42, execute_sympy: bool = True
    ) -> Iterator[Tuple[List[str], Dict[str, torch.Tensor], List["DAGExample"]]]:
        """Create an infinite dataloader for structure examples.

        Args:
            batch_size: Batch size

        Yields:
            Batches of (texts, structure_tensors)
        """
        i = 0
        while True:
            texts, structures, examples = self.generate_batch(
                batch_size, seed=seed + i, execute_sympy=execute_sympy
            )
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
    base: int = 10,  # Number base for digit representation
    allowed_operations: list[str] | None = None,
    printing_style_probs: dict[str, float] | None = None,
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
        base=base,
        allowed_operations=allowed_operations,
        printing_style_probs=printing_style_probs,
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
        base=base,
        allowed_operations=allowed_operations,
        printing_style_probs=printing_style_probs,
    )

    # Create dataloaders
    train_loader = train_dataset.create_dataloader(
        train_batch_size, seed=seed, execute_sympy=False
    )
    val_loader = val_dataset.create_dataloader(
        val_batch_size, seed=seed, execute_sympy=True
    )

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
