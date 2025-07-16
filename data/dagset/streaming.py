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


def convert_number_to_words(number, max_decimal_places: int = 6) -> str:
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


def format_expression_string(
    expression: str,
    conversion_probability: float = 0.3,
    seed: int = 42,
    max_decimal_places: int = 6,
) -> str:
    """Format an expression string with optional english words and spaces.

    Args:
        expression: Expression string
        conversion_probability: Probability of converting each individual token
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
            if rng.random() < conversion_probability:
                converted_tokens.append(rng.choice(symbol_to_english[token]))
            else:
                converted_tokens.append(token)
        elif token in ["(", ")"]:
            # Keep parentheses as symbols - don't convert to English
            converted_tokens.append(token)
        elif re.match(r"-?\d+\.?\d*", token):
            # Randomly convert number to words based on probability
            # Handle both positive and negative numbers, int and float types
            try:
                if "." in token:
                    number = float(token)
                else:
                    number = int(token)
            except ValueError:
                number = float(token)

            if rng.random() < conversion_probability:
                converted_tokens.append(
                    convert_number_to_words(number, max_decimal_places)
                )
            else:
                converted_tokens.append(token)
        else:
            # Keep token as is
            converted_tokens.append(token)

    return " ".join(converted_tokens)


@dataclass
class DAGExample:
    """Lightweight container for a DAG computation example."""

    text: str
    depth: int
    initial_values: list[float]  # for logging
    signs: torch.Tensor  # (D+1)
    digits: torch.Tensor  # (D+1, digits_total, 10)
    log_magnitudes: torch.Tensor  # (D+1) natural-log magnitudes (for legacy tests)
    operations: torch.Tensor  # (D, num_ops)
    seed: int


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


def generate_random_dag_plan(
    depth: int,
    num_initial_values: int = 1,
    seed: int = 42,
    max_digits: int = 4,  # Maximum number of integer digits (1=1-digit, 2=2-digit, etc.)
    max_decimal_places: int = 6,
    allowed_operations: list[str] | None = None,
    identity_cutoff_p: float = 0.1,
) -> tuple[list[float], list[str]]:
    rng = random.Random(seed)
    # Generate random initial values with uniform digit count distribution
    # for both integer part and decimal part
    initial_values = [
        generate_uniform_digit_number(
            max_digits=max_digits, max_decimal_places=max_decimal_places, seed=seed + i
        )
        for i in range(num_initial_values)
    ]

    # Determine the set of operations that can be sampled
    if allowed_operations is None:
        op_choices = OP_NAMES
    else:
        # Check for invalid operations and raise an informative error
        invalid_ops = [op for op in allowed_operations if op not in OP_NAMES]
        if invalid_ops:
            raise ValueError(
                f"Invalid operations provided: {invalid_ops}. Available operations: {OP_NAMES}"
            )

        # All provided operations are valid; use them
        op_choices = list(allowed_operations)

    # Step 1 & 2 – generate operations excluding identity, then (optionally) insert
    # an identity at a random cutoff index and convert all following ops to identity.

    if len(op_choices) == 0:
        raise ValueError("No operations provided")

    op_choices_no_identity = [op for op in op_choices if op != "identity"]

    if len(op_choices_no_identity) == 0:
        # We only have identity operations.
        # Return all identity operations with 1.0 initial values
        operations = ["identity"] * depth
        initial_values = [1.0] * (depth + 1)
        return initial_values, operations

    operations = [rng.choice(op_choices_no_identity) for _ in range(depth)]

    if depth > 0:
        cutoff_idx = rng.randint(0, depth - 1)
        operations[cutoff_idx:] = ["identity"] * (depth - cutoff_idx)

    # Step 4 – constant-based identity replacement on every surviving op.
    for k in range(depth - 1, -1, -1):  # iterate right-to-left
        if k + 1 >= len(initial_values):
            continue

        right_operand = initial_values[k + 1]

        if (
            operations[k] in ["multiply", "divide"] and abs(right_operand - 1.0) < 1e-6
        ) or (operations[k] in ["add", "subtract"] and abs(right_operand) < 1e-6):
            operations[k] = "identity"

    return initial_values, operations


def pad_plan(
    initial_values: list[float], operations: list[str]
) -> tuple[list[float], list[str]]:
    """Pad a DAG plan with systematic identity operations and 1.0 values.

    Now that operations are processed right-to-left (like a stack), padding is simple:
    - Find first identity operation
    - Replace all subsequent operations with identities
    - Replace corresponding rightmost initial values with 1.0s

    Args:
        initial_values: List of initial values
        operations: List of operations (processed right-to-left)

    Returns:
        Tuple of (padded_initial_values, padded_operations)
    """
    # Find first identity operation
    try:
        first_identity_idx = operations.index("identity")
    except ValueError:
        # No identity operations found, return original plan
        return initial_values.copy(), operations.copy()

    # Pad operations: replace everything after first identity with identities
    padded_operations = operations.copy()
    for i in range(first_identity_idx + 1, len(operations)):
        padded_operations[i] = "identity"

    # Pad initial values: since operations are processed right-to-left,
    # the rightmost operations (padding identities) will consume the rightmost initial values
    padded_initial_values = initial_values.copy()
    for i in range(first_identity_idx + 1, len(padded_initial_values)):
        padded_initial_values[i] = 1.0

    return padded_initial_values, padded_operations


def plan_to_string_expression(
    initial_values: list[float],
    operations: list[str],
    seed: int = 42,
    conversion_probability: float = 0.3,
    max_decimal_places: int = 6,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Convert DAG structure to a simple mathematical expression string following stack-based execution."""

    # Generate expression using only absolute values with unique identifiers
    # This avoids sympy's automatic simplification of signs
    abs_values = [abs(v) for v in initial_values]
    unique_symbols = [f"VAL_{i}_{abs_values[i]}" for i in range(len(abs_values))]

    # Create sympy symbols with unique identifiers
    stack = [sympy.Symbol(symbol) for symbol in unique_symbols]

    # Track which initial values each stack position depends on for masking
    num_initial = len(initial_values)
    stack_dependencies = [set([i]) for i in range(num_initial)]

    op_name_to_symbol = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/",
        "identity": "identity",
    }
    op_symbol_to_expression = {
        # Use unevaluated SymPy objects for consistent behavior
        "+": lambda a, b: sympy.Add(a, b, evaluate=False),
        "-": lambda a, b: sympy.Add(a, -b, evaluate=False),
        "*": lambda a, b: sympy.Mul(a, b, evaluate=False),
        "/": lambda a, b: sympy.Mul(
            a, sympy.Pow(b, -1, evaluate=False), evaluate=False
        ),
        "identity": lambda a, b: a,  # Discard b
    }

    # Process operations from right to left (like a stack)
    for op in reversed(operations):
        # Pop operands from both stacks
        b = stack.pop()
        a = stack.pop()
        b_deps = stack_dependencies.pop()
        a_deps = stack_dependencies.pop()

        # Create expression
        op_symbol = op_name_to_symbol[op]
        expr = op_symbol_to_expression[op_symbol](a, b)
        stack.append(expr)

        # Update dependencies
        if op == "identity":
            # Identity keeps first operand, discards second
            result_deps = a_deps
        else:
            # Binary operations use both operands
            result_deps = a_deps.union(b_deps)

        stack_dependencies.append(result_deps)

    # Get expression string with unique symbols
    expr_str = str(stack[0])

    # At this point, since we know our expression contains non-negative values, we can collapse + - into -
    expr_str = expr_str.replace("+ (-", "- (")
    expr_str = expr_str.replace("+-", "-")
    expr_str = expr_str.replace("+ -", "-")

    # Replace symbols with actual signed values
    # This preserves the original operation structure without sympy interference
    for i, original_value in enumerate(initial_values):
        old_symbol = unique_symbols[i]
        if original_value >= 0:
            new_value = str(abs_values[i])
        else:
            new_value = f"-{abs_values[i]}"

        expr_str = expr_str.replace(old_symbol, new_value)

    # Apply final formatting
    result = format_expression_string(
        expr_str, conversion_probability, seed, max_decimal_places
    )

    return result


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
        Tuple of (signs, log_magnitudes, operations_one_hot)
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
    conversion_probability: float = 0.3,
    max_digits: int = 4,
    max_decimal_places: int = 6,
    allowed_operations: list[str] | None = None,
    identity_cutoff_p: float = 0.1,
) -> DAGExample:
    """Generate a single DAG computation example as a simple math expression."""
    # Determine number of initial values to match DAG predictor expectations
    if num_initial_values is None:
        # For DAG with depth n, we need n+1 initial values
        num_initial_values = depth + 1

    initial_values, operations = generate_random_dag_plan(
        depth,
        num_initial_values,
        seed,
        max_digits,
        max_decimal_places,
        allowed_operations=allowed_operations,
        identity_cutoff_p=identity_cutoff_p,
    )

    initial_values, operations = pad_plan(initial_values, operations)

    expression = plan_to_string_expression(
        initial_values=initial_values,
        operations=operations,
        seed=seed,
        conversion_probability=conversion_probability,
        max_decimal_places=max_decimal_places,
    )

    signs, digits_tensor, operations_tensor = plan_to_tensors(
        initial_values=initial_values,
        operations=operations,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
    )

    # Compute log magnitudes (natural log) for legacy compatibility
    log_magnitudes = torch.tensor(
        [math.log(abs(v) if v != 0 else 1e-6) for v in initial_values]
    )

    return DAGExample(
        text=expression,
        depth=depth,
        initial_values=initial_values,
        signs=signs,
        digits=digits_tensor,
        log_magnitudes=log_magnitudes,
        operations=operations_tensor,
        seed=seed,
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
        english_conversion_probability: float = 0.3,
        max_digits: int = 4,
        max_decimal_places: int = 6,
        allowed_operations: list[str] | None = None,
        identity_cutoff_p: float = 0.1,
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
        self.max_digits = max_digits
        self.max_decimal_places = max_decimal_places
        self.identity_cutoff_p = identity_cutoff_p
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
        self, depth: int, seed: int = 42
    ) -> Tuple[str, Dict[str, torch.Tensor]]:
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
            conversion_probability=self.english_conversion_probability,
            max_digits=self.max_digits,
            max_decimal_places=self.max_decimal_places,
            allowed_operations=self.allowed_operations,
            identity_cutoff_p=self.identity_cutoff_p,
        )

        self.num_generated += 1
        # Extract text
        text = example.text

        # Create structure tensors
        structure = self._create_structure_tensors(example)

        return text, structure

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

        # Compute log magnitudes (natural log) from the original initial values
        for i in range(copy_len):
            log_val = example.log_magnitudes[i]
            # Clip to LOG_LIM for safety
            log_val = max(min(log_val.item(), LOG_LIM), -LOG_LIM)
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
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        """Generate a batch of structure examples.

        Args:
            batch_size: Number of examples to generate

        Returns:
            Tuple of (text_list, batched_structure_tensors)
        """
        texts = []
        structures = []
        seeds = []

        for i in range(batch_size):
            # Use fixed depth - all examples in dataset should have the same depth
            # The identity function allows us to handle cases with effective depth < max_depth naturally
            depth = self.max_depth

            # Generate example
            text, structure = self.generate_structure_example(depth, seed=seed + i)
            seeds.append(seed + i)
            texts.append(text)
            structures.append(structure)

        # Batch the structure tensors
        batched_structure = self._batch_structures(structures)

        return texts, batched_structure, seeds

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
    ) -> Iterator[Tuple[List[str], Dict[str, torch.Tensor]]]:
        """Create an infinite dataloader for structure examples.

        Args:
            batch_size: Batch size

        Yields:
            Batches of (texts, structure_tensors)
        """
        i = 0
        while True:
            texts, structures, seeds = self.generate_batch(batch_size, seed=seed + i)
            i += 1
            yield texts, structures, seeds


def create_dag_structure_dataloaders(
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    max_depth: int = 8,
    seed: int = 42,
    english_conversion_rate: float = 0.3,
    max_digits: int = 4,  # Maximum number of integer digits for uniform digit distribution
    max_decimal_places: int = 6,  # Auto-derived from max_digits for uniform string distribution
    allowed_operations: list[str] | None = None,
    identity_cutoff_p: float = 0.1,
) -> Tuple[Iterator, Iterator]:
    """Create train/val DAG structure dataloaders for predictor training.

    Args:
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        max_depth: DAG depth (all examples will have this depth)
        seed: Seed for training data
        english_conversion_rate: Probability of converting to English (0.0 to 1.0)
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
        english_conversion_probability=english_conversion_rate,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        allowed_operations=allowed_operations,
        identity_cutoff_p=identity_cutoff_p,
    )

    val_dataset = DAGStructureDataset(
        max_depth=max_depth,
        seed=seed,
        english_conversion_probability=english_conversion_rate,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
        allowed_operations=allowed_operations,
        identity_cutoff_p=identity_cutoff_p,
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

    padded_values, padded_ops = pad_plan(initial_values, operations)
    print(f"\nAfter padding:")
    print(f"Initial values: {padded_values}")
    print(f"Operations: {padded_ops}")

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
