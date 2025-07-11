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

from models.dag_model import OP_NAMES


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
        # Use high-precision formatting to avoid losing decimal places
        number_str = f"{abs_number:.15f}"

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
    rng: random.Random = None,
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
    if rng is None:
        rng = random

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
    signs: torch.Tensor  # for training - [D+1] tensor
    log_magnitudes: torch.Tensor  # for training - [D+1] tensor
    operations: torch.Tensor  # for training - [D x num_ops] one-hot tensor


def generate_uniform_digit_number(
    rng: random.Random,
    max_digits: int = 4,
    max_decimal_places: int = None,
) -> float:
    """Generate a number with uniform distribution over digit count combinations.

    This creates uniform distribution over string representations that the tokenizer sees,
    rather than uniform distribution over numerical values.

    Args:
        rng: Random number generator
        max_digits: Maximum number of integer digits (1 to max_digits)
        max_decimal_places: Maximum decimal places. If None, derived as max_digits-1
                           to create more uniform string length distribution

    Returns:
        Generated number as float
    """
    if max_decimal_places is None:
        # Derive decimal places to create more uniform string lengths
        # max_digits=4 -> max_decimal_places=3 gives good balance
        max_decimal_places = max(0, max_digits - 1)

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
    rng: random.Random = None,
    max_digits: int = 4,  # Maximum number of integer digits (1=1-digit, 2=2-digit, etc.)
    max_decimal_places: int = None,  # Auto-derived from max_digits for uniform string distribution
) -> tuple[list[float], list[str]]:
    if rng is None:
        rng = random

    # Generate random initial values with uniform digit count distribution
    # for both integer part and decimal part
    initial_values = [
        generate_uniform_digit_number(rng, max_digits, max_decimal_places)
        for _ in range(num_initial_values)
    ]
    operations = [rng.choice(OP_NAMES) for _ in range(depth)]

    # Iterate operations from right-to-left (k indexes that order)
    # and replace multiply/divide operations with identity when right-hand operand is approximately 1.0
    # and replace add/subtract operations with identity when right-hand operand is approximately 0.0
    for k in range(depth - 1, -1, -1):  # k goes from depth-1 to 0
        # Right-hand operand is initial_values[k+1]
        if k + 1 < len(initial_values):
            right_operand = initial_values[k + 1]

            # Check if operation should be replaced with identity
            should_replace = False

            # Check for multiply/divide with operand approximately 1.0
            if (
                operations[k] in ["multiply", "divide"]
                and abs(right_operand - 1.0) < 1e-6
            ):
                should_replace = True

            # Check for add/subtract with operand approximately 0.0
            elif (
                operations[k] in ["add", "subtract"] and abs(right_operand - 0.0) < 1e-6
            ):
                should_replace = True

            if should_replace:
                # Replace with identity operation
                operations[k] = "identity"
                # Leave the constant untouched (don't modify initial_values[k+1])

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
    rng: random.Random = None,
    conversion_probability: float = 0.3,
    max_decimal_places: int = 6,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Convert DAG structure to a simple mathematical expression string following stack-based execution."""
    if rng is None:
        rng = random

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
        expr_str, conversion_probability, rng, max_decimal_places
    )

    return result


def plan_to_tensors(
    initial_values: list[float],
    operations: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a DAG plan to tensors for training.

    Args:
        initial_values: List of initial values
        operations: List of operations

    Returns:
        Tuple of (signs, log_magnitudes, operations_one_hot)
    """
    # Convert initial values to signs and log magnitudes
    signs = torch.tensor([1.0 if v >= 0.0 else -1.0 for v in initial_values])

    # Handle potential zero values by using a small epsilon to avoid log(0) domain error
    epsilon = 1e-8
    log_magnitudes = torch.tensor(
        [math.log(max(abs(v), epsilon)) for v in initial_values]
    )

    # Convert operations to one-hot encoded tensors
    operation_to_index = {op: i for i, op in enumerate(OP_NAMES)}
    depth = len(operations)
    operations_one_hot = torch.zeros(depth, len(OP_NAMES))

    for i, op in enumerate(operations):
        op_idx = operation_to_index[op]
        operations_one_hot[i, op_idx] = 1.0

    return signs, log_magnitudes, operations_one_hot


def generate_single_dag_example(
    depth: int,
    num_initial_values: int = None,
    rng: random.Random = None,
    conversion_probability: float = 0.3,
    max_digits: int = 4,
    max_decimal_places: int = None,  # Auto-derived from max_digits for uniform string distribution
) -> DAGExample:
    """Generate a single DAG computation example as a simple math expression.

    Args:
        depth: Depth of the DAG computation
        num_initial_values: Number of initial values
        rng: Random number generator to use
        conversion_probability: Probability of English conversion
        max_digits: Maximum number of integer digits (1-4 means 1-digit to 4-digit integers)
        max_decimal_places: Maximum decimal places. If None, auto-derived as max_digits-1

    Returns:
        DAG computation example with simple expression format
    """
    if rng is None:
        rng = random

    # Determine number of initial values to match DAG predictor expectations
    if num_initial_values is None:
        # For DAG with depth n, we need n+1 initial values
        num_initial_values = depth + 1

    initial_values, operations = generate_random_dag_plan(
        depth, num_initial_values, rng, max_digits, max_decimal_places
    )

    initial_values, operations = pad_plan(initial_values, operations)

    expression = plan_to_string_expression(
        initial_values=initial_values,
        operations=operations,
        rng=rng,
        conversion_probability=conversion_probability,
        max_decimal_places=max_decimal_places,
    )

    signs, log_magnitudes, operations = plan_to_tensors(
        initial_values=initial_values,
        operations=operations,
    )

    return DAGExample(
        text=expression,
        depth=depth,
        initial_values=initial_values,
        signs=signs,
        log_magnitudes=log_magnitudes,
        operations=operations,
    )


def generate_dag_dataset(
    num_examples: int = 10000,
    max_depth: int = 8,
    num_initial_values: int = None,
    rng: random.Random = None,
    conversion_probability: float = 0.3,
    max_digits: int = 4,  # Maximum number of integer digits for uniform digit distribution
    max_decimal_places: int = None,  # Auto-derived from max_digits for uniform string distribution
) -> list[DAGExample]:
    """Generate a dataset of DAG computation examples (structure-only).

    Args:
        num_examples: Number of examples to generate
        max_depth: DAG depth (all examples will have this depth)
        num_initial_values: Number of initial values per example
        rng: Random number generator to use
        conversion_probability: Probability of English conversion
        max_digits: Maximum number of integer digits (1-4 means 1-digit to 4-digit integers)
        max_decimal_places: Maximum decimal places. If None, auto-derived as max_digits-1

    Returns:
        List of DAG examples with structure only (no computed results)
    """
    if rng is None:
        rng = random

    examples = []

    for _ in range(num_examples):
        # Use fixed depth - all examples should have the same depth as the model
        # The identity function allows us to handle effective shorter computations naturally
        depth = max_depth

        # Generate example (structure-only for training efficiency)
        example = generate_single_dag_example(
            depth,
            num_initial_values,
            rng,
            conversion_probability,
            max_digits,
            max_decimal_places,
        )
        examples.append(example)

    return examples


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
        max_digits: int = 4,  # Maximum number of integer digits for uniform digit distribution
        max_decimal_places: int = None,  # Auto-derived from max_digits for uniform string distribution
    ):
        """Initialize the DAG structure dataset.

        Args:
            max_depth: DAG depth (all examples will have this depth)
            num_initial_values: Number of initial values per example
            seed: Random seed for reproducibility
            tokenizer: Tokenizer to use (default: gpt2)
            max_seq_length: Maximum sequence length for tokenization
            english_conversion_probability: Probability of converting to English (0.0 to 1.0)
            max_digits: Maximum number of integer digits (1-4 means 1-digit to 4-digit integers)
            max_decimal_places: Maximum decimal places. If None, auto-derived as max_digits-1
        """
        self.max_depth = max_depth
        # Set num_initial_values to match DAG predictor expectations
        self.num_initial_values = (
            num_initial_values if num_initial_values is not None else max_depth + 1
        )
        self.seed = seed
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.english_conversion_probability = english_conversion_probability
        self.max_digits = max_digits
        self.max_decimal_places = max_decimal_places

        # Initialize tokenizer
        self.enc = get_encoding(tokenizer)

        # Create random state
        self.random_state = random.Random(seed)

        # Operation name to index mapping
        self.op_name_to_idx = {name: i for i, name in enumerate(OP_NAMES)}
        self.op_idx_to_name = {i: name for i, name in enumerate(OP_NAMES)}

    def generate_structure_example(
        self, depth: int
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
            rng=self.random_state,
            conversion_probability=self.english_conversion_probability,
            max_digits=self.max_digits,
            max_decimal_places=self.max_decimal_places,
        )

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
        initial_log = torch.zeros(num_scratch_nodes)

        # Fill initial values from the example's sign and log magnitude tensors
        signs_tensor = example.signs
        log_mags_tensor = example.log_magnitudes

        # Copy values up to num_scratch_nodes
        copy_len = min(len(signs_tensor), num_scratch_nodes)
        initial_sgn[:copy_len] = signs_tensor[:copy_len]
        initial_log[:copy_len] = log_mags_tensor[:copy_len]

        # Use the example's operation tensor (already one-hot encoded)
        operation_probs = example.operations

        return {
            "initial_sgn": initial_sgn,
            "initial_log": initial_log,
            "operation_probs": operation_probs,
            "depth": torch.tensor(depth, dtype=torch.long),
            "operations": example.operations,  # Include the raw operations list
        }

    def generate_batch(
        self, batch_size: int
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        """Generate a batch of structure examples.

        Args:
            batch_size: Number of examples to generate

        Returns:
            Tuple of (text_list, batched_structure_tensors)
        """
        texts = []
        structures = []

        for _ in range(batch_size):
            # Use fixed depth - all examples in dataset should have the same depth
            # The identity function allows us to handle cases with effective depth < max_depth naturally
            depth = self.max_depth

            # Generate example
            text, structure = self.generate_structure_example(depth)
            texts.append(text)
            structures.append(structure)

        # Batch the structure tensors
        batched_structure = self._batch_structures(structures)

        return texts, batched_structure

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
        batched_operation_probs = torch.zeros(batch_size, max_depth, len(OP_NAMES))
        batched_depths = torch.zeros(batch_size, dtype=torch.long)

        # Fill batched tensors
        for i, structure in enumerate(structures):
            depth = structure["depth"].item()
            nodes = depth + 1

            # Copy initial values
            batched_initial_sgn[i, :nodes] = structure["initial_sgn"][:nodes]
            batched_initial_log[i, :nodes] = structure["initial_log"][:nodes]

            # Copy operation probabilities
            batched_operation_probs[i, :depth] = structure["operation_probs"][:depth]

            # Store depth
            batched_depths[i] = depth

        return {
            "initial_sgn": batched_initial_sgn,
            "initial_log": batched_initial_log,
            "operation_probs": batched_operation_probs,
            "depths": batched_depths,
        }

    def create_dataloader(
        self, batch_size: int = 32
    ) -> Iterator[Tuple[List[str], Dict[str, torch.Tensor]]]:
        """Create an infinite dataloader for structure examples.

        Args:
            batch_size: Batch size

        Yields:
            Batches of (texts, structure_tensors)
        """
        while True:
            texts, structures = self.generate_batch(batch_size)
            yield texts, structures


def create_dag_structure_dataloaders(
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    max_depth: int = 8,
    seed: int = 42,
    english_conversion_rate: float = 0.3,
    max_digits: int = 4,  # Maximum number of integer digits for uniform digit distribution
    max_decimal_places: int = None,  # Auto-derived from max_digits for uniform string distribution
) -> Tuple[Iterator, Iterator]:
    """Create train/val DAG structure dataloaders for predictor training.

    Args:
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        max_depth: DAG depth (all examples will have this depth)
        seed: Seed for training data
        english_conversion_rate: Probability of converting to English (0.0 to 1.0)
        max_digits: Maximum number of integer digits (1-4 means 1-digit to 4-digit integers)
        max_decimal_places: Maximum decimal places. If None, auto-derived as max_digits-1

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
    )

    val_dataset = DAGStructureDataset(
        max_depth=max_depth,
        seed=seed,
        english_conversion_probability=english_conversion_rate,
        max_digits=max_digits,
        max_decimal_places=max_decimal_places,
    )

    # Create dataloaders
    train_loader = train_dataset.create_dataloader(train_batch_size)
    val_loader = val_dataset.create_dataloader(val_batch_size)

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
    texts, structures = structure_dataset.generate_batch(2)
    print(f"Generated {len(texts)} structure examples")
    print(f"Structure keys: {list(structures.keys())}")
    print(f"Initial signs shape: {structures['initial_sgn'].shape}")
    print(f"Operation probs shape: {structures['operation_probs'].shape}")

    print("\n✅ Example completed successfully!")
    print("✅ Operations now processed consistently as stack (right-to-left)")
