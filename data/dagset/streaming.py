#!/usr/bin/env python
"""
streaming.py
On-the-fly DAG dataset generation for training.
"""

import math
import random
import re
import sys
from collections import deque  # fast FIFO buffer for tokens
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import sympy
import torch
from num2words import num2words
from tiktoken import get_encoding

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.dag_model import OP_NAMES


def standardize_float_rounding(value: float, rng: random.Random = None) -> float:
    """Standardize float rounding to 0-5 decimal places.

    Args:
        value: The float value to round
        rng: Random number generator for choosing decimal places

    Returns:
        Float rounded to 0-5 decimal places (never zero)
    """
    if rng is None:
        rng = random

    # Choose number of decimal places (0-5)
    decimal_places = rng.randint(0, 5)

    # Round to the chosen number of decimal places
    rounded_value = round(value, decimal_places)

    # Ensure we never return exactly zero to avoid log(0) domain errors
    if rounded_value == 0.0:
        # Use a small positive value instead
        rounded_value = 0.01 if decimal_places >= 2 else 1

    # Convert to int if no decimal places to avoid .0 at the end
    if decimal_places == 0:
        return int(rounded_value)

    return rounded_value


def convert_number_to_words(number, use_words: bool = True) -> str:
    """Convert a number to its English word equivalent.

    Args:
        number: The number to convert (int or float)
        use_words: Whether to convert to words or keep as digits

    Returns:
        String representation (either words or original format)
    """
    if not use_words:
        return str(number)

    # Handle zero
    if number == 0 or (isinstance(number, float) and abs(number) < 0.00001):
        return "zero"

    # Handle negative numbers
    is_negative = number < 0
    abs_number = abs(number)

    # Handle integers (including those created by standardize_float_rounding)
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
            for digit in decimal_part[:5]:
                result += " " + num2words(int(digit))

            return f"negative {result}" if is_negative else result

    return str(number)


def add_english_to_expression(
    expression: str, conversion_probability: float = 0.3, rng: random.Random = None
) -> str:
    """Convert a mathematical expression to English words with per-token probability.

    Args:
        expression: Mathematical expression like "5.22 - 3.213 / 2.32"
        conversion_probability: Probability of converting each individual token
        rng: Random number generator

    Returns:
        Mixed English/numeric like "five point two two - three point two one three divided by 2.32"
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
            if i == 0 or expression[i - 1] in "(+-*/":
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
                converted_tokens.append(convert_number_to_words(number, True))
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


def generate_random_dag_plan(
    depth: int,
    num_initial_values: int = 1,
    value_range: tuple[float, float] = (-10.0, 10.0),
    rng: random.Random = None,
) -> tuple[list[float], list[str]]:
    if rng is None:
        rng = random
    # Generate random initial values with standardized rounding
    initial_values = [
        standardize_float_rounding(rng.uniform(value_range[0], value_range[1]), rng)
        for _ in range(num_initial_values)
    ]
    operations = [rng.choice(OP_NAMES) for _ in range(depth)]
    return initial_values, operations


def convert_dag_to_expression_string(
    initial_values: list[float],
    operations: list[str],
    rng: random.Random = None,
    convert_to_english: bool = True,
    conversion_probability: float = 0.3,
    permutation_probability: float = 0.5,
) -> str:
    """Convert DAG structure to a simple mathematical expression string following stack-based execution.

    Args:
        initial_values: List of initial values
        operations: List of operations
        rng: Random number generator
        convert_to_english: Whether to convert to English words
        conversion_probability: Probability of English conversion
        permutation_probability: Probability of applying permutation (0.0 = disabled)

    Returns:
        Simple mathematical expression string like "1 * (2 - 3/4)"
    """
    if rng is None:
        rng = random

    # Create symbols using absolute values to avoid double negatives
    # Then handle the sign by wrapping in negation if needed
    stack = []
    for v in initial_values:
        if v < 0:
            # Create symbol with positive value, then negate
            pos_symbol = sympy.Symbol(str(abs(v)))
            stack.append(-pos_symbol)
        else:
            stack.append(sympy.Symbol(str(v)))

    op_name_to_symbol = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/",
        "identity": "identity",
    }
    op_symbol_to_expression = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
        "identity": lambda a, b: a,  # Discard b
    }

    for op in operations:
        b = stack.pop()
        a = stack.pop()
        op_symbol = op_name_to_symbol[op]
        expr = op_symbol_to_expression[op_symbol](a, b)
        stack.append(expr)

    final_expr = stack[0]

    # Apply expression permutation for diversity if probability > 0
    if permutation_probability > 0.0 and rng.random() < permutation_probability:
        final_expr = permute_expression_randomly(final_expr, rng)

    result = str(final_expr)

    # Post-process to clean up any remaining double negatives
    # This is a safety net in case sympy still creates them
    result = result.replace("--", "+")
    result = result.replace("+-", "-")

    # Apply English conversion if requested
    if convert_to_english:
        result = add_english_to_expression(result, conversion_probability, rng)

    return result


def permute_expression_randomly(expr: sympy.Expr, rng: random.Random) -> sympy.Expr:
    """Create a mathematically equivalent but syntactically different expression.

    Args:
        expr: The sympy expression to permute
        rng: Random number generator

    Returns:
        Permuted but equivalent expression
    """
    # List of permutation strategies to try
    strategies = [
        "commute",  # Reorder commutative operations
        "associate",  # Change associative grouping
        "expand",  # Expand products/powers
        "factor",  # Factor expressions
        "simplify",  # Simplify expression
        "reorder_args",  # Reorder function arguments
    ]

    # Randomly select a strategy
    strategy = rng.choice(strategies)

    try:
        if strategy == "commute":
            return commute_operations(expr, rng)
        elif strategy == "associate":
            return change_associative_grouping(expr, rng)
        elif strategy == "expand":
            expanded = sympy.expand(expr)
            # Only use if it's different and not too complex
            if str(expanded) != str(expr) and len(str(expanded)) < len(str(expr)) * 2:
                return expanded
        elif strategy == "factor":
            factored = sympy.factor(expr)
            # Only use if it's different and not too complex
            if str(factored) != str(expr) and len(str(factored)) < len(str(expr)) * 2:
                return factored
        elif strategy == "simplify":
            simplified = sympy.simplify(expr)
            # Only use if it's different
            if str(simplified) != str(expr):
                return simplified
        elif strategy == "reorder_args":
            return reorder_args_recursively(expr, rng)
    except (sympy.SympifyError, AttributeError, TypeError):
        # If any strategy fails, just return the original
        pass

    # Fallback: return original expression
    return expr


def commute_operations(expr: sympy.Expr, rng: random.Random) -> sympy.Expr:
    """Randomly reorder commutative operations (+ and *)."""
    if expr.is_Add:
        # For addition, randomly shuffle the arguments
        args = list(expr.args)
        if len(args) > 1:
            rng.shuffle(args)
            return sympy.Add(*args)
    elif expr.is_Mul:
        # For multiplication, randomly shuffle the arguments
        args = list(expr.args)
        if len(args) > 1:
            rng.shuffle(args)
            return sympy.Mul(*args)
    elif hasattr(expr, "args") and len(expr.args) > 0:
        # Recursively apply to sub-expressions
        new_args = [commute_operations(arg, rng) for arg in expr.args]
        return expr.func(*new_args)

    return expr


def change_associative_grouping(expr: sympy.Expr, rng: random.Random) -> sympy.Expr:
    """Change the associative grouping of operations."""
    if expr.is_Add and len(expr.args) >= 3:
        # For a + b + c, randomly group as (a + b) + c or a + (b + c)
        args = list(expr.args)
        if rng.random() < 0.5:
            # Group first two: (a + b) + rest
            first_group = args[0] + args[1]
            if len(args) > 2:
                return first_group + sympy.Add(*args[2:])
            else:
                return first_group
        else:
            # Group last two: first + (b + c)
            if len(args) > 2:
                last_group = sympy.Add(*args[1:])
                return args[0] + last_group
    elif expr.is_Mul and len(expr.args) >= 3:
        # Similar for multiplication
        args = list(expr.args)
        if rng.random() < 0.5:
            first_group = args[0] * args[1]
            if len(args) > 2:
                return first_group * sympy.Mul(*args[2:])
            else:
                return first_group
        else:
            if len(args) > 2:
                last_group = sympy.Mul(*args[1:])
                return args[0] * last_group
    elif hasattr(expr, "args") and len(expr.args) > 0:
        # Recursively apply to sub-expressions
        new_args = [change_associative_grouping(arg, rng) for arg in expr.args]
        return expr.func(*new_args)

    return expr


def reorder_args_recursively(expr: sympy.Expr, rng: random.Random) -> sympy.Expr:
    """Recursively reorder arguments in commutative operations."""
    if hasattr(expr, "args") and len(expr.args) > 1:
        # Recursively process arguments first
        new_args = [reorder_args_recursively(arg, rng) for arg in expr.args]

        # If this is a commutative operation, randomly reorder
        if expr.is_Add or expr.is_Mul:
            if rng.random() < 0.5:  # 50% chance to reorder
                rng.shuffle(new_args)

        return expr.func(*new_args)

    return expr


def convert_plan_to_tensors(
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
    value_range: tuple[float, float] = (0.1, 100.0),
    rng: random.Random = None,
    convert_to_english: bool = False,
    conversion_probability: float = 0.3,
    permutation_probability: float = 0.5,
) -> DAGExample:
    """Generate a single DAG computation example as a simple math expression.

    Args:
        depth: Depth of the DAG computation
        num_initial_values: Number of initial values
        value_range: Range for values in the expression
        rng: Random number generator to use
        convert_to_english: Whether to convert to English words
        conversion_probability: Probability of English conversion
        permutation_probability: Probability of applying permutation (0.0 = disabled)

    Returns:
        DAG computation example with simple expression format
    """
    if rng is None:
        rng = random

    # Determine number of initial values to match DAG predictor expectations
    if num_initial_values is None:
        # For DAG with depth n, we need n+1 initial values
        num_initial_values = depth + 1

    # Step 1. Generate random dag plan
    initial_values, operations = generate_random_dag_plan(
        depth, num_initial_values, value_range, rng
    )

    # Step 2: Convert DAG plan to simple expression string for data
    expression = convert_dag_to_expression_string(
        initial_values=initial_values,
        operations=operations,
        rng=rng,
        convert_to_english=convert_to_english,
        conversion_probability=conversion_probability,
        permutation_probability=permutation_probability,
    )

    # Step 3: Convert dag plan to a tensor for labels
    signs, log_magnitudes, operations = convert_plan_to_tensors(
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
    value_range: tuple[float, float] = (0.1, 100.0),
    rng: random.Random = None,
    convert_to_english: bool = False,
    conversion_probability: float = 0.3,
    permutation_probability: float = 0.5,
) -> list[DAGExample]:
    """Generate a dataset of DAG computation examples (structure-only).

    Args:
        num_examples: Number of examples to generate
        max_depth: DAG depth (all examples will have this depth)
        num_initial_values: Number of initial values per example
        value_range: Range for initial values
        rng: Random number generator to use
        convert_to_english: Whether to convert to English words
        conversion_probability: Probability of English conversion
        permutation_probability: Probability of applying permutation (0.0 = disabled)

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
            value_range,
            rng,
            convert_to_english,
            conversion_probability,
            permutation_probability,
        )
        examples.append(example)

    return examples


class StreamingDAGDataset:
    """On-the-fly DAG dataset generator that produces infinite streams of examples."""

    def __init__(
        self,
        max_depth: int = 8,
        num_initial_values: int = None,
        value_range: tuple[float, float] = (0.1, 100.0),
        seed: int = 42,
        tokenizer: str = "gpt2",
        convert_to_english: bool = True,
        english_conversion_probability: float = 0.3,
        permutation_probability: float = 0.5,
    ):
        """Initialize the streaming DAG dataset.

        Args:
            max_depth: DAG depth (all examples will have this depth)
            num_initial_values: Number of initial values per example
            value_range: Range for initial values
            seed: Random seed for reproducibility
            tokenizer: Tokenizer to use (default: gpt2)
            convert_to_english: Whether to potentially convert numbers/operators to English
            english_conversion_probability: Probability of converting to English (0.0 to 1.0)
            permutation_probability: Probability of applying permutation (0.0 = disabled)
        """
        self.max_depth = max_depth
        # Set num_initial_values to match DAG predictor expectations
        self.num_initial_values = (
            num_initial_values if num_initial_values is not None else max_depth + 1
        )
        self.value_range = value_range
        self.seed = seed
        self.tokenizer = tokenizer
        self.convert_to_english = convert_to_english
        self.english_conversion_probability = english_conversion_probability
        self.permutation_probability = permutation_probability

        # Initialize tokenizer
        self.enc = get_encoding(tokenizer)

        # Create a dedicated random state for this dataset
        self.random_state = random.Random(seed)
        self.np_random_state = np.random.RandomState(seed)

    def generate_batch(self, batch_size: int) -> tuple[List[int], str]:
        """Generate a batch of examples and return tokens.

        Args:
            batch_size: Number of examples to generate

        Returns:
            Tuple of (tokens, raw_text)
        """
        # Generate examples using dedicated random state
        examples = generate_dag_dataset(
            num_examples=batch_size,
            max_depth=self.max_depth,
            num_initial_values=self.num_initial_values,
            value_range=self.value_range,
            rng=self.random_state,
            convert_to_english=self.convert_to_english,
            conversion_probability=self.english_conversion_probability,
            permutation_probability=self.permutation_probability,
        )

        # Convert to text
        all_text = []
        for example in examples:
            all_text.append(example.text)

        # Join with separators
        full_text = "\n---\n".join(all_text)

        # Tokenize
        tokens = self.enc.encode_ordinary(full_text)

        return tokens, full_text

    def generate_tokens(self, num_tokens: int) -> List[int]:
        """Generate approximately num_tokens tokens.

        Args:
            num_tokens: Target number of tokens

        Returns:
            List of tokens
        """
        all_tokens = []

        while len(all_tokens) < num_tokens:
            # Estimate examples needed (rough estimate: ~200 tokens per example)
            remaining = num_tokens - len(all_tokens)
            estimated_examples = max(10, remaining // 200)

            tokens, _ = self.generate_batch(estimated_examples)
            all_tokens.extend(tokens)

        # Truncate to requested length
        return all_tokens[:num_tokens]

    def stream_tokens(self, batch_size: int = 1000) -> Iterator[List[int]]:
        """Stream tokens in batches indefinitely.

        Args:
            batch_size: Number of examples per batch

        Yields:
            Batches of tokens
        """
        while True:
            tokens, _ = self.generate_batch(batch_size)
            yield tokens

    def get_train_val_split(
        self, train_examples: int, val_examples: int, split_seed: int = None
    ) -> tuple[List[int], List[int]]:
        """Generate train/val splits with different seeds.

        Args:
            train_examples: Number of training examples
            val_examples: Number of validation examples
            split_seed: Optional different seed for val split

        Returns:
            Tuple of (train_tokens, val_tokens)
        """
        # Generate train split
        original_seed = self.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        train_tokens, _ = self.generate_batch(train_examples)

        # Generate val split with different seed if specified
        if split_seed is not None:
            random.seed(split_seed)
            np.random.seed(split_seed)
            torch.manual_seed(split_seed)

        val_tokens, _ = self.generate_batch(val_examples)

        # Restore original seed
        random.seed(original_seed)
        np.random.seed(original_seed)
        torch.manual_seed(original_seed)

        return train_tokens, val_tokens


class DAGDataLoader:
    """DataLoader-like interface for streaming DAG data."""

    def __init__(
        self,
        dataset: StreamingDAGDataset,
        batch_size: int = 32,
        block_size: int = 1024,
        examples_per_batch: int = 100,
    ):
        """Initialize the data loader.

        Args:
            dataset: StreamingDAGDataset instance
            batch_size: Batch size for training
            block_size: Maximum sequence length
            examples_per_batch: Number of DAG examples to generate per batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.block_size = block_size
        self.examples_per_batch = examples_per_batch

        # Buffer for tokens (deque for fast left pops)
        self.token_buffer: deque[int] = deque()
        self.token_stream = dataset.stream_tokens(examples_per_batch)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the next batch of training data.

        Returns:
            Tuple of (inputs, targets) tensors
        """
        # Ensure the buffer has enough tokens for an entire batch
        required_tokens = self.batch_size * (self.block_size + 1)
        while len(self.token_buffer) < required_tokens:
            self.token_buffer.extend(next(self.token_stream))

        # Build batch using fast popleft operations (O(1) each)
        batch_data: List[List[int]] = []
        for _ in range(self.batch_size):
            seq = [self.token_buffer.popleft() for _ in range(self.block_size + 1)]
            batch_data.append(seq)

        # Convert to tensors
        batch_tensor = torch.tensor(batch_data, dtype=torch.long)

        # Split into inputs and targets
        inputs = batch_tensor[:, :-1]  # All but last token
        targets = batch_tensor[:, 1:]  # All but first token

        return inputs, targets


def create_dag_dataloaders(
    train_examples_per_batch: int = 1000,
    val_examples_per_batch: int = 100,
    batch_size: int = 32,
    block_size: int = 1024,
    max_depth: int = 8,
    train_seed: int = 42,
    val_seed: int = 43,
    convert_to_english: bool = False,
    english_conversion_probability: float = 0.3,
    permutation_probability: float = 0.5,
) -> tuple[DAGDataLoader, DAGDataLoader]:
    """Create train and validation data loaders.

    Args:
        train_examples_per_batch: DAG examples per training batch
        val_examples_per_batch: DAG examples per validation batch
        batch_size: Training batch size
        block_size: Maximum sequence length
        max_depth: DAG depth (all examples will have this depth)
        train_seed: Seed for training data
        val_seed: Seed for validation data
        convert_to_english: Whether to potentially convert numbers/operators to English
        english_conversion_probability: Probability of converting to English (0.0 to 1.0)
        permutation_probability: Probability of applying permutation (0.0 = disabled)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = StreamingDAGDataset(
        max_depth=max_depth,
        seed=train_seed,
        convert_to_english=convert_to_english,
        english_conversion_probability=english_conversion_probability,
        permutation_probability=permutation_probability,
    )

    val_dataset = StreamingDAGDataset(
        max_depth=max_depth,
        seed=val_seed,
        convert_to_english=convert_to_english,
        english_conversion_probability=english_conversion_probability,
        permutation_probability=permutation_probability,
    )

    # Create data loaders
    train_loader = DAGDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        block_size=block_size,
        examples_per_batch=train_examples_per_batch,
    )

    val_loader = DAGDataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        block_size=block_size,
        examples_per_batch=val_examples_per_batch,
    )

    return train_loader, val_loader


class DAGStructureDataset:
    """
    Dataset for pretraining DAG predictor on structure prediction.
    Maps text descriptions to DAG structure tensors.
    """

    def __init__(
        self,
        max_depth: int = 8,
        num_initial_values: int = None,
        value_range: tuple[float, float] = (0.1, 100.0),
        seed: int = 42,
        tokenizer: str = "gpt2",
        max_seq_length: int = 512,
        convert_to_english: bool = False,
        english_conversion_probability: float = 0.3,
        permutation_probability: float = 0.5,
    ):
        """Initialize the DAG structure dataset.

        Args:
            max_depth: DAG depth (all examples will have this depth)
            num_initial_values: Number of initial values per example
            value_range: Range for initial values
            seed: Random seed
            tokenizer: Tokenizer to use
            max_seq_length: Maximum sequence length for text inputs
            convert_to_english: Whether to potentially convert numbers/operators to English
            english_conversion_probability: Probability of converting to English (0.0 to 1.0)
            permutation_probability: Probability of applying permutation (0.0 = disabled)
        """
        self.max_depth = max_depth
        # Set num_initial_values to match DAG predictor expectations
        self.num_initial_values = (
            num_initial_values if num_initial_values is not None else max_depth + 1
        )
        self.value_range = value_range
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.convert_to_english = convert_to_english
        self.english_conversion_probability = english_conversion_probability
        self.permutation_probability = permutation_probability

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
        # Generate the DAG example (structure-only, no execution needed)
        example = generate_single_dag_example(
            depth=depth,
            num_initial_values=self.num_initial_values,
            value_range=self.value_range,
            rng=self.random_state,
            convert_to_english=self.convert_to_english,
            conversion_probability=self.english_conversion_probability,
            permutation_probability=self.permutation_probability,
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
    train_seed: int = 42,
    val_seed: int = 43,
    value_range: tuple[float, float] = (-100.0, 100.0),
    english_conversion_rate: float = 0.3,
    permutation_probability: float = 0.5,
) -> Tuple[Iterator, Iterator]:
    """Create train and validation structure dataloaders.

    Args:
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        max_depth: DAG depth (all examples will have this exact depth)
        train_seed: Seed for training data
        val_seed: Seed for validation data
        value_range: Range for initial values (allows negative values for meaningful sign prediction)
        english_conversion_rate: Probability of converting tokens to English (0.0 = disabled, 1.0 = always convert)
        permutation_probability: Probability of applying permutation (0.0 = disabled)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Use configurable english conversion rate
    convert_to_english = english_conversion_rate > 0.0

    # Create datasets with fixed depth
    train_dataset = DAGStructureDataset(
        max_depth=max_depth,
        seed=train_seed,
        value_range=value_range,
        convert_to_english=convert_to_english,
        english_conversion_probability=english_conversion_rate,
        permutation_probability=permutation_probability,
    )

    val_dataset = DAGStructureDataset(
        max_depth=max_depth,
        seed=val_seed,
        value_range=value_range,
        convert_to_english=convert_to_english,
        english_conversion_probability=english_conversion_rate,
        permutation_probability=permutation_probability,
    )

    # Create dataloaders
    train_loader = train_dataset.create_dataloader(train_batch_size)
    val_loader = val_dataset.create_dataloader(val_batch_size)

    return train_loader, val_loader


if __name__ == "__main__":
    # Simple example usage
    print("DAG Streaming Dataset Example")
    print("=" * 40)

    # Test the main streaming dataset (used in train.py)
    print("\nTesting StreamingDAGDataset:")
    dataset = StreamingDAGDataset(max_depth=3, seed=42)
    tokens, text = dataset.generate_batch(3)
    print(f"Generated {len(tokens)} tokens from 3 examples")
    print(f"Sample text:\n{text[:200]}...")

    # Test the structure dataset (used in train_predictor.py)
    print("\nTesting DAGStructureDataset:")
    structure_dataset = DAGStructureDataset(max_depth=3, seed=42)
    texts, structures = structure_dataset.generate_batch(2)
    print(f"Generated {len(texts)} structure examples")
    print(f"Structure keys: {list(structures.keys())}")
    print(f"Initial signs shape: {structures['initial_sgn'].shape}")
    print(f"Operation probs shape: {structures['operation_probs'].shape}")

    print("\n✅ Example completed successfully!")
