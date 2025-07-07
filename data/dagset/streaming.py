#!/usr/bin/env python
"""
streaming.py
On-the-fly DAG dataset generation for training.
"""

import random
import re
import sys
from collections import deque  # fast FIFO buffer for tokens
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from num2words import num2words
from tiktoken import get_encoding

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from dag_model import (LOG_LIM, add_log_space, divide_log_space,
                       identity_log_space, multiply_log_space, op_names,
                       subtract_log_space)


def convert_number_to_words(number: float, use_words: bool = True) -> str:
    """Convert a number to its English word equivalent.

    Args:
        number: The number to convert
        use_words: Whether to convert to words or keep as digits

    Returns:
        String representation (either words or original format)
    """
    if not use_words:
        return str(number)

    # Handle special cases for common decimal values
    if abs(number) < 0.001:
        return "zero"

    # For integers, use num2words directly
    if number == int(number):
        return num2words(int(number))

    # For decimals, convert the parts separately
    if "." in str(number):
        parts = str(number).split(".")
        integer_part = int(parts[0]) if parts[0] else 0
        decimal_part = parts[1]

        result = num2words(integer_part) if integer_part != 0 else "zero"
        result += " point"

        # Convert each decimal digit to words
        for digit in decimal_part[:6]:  # Limit to 6 decimal places
            result += " " + num2words(int(digit))

        return result

    return str(number)


def convert_math_expression_to_english(
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
    tokens = re.findall(r"\d+\.?\d*|\+|\-|\*|/|\(|\)", expression)

    converted_tokens = []
    for token in tokens:
        if token in symbol_to_english:
            # Randomly convert operator to English based on probability
            if rng.random() < conversion_probability:
                converted_tokens.append(rng.choice(symbol_to_english[token]))
            else:
                converted_tokens.append(token)
        elif token in ["(", ")"]:
            # Randomly convert parentheses to words
            if (
                token == "(" and rng.random() < conversion_probability * 0.5
            ):  # Lower probability for parentheses
                converted_tokens.append("open parenthesis")
            elif token == ")" and rng.random() < conversion_probability * 0.5:
                converted_tokens.append("close parenthesis")
            else:
                converted_tokens.append(token)
        elif re.match(r"\d+\.?\d*", token):
            # Randomly convert number to words based on probability
            number = float(token)
            if rng.random() < conversion_probability:
                converted_tokens.append(convert_number_to_words(number, True))
            else:
                converted_tokens.append(token)
        else:
            # Keep token as is
            converted_tokens.append(token)

    return " ".join(converted_tokens)


def convert_dag_text_to_english(
    text: str, conversion_probability: float = 0.3, rng: random.Random = None
) -> str:
    """Convert numbers and operators in DAG text to English words per-token.

    Args:
        text: Original text (now simple math expressions)
        conversion_probability: Probability of converting each individual token (0.0 to 1.0)
        rng: Random number generator

    Returns:
        Text with mixed English/numeric tokens
    """
    if rng is None:
        rng = random

    # For simple mathematical expressions, apply per-token conversion
    return convert_math_expression_to_english(text, conversion_probability, rng)


@dataclass
class DAGExample:
    """Lightweight container for a DAG computation example."""

    text: str
    depth: int
    initial_values: list[tuple[float, float]]
    operations: list[tuple[int, int, str]]


def generate_random_initial_value(
    value_range: tuple[float, float] = (-10.0, 10.0), rng: random.Random = None
) -> tuple[float, float]:
    """Generate a random initial value as (sign, log_magnitude).

    Args:
        value_range: Range for the actual values (not log magnitudes)
        rng: Random number generator to use

    Returns:
        (sign, log_magnitude) where sign is in {-1, 1} and log_magnitude is in [0, LOG_LIM]
    """
    if rng is None:
        rng = random

    # Generate a random actual value in the range, avoiding values too close to zero
    while True:
        actual_value = rng.uniform(value_range[0], value_range[1])
        if abs(actual_value) >= 0.1:  # Ensure we don't get values too close to zero
            break

    # Convert to sign and log magnitude
    sign = 1.0 if actual_value >= 0 else -1.0
    log_magnitude = min(max(np.log(abs(actual_value)), 0.0), LOG_LIM)

    return sign, log_magnitude


def generate_random_dag_plan(
    depth: int, num_initial_values: int = 1, rng: random.Random = None
) -> list[tuple[int, int, str]]:
    """Generate a random DAG execution plan with hard-maxed selections.

    Args:
        depth: Number of computation steps
        num_initial_values: Number of initial values to start with
        rng: Random number generator to use

    Returns:
        List of (operand1_idx, operand2_idx, operation_name) tuples
    """
    if rng is None:
        rng = random

    operations = []
    num_available_values = num_initial_values

    for step in range(depth):
        # Randomly select an operation first
        operation_name = rng.choice(op_names)

        if operation_name == "identity" or num_available_values == 1:
            # For identity operations or when only one value is available,
            # use the same operand twice
            operand1_idx = rng.randint(0, num_available_values - 1)
            operand2_idx = operand1_idx
        else:
            # Select two different operands using random.sample
            operand1_idx, operand2_idx = rng.sample(range(num_available_values), 2)

        operations.append((operand1_idx, operand2_idx, operation_name))

        # After each step, we have one more value available
        num_available_values += 1

    return operations


def execute_dag_computation(
    initial_values: list[tuple[float, float]], operations: list[tuple[int, int, str]]
) -> list[tuple[float, float]]:
    """Execute a DAG computation and return all intermediate values.

    Args:
        initial_values: List of (sign, log_magnitude) initial values
        operations: List of (operand1_idx, operand2_idx, operation_name) operations

    Returns:
        List of all values (initial + computed), each as (sign, log_magnitude)
    """
    # Convert to tensors for computation
    values = []
    for sign, log_mag in initial_values:
        values.append((torch.tensor(sign), torch.tensor(log_mag)))

    # Operation function mapping
    op_func_map = {
        "add": add_log_space,
        "subtract": subtract_log_space,
        "multiply": multiply_log_space,
        "divide": divide_log_space,
        "identity": identity_log_space,
    }

    # Execute each operation
    for operand1_idx, operand2_idx, operation_name in operations:
        # Get operands
        sign1, log1 = values[operand1_idx]
        sign2, log2 = values[operand2_idx]

        # Execute operation
        op_func = op_func_map[operation_name]
        result_sign, result_log = op_func(sign1, log1, sign2, log2, ignore_clip=True)

        # Store result
        values.append((result_sign, result_log))

    # Convert back to Python numbers
    result_values = []
    for sign, log_mag in values:
        result_values.append((float(sign.item()), float(log_mag.item())))

    return result_values


def convert_dag_to_expression_string(
    initial_values: list[tuple[float, float]],
    operations: list[tuple[int, int, str]],
    use_parentheses: bool = True,
    rng: random.Random = None,
    convert_to_english: bool = False,
    conversion_probability: float = 0.3,
) -> str:
    """Convert DAG structure to a simple mathematical expression string following stack-based execution.

    The DAG executes as follows:
    1. Start with initial values on a stack
    2. For each operation, use the specified operand indices
    3. Apply operation with operand1 as first operand, operand2 as second operand
    4. Push result back onto stack
    5. For identity operations, keep first operand and discard second operand

    Args:
        initial_values: List of (sign, log_magnitude) initial values
        operations: List of (operand1_idx, operand2_idx, operation_name) operations
        use_parentheses: Whether to add parentheses for clarity
        rng: Random number generator
        convert_to_english: Whether to potentially convert numbers/operators to English
        conversion_probability: Probability of converting to English (0.0 to 1.0)

    Returns:
        Simple mathematical expression string like "1 * (2 - 3/4)"
    """
    if rng is None:
        rng = random

    # Convert initial values back to regular numbers and round for display
    values = []  # List of (value, expression) tuples
    for sign, log_mag in initial_values:
        if log_mag == 0.0:
            number = 0.0 if sign == 0.0 else sign * 1.0
        else:
            number = sign * np.exp(log_mag)
        # Round to 3 decimal places for consistent display
        number = round(number, 3)
        values.append((number, f"{number:.3f}"))

    if len(operations) == 0:
        # Just a single number
        result = values[0][1] if values else "1.0"
    else:
        # Operation symbols for display
        op_symbols = {
            "add": "+",
            "subtract": "-",
            "multiply": "*",
            "divide": "/",
            "identity": None,  # Special case - no operator needed
        }

        # Process each operation in sequence
        for op_idx, (operand1_idx, operand2_idx, operation_name) in enumerate(
            operations
        ):
            # Get the operands - note that operand indices refer to positions in the values list
            val1, expr1 = values[operand1_idx]
            val2, expr2 = values[operand2_idx]

            if operation_name == "identity":
                # For identity, just keep the first operand
                result_val = val1
                result_expr = expr1
            else:
                symbol = op_symbols[operation_name]

                # Compute the actual result
                if operation_name == "add":
                    result_val = val1 + val2
                elif operation_name == "subtract":
                    result_val = val1 - val2
                elif operation_name == "multiply":
                    result_val = val1 * val2
                else:  # divide
                    result_val = val1 / val2

                # Round the result consistently
                result_val = round(result_val, 3)

                # Add parentheses based on operation precedence
                if operation_name in ["multiply", "divide"]:
                    # Higher precedence operations - check if operands need parentheses
                    if "+" in expr1 or "-" in expr1:
                        expr1 = f"({expr1})"
                    if "+" in expr2 or "-" in expr2:
                        expr2 = f"({expr2})"
                elif operation_name in ["subtract", "divide"]:
                    # Non-associative operations - always parenthesize second operand if it has operators
                    if any(op in expr2 for op in ["+", "-", "*", "/"]):
                        expr2 = f"({expr2})"

                # Combine operands with operator
                result_expr = f"{expr1} {symbol} {expr2}"

                # Add parentheses around the entire expression if needed
                if op_idx < len(operations) - 1:
                    next_op = operations[op_idx + 1][2]
                    if next_op in ["multiply", "divide"] and operation_name in [
                        "add",
                        "subtract",
                    ]:
                        result_expr = f"({result_expr})"

            # Add result to values list - this becomes available for next operation
            values.append((result_val, result_expr))

        # Return the final expression
        result = values[-1][1]

    # Apply English conversion if requested
    if convert_to_english:
        result = convert_dag_text_to_english(result, conversion_probability, rng)

    return result


def generate_single_dag_example(
    depth: int,
    num_initial_values: int = None,
    value_range: tuple[float, float] = (0.1, 100.0),
    rng: random.Random = None,
    convert_to_english: bool = False,
    conversion_probability: float = 0.3,
) -> DAGExample:
    """Generate a single DAG computation example as a simple math expression.

    Args:
        depth: Depth of the DAG computation
        num_initial_values: Number of initial values
        value_range: Range for values in the expression
        rng: Random number generator to use

    Returns:
        DAG computation example with simple expression format
    """
    if rng is None:
        rng = random

    # Determine number of initial values to match DAG predictor expectations
    if num_initial_values is None:
        # For DAG with depth n, we need n+1 initial values
        num_initial_values = depth + 1

    # Step 1: Generate initial values using existing logic
    initial_values = []
    for _ in range(num_initial_values):
        initial_values.append(
            generate_random_initial_value(value_range=value_range, rng=rng)
        )

    # Step 2: Generate DAG operations using existing logic
    operations = generate_random_dag_plan(depth, num_initial_values, rng)

    # Step 3: Convert DAG structure to simple expression string
    expression = convert_dag_to_expression_string(
        initial_values=initial_values,
        operations=operations,
        use_parentheses=True,
        rng=rng,
        convert_to_english=convert_to_english,
        conversion_probability=conversion_probability,
    )

    return DAGExample(
        text=expression,
        depth=depth,
        initial_values=initial_values,
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
) -> list[DAGExample]:
    """Generate a dataset of DAG computation examples (structure-only).

    Args:
        num_examples: Number of examples to generate
        max_depth: DAG depth (all examples will have this depth)
        num_initial_values: Number of initial values per example
        value_range: Range for initial values
        rng: Random number generator to use

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
        convert_to_english: bool = False,
        english_conversion_probability: float = 0.3,
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

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = StreamingDAGDataset(
        max_depth=max_depth,
        seed=train_seed,
        convert_to_english=convert_to_english,
        english_conversion_probability=english_conversion_probability,
    )

    val_dataset = StreamingDAGDataset(
        max_depth=max_depth,
        seed=val_seed,
        convert_to_english=convert_to_english,
        english_conversion_probability=english_conversion_probability,
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

        # Initialize tokenizer
        self.enc = get_encoding(tokenizer)

        # Create random state
        self.random_state = random.Random(seed)

        # Operation name to index mapping
        self.op_name_to_idx = {name: i for i, name in enumerate(op_names)}
        self.op_idx_to_name = {i: name for i, name in enumerate(op_names)}

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

        # Fill initial values (typically just one for slot 0)
        for i, (sign, log_mag) in enumerate(example.initial_values):
            if i < num_scratch_nodes:
                initial_sgn[i] = sign
                initial_log[i] = log_mag

        # Create operation probabilities (one-hot for ground truth)
        operation_probs = torch.zeros(depth, len(op_names))

        for step, (operand1_idx, operand2_idx, operation_name) in enumerate(
            example.operations
        ):
            if step < depth:
                op_idx = self.op_name_to_idx[operation_name]
                operation_probs[step, op_idx] = 1.0

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
        batched_operation_probs = torch.zeros(batch_size, max_depth, len(op_names))
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
    convert_to_english: bool = False,
    english_conversion_probability: float = 0.3,
) -> Tuple[Iterator, Iterator]:
    """Create train and validation structure dataloaders.

    Args:
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        max_depth: DAG depth (all examples will have this exact depth)
        train_seed: Seed for training data
        val_seed: Seed for validation data
        convert_to_english: Whether to potentially convert numbers/operators to English
        english_conversion_probability: Probability of converting to English (0.0 to 1.0)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets with fixed depth
    train_dataset = DAGStructureDataset(
        max_depth=max_depth,
        seed=train_seed,
        convert_to_english=convert_to_english,
        english_conversion_probability=english_conversion_probability,
    )

    val_dataset = DAGStructureDataset(
        max_depth=max_depth,
        seed=val_seed,
        convert_to_english=convert_to_english,
        english_conversion_probability=english_conversion_probability,
    )

    # Create dataloaders
    train_loader = train_dataset.create_dataloader(train_batch_size)
    val_loader = val_dataset.create_dataloader(val_batch_size)

    return train_loader, val_loader


if __name__ == "__main__":
    # Simple example usage
    print("DAG Streaming Dataset Example")
    print("=" * 40)

    # Create a dataset
    dataset = StreamingDAGDataset(max_depth=3, seed=42)

    # Generate a small batch
    tokens, text = dataset.generate_batch(3)
    print(f"Generated {len(tokens)} tokens from 3 examples")

    # Show examples
    print("\nGenerated examples:")
    for i, example in enumerate(text.split("\n---\n")):
        print(f"Example {i+1}: {example}")

    print("\n✅ Example completed successfully!")
