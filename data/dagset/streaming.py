#!/usr/bin/env python
"""
streaming.py
On-the-fly DAG dataset generation for training.

NOTE: Currently this is incomplete and not used until we have the DAG structure more stable.
"""

import random
import sys
from pathlib import Path
from typing import Iterator, List

import numpy as np
import torch
from tiktoken import get_encoding

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from dag_model import (LOG_LIM, add_log_space, divide_log_space,
                       identity_log_space, multiply_log_space, op_names,
                       subtract_log_space)


class DAGExample:
    """Represents a single DAG computation example."""

    def __init__(self, text: str, depth: int, initial_values: list, operations: list):
        self.text = text
        self.depth = depth
        self.initial_values = initial_values
        self.operations = operations


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
        # Randomly select two operands from available values
        operand1_idx = rng.randint(0, num_available_values - 1)
        operand2_idx = rng.randint(0, num_available_values - 1)

        # Randomly select an operation
        operation_name = rng.choice(op_names)

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


def generate_single_dag_example(
    depth: int,
    num_initial_values: int = 1,
    value_range: tuple[float, float] = (-10.0, 10.0),
    rng: random.Random = None,
) -> DAGExample:
    """Generate a single DAG computation example.

    Args:
        depth: Depth of the DAG computation
        num_initial_values: Number of initial values
        value_range: Range for initial values
        rng: Random number generator to use

    Returns:
        DAG computation example
    """
    if rng is None:
        rng = random

    # Generate initial values
    initial_values = []
    for _ in range(num_initial_values):
        initial_values.append(
            generate_random_initial_value(value_range=value_range, rng=rng)
        )

    # Generate DAG plan
    operations = generate_random_dag_plan(depth, num_initial_values, rng)

    # Execute the computation
    all_values = execute_dag_computation(initial_values, operations)

    # Format as text
    text_lines = []
    text_lines.append(f"DAG Computation (depth={depth}):")
    text_lines.append("")

    # Show initial values
    for i, (sign, log_mag) in enumerate(initial_values):
        actual_value = sign * np.exp(log_mag)
        text_lines.append(
            f"v{i} = {actual_value:.6f} (sign={sign:+.1f}, log_mag={log_mag:.6f})"
        )

    text_lines.append("")

    # Show computation steps
    for step, (operand1_idx, operand2_idx, operation_name) in enumerate(operations):
        result_idx = num_initial_values + step
        result_sign, result_log = all_values[result_idx]
        result_value = result_sign * np.exp(result_log)

        text_lines.append(
            f"Step {step + 1}: v{result_idx} = v{operand1_idx} {operation_name} v{operand2_idx}"
        )
        text_lines.append(
            f"  Result: {result_value:.6f} (sign={result_sign:+.1f}, log_mag={result_log:.6f})"
        )

    text_lines.append("")

    # Show final result
    final_idx = len(all_values) - 1
    final_sign, final_log = all_values[final_idx]
    final_value = final_sign * np.exp(final_log)
    text_lines.append(f"Final result: v{final_idx} = {final_value:.6f}")
    text_lines.append("")

    text = "\n".join(text_lines)

    return DAGExample(
        text=text, depth=depth, initial_values=initial_values, operations=operations
    )


def generate_dag_dataset(
    num_examples: int = 10000,
    max_depth: int = 8,
    min_depth: int = 1,
    num_initial_values: int = 1,
    value_range: tuple[float, float] = (-10.0, 10.0),
    rng: random.Random = None,
) -> list[DAGExample]:
    """Generate a dataset of DAG computation examples.

    Args:
        num_examples: Number of examples to generate
        max_depth: Maximum DAG depth
        min_depth: Minimum DAG depth
        num_initial_values: Number of initial values per example
        value_range: Range for initial values
        rng: Random number generator to use

    Returns:
        List of DAG examples
    """
    if rng is None:
        rng = random

    examples = []

    for _ in range(num_examples):
        # Choose random depth
        depth = rng.randint(min_depth, max_depth)

        # Generate example
        example = generate_single_dag_example(
            depth, num_initial_values, value_range, rng
        )
        examples.append(example)

    return examples


def format_dag_as_text(example: DAGExample) -> str:
    """Format a DAG computation as text for language modeling.

    Args:
        example: DAG computation example

    Returns:
        Formatted text representation
    """
    return example.text


class StreamingDAGDataset:
    """On-the-fly DAG dataset generator that produces infinite streams of examples."""

    def __init__(
        self,
        max_depth: int = 8,
        min_depth: int = 1,
        num_initial_values: int = 1,
        value_range: tuple[float, float] = (-10.0, 10.0),
        seed: int = 42,
        tokenizer: str = "gpt2",
    ):
        """Initialize the streaming DAG dataset.

        Args:
            max_depth: Maximum DAG depth
            min_depth: Minimum DAG depth
            num_initial_values: Number of initial values per example
            value_range: Range for initial values
            seed: Random seed for reproducibility
            tokenizer: Tokenizer to use (default: gpt2)
        """
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.num_initial_values = num_initial_values
        self.value_range = value_range
        self.seed = seed
        self.tokenizer = tokenizer

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
            min_depth=self.min_depth,
            num_initial_values=self.num_initial_values,
            value_range=self.value_range,
            rng=self.random_state,
        )

        # Convert to text
        all_text = []
        for example in examples:
            all_text.append(format_dag_as_text(example))

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

        # Buffer for tokens
        self.token_buffer = []
        self.token_stream = dataset.stream_tokens(examples_per_batch)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the next batch of training data.

        Returns:
            Tuple of (inputs, targets) tensors
        """
        # Ensure we have enough tokens
        while len(self.token_buffer) < self.batch_size * (self.block_size + 1):
            new_tokens = next(self.token_stream)
            self.token_buffer.extend(new_tokens)

        # Create batch
        batch_data = []
        for _ in range(self.batch_size):
            if len(self.token_buffer) >= self.block_size + 1:
                # Take a sequence of block_size + 1 tokens
                seq = self.token_buffer[: self.block_size + 1]
                self.token_buffer = self.token_buffer[self.block_size + 1 :]
                batch_data.append(seq)
            else:
                # Not enough tokens, refill buffer
                new_tokens = next(self.token_stream)
                self.token_buffer.extend(new_tokens)
                seq = self.token_buffer[: self.block_size + 1]
                self.token_buffer = self.token_buffer[self.block_size + 1 :]
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
    min_depth: int = 1,
    train_seed: int = 42,
    val_seed: int = 43,
) -> tuple[DAGDataLoader, DAGDataLoader]:
    """Create train and validation data loaders.

    Args:
        train_examples_per_batch: DAG examples per training batch
        val_examples_per_batch: DAG examples per validation batch
        batch_size: Training batch size
        block_size: Maximum sequence length
        max_depth: Maximum DAG depth
        min_depth: Minimum DAG depth
        train_seed: Seed for training data
        val_seed: Seed for validation data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = StreamingDAGDataset(
        max_depth=max_depth,
        min_depth=min_depth,
        seed=train_seed,
    )

    val_dataset = StreamingDAGDataset(
        max_depth=max_depth,
        min_depth=min_depth,
        seed=val_seed,
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


if __name__ == "__main__":
    # Quick test
    print("Testing streaming DAG dataset...")

    dataset = StreamingDAGDataset(max_depth=3, min_depth=1, seed=42)

    # Generate a small batch
    tokens, text = dataset.generate_batch(5)
    print(f"Generated {len(tokens)} tokens from 5 examples")

    # Show first example
    lines = text.split("\n")[:10]
    print("\nFirst few lines:")
    for line in lines:
        print(f"  {line}")

    print("\n✅ Streaming dataset test passed!")
