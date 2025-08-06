"""Streaming NALM dataset implementation.

This module implements the Single Module Arithmetic Task dataset from the NALM benchmark.
It generates synthetic arithmetic expressions for training and evaluation.
"""

import random
from typing import Dict, List, Optional, Tuple

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class NALMDataset(Dataset):
    """Streaming dataset for NALM arithmetic tasks.

    Generates synthetic arithmetic expressions on-the-fly for training.
    Supports addition, subtraction, multiplication, and division operations.
    """

    def __init__(
        self,
        split: str = "train",
        operations: List[str] = ["add", "sub", "mul", "div"],
        train_range: Tuple[float, float] = (-10.0, 10.0),
        val_range: Tuple[float, float] = (-10.0, 10.0),
        extrapolation_range: Tuple[float, float] = (-100.0, 100.0),
        num_examples: int = 10000,
        seed: int = 42,
    ):
        """Initialize the NALM dataset.

        Args:
            split: Dataset split ("train" or "val")
            operations: List of operations to include ("add", "sub", "mul", "div")
            train_range: Range for training data generation
            val_range: Range for validation data generation
            extrapolation_range: Range for extrapolation testing
            num_examples: Number of examples to generate (for length calculation)
            seed: Random seed for reproducibility
        """
        self.split = split
        self.operations = operations
        self.train_range = train_range
        self.val_range = val_range
        self.extrapolation_range = extrapolation_range
        self.num_examples = num_examples
        self.seed = seed

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # Set random seed
        random.seed(seed)

        # Operation symbols
        self.op_symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/"}

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return self.num_examples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a single arithmetic example.

        Args:
            idx: Index of the example to generate

        Returns:
            Dictionary containing the example data
        """
        # Set seed for this specific example
        random.seed(self.seed + idx)

        # Choose operation
        op = random.choice(self.operations)
        op_symbol = self.op_symbols[op]

        # Generate operands based on split
        if self.split == "train":
            a = random.uniform(*self.train_range)
            b = random.uniform(*self.train_range)
        else:  # val
            # For validation, use extrapolation range to test generalization
            a = random.uniform(*self.extrapolation_range)
            b = random.uniform(*self.extrapolation_range)

        # Handle division by zero
        if op == "div" and abs(b) < 1e-8:
            b = random.uniform(0.1, 10.0) if b >= 0 else random.uniform(-10.0, -0.1)

        # Calculate result
        if op == "add":
            result = a + b
        elif op == "sub":
            result = a - b
        elif op == "mul":
            result = a * b
        elif op == "div":
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {op}")

        # Format expression as text
        expression = f"{a:.6f} {op_symbol} {b:.6f} = {result:.6f}"

        # Tokenize
        tokens = self.tokenizer.encode(expression)

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "expression": expression,
            "a": a,
            "b": b,
            "operation": op,
            "result": result,
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    # Separate different types of data
    tokens_list = [item["tokens"] for item in batch]
    expressions = [item["expression"] for item in batch]
    a_values = [item["a"] for item in batch]
    b_values = [item["b"] for item in batch]
    operations = [item["operation"] for item in batch]
    results = [item["result"] for item in batch]

    # Pad tokens to the same length
    max_len = max(len(tokens) for tokens in tokens_list)
    padded_tokens = []

    for tokens in tokens_list:
        # Pad with GPT-2 end-of-text token (50256)
        padding_length = max_len - len(tokens)
        padded = torch.cat(
            [tokens, torch.full((padding_length,), 50256, dtype=tokens.dtype)]
        )
        padded_tokens.append(padded)

    # Stack tensors
    tokens_tensor = torch.stack(padded_tokens)

    return {
        "tokens": tokens_tensor,
        "expression": expressions,
        "a": torch.tensor(a_values, dtype=torch.float32),
        "b": torch.tensor(b_values, dtype=torch.float32),
        "operation": operations,
        "result": torch.tensor(results, dtype=torch.float32),
    }


def create_nalm_dataloaders(
    train_range: Tuple[float, float] = (-10.0, 10.0),
    val_range: Tuple[float, float] = (-10.0, 10.0),
    extrapolation_range: Tuple[float, float] = (-100.0, 100.0),
    operations: List[str] = ["add", "sub", "mul", "div"],
    batch_size: int = 32,
    num_workers: int = 0,
    train_examples: int = 10000,
    val_examples: int = 1000,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create NALM dataloaders for training and validation.

    Args:
        train_range: Range for training data generation
        val_range: Range for validation data generation
        extrapolation_range: Range for extrapolation testing
        operations: List of operations to include
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_examples: Number of training examples
        val_examples: Number of validation examples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = NALMDataset(
        split="train",
        operations=operations,
        train_range=train_range,
        val_range=val_range,
        extrapolation_range=extrapolation_range,
        num_examples=train_examples,
        seed=seed,
    )

    val_dataset = NALMDataset(
        split="val",
        operations=operations,
        train_range=train_range,
        val_range=val_range,
        extrapolation_range=extrapolation_range,
        num_examples=val_examples,
        seed=seed + 1,  # Different seed for validation
    )

    # Create dataloaders with custom collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_dataloader, val_dataloader


def evaluate_nalm_model(
    model,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate a model on the NALM dataset.

    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        device: Device to run evaluation on

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_examples = 0
    correct_predictions = 0
    total_mse = 0.0

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            targets = batch["result"].to(device)

            # Forward pass
            outputs = model(tokens)

            # Extract predicted results (assuming model outputs the result directly)
            # This is a simplified evaluation - you may need to adapt based on your model's output format
            predicted_results = outputs.squeeze(-1)

            # Calculate metrics
            mse = torch.nn.functional.mse_loss(predicted_results, targets)
            total_mse += mse.item() * len(targets)

            # For accuracy, we'll use a tolerance
            tolerance = 1e-3
            correct = torch.abs(predicted_results - targets) < tolerance
            correct_predictions += correct.sum().item()

            total_examples += len(targets)

    accuracy = correct_predictions / total_examples if total_examples > 0 else 0.0
    avg_mse = total_mse / total_examples if total_examples > 0 else float("inf")

    return {
        "accuracy": accuracy,
        "mse": avg_mse,
        "total_examples": total_examples,
    }
