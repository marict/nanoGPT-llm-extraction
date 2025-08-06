"""Tests for the NALM dataset integration."""

from pathlib import Path

import pytest
import torch

from data import prepare_dataset
from data.nalm.streaming import (
    NALMDataset,
    create_nalm_dataloaders,
    evaluate_nalm_model,
)


def test_nalm_dataset_creation():
    """Test that NALM dataset can be created."""
    dataset = NALMDataset(
        split="train",
        operations=["add", "sub"],
        train_range=(-5.0, 5.0),
        num_examples=100,
        seed=42,
    )

    assert len(dataset) == 100
    assert dataset.operations == ["add", "sub"]
    assert dataset.train_range == (-5.0, 5.0)


def test_nalm_dataset_item():
    """Test that NALM dataset returns correct item format."""
    dataset = NALMDataset(
        split="train",
        operations=["add"],
        train_range=(-1.0, 1.0),
        num_examples=10,
        seed=42,
    )

    item = dataset[0]

    # Check required keys
    assert "tokens" in item
    assert "expression" in item
    assert "a" in item
    assert "b" in item
    assert "operation" in item
    assert "result" in item

    # Check data types
    assert isinstance(item["tokens"], torch.Tensor)
    assert isinstance(item["expression"], str)
    assert isinstance(item["a"], float)
    assert isinstance(item["b"], float)
    assert isinstance(item["operation"], str)
    assert isinstance(item["result"], float)

    # Check operation
    assert item["operation"] == "add"

    # Check expression format
    assert "+" in item["expression"]
    assert "=" in item["expression"]

    # Check result calculation
    expected_result = item["a"] + item["b"]
    assert abs(item["result"] - expected_result) < 1e-6


def test_nalm_dataloaders():
    """Test that NALM dataloaders can be created."""
    train_dataloader, val_dataloader = create_nalm_dataloaders(
        train_range=(-2.0, 2.0),
        val_range=(-2.0, 2.0),
        extrapolation_range=(-10.0, 10.0),
        operations=["add", "mul"],
        batch_size=4,
        train_examples=20,
        val_examples=10,
        seed=42,
    )

    # Check dataloaders
    assert train_dataloader is not None
    assert val_dataloader is not None

    # Check batch format
    train_batch = next(iter(train_dataloader))
    assert "tokens" in train_batch
    assert "expression" in train_batch
    assert "result" in train_batch

    # Check batch shapes
    assert train_batch["tokens"].shape[0] == 4  # batch_size
    assert len(train_batch["expression"]) == 4
    assert train_batch["result"].shape[0] == 4


def test_nalm_dataset_prepare():
    """Test that NALM dataset can be prepared through the main data module."""
    # Create temporary data directory
    temp_data_dir = Path("temp_data")
    temp_data_dir.mkdir(exist_ok=True)

    try:
        # Prepare NALM dataset
        train_tokens, val_tokens = prepare_dataset(
            dataset="nalm",
            data_dir=temp_data_dir,
            subset=1.0,
            force=True,
        )

        # Check that metadata was created
        meta_file = temp_data_dir / "nalm" / "meta.pkl"
        assert meta_file.exists()

        # Check token counts
        assert train_tokens > 0
        assert val_tokens > 0

    finally:
        # Clean up
        import shutil

        shutil.rmtree(temp_data_dir, ignore_errors=True)


def test_nalm_evaluation():
    """Test NALM evaluation function."""

    # Create a dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, tokens):
            batch_size = tokens.shape[0]
            return torch.randn(batch_size, 1)

    # Create dataloaders
    _, val_dataloader = create_nalm_dataloaders(
        batch_size=4,
        val_examples=20,
        seed=123,
    )

    # Create model
    model = DummyModel()

    # Test evaluation
    metrics = evaluate_nalm_model(model, val_dataloader, device="cpu")

    # Check metrics
    assert "accuracy" in metrics
    assert "mse" in metrics
    assert "total_examples" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["mse"] >= 0.0
    assert metrics["total_examples"] > 0


def test_nalm_operations():
    """Test all NALM operations."""
    operations = ["add", "sub", "mul", "div"]

    for op in operations:
        dataset = NALMDataset(
            split="train",
            operations=[op],
            train_range=(-2.0, 2.0),
            num_examples=10,
            seed=42,
        )

        item = dataset[0]
        assert item["operation"] == op

        # Test result calculation
        a, b = item["a"], item["b"]
        if op == "add":
            expected = a + b
        elif op == "sub":
            expected = a - b
        elif op == "mul":
            expected = a * b
        elif op == "div":
            expected = a / b

        assert abs(item["result"] - expected) < 1e-6


def test_nalm_extrapolation():
    """Test that validation uses extrapolation range."""
    train_dataset = NALMDataset(
        split="train",
        operations=["add"],
        train_range=(-1.0, 1.0),
        extrapolation_range=(-10.0, 10.0),
        num_examples=100,
        seed=42,
    )

    val_dataset = NALMDataset(
        split="val",
        operations=["add"],
        train_range=(-1.0, 1.0),
        extrapolation_range=(-10.0, 10.0),
        num_examples=100,
        seed=42,
    )

    # Check that validation examples use extrapolation range
    train_values = []
    val_values = []

    for i in range(50):
        train_item = train_dataset[i]
        val_item = val_dataset[i]

        train_values.extend([train_item["a"], train_item["b"]])
        val_values.extend([val_item["a"], val_item["b"]])

    # Training values should be within train_range
    assert all(-1.0 <= v <= 1.0 for v in train_values)

    # Validation values should be within extrapolation_range
    assert all(-10.0 <= v <= 10.0 for v in val_values)

    # Some validation values should be outside train_range
    assert any(abs(v) > 1.0 for v in val_values)
