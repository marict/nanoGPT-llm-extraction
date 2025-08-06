#!/usr/bin/env python3
"""Test script for the NALM dataset."""

import torch
from streaming import create_nalm_dataloaders, evaluate_nalm_model


def test_nalm_dataset():
    """Test the NALM dataset creation and basic functionality."""
    print("🧪 Testing NALM dataset...")

    # Create dataloaders
    train_dataloader, val_dataloader = create_nalm_dataloaders(
        train_range=(-5.0, 5.0),
        val_range=(-5.0, 5.0),
        extrapolation_range=(-50.0, 50.0),
        operations=["add", "sub", "mul", "div"],
        batch_size=4,
        train_examples=100,
        val_examples=20,
        seed=42,
    )

    print(
        f"✅ Created dataloaders - Train: {len(train_dataloader)} batches, Val: {len(val_dataloader)} batches"
    )

    # Test a few examples
    print("\n📝 Sample training examples:")
    for i, batch in enumerate(train_dataloader):
        if i >= 3:  # Show first 3 batches
            break

        print(f"Batch {i + 1}:")
        for j in range(
            min(2, len(batch["expression"]))
        ):  # Show first 2 examples per batch
            expr = batch["expression"][j]
            result = batch["result"][j].item()
            print(f"  {expr} (result: {result:.6f})")
        print()

    print("📝 Sample validation examples (extrapolation):")
    for i, batch in enumerate(val_dataloader):
        if i >= 2:  # Show first 2 batches
            break

        print(f"Batch {i + 1}:")
        for j in range(
            min(2, len(batch["expression"]))
        ):  # Show first 2 examples per batch
            expr = batch["expression"][j]
            result = batch["result"][j].item()
            print(f"  {expr} (result: {result:.6f})")
        print()

    # Test tokenization
    print("🔤 Testing tokenization:")
    sample_batch = next(iter(train_dataloader))
    tokens = sample_batch["tokens"][0]
    expression = sample_batch["expression"][0]
    print(f"Expression: {expression}")
    print(f"Tokens: {tokens.tolist()}")
    print(f"Token count: {len(tokens)}")

    print("\n✅ NALM dataset test completed successfully!")


def test_nalm_evaluation():
    """Test the NALM evaluation function with a dummy model."""
    print("\n🧪 Testing NALM evaluation...")

    # Create a dummy model that just returns the target values
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, tokens):
            # This is a dummy model that doesn't actually process tokens
            # In a real scenario, you'd have a proper model here
            batch_size = tokens.shape[0]
            # Return random values for testing
            return torch.randn(batch_size, 1)

    # Create dataloaders
    _, val_dataloader = create_nalm_dataloaders(
        batch_size=8,
        val_examples=50,
        seed=123,
    )

    # Create dummy model
    model = DummyModel()

    # Test evaluation
    metrics = evaluate_nalm_model(model, val_dataloader, device="cpu")

    print(f"📊 Evaluation metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  Total examples: {metrics['total_examples']}")

    print("\n✅ NALM evaluation test completed successfully!")


if __name__ == "__main__":
    test_nalm_dataset()
    test_nalm_evaluation()
