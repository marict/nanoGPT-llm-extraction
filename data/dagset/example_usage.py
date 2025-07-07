#!/usr/bin/env python
"""
example_usage.py
Example of how to use the streaming DAG dataset for training.
"""

import sys
import time
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from streaming import StreamingDAGDataset, create_dag_dataloaders


def example_simple_streaming():
    """Example 1: Simple streaming dataset usage."""
    print("=== Example 1: Simple Streaming Dataset ===")

    # Create a streaming dataset
    dataset = StreamingDAGDataset(
        max_depth=5,
        seed=42,
    )

    # Generate a batch of examples
    tokens, text = dataset.generate_batch(batch_size=10)
    print(f"Generated {len(tokens)} tokens from 10 examples")

    # Show first example
    examples = text.split("\n---\n")
    print(f"\nFirst example:")
    print(examples[0])

    # Generate specific number of tokens
    tokens = dataset.generate_tokens(5000)
    print(f"\nGenerated exactly {len(tokens)} tokens")


def example_dataloader_usage():
    """Example 2: Using the dataloader for training."""
    print("\n=== Example 2: DataLoader Usage ===")

    # Create train and validation loaders
    train_loader, val_loader = create_dag_dataloaders(
        train_examples_per_batch=500,  # Generate 500 examples per batch
        val_examples_per_batch=100,  # Generate 100 examples per batch
        batch_size=8,  # Training batch size
        block_size=512,  # Sequence length
        max_depth=4,
        train_seed=42,
        val_seed=43,
    )

    # Simulate training for a few batches
    print("Simulating training...")
    for i, (inputs, targets) in enumerate(train_loader):
        print(f"Training batch {i}: inputs {inputs.shape}, targets {targets.shape}")

        # Here you would do your forward pass, loss computation, etc.
        # loss = model(inputs, targets)
        # loss.backward()
        # optimizer.step()

        if i >= 3:  # Just test a few batches
            break

    # Test validation
    print("\nTesting validation...")
    for i, (inputs, targets) in enumerate(val_loader):
        print(f"Validation batch {i}: inputs {inputs.shape}, targets {targets.shape}")
        if i >= 1:  # Just test a couple
            break


def example_training_loop():
    """Example 3: Simple training loop with DAG data."""
    print("\n=== Example 3: Simple Training Loop ===")

    # Create a tiny model for demonstration
    class TinyModel(nn.Module):
        def __init__(self, vocab_size, embed_dim=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embed_dim, nhead=4, batch_first=True),
                num_layers=2,
            )
            self.lm_head = nn.Linear(embed_dim, vocab_size)

        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            return self.lm_head(x)

    # Create model and optimizer
    model = TinyModel(vocab_size=50257)  # GPT-2 vocab size
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Create dataloader
    train_loader, _ = create_dag_dataloaders(
        train_examples_per_batch=100,
        batch_size=4,
        block_size=128,
        max_depth=3,
        train_seed=42,
    )

    # Training loop
    model.train()
    print("Training tiny model on DAG data...")

    start_time = time.time()
    for step, (inputs, targets) in enumerate(train_loader):
        # Forward pass
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

        if step >= 50:  # Just a few steps for demo
            break

    duration = time.time() - start_time
    print(f"Completed 50 training steps in {duration:.2f}s")
    print(f"Average time per step: {duration/50:.3f}s")


def example_benchmarking():
    """Example 4: Benchmarking generation speed."""
    print("\n=== Example 4: Benchmarking ===")

    dataset = StreamingDAGDataset(max_depth=6, seed=42)

    # Test different batch sizes
    batch_sizes = [10, 100, 1000]
    for batch_size in batch_sizes:
        print(f"\nTesting batch size {batch_size}...")

        start_time = time.time()
        tokens, _ = dataset.generate_batch(batch_size)
        duration = time.time() - start_time

        print(f"  Generated {len(tokens)} tokens in {duration:.3f}s")
        print(f"  Rate: {len(tokens)/duration:.0f} tokens/sec")
        print(f"  Rate: {batch_size/duration:.0f} examples/sec")


if __name__ == "__main__":
    print("🚀 DAG Dataset Streaming Examples")
    print("=" * 50)

    # Run all examples
    example_simple_streaming()
    example_dataloader_usage()
    example_training_loop()
    example_benchmarking()

    print(f"\n✅ All examples completed successfully!")
    print("\nKey benefits of streaming DAG dataset:")
    print("  • Zero disk storage required")
    print("  • Infinite variety with different seeds")
    print("  • Fast generation (~5000 examples/sec)")
    print("  • Perfect reproducibility")
    print("  • Memory efficient")
    print("  • Drop-in replacement for file-based datasets")
