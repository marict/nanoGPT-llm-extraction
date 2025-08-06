#!/usr/bin/env python3
"""Example script for benchmarking DAG-GPT vs standard GPT on NALM dataset."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from data.nalm.streaming import create_nalm_dataloaders, evaluate_nalm_model
from models import GPT, GPTConfig


def create_standard_gpt():
    """Create a standard GPT model for comparison."""
    config = GPTConfig(
        model_type="gpt_mini",
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=64,
        bias=False,
        dropout=0.0,
    )
    return GPT(config)


def create_dag_gpt():
    """Create a DAG-GPT model."""
    config = GPTConfig(
        model_type="gpt_mini",
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=64,
        bias=False,
        dropout=0.0,
        use_dag=True,
        dag_depth=3,
        max_digits=6,
        max_decimal_places=3,
    )
    return GPT(config)


def train_model(model, train_dataloader, val_dataloader, device, epochs=5):
    """Train a model on the NALM dataset."""
    print(f"🎯 Training {model.__class__.__name__}...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= 50:  # Limit batches per epoch for demo
                break

            tokens = batch["tokens"].to(device)
            targets = batch["result"].to(device)

            # Forward pass
            logits = model(tokens)

            # Simplified loss calculation (predict result from last token)
            result_logits = logits[:, -1, :]
            target_tokens = (
                torch.round(targets * 1000).long().clamp(0, model.config.vocab_size - 1)
            )
            loss = F.cross_entropy(result_logits, target_tokens)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"  Epoch {epoch + 1}/{epochs}: Avg loss = {avg_loss:.4f}")

    return model


def benchmark_models():
    """Benchmark standard GPT vs DAG-GPT on NALM dataset."""
    print("🚀 Starting NALM Benchmark...")

    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device}")

    # Create NALM dataloaders
    print("📊 Creating NALM dataloaders...")
    train_dataloader, val_dataloader = create_nalm_dataloaders(
        train_range=(-5.0, 5.0),
        val_range=(-5.0, 5.0),
        extrapolation_range=(-20.0, 20.0),
        operations=["add", "sub", "mul", "div"],
        batch_size=16,
        train_examples=1000,
        val_examples=200,
        seed=42,
    )

    # Create models
    print("🤖 Creating models...")
    standard_gpt = create_standard_gpt().to(device)
    dag_gpt = create_dag_gpt().to(device)

    print(
        f"📈 Standard GPT parameters: {sum(p.numel() for p in standard_gpt.parameters()):,}"
    )
    print(f"📈 DAG-GPT parameters: {sum(p.numel() for p in dag_gpt.parameters()):,}")

    # Train models
    print("\n" + "=" * 50)
    print("TRAINING STANDARD GPT")
    print("=" * 50)
    standard_gpt = train_model(
        standard_gpt, train_dataloader, val_dataloader, device, epochs=3
    )

    print("\n" + "=" * 50)
    print("TRAINING DAG-GPT")
    print("=" * 50)
    dag_gpt = train_model(dag_gpt, train_dataloader, val_dataloader, device, epochs=3)

    # Evaluate models
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    # Standard GPT evaluation
    print("\n📊 Standard GPT Results:")
    standard_metrics = evaluate_nalm_model(standard_gpt, val_dataloader, device)
    print(f"  Accuracy: {standard_metrics['accuracy']:.4f}")
    print(f"  MSE: {standard_metrics['mse']:.6f}")
    print(f"  Total examples: {standard_metrics['total_examples']}")

    # DAG-GPT evaluation
    print("\n📊 DAG-GPT Results:")
    dag_metrics = evaluate_nalm_model(dag_gpt, val_dataloader, device)
    print(f"  Accuracy: {dag_metrics['accuracy']:.4f}")
    print(f"  MSE: {dag_metrics['mse']:.6f}")
    print(f"  Total examples: {dag_metrics['total_examples']}")

    # Comparison
    print("\n📊 Comparison:")
    accuracy_improvement = dag_metrics["accuracy"] - standard_metrics["accuracy"]
    mse_improvement = standard_metrics["mse"] - dag_metrics["mse"]

    print(f"  Accuracy improvement: {accuracy_improvement:+.4f}")
    print(f"  MSE improvement: {mse_improvement:+.6f}")

    if accuracy_improvement > 0:
        print("  ✅ DAG-GPT performs better on accuracy!")
    else:
        print("  ❌ Standard GPT performs better on accuracy")

    if mse_improvement > 0:
        print("  ✅ DAG-GPT performs better on MSE!")
    else:
        print("  ❌ Standard GPT performs better on MSE")

    print("\n✅ Benchmark completed!")


def test_extrapolation():
    """Test extrapolation performance specifically."""
    print("\n🔍 Testing Extrapolation Performance...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataloaders with different ranges
    train_dataloader, _ = create_nalm_dataloaders(
        train_range=(-2.0, 2.0),  # Train on small range
        val_range=(-2.0, 2.0),
        extrapolation_range=(-10.0, 10.0),  # Test on larger range
        operations=["add", "mul"],
        batch_size=16,
        train_examples=500,
        val_examples=100,
        seed=42,
    )

    extrapolation_dataloader, _ = create_nalm_dataloaders(
        train_range=(-2.0, 2.0),
        val_range=(-2.0, 2.0),
        extrapolation_range=(-10.0, 10.0),
        operations=["add", "mul"],
        batch_size=16,
        train_examples=500,
        val_examples=100,
        seed=123,  # Different seed for extrapolation
    )

    # Create and train models
    standard_gpt = create_standard_gpt().to(device)
    dag_gpt = create_dag_gpt().to(device)

    # Quick training
    standard_gpt = train_model(
        standard_gpt, train_dataloader, extrapolation_dataloader, device, epochs=2
    )
    dag_gpt = train_model(
        dag_gpt, train_dataloader, extrapolation_dataloader, device, epochs=2
    )

    # Evaluate on extrapolation
    print("\n📊 Extrapolation Results:")

    standard_extrapolation = evaluate_nalm_model(
        standard_gpt, extrapolation_dataloader, device
    )
    dag_extrapolation = evaluate_nalm_model(dag_gpt, extrapolation_dataloader, device)

    print(
        f"Standard GPT extrapolation accuracy: {standard_extrapolation['accuracy']:.4f}"
    )
    print(f"DAG-GPT extrapolation accuracy: {dag_extrapolation['accuracy']:.4f}")

    extrapolation_improvement = (
        dag_extrapolation["accuracy"] - standard_extrapolation["accuracy"]
    )
    print(f"Extrapolation improvement: {extrapolation_improvement:+.4f}")


if __name__ == "__main__":
    benchmark_models()
    test_extrapolation()
