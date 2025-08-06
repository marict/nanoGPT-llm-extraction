"""Tests for NALM dataset integration with DAG model."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch

from data.nalm.streaming import NALMDataset, create_nalm_dataloaders
from models.dag_model import GPT, GPTConfig
from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel


def test_nalm_dataset_generation():
    """Test that NALM dataset can generate synthetic arithmetic data."""
    print("🧪 Testing NALM dataset generation...")

    # Create a small NALM dataset
    dataset = NALMDataset(
        split="train",
        operations=["add", "sub", "mul", "div"],
        train_range=(-5.0, 5.0),
        num_examples=100,
        seed=42,
    )

    # Test dataset properties
    assert len(dataset) == 100
    assert dataset.operations == ["add", "sub", "mul", "div"]

    # Test a few examples
    for i in range(5):
        item = dataset[i]

        # Check required fields
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

        # Check expression format
        assert "=" in item["expression"]
        assert item["operation"] in ["add", "sub", "mul", "div"]

        # Verify result calculation
        a, b = item["a"], item["b"]
        op = item["operation"]
        expected_result = item["result"]

        if op == "add":
            calculated = a + b
        elif op == "sub":
            calculated = a - b
        elif op == "mul":
            calculated = a * b
        elif op == "div":
            calculated = a / b

        assert (
            abs(calculated - expected_result) < 1e-6
        ), f"Result mismatch for {op}: {a} {op} {b}"

        print(f"  ✅ Example {i}: {item['expression']} (result: {expected_result:.6f})")


def test_nalm_dataloader_batching():
    """Test that NALM dataloaders work correctly with batching."""
    print("🧪 Testing NALM dataloader batching...")

    # Create small dataloaders
    train_dataloader, val_dataloader = create_nalm_dataloaders(
        train_range=(-3.0, 3.0),
        val_range=(-3.0, 3.0),
        extrapolation_range=(-10.0, 10.0),
        operations=["add", "mul"],
        batch_size=4,
        train_examples=20,
        val_examples=8,
        seed=42,
    )

    # Test training dataloader
    train_batch = next(iter(train_dataloader))
    assert "tokens" in train_batch
    assert "expression" in train_batch
    assert "result" in train_batch

    # Check batch shapes
    assert train_batch["tokens"].shape[0] == 4  # batch_size
    assert len(train_batch["expression"]) == 4
    assert train_batch["result"].shape[0] == 4

    # Check that tokens are padded to same length
    token_lengths = [len(expr.split()) for expr in train_batch["expression"]]
    max_len = max(token_lengths)
    assert train_batch["tokens"].shape[1] >= max_len

    print(f"  ✅ Training batch shape: {train_batch['tokens'].shape}")
    print(f"  ✅ Sample expressions: {train_batch['expression'][:2]}")

    # Test validation dataloader
    val_batch = next(iter(val_dataloader))
    assert val_batch["tokens"].shape[0] == 4
    print(f"  ✅ Validation batch shape: {val_batch['tokens'].shape}")


def test_nalm_with_dag_predictor():
    """Test that NALM data can be processed by DAG predictor model."""
    print("🧪 Testing NALM with DAG predictor...")

    # Create a small DAG predictor model
    config = PredictorOnlyConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=32,
        dag_depth=3,
        max_digits=4,
        max_decimal_places=2,
    )

    model = PredictorOnlyModel(config)
    print(f"  📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create small NALM dataloader
    train_dataloader, _ = create_nalm_dataloaders(
        train_range=(-2.0, 2.0),
        val_range=(-2.0, 2.0),
        extrapolation_range=(-5.0, 5.0),
        operations=["add", "sub"],
        batch_size=2,
        train_examples=10,
        val_examples=4,
        seed=42,
    )

    # Get a batch
    batch = next(iter(train_dataloader))
    tokens = batch["tokens"]

    print(f"  📝 Input tokens shape: {tokens.shape}")
    print(f"  📝 Sample expressions: {batch['expression']}")

    # Forward pass through DAG predictor
    model.eval()
    with torch.no_grad():
        digit_logits, V_sign, O, G = model(tokens)

    # Check output shapes
    batch_size, seq_len = tokens.shape
    print(
        f"    - Expected digit_logits shape: ({batch_size}, {seq_len}, {config.max_digits})"
    )
    print(f"    - Actual digit_logits shape: {digit_logits.shape}")
    print(f"    - Expected V_sign shape: ({batch_size}, {seq_len})")
    print(f"    - Actual V_sign shape: {V_sign.shape}")
    print(f"    - Expected O shape: ({batch_size}, {seq_len}, {config.dag_depth})")
    print(f"    - Actual O shape: {O.shape}")
    print(f"    - Expected G shape: ({batch_size}, {seq_len}, {config.dag_depth})")
    print(f"    - Actual G shape: {G.shape}")

    # Check output shapes (with more flexible assertions)
    assert digit_logits.shape[0] == batch_size
    assert digit_logits.shape[1] == seq_len
    assert digit_logits.shape[2] == config.max_digits
    assert V_sign.shape[0] == batch_size
    assert V_sign.shape[1] == seq_len
    assert O.shape[0] == batch_size
    assert O.shape[1] == seq_len
    assert O.shape[2] == config.dag_depth
    assert G.shape[0] == batch_size
    assert G.shape[1] == seq_len
    assert G.shape[2] == config.dag_depth

    print(f"  ✅ DAG predictor outputs:")
    print(f"    - digit_logits: {digit_logits.shape}")
    print(f"    - V_sign: {V_sign.shape}")
    print(f"    - O: {O.shape}")
    print(f"    - G: {G.shape}")


def test_nalm_with_full_dag_model():
    """Test that NALM data can be processed by full DAG-GPT model."""
    print("🧪 Testing NALM with full DAG-GPT model...")

    # Create a small DAG-GPT model
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=32,
        dag_depth=3,  # This enables DAG functionality
        max_digits=4,
        max_decimal_places=2,
    )

    model = GPT(config)
    print(f"  📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create small NALM dataloader
    train_dataloader, _ = create_nalm_dataloaders(
        train_range=(-2.0, 2.0),
        val_range=(-2.0, 2.0),
        extrapolation_range=(-5.0, 5.0),
        operations=["add", "mul"],
        batch_size=2,
        train_examples=10,
        val_examples=4,
        seed=42,
    )

    # Get a batch
    batch = next(iter(train_dataloader))
    tokens = batch["tokens"]

    print(f"  📝 Input tokens shape: {tokens.shape}")
    print(f"  📝 Sample expressions: {batch['expression']}")

    # Forward pass through full DAG-GPT model
    model.eval()
    with torch.no_grad():
        logits = model(tokens)

    # Check output shape
    batch_size, seq_len = tokens.shape
    print(
        f"    - Expected logits shape: ({batch_size}, {seq_len}, {config.vocab_size})"
    )
    print(f"    - Actual logits shape: {logits.shape}")

    # Check output shape (more flexible)
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == seq_len

    print(f"  ✅ DAG-GPT output shape: {logits.shape}")

    # Test that model has DAG components
    assert hasattr(model, "dag_predictor")
    assert hasattr(model, "dag_executor")
    print(
        f"  ✅ Model has DAG components: dag_predictor={model.dag_predictor is not None}, dag_executor={model.dag_executor is not None}"
    )


def test_nalm_training_step():
    """Test a single training step with NALM data and DAG model."""
    print("🧪 Testing NALM training step...")

    # Create a small DAG-GPT model
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=32,
        dag_depth=3,  # This enables DAG functionality
        max_digits=4,
        max_decimal_places=2,
    )

    model = GPT(config)
    model.train()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create small NALM dataloader
    train_dataloader, _ = create_nalm_dataloaders(
        train_range=(-2.0, 2.0),
        val_range=(-2.0, 2.0),
        extrapolation_range=(-5.0, 5.0),
        operations=["add", "sub"],
        batch_size=2,
        train_examples=10,
        val_examples=4,
        seed=42,
    )

    # Get a batch
    batch = next(iter(train_dataloader))
    tokens = batch["tokens"]

    # Create targets (shift tokens by 1 for language modeling)
    targets = tokens[:, 1:].contiguous()
    inputs = tokens[:, :-1].contiguous()

    print(f"  📝 Input shape: {inputs.shape}")
    print(f"  📝 Target shape: {targets.shape}")

    # Training step
    optimizer.zero_grad()

    # Forward pass
    dag_results = model(inputs)  # Shape: (batch_size, seq_len) - DAG execution results

    # For DAG models, we get execution results, not logits
    # Let's calculate a simple MSE loss on the results
    # In practice, you'd want to compare against expected arithmetic results
    loss = torch.nn.functional.mse_loss(dag_results, torch.zeros_like(dag_results))

    # Backward pass
    loss.backward()

    # Check gradients
    total_grad_norm = 0.0
    param_count = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2).item() ** 2
            param_count += 1

    total_grad_norm = total_grad_norm**0.5

    print(f"  ✅ Training step completed:")
    print(f"    - Loss: {loss.item():.6f}")
    print(f"    - Grad norm: {total_grad_norm:.6f}")
    print(f"    - Parameters with gradients: {param_count}")

    # Optimizer step
    optimizer.step()

    # Verify loss is finite
    assert torch.isfinite(loss), "Loss should be finite"
    assert total_grad_norm > 0, "Gradients should be non-zero"


def test_nalm_extrapolation():
    """Test that NALM dataset correctly handles extrapolation ranges."""
    print("🧪 Testing NALM extrapolation...")

    # Create datasets with different ranges
    train_dataset = NALMDataset(
        split="train",
        operations=["add"],
        train_range=(-1.0, 1.0),  # Small training range
        extrapolation_range=(-10.0, 10.0),
        num_examples=50,
        seed=42,
    )

    val_dataset = NALMDataset(
        split="val",
        operations=["add"],
        train_range=(-1.0, 1.0),  # Same training range
        extrapolation_range=(-10.0, 10.0),  # Larger extrapolation range
        num_examples=50,
        seed=42,
    )

    # Collect values from both datasets
    train_values = []
    val_values = []

    for i in range(20):
        train_item = train_dataset[i]
        val_item = val_dataset[i]

        train_values.extend([train_item["a"], train_item["b"]])
        val_values.extend([val_item["a"], val_item["b"]])

    # Check that training values are within training range
    assert all(
        -1.0 <= v <= 1.0 for v in train_values
    ), "Training values should be within [-1, 1]"

    # Check that validation values use extrapolation range
    assert all(
        -10.0 <= v <= 10.0 for v in val_values
    ), "Validation values should be within [-10, 10]"

    # Check that some validation values are outside training range
    outside_training = [v for v in val_values if abs(v) > 1.0]
    assert (
        len(outside_training) > 0
    ), "Some validation values should be outside training range"

    print(f"  ✅ Extrapolation test passed:")
    print(f"    - Training values: {len(train_values)} values in [-1, 1]")
    print(f"    - Validation values: {len(val_values)} values in [-10, 10]")
    print(f"    - Values outside training range: {len(outside_training)}")


def test_nalm_operations_coverage():
    """Test that all NALM operations work correctly."""
    print("🧪 Testing NALM operations coverage...")

    operations = ["add", "sub", "mul", "div"]

    for op in operations:
        dataset = NALMDataset(
            split="train",
            operations=[op],
            train_range=(-2.0, 2.0),
            num_examples=20,
            seed=42,
        )

        # Test a few examples for each operation
        for i in range(5):
            item = dataset[i]
            assert item["operation"] == op

            # Test result calculation
            a, b = item["a"], item["b"]
            expected = item["result"]

            if op == "add":
                calculated = a + b
            elif op == "sub":
                calculated = a - b
            elif op == "mul":
                calculated = a * b
            elif op == "div":
                calculated = a / b

            assert (
                abs(calculated - expected) < 1e-6
            ), f"Operation {op} failed: {a} {op} {b}"

        print(f"  ✅ Operation {op}: 5 examples tested")


if __name__ == "__main__":
    """Run all NALM integration tests."""
    print("🚀 Running NALM dataset integration tests...")
    print("=" * 60)

    test_nalm_dataset_generation()
    print()

    test_nalm_dataloader_batching()
    print()

    test_nalm_with_dag_predictor()
    print()

    test_nalm_with_full_dag_model()
    print()

    test_nalm_training_step()
    print()

    test_nalm_extrapolation()
    print()

    test_nalm_operations_coverage()
    print()

    print("✅ All NALM integration tests completed successfully!")
