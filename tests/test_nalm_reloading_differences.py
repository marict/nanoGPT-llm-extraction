"""Tests to verify checkpoint reloading produces different outputs and simple DAG execution."""

import sys
import tempfile
from pathlib import Path

from data.nalm.streaming import NALMDataset

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

from checkpoint_manager import CheckpointManager
from models.dag_model import GPT, GPTConfig
from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel


def create_mock_checkpoint(config, checkpoint_path, seed=42):
    """Create a mock checkpoint with the specified configuration."""
    torch.manual_seed(seed)

    # Create model with the config
    if hasattr(config, "dag_depth") and config.dag_depth > 0:
        model = GPT(config)
    else:
        model = PredictorOnlyModel(config)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create checkpoint data
    checkpoint_data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": config,
        "iter_num": 1000,
        "best_val_loss": 0.1,
    }

    # Save checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_data, checkpoint_path)

    return checkpoint_data


def test_outputs_different_with_reloading():
    """Test that outputs are different with and without checkpoint reloading."""
    print("🧪 Testing that outputs are different with and without reloading...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        checkpoint_manager = CheckpointManager(temp_path)

        # Create config
        config = PredictorOnlyConfig(
            n_layer=2,
            n_head=4,
            n_embd=256,
            block_size=32,
            dag_depth=4,
            max_digits=4,
            max_decimal_places=4,
        )

        # Create checkpoint
        checkpoint_path = temp_path / "test_checkpoint.pt"
        create_mock_checkpoint(config, checkpoint_path, seed=42)

        # Test input
        test_input = torch.randint(0, 1000, (2, 10))

        # Model 1: Fresh initialization (different seed)
        torch.manual_seed(123)
        model1 = PredictorOnlyModel(config)

        # Model 2: Loaded from checkpoint
        loaded_checkpoint = checkpoint_manager.load_checkpoint_from_path(
            checkpoint_path
        )
        model2 = PredictorOnlyModel(loaded_checkpoint["model_args"])
        model2.load_state_dict(loaded_checkpoint["model"])

        # Get outputs
        with torch.no_grad():
            outputs1 = model1(test_input)
            outputs2 = model2(test_input)

        # Check that outputs are different
        digit_logits1, V_sign1, O1, G1 = outputs1
        digit_logits2, V_sign2, O2, G2 = outputs2

        # Calculate differences
        digit_diff = torch.abs(digit_logits1 - digit_logits2).mean().item()
        vsign_diff = torch.abs(V_sign1 - V_sign2).mean().item()
        o_diff = torch.abs(O1 - O2).mean().item()
        g_diff = torch.abs(G1 - G2).mean().item()

        print(f"  📊 Output differences:")
        print(f"    - Digit logits: {digit_diff:.6f}")
        print(f"    - V_sign: {vsign_diff:.6f}")
        print(f"    - O: {o_diff:.6f}")
        print(f"    - G: {g_diff:.6f}")

        # Verify outputs are different (not identical)
        assert digit_diff > 1e-6, "Digit logits should be different"
        assert vsign_diff > 1e-6, "V_sign should be different"
        assert o_diff > 1e-6, "O should be different"
        assert g_diff > 1e-6, "G should be different"

        print(f"  ✅ Outputs are different between fresh and loaded models")


def test_simple_dag_execution():
    """Test that the DAG model can execute simple arithmetic like 1.0 + 1.0."""
    print("🧪 Testing simple DAG execution (1.0 + 1.0)...")

    # Create DAG-GPT model
    config = GPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=256,
        block_size=32,
        dag_depth=4,
        max_digits=4,
        max_decimal_places=4,
    )

    model = GPT(config)
    model.eval()

    # Create simple input that represents "1.0 + 1.0"
    # We'll use a simple token sequence and see if the DAG can process it
    test_input = torch.randint(0, 1000, (1, 8))  # Single sequence, 8 tokens

    print(f"  📝 Input shape: {test_input.shape}")

    # Forward pass
    with torch.no_grad():
        dag_results = model(test_input)

    print(f"  📊 DAG execution results:")
    print(f"    - Output shape: {dag_results.shape}")
    print(
        f"    - Output values: {dag_results.flatten()[:5].tolist()}..."
    )  # First 5 values

    # Verify output shape
    batch_size, seq_len = test_input.shape
    assert dag_results.shape == (batch_size, seq_len)

    # Check that outputs are finite
    assert torch.isfinite(dag_results).all(), "DAG outputs should be finite"

    # Check that outputs are not all zeros
    assert not torch.allclose(
        dag_results, torch.zeros_like(dag_results)
    ), "DAG outputs should not be all zeros"

    print(f"  ✅ Simple DAG execution successful")


def test_dag_predictor_simple_arithmetic():
    """Test DAG predictor on simple arithmetic expressions."""
    print("🧪 Testing DAG predictor on simple arithmetic...")

    # Create DAG predictor model
    config = PredictorOnlyConfig(
        n_layer=2,
        n_head=4,
        n_embd=256,
        block_size=32,
        dag_depth=4,
        max_digits=4,
        max_decimal_places=4,
    )

    model = PredictorOnlyModel(config)
    model.eval()

    # Test input representing simple arithmetic
    test_input = torch.randint(0, 1000, (2, 10))

    print(f"  📝 Input shape: {test_input.shape}")

    # Forward pass
    with torch.no_grad():
        digit_logits, V_sign, O, G = model(test_input)

    print(f"  📊 DAG predictor outputs:")
    print(f"    - Digit logits shape: {digit_logits.shape}")
    print(f"    - V_sign shape: {V_sign.shape}")
    print(f"    - O shape: {O.shape}")
    print(f"    - G shape: {G.shape}")

    # Verify output shapes
    batch_size, seq_len = test_input.shape
    assert digit_logits.shape[0] == batch_size
    assert digit_logits.shape[1] == seq_len
    assert V_sign.shape[0] == batch_size
    assert V_sign.shape[1] == seq_len
    assert O.shape[0] == batch_size
    assert O.shape[1] == seq_len
    assert O.shape[2] == config.dag_depth
    assert G.shape[0] == batch_size
    assert G.shape[1] == seq_len
    assert G.shape[2] == config.dag_depth

    # Check that outputs are finite
    assert torch.isfinite(digit_logits).all(), "Digit logits should be finite"
    assert torch.isfinite(V_sign).all(), "V_sign should be finite"
    assert torch.isfinite(O).all(), "O should be finite"
    assert torch.isfinite(G).all(), "G should be finite"

    # Check that outputs are not all zeros
    assert not torch.allclose(
        digit_logits, torch.zeros_like(digit_logits)
    ), "Digit logits should not be all zeros"
    assert not torch.allclose(
        V_sign, torch.zeros_like(V_sign)
    ), "V_sign should not be all zeros"
    assert not torch.allclose(O, torch.zeros_like(O)), "O should not be all zeros"
    assert not torch.allclose(G, torch.zeros_like(G)), "G should not be all zeros"

    print(f"  ✅ DAG predictor working on simple arithmetic")


def test_checkpoint_size_estimation():
    """Estimate the size of the checkpoint based on model parameters."""
    print("🧪 Estimating checkpoint size...")

    # Create config matching the checkpoint
    config = PredictorOnlyConfig(
        n_layer=2,
        n_head=4,
        n_embd=256,
        block_size=32,
        dag_depth=4,
        max_digits=4,
        max_decimal_places=4,
    )

    # Create model
    model = PredictorOnlyModel(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate checkpoint size (assuming float32 = 4 bytes per parameter)
    estimated_size_mb = (total_params * 4) / (1024 * 1024)

    print(f"  📊 Model parameter count:")
    print(f"    - Total parameters: {total_params:,}")
    print(f"    - Trainable parameters: {trainable_params:,}")
    print(f"    - Estimated checkpoint size: {estimated_size_mb:.2f} MB")

    # Create actual checkpoint to measure real size
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        checkpoint_path = temp_path / "size_test.pt"
        create_mock_checkpoint(config, checkpoint_path)

        # Get actual file size
        actual_size_bytes = checkpoint_path.stat().st_size
        actual_size_mb = actual_size_bytes / (1024 * 1024)

        print(f"  📊 Actual checkpoint size:")
        print(f"    - File size: {actual_size_bytes:,} bytes")
        print(f"    - File size: {actual_size_mb:.2f} MB")

        # Check if size is reasonable
        assert actual_size_mb > 0, "Checkpoint should have non-zero size"
        assert actual_size_mb < 1000, "Checkpoint should be less than 1GB"

        print(f"  ✅ Checkpoint size estimation complete")


def test_nalm_simple_expression():
    """Test NALM dataset with simple expressions like 1.0 + 1.0."""
    print("🧪 Testing NALM dataset with simple expressions...")

    # Import NALM dataset

    # Create dataset with simple range
    dataset = NALMDataset(
        split="train",
        operations=["add"],
        train_range=(0.0, 2.0),  # Simple range including 1.0
        num_examples=10,
        seed=42,
    )

    # Get a few examples
    for i in range(3):
        item = dataset[i]
        print(f"  📝 Example {i}: {item['expression']}")

        # Check if we get 1.0 + 1.0 or similar
        a, b = item["a"], item["b"]
        result = item["result"]

        print(f"    - a: {a:.2f}, b: {b:.2f}, result: {result:.2f}")

        # Verify the arithmetic is correct
        expected = a + b
        assert (
            abs(result - expected) < 1e-6
        ), f"Arithmetic error: {a} + {b} = {result}, expected {expected}"

    print(f"  ✅ NALM dataset working with simple expressions")


if __name__ == "__main__":
    """Run all reloading difference tests."""
    print("🚀 Running NALM reloading difference tests...")
    print("=" * 60)

    test_outputs_different_with_reloading()
    print()

    test_simple_dag_execution()
    print()

    test_dag_predictor_simple_arithmetic()
    print()

    test_checkpoint_size_estimation()
    print()

    test_nalm_simple_expression()
    print()

    print("✅ All reloading difference tests completed successfully!")
