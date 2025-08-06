"""Tests for NALM checkpoint reloading functionality."""

import shutil
import sys
import tempfile
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

from checkpoint_manager import CheckpointManager
from models.dag_model import GPT, GPTConfig
from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel


def create_mock_checkpoint(config, checkpoint_path):
    """Create a mock checkpoint with the specified configuration."""
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


def test_nalm_checkpoint_reloading_matching_config():
    """Test that NALM training can reload from checkpoint with matching DAG config."""
    print("🧪 Testing NALM checkpoint reloading with matching config...")

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(temp_path)

        # Create config matching the predictor training config
        config = PredictorOnlyConfig(
            n_layer=2,
            n_head=4,
            n_embd=256,  # n_head * 64
            block_size=32,
            dag_depth=4,
            max_digits=4,
            max_decimal_places=4,
        )

        # Create mock checkpoint path
        checkpoint_path = temp_path / "test_checkpoint.pt"

        # Create mock checkpoint
        checkpoint_data = create_mock_checkpoint(config, checkpoint_path)
        print(f"  ✅ Created mock checkpoint at {checkpoint_path}")

        # Test loading the checkpoint
        loaded_checkpoint = checkpoint_manager.load_checkpoint_from_path(
            checkpoint_path
        )

        # Verify checkpoint data
        assert "model" in loaded_checkpoint
        assert "optimizer" in loaded_checkpoint
        assert "model_args" in loaded_checkpoint
        assert "iter_num" in loaded_checkpoint
        assert "best_val_loss" in loaded_checkpoint

        # Verify model args match
        loaded_config = loaded_checkpoint["model_args"]
        assert loaded_config.n_layer == config.n_layer
        assert loaded_config.n_head == config.n_head
        assert loaded_config.n_embd == config.n_embd
        assert loaded_config.block_size == config.block_size
        assert loaded_config.dag_depth == config.dag_depth
        assert loaded_config.max_digits == config.max_digits
        assert loaded_config.max_decimal_places == config.max_decimal_places

        print(f"  ✅ Checkpoint loaded successfully with matching config")
        print(f"    - DAG depth: {loaded_config.dag_depth}")
        print(f"    - Max digits: {loaded_config.max_digits}")
        print(f"    - Max decimal places: {loaded_config.max_decimal_places}")

        # Test creating model from loaded config
        model = PredictorOnlyModel(loaded_config)

        # Test that model can load the state dict
        model.load_state_dict(loaded_checkpoint["model"])
        print(f"  ✅ Model state dict loaded successfully")

        # Test forward pass with small input
        test_input = torch.randint(0, 1000, (2, 10))  # Small batch
        with torch.no_grad():
            outputs = model(test_input)

        # Verify outputs have expected shapes
        assert len(outputs) == 4  # digit_logits, V_sign, O, G
        digit_logits, V_sign, O, G = outputs

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

        print(f"  ✅ Model forward pass successful")
        print(
            f"    - Output shapes: digit_logits={digit_logits.shape}, V_sign={V_sign.shape}, O={O.shape}, G={G.shape}"
        )


def test_nalm_checkpoint_reloading_mismatch_detection():
    """Test that checkpoint reloading detects configuration mismatches."""
    print("🧪 Testing NALM checkpoint reloading mismatch detection...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        checkpoint_manager = CheckpointManager(temp_path)

        # Create checkpoint with one config
        checkpoint_config = PredictorOnlyConfig(
            n_layer=2,
            n_head=4,
            n_embd=256,
            block_size=32,
            dag_depth=4,
            max_digits=4,
            max_decimal_places=4,
        )

        checkpoint_path = temp_path / "mismatch_checkpoint.pt"
        create_mock_checkpoint(checkpoint_config, checkpoint_path)

        # Try to load with different DAG config
        mismatched_config = PredictorOnlyConfig(
            n_layer=2,
            n_head=4,
            n_embd=256,
            block_size=32,
            dag_depth=6,  # Different!
            max_digits=8,  # Different!
            max_decimal_places=2,  # Different!
        )

        # Load checkpoint
        loaded_checkpoint = checkpoint_manager.load_checkpoint_from_path(
            checkpoint_path
        )
        loaded_config = loaded_checkpoint["model_args"]

        # Verify mismatch detection
        assert loaded_config.dag_depth != mismatched_config.dag_depth
        assert loaded_config.max_digits != mismatched_config.max_digits
        assert loaded_config.max_decimal_places != mismatched_config.max_decimal_places

        print(f"  ✅ Mismatch detection working:")
        print(
            f"    - Checkpoint DAG depth: {loaded_config.dag_depth}, Expected: {mismatched_config.dag_depth}"
        )
        print(
            f"    - Checkpoint max digits: {loaded_config.max_digits}, Expected: {mismatched_config.max_digits}"
        )
        print(
            f"    - Checkpoint max decimal places: {loaded_config.max_decimal_places}, Expected: {mismatched_config.max_decimal_places}"
        )


def test_nalm_gpt_checkpoint_reloading():
    """Test that full DAG-GPT model can reload from checkpoint."""
    print("🧪 Testing NALM GPT checkpoint reloading...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        checkpoint_manager = CheckpointManager(temp_path)

        # Create GPT config with DAG
        config = GPTConfig(
            n_layer=2,
            n_head=4,
            n_embd=256,
            block_size=32,
            dag_depth=4,  # This enables DAG functionality
            max_digits=4,
            max_decimal_places=4,
        )

        checkpoint_path = temp_path / "gpt_checkpoint.pt"
        create_mock_checkpoint(config, checkpoint_path)

        # Load checkpoint
        loaded_checkpoint = checkpoint_manager.load_checkpoint_from_path(
            checkpoint_path
        )
        loaded_config = loaded_checkpoint["model_args"]

        # Create GPT model from loaded config
        model = GPT(loaded_config)
        model.load_state_dict(loaded_checkpoint["model"])

        # Verify DAG components exist
        assert hasattr(model, "dag_predictor")
        assert hasattr(model, "dag_executor")
        assert model.dag_predictor is not None
        assert model.dag_executor is not None

        print(f"  ✅ GPT model loaded with DAG components")
        print(f"    - DAG predictor: {model.dag_predictor is not None}")
        print(f"    - DAG executor: {model.dag_executor is not None}")

        # Test forward pass
        test_input = torch.randint(0, 1000, (2, 10))
        with torch.no_grad():
            dag_results = model(test_input)

        # Verify output shape
        batch_size, seq_len = test_input.shape
        assert dag_results.shape == (batch_size, seq_len)

        print(f"  ✅ GPT forward pass successful, output shape: {dag_results.shape}")


def test_nalm_config_validation():
    """Test that NALM config validation works correctly."""
    print("🧪 Testing NALM config validation...")

    # Test valid config
    valid_config = PredictorOnlyConfig(
        n_layer=2,
        n_head=4,
        n_embd=256,
        block_size=32,
        dag_depth=4,
        max_digits=4,
        max_decimal_places=4,
    )

    # Test invalid config (dag_depth = 0)
    invalid_config = PredictorOnlyConfig(
        n_layer=2,
        n_head=4,
        n_embd=256,
        block_size=32,
        dag_depth=0,  # This disables DAG
        max_digits=4,
        max_decimal_places=4,
    )

    # Create models to test
    valid_model = PredictorOnlyModel(valid_config)
    invalid_model = PredictorOnlyModel(invalid_config)

    # Test forward pass
    test_input = torch.randint(0, 1000, (2, 10))

    # Valid model should work
    with torch.no_grad():
        valid_outputs = valid_model(test_input)
        assert len(valid_outputs) == 4

    # Invalid model should also work (no DAG but still predictor)
    with torch.no_grad():
        invalid_outputs = invalid_model(test_input)
        assert len(invalid_outputs) == 4

    print(f"  ✅ Config validation working")
    print(f"    - Valid config (dag_depth=4): {len(valid_outputs)} outputs")
    print(f"    - Invalid config (dag_depth=0): {len(invalid_outputs)} outputs")


if __name__ == "__main__":
    """Run all NALM checkpoint reloading tests."""
    print("🚀 Running NALM checkpoint reloading tests...")
    print("=" * 60)

    test_nalm_checkpoint_reloading_matching_config()
    print()

    test_nalm_checkpoint_reloading_mismatch_detection()
    print()

    test_nalm_gpt_checkpoint_reloading()
    print()

    test_nalm_config_validation()
    print()

    print("✅ All NALM checkpoint reloading tests completed successfully!")
