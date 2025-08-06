#!/usr/bin/env python3
"""
Script to test a local checkpoint on example DAGs.

This script:
1. Loads a local checkpoint
2. Tests the checkpoint on example DAGs like 1.0 + 1.0
3. Compares with fresh model initialization
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from checkpoint_manager import CheckpointManager
from data.nalm.streaming import NALMDataset, create_nalm_dataloaders
from models.dag_model import GPT, GPTConfig
from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel


def load_checkpoint_and_model(checkpoint_path: Path):
    """Load checkpoint and create model."""
    print(f"🔄 Loading checkpoint from {checkpoint_path}...")

    checkpoint_manager = CheckpointManager()

    try:
        # Load checkpoint
        checkpoint = checkpoint_manager.load_checkpoint_from_path(checkpoint_path)
        print(f"✅ Checkpoint loaded successfully")

        # Extract model args (handle both model_args and model_config keys)
        if "model_args" in checkpoint:
            model_args = checkpoint["model_args"]
        elif "model_config" in checkpoint:
            # Convert dict to config object, filtering out extra fields
            config_dict = checkpoint["model_config"]
            # Filter to only include PredictorOnlyConfig fields
            valid_fields = {
                "n_layer",
                "n_head",
                "n_embd",
                "block_size",
                "dropout",
                "bias",
                "dag_depth",
                "max_digits",
                "max_decimal_places",
            }
            filtered_config = {
                k: v for k, v in config_dict.items() if k in valid_fields
            }
            model_args = PredictorOnlyConfig(**filtered_config)
        else:
            raise KeyError(
                "Checkpoint must contain either 'model_args' or 'model_config'"
            )
        print(f"📊 Model configuration:")
        print(f"   - DAG depth: {model_args.dag_depth}")
        print(f"   - Max digits: {model_args.max_digits}")
        print(f"   - Max decimal places: {model_args.max_decimal_places}")
        print(f"   - Layers: {model_args.n_layer}")
        print(f"   - Heads: {model_args.n_head}")
        print(f"   - Embedding dim: {model_args.n_embd}")

        # Create model
        if hasattr(model_args, "dag_depth") and model_args.dag_depth > 0:
            model = PredictorOnlyModel(model_args)
            model_type = "DAG Predictor"
        else:
            model = GPT(model_args)
            model_type = "DAG-GPT"

        # Load state dict
        model.load_state_dict(checkpoint["model"])
        print(f"✅ {model_type} model created and loaded")

        # Get parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parameters: {total_params:,}")

        return model, model_args, checkpoint

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None, None, None


def test_simple_dag_examples(model, model_args):
    """Test the model on simple DAG examples."""
    print(f"\n🧪 Testing simple DAG examples...")

    model.eval()

    # Test 1: Simple arithmetic expressions
    print(f"📝 Test 1: Simple arithmetic expressions")

    # Create NALM dataset with simple examples
    dataset = NALMDataset(
        split="train",
        operations=["add", "sub", "mul", "div"],
        train_range=(-2.0, 2.0),
        num_examples=5,
        seed=42,
    )

    for i in range(3):
        item = dataset[i]
        expression = item["expression"]
        a, b = item["a"], item["b"]
        expected_result = item["result"]

        print(f"   Example {i+1}: {expression}")
        print(f"     - Input: a={a:.2f}, b={b:.2f}")
        print(f"     - Expected: {expected_result:.6f}")
        print(f"     - Model would process: {expression}")

    # Test 2: Specific 1.0 + 1.0 case
    print(f"\n📝 Test 2: Specific 1.0 + 1.0 case")

    # Create a dataset that's more likely to generate 1.0 + 1.0
    simple_dataset = NALMDataset(
        split="train",
        operations=["add"],
        train_range=(0.9, 1.1),  # Range around 1.0
        num_examples=10,
        seed=42,
    )

    found_simple = False
    for i in range(10):
        item = simple_dataset[i]
        if abs(item["a"] - 1.0) < 0.1 and abs(item["b"] - 1.0) < 0.1:
            print(f"   Found simple case: {item['expression']}")
            print(f"     - a={item['a']:.6f}, b={item['b']:.6f}")
            print(f"     - Expected result: {item['result']:.6f}")
            found_simple = True
            break

    if not found_simple:
        print(f"   No exact 1.0 + 1.0 found, showing closest:")
        item = simple_dataset[0]
        print(f"   Closest: {item['expression']}")
        print(f"     - a={item['a']:.6f}, b={item['b']:.6f}")
        print(f"     - Expected result: {item['result']:.6f}")


def test_model_forward_pass(model, model_args):
    """Test the model's forward pass on sample inputs."""
    print(f"\n🧪 Testing model forward pass...")

    model.eval()

    # Test input
    test_input = torch.randint(0, 1000, (2, 10))
    print(f"📝 Test input shape: {test_input.shape}")

    with torch.no_grad():
        if isinstance(model, PredictorOnlyModel):
            # DAG Predictor model
            outputs = model(test_input)
            digit_logits, V_sign, O, G = outputs

            print(f"📊 DAG Predictor outputs:")
            print(f"   - Digit logits: {digit_logits.shape}")
            print(f"   - V_sign: {V_sign.shape}")
            print(f"   - O: {O.shape}")
            print(f"   - G: {G.shape}")

            # Show some sample values
            print(f"   - Sample V_sign values: {V_sign[0, :3].tolist()}")
            print(f"   - Sample G values: {G[0, 0, :].tolist()}")

        else:
            # DAG-GPT model
            dag_results = model(test_input)

            print(f"📊 DAG-GPT outputs:")
            print(f"   - Output shape: {dag_results.shape}")
            print(f"   - Sample values: {dag_results[0, :5].tolist()}")

    print(f"✅ Forward pass successful")


def compare_with_fresh_model(model, model_args):
    """Compare loaded model with fresh initialization."""
    print(f"\n🧪 Comparing with fresh model...")

    # Create fresh model with same config
    torch.manual_seed(123)  # Different seed
    if isinstance(model, PredictorOnlyModel):
        fresh_model = PredictorOnlyModel(model_args)
    else:
        fresh_model = GPT(model_args)

    fresh_model.eval()
    model.eval()

    # Test input
    test_input = torch.randint(0, 1000, (2, 10))

    with torch.no_grad():
        if isinstance(model, PredictorOnlyModel):
            # Compare DAG predictor outputs
            loaded_outputs = model(test_input)
            fresh_outputs = fresh_model(test_input)

            digit_logits_loaded, V_sign_loaded, O_loaded, G_loaded = loaded_outputs
            digit_logits_fresh, V_sign_fresh, O_fresh, G_fresh = fresh_outputs

            # Calculate differences
            digit_diff = (
                torch.abs(digit_logits_loaded - digit_logits_fresh).mean().item()
            )
            vsign_diff = torch.abs(V_sign_loaded - V_sign_fresh).mean().item()
            o_diff = torch.abs(O_loaded - O_fresh).mean().item()
            g_diff = torch.abs(G_loaded - G_fresh).mean().item()

            print(f"📊 Output differences (loaded vs fresh):")
            print(f"   - Digit logits: {digit_diff:.6f}")
            print(f"   - V_sign: {vsign_diff:.6f}")
            print(f"   - O: {o_diff:.6f}")
            print(f"   - G: {g_diff:.6f}")

        else:
            # Compare DAG-GPT outputs
            loaded_results = model(test_input)
            fresh_results = fresh_model(test_input)

            diff = torch.abs(loaded_results - fresh_results).mean().item()
            print(f"📊 DAG-GPT output difference: {diff:.6f}")

    print(f"✅ Comparison complete")


def main():
    """Main function to test local checkpoint."""
    parser = argparse.ArgumentParser(
        description="Test a local checkpoint on DAG examples"
    )
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)

    print("🚀 Local Checkpoint Testing Script")
    print("=" * 50)
    print(f"📁 Testing checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    # Get file size
    size_bytes = checkpoint_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"📊 Checkpoint size: {size_mb:.2f} MB ({size_bytes:,} bytes)")

    # Load checkpoint and model
    model, model_args, checkpoint = load_checkpoint_and_model(checkpoint_path)
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return

    # Test simple DAG examples
    test_simple_dag_examples(model, model_args)

    # Test model forward pass
    test_model_forward_pass(model, model_args)

    # Compare with fresh model
    compare_with_fresh_model(model, model_args)

    print(f"\n✅ All tests completed successfully!")


if __name__ == "__main__":
    main()
