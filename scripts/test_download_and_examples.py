#!/usr/bin/env python3
"""
Comprehensive test script that demonstrates the full workflow:
1. Download checkpoint from RunPod (or simulate it)
2. Test the checkpoint on DAG examples
3. Show the results

This script can work with:
- Real RunPod checkpoints (when on RunPod instance)
- Downloaded checkpoints (when available locally)
- Mock checkpoints (for testing the workflow)
"""

import sys
from pathlib import Path

import torch

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from checkpoint_manager import CheckpointManager
from scripts.test_checkpoint_examples import (
    load_existing_checkpoint,
    test_dag_examples,
    test_model_behavior,
)


def simulate_runpod_workflow():
    """Simulate the complete RunPod workflow."""
    print("🚀 Simulating RunPod Checkpoint Workflow")
    print("=" * 60)

    # Step 1: Try to download from RunPod
    print("📥 Step 1: Attempting to download checkpoint from RunPod...")

    checkpoint_manager = CheckpointManager()
    runpod_path = "/runpod-volume/checkpoints/rhfuok9btu6t8m-pretrained_sharp/ckpt_predictor_pretrain_8600_99.98acc.pt"

    # Try to download
    local_path = checkpoint_manager.download_checkpoint_from_runpod(
        runpod_path, local_dir=Path("runpod_checkpoints")
    )

    if local_path and local_path.exists():
        print(f"✅ Successfully downloaded checkpoint to: {local_path}")
        checkpoint_path = local_path
    else:
        print(
            "⚠️  Could not download from RunPod (expected when not on RunPod instance)"
        )
        print("🔄 Falling back to local checkpoint or mock...")
        checkpoint_path = None

    # Step 2: Load checkpoint (real or mock)
    print(f"\n📂 Step 2: Loading checkpoint...")
    model, config, checkpoint = load_existing_checkpoint(checkpoint_path)

    # Step 3: Test on examples
    print(f"\n🧪 Step 3: Testing on DAG examples...")
    test_dag_examples(model, config)

    # Step 4: Test model behavior
    print(f"\n🔍 Step 4: Testing model behavior...")
    test_model_behavior(model, config)

    print(f"\n✅ Workflow completed successfully!")

    # Summary
    print(f"\n📋 Summary:")
    print(
        f"   - Checkpoint source: {'RunPod' if local_path and local_path.exists() else 'Local/Mock'}"
    )
    print(f"   - Model type: {type(model).__name__}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - DAG depth: {config.dag_depth}")
    print(f"   - Max digits: {config.max_digits}")

    return model, config


def test_with_specific_checkpoint(checkpoint_path: str):
    """Test with a specific checkpoint path."""
    print(f"🚀 Testing with specific checkpoint: {checkpoint_path}")
    print("=" * 60)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, None

    # Load checkpoint
    model, config, checkpoint = load_existing_checkpoint(checkpoint_path)

    # Test on examples
    test_dag_examples(model, config)
    test_model_behavior(model, config)

    print(f"\n✅ Testing completed!")
    return model, config


def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Test with specific checkpoint
        checkpoint_path = sys.argv[1]
        test_with_specific_checkpoint(checkpoint_path)
    else:
        # Simulate full workflow
        simulate_runpod_workflow()


if __name__ == "__main__":
    main()
