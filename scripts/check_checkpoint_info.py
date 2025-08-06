#!/usr/bin/env python3
"""
Simple script to check checkpoint information.

This script:
1. Loads a checkpoint and shows its basic information
2. Shows the model configuration
3. Shows file size and other metadata
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

import torch


def check_checkpoint_info(checkpoint_path: Path):
    """Check basic information about a checkpoint."""
    print(f"📁 Checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    # Get file size
    size_bytes = checkpoint_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"📊 File size: {size_mb:.2f} MB ({size_bytes:,} bytes)")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"✅ Checkpoint loaded successfully")

        # Show keys
        print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")

        # Show model configuration
        if "model_args" in checkpoint:
            model_args = checkpoint["model_args"]
            print(f"📊 Model args type: {type(model_args)}")
            if hasattr(model_args, "__dict__"):
                print(f"📊 Model args attributes: {list(model_args.__dict__.keys())}")
        elif "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
            print(f"📊 Model config: {model_config}")

        # Show other metadata
        if "iter_num" in checkpoint:
            print(f"📊 Iteration number: {checkpoint['iter_num']}")
        if "best_val_loss" in checkpoint:
            print(f"📊 Best validation loss: {checkpoint['best_val_loss']}")

        # Show model state dict info
        if "model" in checkpoint:
            model_state = checkpoint["model"]
            print(f"📊 Model state dict keys: {len(model_state)} keys")
            print(f"📊 Sample model keys: {list(model_state.keys())[:5]}...")

            # Count parameters
            total_params = sum(
                p.numel() for p in model_state.values() if isinstance(p, torch.Tensor)
            )
            print(f"📊 Estimated parameters: {total_params:,}")

        print(f"✅ Checkpoint analysis complete")

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check checkpoint information")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)

    print("🔍 Checkpoint Information Script")
    print("=" * 40)

    check_checkpoint_info(checkpoint_path)


if __name__ == "__main__":
    main()
