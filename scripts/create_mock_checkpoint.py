#!/usr/bin/env python3
"""
Script to create a mock checkpoint for testing.

This creates a checkpoint with the same architecture as our current models
so we can test the checkpoint loading and DAG functionality.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from models.predictor_only_model import PredictorOnlyConfig, PredictorOnlyModel


def create_mock_checkpoint():
    """Create a mock checkpoint for testing."""
    print("🔄 Creating mock checkpoint...")

    # Create config matching the RunPod checkpoint
    config = PredictorOnlyConfig(
        n_layer=2,
        n_head=4,
        n_embd=256,  # n_head * 64
        block_size=32,
        dag_depth=4,
        max_digits=4,
        max_decimal_places=4,
    )

    # Create model
    model = PredictorOnlyModel(config)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create checkpoint data
    checkpoint_data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": config,
        "iter_num": 8600,
        "best_val_loss": 0.01,  # Simulating 99.98% accuracy
    }

    # Save checkpoint
    output_dir = Path("runpod_checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "mock_checkpoint_99.98acc.pt"
    torch.save(checkpoint_data, checkpoint_path)

    # Get file size
    size_bytes = checkpoint_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    print(f"✅ Mock checkpoint created:")
    print(f"   - Path: {checkpoint_path}")
    print(f"   - Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - DAG depth: {config.dag_depth}")
    print(f"   - Max digits: {config.max_digits}")

    return checkpoint_path


if __name__ == "__main__":
    create_mock_checkpoint()
