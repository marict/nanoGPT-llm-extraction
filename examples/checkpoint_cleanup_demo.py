#!/usr/bin/env python3
"""Demo script to show checkpoint cleanup functionality."""

import sys
import tempfile
import time
from pathlib import Path

import torch

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from checkpoint_manager import CheckpointManager


def demo_checkpoint_cleanup():
    """Demonstrate the checkpoint cleanup functionality."""
    print("🧹 Checkpoint Cleanup Demo")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary checkpoint directory
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        # Mock the checkpoint directory for demo
        original_checkpoint_dir = CheckpointManager.checkpoint_dir
        CheckpointManager.checkpoint_dir = checkpoint_dir

        try:
            manager = CheckpointManager()
            config_name = "demo_config"

            print(f"📁 Created temporary checkpoint directory: {checkpoint_dir}")
            print()

            # Create multiple "best" checkpoints to simulate the problem
            print("📝 Creating multiple 'best' checkpoints...")
            checkpoints = [
                ("ckpt_demo_config_best_90.00acc.pt", 90.0),
                ("ckpt_demo_config_best_92.50acc.pt", 92.5),
                ("ckpt_demo_config_best_95.00acc.pt", 95.0),
                ("ckpt_demo_config_best_97.50acc.pt", 97.5),
                ("ckpt_demo_config_best_99.00acc.pt", 99.0),
            ]

            for i, (filename, accuracy) in enumerate(checkpoints):
                checkpoint_path = checkpoint_dir / filename
                dummy_data = {
                    "iter": i * 100,
                    "accuracy": accuracy,
                    "model_state": torch.randn(10, 10),  # Dummy model weights
                }
                torch.save(dummy_data, checkpoint_path)
                time.sleep(0.01)  # Ensure different timestamps
                print(f"  ✅ Created: {filename}")

            # Create one regular checkpoint (should not be affected)
            regular_checkpoint = checkpoint_dir / "ckpt_demo_config_500_98.50acc.pt"
            torch.save({"iter": 500, "accuracy": 98.5}, regular_checkpoint)
            print(f"  ✅ Created: {regular_checkpoint.name}")

            print()
            print(
                f"📊 Before cleanup: {len(list(checkpoint_dir.glob('*.pt')))} checkpoint files"
            )
            print("Files:")
            for file in sorted(checkpoint_dir.glob("*.pt")):
                print(f"  - {file.name}")

            print()
            print("🧹 Running checkpoint cleanup...")
            manager.clean_previous_best_checkpoints(config_name)

            print()
            print(
                f"📊 After cleanup: {len(list(checkpoint_dir.glob('*.pt')))} checkpoint files"
            )
            print("Files:")
            for file in sorted(checkpoint_dir.glob("*.pt")):
                print(f"  - {file.name}")

            print()
            print("✅ Cleanup completed!")
            print("   - Only the most recent 'best' checkpoint remains")
            print("   - Regular checkpoints are unaffected")
            print("   - Disk space is saved")

        finally:
            # Restore original checkpoint directory
            CheckpointManager.checkpoint_dir = original_checkpoint_dir


def demo_cleanup_integration():
    """Show how cleanup integrates with training."""
    print("\n" + "=" * 50)
    print("🔧 Integration with Training")
    print("=" * 50)

    print("When training with the updated scripts:")
    print()
    print("1. 🎯 New best validation loss achieved")
    print("2. 🧹 Previous 'best' checkpoints are automatically cleaned up")
    print("3. 💾 New best checkpoint is saved")
    print("4. 💿 Disk space is preserved")
    print()
    print("Example training output:")
    print(
        "  [123.45s] 🎯 BEST validation loss: 0.123456 (iter 1000) - saving checkpoint: run_name/ckpt_config_best_99.50acc.pt"
    )
    print("  Removed previous best checkpoint: ckpt_config_best_99.00acc.pt")
    print("  Removed previous best checkpoint: ckpt_config_best_98.50acc.pt")
    print("  Cleaned up 2 previous best checkpoint(s)")
    print("  Saved checkpoint: run_name/ckpt_config_best_99.50acc.pt")


if __name__ == "__main__":
    demo_checkpoint_cleanup()
    demo_cleanup_integration()
