#!/usr/bin/env python3
"""
Simple script to download RunPod checkpoint using checkpoint manager.

Usage:
    python scripts/download_checkpoint.py [source_path] [local_dir]
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from checkpoint_manager import CheckpointManager


def main():
    """Download checkpoint using checkpoint manager."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/download_checkpoint.py <source_path> [local_dir]")
        print("\nExample:")
        print(
            "  python scripts/download_checkpoint.py /runpod-volume/checkpoints/rhfuok9btu6t8m-pretrained_sharp/ckpt_predictor_pretrain_8600_99.98acc.pt"
        )
        return

    source_path = sys.argv[1]
    local_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    print("🚀 Downloading checkpoint using checkpoint manager...")
    print(f"📁 Source: {source_path}")
    if local_dir:
        print(f"📁 Local dir: {local_dir}")

    checkpoint_manager = CheckpointManager()
    result = checkpoint_manager.download_checkpoint_from_runpod(source_path, local_dir)

    if result:
        print(f"\n✅ Checkpoint downloaded successfully to: {result}")
        print(f"💡 You can now test it with:")
        print(f"   python scripts/test_local_checkpoint.py {result}")
    else:
        print(f"\n❌ Failed to download checkpoint")
        print(f"💡 Check the manual instructions above")


if __name__ == "__main__":
    main()
