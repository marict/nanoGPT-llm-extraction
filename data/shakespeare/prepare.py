#!/usr/bin/env python
"""
prepare_shakespeare.py
Prepare the tiny Shakespeare dataset for language-model training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import requests

sys.path.append(str(Path(__file__).parent.parent))
from data.common_prep import DataPrep, get_common_parser


def prepare(
    data_dir: Path, subset: float = 1.0, force: bool = False
) -> tuple[int, int]:
    """Prepare the tiny Shakespeare dataset for training.

    Downloads the dataset if needed, splits into train/val, and exports to binary files.

    Args:
        data_dir: Directory to save the prepared dataset
        subset: Fraction of each split to keep (0 < subset ≤ 1)
        force: Force re-preparation even if files already exist

    Returns:
        Tuple of (train_tokens, val_tokens)
    """
    # Initialize DataPrep with dataset-specific subfolder
    prep = DataPrep(data_dir, dataset_name="shakespeare")
    subset = prep.validate_subset(subset)

    # Check if files already exist (unless force is specified)
    if not force:
        if prep.check_existing_files():
            token_counts = prep.get_existing_token_counts()
            if token_counts:
                train_tokens, val_tokens = token_counts
                print(
                    f"📁 Using existing files - Train: {train_tokens:,} tokens, Val: {val_tokens:,} tokens"
                )
                return train_tokens, val_tokens
            else:
                print(
                    "⚠️  Could not read token counts from existing files, proceeding with preparation"
                )

    print(f"🔄 Starting Shakespeare dataset preparation (subset: {subset})")

    # Download the tiny shakespeare dataset
    input_file_path = prep.data_dir / "input.txt"
    if not input_file_path.exists():
        print("📥 Downloading Shakespeare dataset...")
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)
        print("📥 Download complete")

    with open(input_file_path, "r") as f:
        data = f.read()

    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # Apply subset if needed
    if subset < 1.0:
        train_len = len(train_data)
        val_len = len(val_data)
        train_keep = int(train_len * subset)
        val_keep = int(val_len * subset)
        train_data = train_data[:train_keep]
        val_data = val_data[:val_keep]
        print(f"  train: {train_len:,} → {train_keep:,} characters")
        print(f"  val: {val_len:,} → {val_keep:,} characters")
    else:
        print(f"Using full dataset")
        print(f"  train: {len(train_data):,} characters")
        print(f"  val: {len(val_data):,} characters")

    # Tokenize using common utility
    train_ids = prep.tokenize_text(train_data, add_eot=False)
    val_ids = prep.tokenize_text(val_data, add_eot=False)

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # Convert to numpy arrays and write binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    prep.write_binary_file(train_ids, "train")
    prep.write_binary_file(val_ids, "val")

    # Save metadata
    prep.save_meta()

    return len(train_ids), len(val_ids)


if __name__ == "__main__":
    parser = get_common_parser("Prepare the Shakespeare dataset.")
    args = parser.parse_args()

    train_tokens, val_tokens = prepare(args.data_dir, args.subset, args.force)

    prep = DataPrep(args.data_dir, dataset_name="shakespeare")
    prep.print_completion("shakespeare", train_tokens, val_tokens)

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
