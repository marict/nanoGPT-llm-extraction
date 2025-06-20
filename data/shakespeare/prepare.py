import os
import pickle
from pathlib import Path

import numpy as np
import requests
from tiktoken import get_encoding


def prepare(data_dir: Path, subset: float = 1.0) -> tuple[int, int]:
    """Prepare the tiny Shakespeare dataset for training.

    Downloads the dataset if needed, splits into train/val, and exports to binary files.

    Args:
        data_dir: Directory to save the prepared dataset
        subset: Fraction of each split to keep (0 < subset ≤ 1)

    Returns:
        Tuple of (train_tokens, val_tokens)
    """
    # download the tiny shakespeare dataset
    input_file_path = data_dir / "input.txt"
    if not input_file_path.exists():
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, "r") as f:
        data = f.read()
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # Optional: keep only a subset of each split.
    subset = max(min(subset, 1.0), 0.0)
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

    # encode with tiktoken gpt2 bpe
    enc = get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(data_dir / "train.bin")
    val_ids.tofile(data_dir / "val.bin")

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": 50257,  # GPT-2 vocab size
        "tokenizer": "gpt2",
    }
    with open(data_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    return len(train_ids), len(val_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare the Shakespeare dataset.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).parent, help="Output directory."
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of each split to keep (0 < subset ≤ 1).",
    )

    args = parser.parse_args()
    train_tokens, val_tokens = prepare(args.data_dir, args.subset)
    print(f"✅ Preparation complete for shakespeare")
    print(f"Train tokens: {train_tokens:,}")
    print(f"Val tokens:   {val_tokens:,}")

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
