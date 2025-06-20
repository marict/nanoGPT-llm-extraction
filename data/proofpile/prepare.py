"""
prepare_proofpile.py
Prepare the Proof‑Pile dataset for language model training.

This script mirrors prepare.py for OpenWebText but targets the Proof‑Pile
(≈8.3 B math tokens) dataset hosted on Hugging Face. It writes contiguous
uint16 token streams to train.bin and val.bin plus a meta.pkl file.

Usage:
    python prepare_proofpile.py --data-dir /path/to/output --num-proc 16
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from datasets import load_dataset
from tiktoken import get_encoding


def prepare(data_dir: Path, num_proc: int = 8, subset: float = 1.0) -> Tuple[int, int]:
    """Prepare the Proof‑Pile dataset for training.

    Downloads the dataset, ensures train and validation splits exist,
    tokenizes with the GPT‑2 tokenizer, and writes contiguous uint16 token
    streams to train.bin and val.bin.

    Args:
        data_dir: Target directory for output files.
        num_proc: Worker processes for tokenization and dataset loading.

    Returns:
        Tuple (train_tokens, val_tokens).
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load the Proof‑Pile dataset. Splits already include train/validation/test.
    dataset = load_dataset(
        "hoskinson-center/proof-pile", num_proc=num_proc, trust_remote_code=True
    )

    # Ensure we have both train and validation splits.
    if "validation" not in dataset:
        split_dataset = dataset["train"].train_test_split(
            test_size=0.05, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")
    else:
        split_dataset = {"train": dataset["train"], "val": dataset["validation"]}

    # Optional: keep only a subset of each split.
    subset = max(min(subset, 1.0), 0.0)
    if subset < 1.0:
        for key in ("train", "val"):
            dset = split_dataset[key].shuffle(seed=42)
            keep = int(len(dset) * subset)
            split_dataset[key] = dset.select(range(keep))

    enc = get_encoding("gpt2")

    def _encode(example: Dict[str, str]):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    tokenized = {
        split: dset.map(
            _encode,
            remove_columns=[col for col in dset.column_names if col != "text"],
            desc=f"tokenizing {split} split",
            num_proc=num_proc,
        )
        for split, dset in split_dataset.items()
    }

    # Write binary token streams.
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        mmap = np.memmap(
            data_dir / f"{split}.bin", dtype=np.uint16, mode="w+", shape=(arr_len,)
        )
        # Use adaptive batching based on dataset size
        dataset_size = len(dset)
        total_batches = min(
            1024, max(1, dataset_size // 100)
        )  # At least 1 batch, at most 1024
        idx = 0
        for batch_idx in range(total_batches):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            mmap[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        mmap.flush()

    # Save tokenizer meta.
    meta = {"vocab_size": 50257, "tokenizer": "gpt2"}
    with (data_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    return int(np.sum(tokenized["train"]["len"])), int(np.sum(tokenized["val"]["len"]))


def parse_args() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Prepare the Proof‑Pile dataset.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("."), help="Output directory."
    )
    parser.add_argument(
        "--num-proc", type=int, default=8, help="Parallel worker processes."
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of each split to keep (0 < subset ≤ 1).",
    )
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    train_tokens, val_tokens = prepare(args.data_dir, args.num_proc, args.subset)
    print(f"✅ Preparation complete for proofpile")
    print(f"Train tokens: {train_tokens:,}")
    print(f"Val tokens:   {val_tokens:,}")
