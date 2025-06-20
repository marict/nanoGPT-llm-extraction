#!/usr/bin/env python
"""
prepare_proofpile.py
Prepare the Proof-Pile dataset for language-model training.

Writes train.bin / val.bin (uint16 token streams) and meta.pkl.

Example:
    python prepare_proofpile.py --data-dir /runpod-volume/data --num-proc 16 --subset 0.00002
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from datasets import load_dataset
from tiktoken import get_encoding


# --------------------------------------------------------------------------- #
# Core preparation routine
# --------------------------------------------------------------------------- #
def prepare(data_dir: Path, num_proc: int = 8, subset: float = 1.0) -> Tuple[int, int]:
    """Download, tokenize, and binarize Proof-Pile.

    Args:
        data_dir:  Output directory.
        num_proc:  HF datasets worker processes.
        subset:    Fraction of each split to keep (0 < subset ≤ 1).

    Returns:
        (train_tokens, val_tokens)
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    subset = max(min(subset, 1.0), 0.0)  # clamp to [0,1]
    pct = subset * 100.0

    # --------------------------------------------------------------------- #
    # Load (or slice-load) the dataset
    # --------------------------------------------------------------------- #
    if pct < 100.0:
        train_split = f"train[:{pct:.5f}%]"
        val_split = f"validation[:{pct:.5f}%]"
        print(f"Loading Proof-Pile slices: {train_split}, {val_split}")
        split_dataset = {
            "train": load_dataset(
                "hoskinson-center/proof-pile",
                split=train_split,
                num_proc=num_proc,
                trust_remote_code=True,
            ),
            "val": load_dataset(
                "hoskinson-center/proof-pile",
                split=val_split,
                num_proc=num_proc,
                trust_remote_code=True,
            ),
        }
    else:
        print("Loading full Proof-Pile dataset")
        full = load_dataset(
            "hoskinson-center/proof-pile", num_proc=num_proc, trust_remote_code=True
        )
        split_dataset = {
            "train": full["train"],
            "val": (
                full["validation"]
                if "validation" in full
                else full["train"].train_test_split(
                    test_size=0.05, seed=2357, shuffle=True
                )["test"]
            ),
        }

    for k, d in split_dataset.items():
        print(f"  {k}: {len(d):,} examples")

    # --------------------------------------------------------------------- #
    # Tokenize
    # --------------------------------------------------------------------- #
    enc = get_encoding("gpt2")

    def _encode(example: Dict[str, str]):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    tokenized = {
        split: d.map(
            _encode,
            remove_columns=[col for col in d.column_names if col != "text"],
            desc=f"tokenizing {split}",
            num_proc=num_proc,
        )
        for split, d in split_dataset.items()
    }

    # --------------------------------------------------------------------- #
    # Write contiguous uint16 token streams
    # --------------------------------------------------------------------- #
    for split, d in tokenized.items():
        arr_len = np.sum(d["len"], dtype=np.uint64)
        mmap = np.memmap(
            data_dir / f"{split}.bin", dtype=np.uint16, mode="w+", shape=(arr_len,)
        )

        total_batches = min(1024, max(1, len(d) // 100))
        idx = 0
        for batch_idx in range(total_batches):
            shard = d.shard(total_batches, batch_idx, contiguous=True).with_format(
                "numpy"
            )
            batch_ids = np.concatenate(shard["ids"])
            mmap[idx : idx + len(batch_ids)] = batch_ids
            idx += len(batch_ids)
        mmap.flush()

    # --------------------------------------------------------------------- #
    # Save tokenizer meta
    # --------------------------------------------------------------------- #
    with (data_dir / "meta.pkl").open("wb") as f:
        pickle.dump({"vocab_size": 50257, "tokenizer": "gpt2"}, f)

    return int(np.sum(tokenized["train"]["len"])), int(np.sum(tokenized["val"]["len"]))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the Proof-Pile dataset.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("."), help="Output directory"
    )
    parser.add_argument(
        "--num-proc", type=int, default=8, help="Parallel worker processes"
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of each split to keep (0 < s ≤ 1)",
    )
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    train_tok, val_tok = prepare(args.data_dir, args.num_proc, args.subset)
    print("✅ Preparation complete")
    print(f"Train tokens: {train_tok:,}")
    print(f"Val tokens:   {val_tok:,}")
