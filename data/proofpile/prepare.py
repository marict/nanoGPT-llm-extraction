#!/usr/bin/env python
"""
prepare_proofpile.py
Prepare the Proof-Pile dataset for language-model training.

Writes train.bin / val.bin (uint16 token streams) and meta.pkl.

Example
-------
python prepare_proofpile.py --data-dir /path/to/out --num-proc 8 --subset 0.000002
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
# Core routine
# --------------------------------------------------------------------------- #
def prepare(data_dir: Path, num_proc: int = 8, subset: float = 1.0) -> Tuple[int, int]:
    """Download, tokenize, and binarize Proof-Pile.

    Args
    ----
    data_dir : Path   – output directory
    num_proc : int    – tokenization workers
    subset   : float  – fraction of each split (0 < subset ≤ 1)

    Returns
    -------
    (train_tokens, val_tokens)
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    subset = max(min(subset, 1.0), 0.0)
    pct = subset * 100.0

    # constants are approximate dataset sizes at time of writing
    TRAIN_N = 2_035_895
    VAL_N = 46_251

    if pct >= 0.01:  # ≥ 0.01 %
        train_split = f"train[:{pct:.5f}%]"
        val_split = f"validation[:{pct:.5f}%]"
    else:  # fallback to absolute counts
        n_train = max(1, int(TRAIN_N * subset))
        n_val = max(1, int(VAL_N * subset))
        train_split = f"train[:{n_train}]"
        val_split = f"validation[:{n_val}]"

    print(f"Loading Proof-Pile slices:\n  {train_split}\n  {val_split}")

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
            remove_columns=[c for c in d.column_names if c != "text"],
            desc=f"tokenizing {split}",
            num_proc=num_proc,
        )
        for split, d in split_dataset.items()
    }

    # --------------------------------------------------------------------- #
    # Write uint16 streams
    # --------------------------------------------------------------------- #
    for split, d in tokenized.items():
        arr_len = np.sum(d["len"], dtype=np.uint64)
        mmap = np.memmap(
            data_dir / f"{split}.bin", dtype=np.uint16, mode="w+", shape=(arr_len,)
        )

        total_batches = min(1024, max(1, len(d) // 100))
        idx = 0
        for b in range(total_batches):
            shard = d.shard(total_batches, b, contiguous=True).with_format("numpy")
            batch_ids = np.concatenate(shard["ids"])
            mmap[idx : idx + len(batch_ids)] = batch_ids
            idx += len(batch_ids)
        mmap.flush()

    # save tokenizer meta
    with (data_dir / "meta.pkl").open("wb") as f:
        pickle.dump({"vocab_size": 50257, "tokenizer": "gpt2"}, f)

    return int(np.sum(tokenized["train"]["len"])), int(np.sum(tokenized["val"]["len"]))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare the Proof-Pile dataset.")
    p.add_argument("--data-dir", type=Path, default=Path("."), help="Output directory")
    p.add_argument("--num-proc", type=int, default=8, help="Tokenization workers")
    p.add_argument(
        "--subset", type=float, default=1.0, help="Fraction of each split to keep"
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    tr_tok, va_tok = prepare(args.data_dir, args.num_proc, args.subset)
    print("✅ Preparation complete")
    print(f"Train tokens: {tr_tok:,}")
    print(f"Val tokens:   {va_tok:,}")
