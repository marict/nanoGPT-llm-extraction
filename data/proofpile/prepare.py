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

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent.parent))
from data.common_prep import DataPrep, add_num_proc_arg, get_common_parser


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
    prep = DataPrep(data_dir)
    subset = prep.validate_subset(subset)
    pct = subset * 100.0

    # constants are approximate dataset sizes at time of writing
    TRAIN_N = 2_035_895
    VAL_N = 46_251

    if pct >= 1.0:  # huggingface doesn't support < 1%
        if subset >= 1.0:
            train_split = "train"
            val_split = "validation"
        else:
            train_split = f"train[:{pct:.0f}%]"
            val_split = f"validation[:{pct:.0f}%]"
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

    # Tokenize using common utility
    process_fn = prep.create_tokenization_function("text")

    tokenized = {
        split: d.map(
            process_fn,
            remove_columns=[c for c in d.column_names if c != "text"],
            desc=f"tokenizing {split}",
            num_proc=num_proc,
        )
        for split, d in split_dataset.items()
    }

    # Write binary files using common utility
    for split, d in tokenized.items():
        prep.write_binary_file(d, split)

    # Save metadata
    prep.save_meta()

    return int(np.sum(tokenized["train"]["len"])), int(np.sum(tokenized["val"]["len"]))


def _get_parser():
    """Get argument parser for ProofPile preparation."""
    parser = get_common_parser("Prepare the Proof-Pile dataset.")
    add_num_proc_arg(parser, default=8)
    return parser


if __name__ == "__main__":
    args = _get_parser().parse_args()
    tr_tok, va_tok = prepare(args.data_dir, args.num_proc, args.subset)

    prep = DataPrep(args.data_dir)
    prep.print_completion("proof-pile", tr_tok, va_tok)
