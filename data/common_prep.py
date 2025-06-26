#!/usr/bin/env python
"""
common_prep.py
Common functionality for data preparation scripts.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from tiktoken import get_encoding

# Disable progress bars to prevent broken newlines in containers/remote terminals
os.environ.setdefault("TQDM_DISABLE", "1")

# Also disable datasets library progress bars
try:
    import datasets

    datasets.disable_progress_bars()
except ImportError:
    pass


class DataPrep:
    """Common data preparation utilities."""

    def __init__(self, data_dir: Path):
        """Initialize data preparation utility.

        Args:
            data_dir: Directory to save prepared datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.enc = get_encoding("gpt2")

    def validate_subset(self, subset: float) -> float:
        """Validate and clamp subset parameter.

        Args:
            subset: Fraction of data to keep

        Returns:
            Validated subset value between 0 and 1
        """
        return max(min(subset, 1.0), 0.0)

    def tokenize_text(self, text: str, add_eot: bool = True) -> list[int]:
        """Tokenize text using GPT-2 BPE encoding.

        Args:
            text: Text to tokenize
            add_eot: Whether to append end-of-text token

        Returns:
            List of token IDs
        """
        ids = self.enc.encode_ordinary(text)
        if add_eot:
            ids.append(self.enc.eot_token)
        return ids

    def create_tokenization_function(self, text_column: str = "text"):
        """Create a tokenization function for datasets.map().

        Args:
            text_column: Name of the text column in dataset

        Returns:
            Function that can be used with datasets.map()
        """

        def _encode(example: Dict[str, Any]):
            ids = self.tokenize_text(example[text_column])
            return {"ids": ids, "len": len(ids)}

        return _encode

    def write_binary_file(self, token_data, split_name: str, batch_size: int = 1024):
        """Write tokenized data to binary file.

        Args:
            token_data: Data with 'ids' and 'len' columns or numpy array
            split_name: Name of the split (train/val)
            batch_size: Number of batches for processing
        """
        if isinstance(token_data, np.ndarray):
            # Simple numpy array case (e.g., Shakespeare)
            token_data.tofile(self.data_dir / f"{split_name}.bin")
        else:
            # Dataset case (e.g., OpenWebText, ProofPile)
            arr_len = np.sum(token_data["len"], dtype=np.uint64)
            mmap = np.memmap(
                self.data_dir / f"{split_name}.bin",
                dtype=np.uint16,
                mode="w+",
                shape=(arr_len,),
            )

            dataset_size = len(token_data)
            total_batches = min(batch_size, max(1, dataset_size // 100))

            idx = 0
            for batch_idx in range(total_batches):
                shard = token_data.shard(total_batches, batch_idx, contiguous=True)
                shard = shard.with_format("numpy")
                batch_ids = np.concatenate(shard["ids"])
                mmap[idx : idx + len(batch_ids)] = batch_ids
                idx += len(batch_ids)
            mmap.flush()

    def save_meta(self, vocab_size: int = 50257, tokenizer: str = "gpt2"):
        """Save tokenizer metadata.

        Args:
            vocab_size: Size of the vocabulary
            tokenizer: Name of the tokenizer
        """
        meta = {
            "vocab_size": vocab_size,
            "tokenizer": tokenizer,
        }
        with (self.data_dir / "meta.pkl").open("wb") as f:
            pickle.dump(meta, f)

    def print_completion(self, dataset_name: str, train_tokens: int, val_tokens: int):
        """Print completion message with token counts.

        Args:
            dataset_name: Name of the dataset
            train_tokens: Number of training tokens
            val_tokens: Number of validation tokens
        """
        print(f"✅ Preparation complete for {dataset_name}")
        print(f"Train tokens: {train_tokens:,}")
        print(f"Val tokens:   {val_tokens:,}")


def get_common_parser(description: str):
    """Get common argument parser with standard arguments.

    Args:
        description: Description for the parser

    Returns:
        Configured ArgumentParser
    """
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("."), help="Output directory"
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of each split to keep (0 < subset ≤ 1)",
    )
    return parser


def add_num_proc_arg(parser, default: int = 8):
    """Add num-proc argument to parser.

    Args:
        parser: ArgumentParser to add argument to
        default: Default number of processes
    """
    parser.add_argument(
        "--num-proc",
        type=int,
        default=default,
        help="Number of processes for tokenization",
    )
