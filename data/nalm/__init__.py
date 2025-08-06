"""NALM (Neural Arithmetic Logic Modules) dataset.

This module implements the Single Module Arithmetic Task dataset from the NALM benchmark.
It generates synthetic arithmetic expressions for training and evaluation.
"""

import pickle
from pathlib import Path
from typing import Tuple

from .streaming import create_nalm_dataloaders


def prepare(
    data_dir: Path, num_proc: int = 8, subset: float = 1.0, force: bool = False
) -> Tuple[int, int]:
    """Prepare the NALM dataset for training.

    Since this is a streaming dataset, we just create minimal metadata files
    to indicate the dataset is ready, and return estimated token counts.

    Args:
        data_dir: Directory to save the prepared dataset
        num_proc: Number of processes (ignored for streaming dataset)
        subset: Fraction of dataset to use (ignored for streaming dataset)
        force: Force re-preparation even if files already exist

    Returns:
        Tuple of (train_tokens, val_tokens) - estimated counts
    """
    # Create nalm-specific directory
    nalm_dir = data_dir / "nalm"
    nalm_dir.mkdir(parents=True, exist_ok=True)

    # Check if already prepared (unless force is specified)
    meta_file = nalm_dir / "meta.pkl"
    if not force and meta_file.exists():
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
        print(
            f"📁 Using existing NALM dataset - Train: {meta['train_tokens']:,} tokens, Val: {meta['val_tokens']:,} tokens"
        )
        return meta["train_tokens"], meta["val_tokens"]

    print("🔄 Preparing NALM streaming dataset...")

    # Create minimal metadata file to indicate dataset is ready
    # Since it's streaming, we estimate token counts based on typical usage
    estimated_train_tokens = 5_000_000  # 5M tokens (streaming generates infinite)
    estimated_val_tokens = 500_000  # 500K tokens (streaming generates infinite)

    # Create metadata with GPT-2 vocab info (nalm uses GPT-2 tokenizer)
    meta = {
        "vocab_size": 50257,  # GPT-2 vocab size
        "train_tokens": estimated_train_tokens,
        "val_tokens": estimated_val_tokens,
        "streaming": True,
        "byte_level": False,
    }

    with open(meta_file, "wb") as f:
        pickle.dump(meta, f)

    print(
        f"✅ NALM dataset prepared - Train: {estimated_train_tokens:,} tokens, Val: {estimated_val_tokens:,} tokens"
    )
    print("📡 Dataset is streaming-based - generates infinite examples on-the-fly")

    return estimated_train_tokens, estimated_val_tokens


__all__ = [
    "create_nalm_dataloaders",
    "prepare",
]
