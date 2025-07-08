#!/usr/bin/env python
"""Common functionality for data preparation scripts."""

from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from tiktoken import get_encoding

os.environ.setdefault("TQDM_DISABLE", "1")

try:
    import datasets

    datasets.disable_progress_bars()
except ImportError:
    pass


def _get_runpod_storage_path() -> Path | None:
    """Get RunPod persistent storage path if available."""
    runpod_volume = Path("/runpod-volume")
    return runpod_volume if runpod_volume.exists() else None


class DataPrep:
    """Common data preparation utilities."""

    def __init__(self, data_dir: Path, dataset_name: str | None = None):
        """Initialize data preparation utility."""
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name

        if dataset_name:
            self.data_dir = self.data_dir / dataset_name

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.enc = get_encoding("gpt2")

    def check_existing_files(self, required_files: list[str] | None = None) -> bool:
        """Check if preparation files already exist locally or in RunPod storage."""
        if required_files is None:
            required_files = ["train.bin", "val.bin", "meta.pkl"]

        local_missing = []
        for filename in required_files:
            file_path = self.data_dir / filename
            if not file_path.exists():
                local_missing.append(filename)

        if not local_missing:
            print(f"✅ All required files found locally in {self.data_dir}")
            return True

        runpod_volume = _get_runpod_storage_path()
        if runpod_volume is None:
            print(f"Missing files: {local_missing}")
            return False

        runpod_data_dir = runpod_volume / "data" / self.data_dir.name
        runpod_missing = []
        runpod_available = []

        for filename in local_missing:
            runpod_file_path = runpod_data_dir / filename
            if runpod_file_path.exists():
                runpod_available.append(filename)
            else:
                runpod_missing.append(filename)

        if runpod_missing:
            print(f"Missing files (local and RunPod): {runpod_missing}")
            return False

        print(f"📁 Found all files in RunPod storage: {runpod_data_dir}")
        self._copy_from_runpod_storage(runpod_data_dir, local_missing)
        print(f"✅ All required files restored from RunPod storage to {self.data_dir}")
        return True

    def _copy_from_runpod_storage(self, runpod_data_dir: Path, filenames: list[str]):
        """Copy files from RunPod persistent storage to local directory."""
        import shutil

        self.data_dir.mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            runpod_file_path = runpod_data_dir / filename
            local_file_path = self.data_dir / filename

            if runpod_file_path.exists():
                shutil.copy2(runpod_file_path, local_file_path)
                print(f"📥 Copied {filename} from RunPod storage to local")
            else:
                print(f"⚠️ Warning: {filename} not found in RunPod storage")

    def get_existing_token_counts(self) -> Tuple[int, int] | None:
        """Get token counts from existing bin files (local or RunPod)."""
        try:
            train_path = self.data_dir / "train.bin"
            val_path = self.data_dir / "val.bin"

            if not (train_path.exists() and val_path.exists()):
                runpod_volume = _get_runpod_storage_path()
                if runpod_volume:
                    runpod_data_dir = runpod_volume / "data" / self.data_dir.name
                    runpod_train_path = runpod_data_dir / "train.bin"
                    runpod_val_path = runpod_data_dir / "val.bin"

                    if runpod_train_path.exists() and runpod_val_path.exists():
                        self._copy_from_runpod_storage(
                            runpod_data_dir, ["train.bin", "val.bin", "meta.pkl"]
                        )
                    else:
                        return None
                else:
                    return None

            meta_path = self.data_dir / "meta.pkl"
            dtype = np.uint16
            if meta_path.exists():
                with meta_path.open("rb") as f:
                    meta = pickle.load(f)
                    if meta.get("byte_level", False):
                        dtype = np.uint8

            train_data = np.memmap(train_path, dtype=dtype, mode="r")
            val_data = np.memmap(val_path, dtype=dtype, mode="r")

            return len(train_data), len(val_data)

        except Exception as e:
            print(f"Warning: Could not read existing files: {e}")
            return None

    def validate_subset(self, subset: float) -> float:
        """Validate and clamp subset parameter."""
        return max(min(subset, 1.0), 0.0)

    def tokenize_text(self, text: str, add_eot: bool = True) -> list[int]:
        """Tokenize text using GPT-2 BPE encoding."""
        ids = self.enc.encode_ordinary(text)
        if add_eot:
            ids.append(self.enc.eot_token)
        return ids

    def create_tokenization_function(self, text_column: str = "text"):
        """Create a tokenization function for datasets.map()."""

        def _encode(example: Dict[str, Any]):
            ids = self.tokenize_text(example[text_column])
            return {"ids": ids, "len": len(ids)}

        return _encode

    def write_binary_file(self, token_data, split_name: str, batch_size: int = 1024):
        """Write tokenized data to binary file."""
        local_bin_path = self.data_dir / f"{split_name}.bin"

        if isinstance(token_data, np.ndarray):
            token_data.tofile(local_bin_path)
        else:
            arr_len = np.sum(token_data["len"], dtype=np.uint64)
            mmap = np.memmap(
                local_bin_path,
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

        self._copy_to_runpod_storage(local_bin_path)

    def _copy_to_runpod_storage(self, local_file_path: Path):
        """Copy file to RunPod persistent storage if available."""
        runpod_volume = _get_runpod_storage_path()
        if runpod_volume is None:
            return

        runpod_data_dir = runpod_volume / "data" / self.data_dir.name
        runpod_data_dir.mkdir(parents=True, exist_ok=True)
        runpod_file_path = runpod_data_dir / local_file_path.name
        shutil.copy2(local_file_path, runpod_file_path)
        print(f"📁 Copied {local_file_path.name} to RunPod storage: {runpod_file_path}")

    def save_meta(self, vocab_size: int = 50257, tokenizer: str = "gpt2"):
        """Save tokenizer metadata."""
        meta = {
            "vocab_size": vocab_size,
            "tokenizer": tokenizer,
        }
        local_meta_path = self.data_dir / "meta.pkl"
        with local_meta_path.open("wb") as f:
            pickle.dump(meta, f)

        self._copy_to_runpod_storage(local_meta_path)

    def print_completion(self, dataset_name: str, train_tokens: int, val_tokens: int):
        """Print completion message with token counts."""
        print(f"✅ Preparation complete for {dataset_name}")
        print(f"Train tokens: {train_tokens:,}")
        print(f"Val tokens:   {val_tokens:,}")

        runpod_volume = _get_runpod_storage_path()
        if runpod_volume:
            runpod_data_dir = runpod_volume / "data" / self.data_dir.name
            print(f"📁 Files also saved to RunPod storage: {runpod_data_dir}")


def get_common_parser(description: str):
    """Get common argument parser with standard arguments."""
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-preparation even if files already exist",
    )
    return parser


def add_num_proc_arg(parser, default: int = 8):
    """Add num-proc argument to parser."""
    parser.add_argument(
        "--num-proc",
        type=int,
        default=default,
        help="Number of processes for tokenization",
    )
