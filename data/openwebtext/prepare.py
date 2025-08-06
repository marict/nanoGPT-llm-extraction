#!/usr/bin/env python
"""
prepare_openwebtext.py
Prepare the OpenWebText dataset for language-model training.

Saves the openwebtext dataset to a binary file for training. Following was helpful:
https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset

sys.path.append(str(Path(__file__).parent.parent))
from data.common_prep import DataPrep, add_num_proc_arg, get_common_parser


def prepare(
    data_dir: Path, num_proc: int = 8, subset: float = 1.0, force: bool = False
) -> tuple[int, int]:
    """Prepare the OpenWebText dataset for training.

    Downloads the dataset, splits into train/val, and exports to binary files.

    Args:
        data_dir: Directory to save the prepared dataset
        num_proc: Number of processes to use for tokenization
        subset: Fraction of each split to keep (0 < subset ≤ 1)
        force: Force re-preparation even if files already exist

    Returns:
        Tuple of (train_tokens, val_tokens)
    """

    prep = DataPrep(data_dir, dataset_name="openwebtext")
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

    print(f"🔄 Starting OpenWebText dataset preparation (subset: {subset})")

    # number of workers in load_dataset() call
    # best number might be different from num_proc above as it also depends on NW speed.
    # it is better than 1 usually though
    num_proc_load_dataset = num_proc

    if subset < 1.0:
        print(f"Using subset {subset:.6f} - will download only required examples")
    else:
        print(f"Using full dataset")

    # Load the OpenWebText dataset with streaming for subset selection
    if subset < 1.0:
        # For subsets, use streaming to avoid downloading full dataset
        print("📥 Loading OpenWebText dataset with streaming...")
        dataset = load_dataset("openwebtext", streaming=True, trust_remote_code=True)

        # Calculate how many examples to take from train split
        # OpenWebText has ~8M train examples, we'll create val from train
        train_size = 8009762

        train_take = max(1, int(train_size * subset))
        val_take = max(1, int(train_take * 0.0005))  # Same ratio as original

        print(
            f"  Will take {train_take:,} train examples and {val_take:,} val examples"
        )

        # Take only the subset we need
        train_data = dataset["train"].take(train_take)

        # Convert streaming dataset to regular dataset for processing
        train_dataset = Dataset.from_list(list(train_data))

        # Create validation split from train data
        split_dataset = train_dataset.train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

    else:
        # For full dataset, use regular loading
        # takes 54GB in huggingface .cache dir, at least 30GB more to run
        print("📥 Loading full OpenWebText dataset...")
        dataset = load_dataset(
            "openwebtext", num_proc=num_proc_load_dataset, trust_remote_code=True
        )

        # owt by default only contains the 'train' split, so create a test split
        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

    # Print final sizes
    for key in ("train", "val"):
        size = len(split_dataset[key])
        print(f"  {key}: {size:,} examples")

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    print("🔄 Tokenizing dataset...")
    # Tokenize the dataset using common utility
    process_fn = prep.create_tokenization_function("text")

    # tokenize the dataset
    tokenized = {
        split: dset.map(
            process_fn,
            remove_columns=["text"],
            desc=f"tokenizing {split} split",
            num_proc=num_proc,
        )
        for split, dset in split_dataset.items()
    }

    print("💾 Writing binary files...")
    # Write binary files using common utility
    for split, dset in tokenized.items():
        prep.write_binary_file(dset, split)

    # Save metadata
    prep.save_meta()

    return int(np.sum(tokenized["train"]["len"])), int(np.sum(tokenized["val"]["len"]))


if __name__ == "__main__":
    parser = get_common_parser("Prepare the OpenWebText dataset.")
    add_num_proc_arg(parser, default=8)
    args = parser.parse_args()

    train_tokens, val_tokens = prepare(
        args.data_dir, args.num_proc, args.subset, args.force
    )

    prep = DataPrep(args.data_dir, dataset_name="openwebtext")
    prep.print_completion("openwebtext", train_tokens, val_tokens)

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
