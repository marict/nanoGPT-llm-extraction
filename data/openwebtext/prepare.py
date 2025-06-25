# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
import pickle
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from tiktoken import get_encoding

# Disable progress bars to prevent broken newlines in containers/remote terminals
os.environ.setdefault("TQDM_DISABLE", "1")

# Also disable datasets library progress bars
import datasets

datasets.disable_progress_bars()


def prepare(data_dir: Path, num_proc: int = 8, subset: float = 1.0) -> tuple[int, int]:
    """Prepare the OpenWebText dataset for training.

    Downloads the dataset, splits into train/val, and exports to binary files.

    Args:
        data_dir: Directory to save the prepared dataset
        num_proc: Number of processes to use for tokenization
        subset: Fraction of each split to keep (0 < subset ≤ 1)

    Returns:
        Tuple of (train_tokens, val_tokens)
    """
    # Create output directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # number of workers in .map() call
    # good number to use is ~order number of cpu cores // 2
    num_proc = num_proc

    # number of workers in load_dataset() call
    # best number might be different from num_proc above as it also depends on NW speed.
    # it is better than 1 usually though
    num_proc_load_dataset = num_proc

    # Validate subset parameter
    subset = max(min(subset, 1.0), 0.0)

    if subset < 1.0:
        print(f"Using subset {subset:.6f} - will download only required examples")
    else:
        print(f"Using full dataset")

    # Load the OpenWebText dataset with streaming for subset selection
    if subset < 1.0:
        # For subsets, use streaming to avoid downloading full dataset
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

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    enc = get_encoding("gpt2")

    def process(example):
        ids = enc.encode_ordinary(
            example["text"]
        )  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = {
        split: dset.map(
            process,
            remove_columns=["text"],
            desc=f"tokenizing {split} split",
            num_proc=num_proc,
        )
        for split, dset in split_dataset.items()
    }

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = data_dir / f"{split}.bin"
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        # Use adaptive batching based on dataset size
        dataset_size = len(dset)
        total_batches = min(
            1024, max(1, dataset_size // 100)
        )  # At least 1 batch, at most 1024

        idx = 0
        for batch_idx in range(total_batches):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": 50257,  # GPT-2 vocab size
        "tokenizer": "gpt2",
    }
    with open(data_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    return int(np.sum(tokenized["train"]["len"])), int(np.sum(tokenized["val"]["len"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare the OpenWebText dataset.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).parent, help="Output directory."
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

    args = parser.parse_args()
    train_tokens, val_tokens = prepare(args.data_dir, args.num_proc, args.subset)
    print(f"✅ Preparation complete for openwebtext")
    print(f"Train tokens: {train_tokens:,}")
    print(f"Val tokens:   {val_tokens:,}")

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
