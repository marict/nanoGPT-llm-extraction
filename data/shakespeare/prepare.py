import os
import pickle
import requests
import numpy as np
from pathlib import Path
from tiktoken import get_encoding

from data import register_dataset

@register_dataset("shakespeare")
def prepare(data_dir: Path) -> tuple[int, int]:
    """Prepare the tiny Shakespeare dataset for training.
    
    Downloads the dataset if needed, splits into train/val, and exports to binary files.
    
    Args:
        data_dir: Directory to save the prepared dataset
        
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
    prepare(Path(__file__).parent)

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
