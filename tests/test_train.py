import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


import pytest


@pytest.mark.parametrize("batch_size", [1, 2])
def test_train_script_runs(tmp_path: Path, batch_size: int):
    """Run ``train.py`` on a tiny synthetic dataset to ensure the script works."""
    # Create a minimal dataset locally to avoid network downloads from prepare.py
    data_dir = tmp_path / "data" / "shakespeare_char"
    data_dir.mkdir(parents=True)

    vocab_size = 10
    block_size = 32

    # Generate a very small token stream just long enough for ``get_batch``
    arr = np.arange(block_size + 2, dtype=np.uint16) % vocab_size
    arr.tofile(data_dir / "train.bin")
    arr.tofile(data_dir / "val.bin")

    meta = {
        "vocab_size": vocab_size,
        "itos": {i: str(i) for i in range(vocab_size)},
        "stoi": {str(i): i for i in range(vocab_size)},
    }
    with open(data_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    train_script = REPO_ROOT / "train.py"
    config_file = REPO_ROOT / "config" / "train_default.py"

    cmd = [
        sys.executable,
        train_script,
        config_file,
        f"--out_dir={tmp_path / 'out'}",
        "--device=cpu",
        "--compile=False",
        "--eval_interval=1",
        "--eval_iters=1",
        "--log_interval=1",
        "--max_iters=0",
        "--dataset=shakespeare_char",
        f"--batch_size={batch_size}",
        "--n_layer=1",
        "--n_head=1",
        "--n_embd=32",
        "--block_size=32",
        "--wandb_log=False",
    ]
    # Run the training script in ``tmp_path`` so it picks up the synthetic dataset
    subprocess.check_call(cmd, cwd=tmp_path)
