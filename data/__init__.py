from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from .openwebtext.prepare import prepare as prepare_openwebtext
from .proofpile.prepare import prepare as prepare_proofpile
from .shakespeare.prepare import prepare as prepare_shakespeare

# Available datasets and their prepare functions
DATASETS: Dict[str, Callable] = {
    "shakespeare": prepare_shakespeare,
    "openwebtext": prepare_openwebtext,
    "proofpile": prepare_proofpile,
}


def get_dataset_info(dataset: str, subset: float = 1.0) -> Dict[str, int]:
    """Get information about dataset sizes for a given subset.

    Args:
        dataset: Name of the dataset
        subset: Fraction of each split to keep (0 < subset ≤ 1)

    Returns:
        Dictionary with expected example counts for train and val splits
    """
    # Known dataset sizes (approximate)
    dataset_sizes = {
        "shakespeare": {"train": 1003234, "val": 111539},  # characters
        "openwebtext": {"train": 8009762, "val": 4007},  # documents
        "proofpile": {"train": 270000, "val": 46300},  # documents
    }

    if dataset not in dataset_sizes:
        return {"train": 0, "val": 0}

    sizes = dataset_sizes[dataset]
    subset = max(min(subset, 1.0), 0.0)

    return {"train": int(sizes["train"] * subset), "val": int(sizes["val"] * subset)}


def prepare_dataset(
    dataset: str,
    data_dir: Optional[Path] = None,
    subset: float = 1.0,
    num_proc: int = 8,
) -> Tuple[int, int]:
    """Prepare a dataset for training.

    Args:
        dataset: Name of the dataset to prepare
        data_dir: Optional path to prepare the dataset in. If None, uses the dataset's default location.
        subset: Fraction of each split to keep (0 < subset ≤ 1)
        num_proc: Number of processes to use for tokenization (only used by openwebtext and proofpile)

    Returns:
        Tuple of (train_tokens, val_tokens)

    Raises:
        ValueError: If dataset is not registered
    """
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset}. Available datasets: {list(DATASETS.keys())}"
        )

    if data_dir is None:
        data_dir = Path("data") / dataset

    # Show expected example counts before starting preparation
    info = get_dataset_info(dataset, subset)
    if subset < 1.0:
        print(f"Expected examples with subset {subset:.6f}:")
        print(f"  train: {info['train']:,} examples")
        print(f"  val: {info['val']:,} examples")
    else:
        print(f"Using full dataset:")
        print(f"  train: {info['train']:,} examples")
        print(f"  val: {info['val']:,} examples")

    # Shakespeare only takes data_dir and subset, others take num_proc as well
    if dataset == "shakespeare":
        return DATASETS[dataset](data_dir, subset)
    else:
        return DATASETS[dataset](data_dir, num_proc, subset)
