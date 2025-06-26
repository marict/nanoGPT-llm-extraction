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


def prepare_dataset(
    dataset: str,
    data_dir: Optional[Path] = None,
    subset: float = 1.0,
    num_proc: int = 8,
    force: bool = False,
) -> Tuple[int, int]:
    """Prepare a dataset for training.

    Args:
        dataset: Name of the dataset to prepare
        data_dir: Optional path to prepare the dataset in. If None, uses the dataset's default location.
        subset: Fraction of each split to keep (0 < subset ≤ 1)
        num_proc: Number of processes to use for tokenization (only used by openwebtext and proofpile)
        force: Force re-preparation even if files already exist

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
        data_dir = Path("data")

    # Shakespeare only takes data_dir, subset, and force; others take num_proc as well
    if dataset == "shakespeare":
        return DATASETS[dataset](data_dir, subset, force)
    else:
        return DATASETS[dataset](data_dir, num_proc, subset, force)
