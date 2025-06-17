from pathlib import Path
from typing import Dict, Callable, Tuple, Optional

# Registry of available datasets and their prepare functions
DATASETS: Dict[str, Callable[[Path], Tuple[int, int]]] = {}

def register_dataset(name: str) -> Callable:
    """Decorator to register a dataset's prepare function."""
    def decorator(prepare_func: Callable[[Path], Tuple[int, int]]) -> Callable:
        DATASETS[name] = prepare_func
        return prepare_func
    return decorator

def prepare_dataset(dataset: str, data_dir: Optional[Path] = None) -> Tuple[int, int]:
    """Prepare a dataset for training.
    
    Args:
        dataset: Name of the dataset to prepare
        data_dir: Optional path to prepare the dataset in. If None, uses the dataset's default location.
    
    Returns:
        Tuple of (train_tokens, val_tokens)
    
    Raises:
        ValueError: If dataset is not registered
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Available datasets: {list(DATASETS.keys())}")
    
    if data_dir is None:
        data_dir = Path("data") / dataset
    
    return DATASETS[dataset](data_dir) 