"""
DAG dataset for training language models on structured reasoning tasks.
Streaming-only implementation - no file storage required.
"""

from .streaming import (DAGDataLoader, StreamingDAGDataset,
                        create_dag_dataloaders)

__all__ = [
    "StreamingDAGDataset",
    "DAGDataLoader",
    "create_dag_dataloaders",
]
