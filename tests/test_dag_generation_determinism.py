import pytest
import torch

from data.dagset.streaming import create_dag_structure_dataloaders


@pytest.mark.parametrize("depth,seed", [(3, 42), (3, 43), (4, 321), (5, 123)])
def test_dag_generation_determinism(depth: int, seed: int):
    """Ensure that repeated calls with identical arguments yield identical data batches."""
    kwargs = dict(
        train_batch_size=2,
        val_batch_size=2,
        max_depth=depth,
        seed=seed,
        max_digits=4,
        max_decimal_places=6,
        block_size=25,  # Increased to accommodate more expressions
    )

    # Generate two identical dataloaders
    train_loader1, val_loader1 = create_dag_structure_dataloaders(**kwargs)
    train_loader2, val_loader2 = create_dag_structure_dataloaders(**kwargs)

    # Get first batch from each
    texts1, target_tensors1 = next(train_loader1)
    texts2, target_tensors2 = next(train_loader2)

    # Check that texts are identical
    assert texts1 == texts2

    # Check that target tensors are identical
    assert len(target_tensors1) == len(target_tensors2)
    for key in target_tensors1.keys():
        assert key in target_tensors2, f"Missing key {key}"
        if isinstance(target_tensors1[key], torch.Tensor):
            assert torch.equal(
                target_tensors1[key], target_tensors2[key]
            ), f"Mismatch in {key}"
        else:
            assert target_tensors1[key] == target_tensors2[key], f"Mismatch in {key}"
