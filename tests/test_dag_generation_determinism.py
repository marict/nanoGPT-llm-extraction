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
    )

    # Generate two identical dataloaders
    train_loader1, val_loader1 = create_dag_structure_dataloaders(**kwargs)
    train_loader2, val_loader2 = create_dag_structure_dataloaders(**kwargs)

    # Get first batch from each
    texts1, target_tensors1, valid_mask1 = next(train_loader1)
    texts2, target_tensors2, valid_mask2 = next(train_loader2)

    # Check that texts are identical
    assert texts1 == texts2

    # Check that valid masks are identical
    assert torch.equal(valid_mask1, valid_mask2)

    # Check that target tensors are identical (for valid positions)
    assert len(target_tensors1) == len(target_tensors2)
    for i, (target1, target2) in enumerate(zip(target_tensors1, target_tensors2)):
        for key in target1.keys():
            assert key in target2, f"Missing key {key} in target {i}"
            if isinstance(target1[key], torch.Tensor):
                assert torch.equal(
                    target1[key], target2[key]
                ), f"Mismatch in {key} for target {i}"
            else:
                assert target1[key] == target2[key], f"Mismatch in {key} for target {i}"
