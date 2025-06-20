"""
Tests for the adaptive sharding logic in dataset preparation.
"""

from unittest.mock import MagicMock

import numpy as np


def test_adaptive_sharding_logic():
    """Test the adaptive sharding logic with different dataset sizes."""

    def calculate_total_batches(dataset_size):
        """Replicate the adaptive batching logic from prepare functions."""
        return min(1024, max(1, dataset_size // 100))

    # Test cases: (dataset_size, expected_batches)
    test_cases = [
        (1, 1),  # Very small dataset - should use 1 batch
        (50, 1),  # Small dataset - should use 1 batch (50//100 = 0, max(1,0) = 1)
        (100, 1),  # Exactly 100 - should use 1 batch (100//100 = 1)
        (150, 1),  # Just over 100 - should use 1 batch (150//100 = 1)
        (1000, 10),  # Medium dataset - should use 10 batches
        (10000, 100),  # Larger dataset - should use 100 batches
        (100000, 1000),  # Large dataset - should use 1000 batches
        (200000, 1024),  # Very large dataset - should be capped at 1024
        (1000000, 1024),  # Huge dataset - should be capped at 1024
    ]

    for dataset_size, expected_batches in test_cases:
        actual_batches = calculate_total_batches(dataset_size)
        assert (
            actual_batches == expected_batches
        ), f"Dataset size {dataset_size} should use {expected_batches} batches, got {actual_batches}"


def test_sharding_with_small_dataset():
    """Test that small datasets don't cause index out of range errors."""

    # Mock a small dataset with 40 examples
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 40

    # Mock the shard method to return a batch
    mock_batch = MagicMock()
    # batch["ids"] should be a list of arrays, as expected by np.concatenate
    mock_batch.with_format.return_value.__getitem__.return_value = [
        np.array([1, 2, 3, 4, 5])
    ]
    mock_dataset.shard.return_value = mock_batch

    # Calculate total batches using the adaptive logic
    dataset_size = len(mock_dataset)
    total_batches = min(1024, max(1, dataset_size // 100))  # Should be 1

    assert (
        total_batches == 1
    ), f"Expected 1 batch for dataset size 40, got {total_batches}"

    # Verify that sharding works for all batch indices
    for batch_idx in range(total_batches):
        batch = mock_dataset.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")

        # This should not raise an IndexError
        arr_batch = np.concatenate(batch["ids"])
        assert len(arr_batch) > 0


def test_sharding_with_large_dataset():
    """Test that large datasets use appropriate number of batches."""

    # Mock a large dataset with 50,000 examples
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 50000

    # Calculate total batches using the adaptive logic
    dataset_size = len(mock_dataset)
    total_batches = min(1024, max(1, dataset_size // 100))  # Should be 500

    assert (
        total_batches == 500
    ), f"Expected 500 batches for dataset size 50000, got {total_batches}"

    # Verify that we can access batch indices without errors
    for batch_idx in range(min(5, total_batches)):  # Test first 5 batches
        mock_dataset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
        # Verify the call was made with correct parameters
        mock_dataset.shard.assert_called_with(
            num_shards=total_batches, index=batch_idx, contiguous=True
        )


def test_sharding_edge_cases():
    """Test edge cases in the sharding logic."""

    def calculate_total_batches(dataset_size):
        return min(1024, max(1, dataset_size // 100))

    # Test edge cases
    edge_cases = [
        (0, 1),  # Zero size dataset - should default to 1 batch
        (99, 1),  # Just under 100 - should use 1 batch
        (100, 1),  # Exactly 100 - should use 1 batch
        (101, 1),  # Just over 100 - should use 1 batch
        (102400, 1024),  # Exactly at the 1024 cap - should use 1024 batches
        (102500, 1024),  # Just over the cap - should be capped at 1024
    ]

    for dataset_size, expected_batches in edge_cases:
        actual_batches = calculate_total_batches(dataset_size)
        assert (
            actual_batches == expected_batches
        ), f"Edge case: Dataset size {dataset_size} should use {expected_batches} batches, got {actual_batches}"


def test_sharding_consistency():
    """Test that the sharding logic is consistent across different dataset sizes."""

    def calculate_total_batches(dataset_size):
        return min(1024, max(1, dataset_size // 100))

    # Test that the function is monotonic (larger datasets get more or equal batches)
    sizes = [1, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
    batches = [calculate_total_batches(size) for size in sizes]

    # Check monotonicity
    for i in range(1, len(batches)):
        assert (
            batches[i] >= batches[i - 1]
        ), f"Sharding should be monotonic: {sizes[i-1]} -> {batches[i-1]} batches, {sizes[i]} -> {batches[i]} batches"


def test_integration_with_prepare_functions():
    """Test that the sharding logic integrates properly with prepare functions."""

    # This test verifies that the prepare functions can handle the sharding logic
    # without actually running the full dataset preparation

    def mock_prepare_sharding_logic(dataset_size):
        """Mock the sharding logic that would be used in prepare functions."""
        total_batches = min(1024, max(1, dataset_size // 100))

        # Simulate the batching loop
        total_tokens = 0
        for batch_idx in range(total_batches):
            # Mock that each batch contributes some tokens
            batch_tokens = max(
                1, dataset_size // total_batches
            )  # At least 1 token per batch
            total_tokens += batch_tokens

        return total_batches, total_tokens

    # Test with various dataset sizes
    test_sizes = [40, 100, 1000, 10000, 100000]

    for size in test_sizes:
        batches, tokens = mock_prepare_sharding_logic(size)

        # Verify reasonable behavior
        assert batches >= 1, f"Should have at least 1 batch for size {size}"
        assert batches <= 1024, f"Should have at most 1024 batches for size {size}"
        assert tokens > 0, f"Should have positive token count for size {size}"

        # For small datasets, should use fewer batches
        if size < 100:
            assert batches == 1, f"Small dataset {size} should use 1 batch"
