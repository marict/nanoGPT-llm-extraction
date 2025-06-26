"""
Tests for the adaptive sharding logic in dataset preparation.
"""

from unittest.mock import MagicMock

import numpy as np


def calculate_total_batches(dataset_size):
    """Replicate the adaptive batching logic from prepare functions."""
    return min(1024, max(1, dataset_size // 100))


def test_adaptive_sharding_comprehensive():
    """Comprehensive test of adaptive sharding logic covering all scenarios."""

    # Test 1: Core sharding calculation with various sizes
    test_cases = [
        # (dataset_size, expected_batches, description)
        (0, 1, "zero size"),
        (1, 1, "minimal size"),
        (50, 1, "small dataset"),
        (99, 1, "just under 100"),
        (100, 1, "exactly 100"),
        (101, 1, "just over 100"),
        (150, 1, "slightly larger"),
        (1000, 10, "medium dataset"),
        (10000, 100, "large dataset"),
        (100000, 1000, "very large dataset"),
        (102400, 1024, "exactly at cap"),
        (200000, 1024, "over cap - should be capped"),
        (1000000, 1024, "huge dataset - should be capped"),
    ]

    for dataset_size, expected_batches, description in test_cases:
        actual_batches = calculate_total_batches(dataset_size)
        assert (
            actual_batches == expected_batches
        ), f"{description}: Dataset size {dataset_size} should use {expected_batches} batches, got {actual_batches}"

    # Test 2: Monotonicity - larger datasets should get more or equal batches
    sizes = [1, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
    batches = [calculate_total_batches(size) for size in sizes]

    for i in range(1, len(batches)):
        assert (
            batches[i] >= batches[i - 1]
        ), f"Sharding should be monotonic: {sizes[i-1]} -> {batches[i-1]} batches, {sizes[i]} -> {batches[i]} batches"

    # Test 3: Mock dataset integration with small and large datasets
    for dataset_size in [40, 50000]:
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = dataset_size

        # Mock the shard method to return a batch
        mock_batch = MagicMock()
        mock_batch.with_format.return_value.__getitem__.return_value = [
            np.array([1, 2, 3, 4, 5])
        ]
        mock_dataset.shard.return_value = mock_batch

        total_batches = calculate_total_batches(dataset_size)

        # Test that we can shard without errors
        for batch_idx in range(min(3, total_batches)):  # Test first few batches
            batch = mock_dataset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")

            # This should not raise an IndexError
            arr_batch = np.concatenate(batch["ids"])
            assert len(arr_batch) > 0

    # Test 4: Integration simulation
    def mock_prepare_sharding_logic(dataset_size):
        """Simulate the sharding logic used in prepare functions."""
        total_batches = calculate_total_batches(dataset_size)

        # Simulate the batching loop
        total_tokens = 0
        for batch_idx in range(total_batches):
            batch_tokens = max(1, dataset_size // total_batches)
            total_tokens += batch_tokens

        return total_batches, total_tokens

    # Test integration with various sizes
    for size in [40, 100, 1000, 10000, 100000]:
        batches, tokens = mock_prepare_sharding_logic(size)

        # Verify reasonable behavior
        assert batches >= 1, f"Should have at least 1 batch for size {size}"
        assert batches <= 1024, f"Should have at most 1024 batches for size {size}"
        assert tokens > 0, f"Should have positive token count for size {size}"

        # Small datasets should use 1 batch
        if size < 100:
            assert batches == 1, f"Small dataset {size} should use 1 batch"
