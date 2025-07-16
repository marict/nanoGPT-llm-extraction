import re

import pytest

from data.dagset.streaming import DAGStructureDataset


@pytest.mark.parametrize("depth", [3])
def test_dataset_expression_permutation(depth):
    """Dataset should generate permuted operand order for at least one seed when probability is 1.0."""
    ds_no_perm = DAGStructureDataset(
        max_depth=depth,
        seed=0,
        english_conversion_probability=0.0,
        integer_no_decimal_probability=0.0,
        expression_permutation_probability=0.0,
        allowed_operations=["add"],
    )

    ds_perm = DAGStructureDataset(
        max_depth=depth,
        seed=0,
        english_conversion_probability=0.0,
        integer_no_decimal_probability=0.0,
        expression_permutation_probability=1.0,
        allowed_operations=["add"],
    )

    # Generate multiple examples across seeds to ensure at least one permutation difference
    difference_found = False
    for s in range(1000, 1010):
        expr_no, _ = ds_no_perm.generate_structure_example(depth=depth, seed=s)
        expr_perm, _ = ds_perm.generate_structure_example(depth=depth, seed=s)

        # Skip seeds where numeric literals differ (unlikely but safe check)
        nums_no = sorted(re.findall(r"-?\d+\.?\d*", expr_no))
        nums_perm = sorted(re.findall(r"-?\d+\.?\d*", expr_perm))

        if nums_no != nums_perm:
            continue  # Numeric values changed, not a pure permutation

        if expr_no != expr_perm:
            difference_found = True
            break

    assert (
        difference_found
    ), "Permutation probability did not create any observable operand reordering across seeds"
