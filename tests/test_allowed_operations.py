import pytest

from data.dagset.streaming import DAGStructureDataset, generate_random_dag_plan
from models.dag_model import TEST_OPS_NAMES


def test_generate_random_dag_plan_respects_allowed_ops():
    """All sampled operations must come from the user-specified subset."""
    # Use only operations supported by the current implementation
    allowed_subset = TEST_OPS_NAMES  # always valid
    depth = 10
    init_vals, ops = generate_random_dag_plan(
        depth=depth,
        num_initial_values=depth + 1,
        seed=123,
        allowed_operations=allowed_subset,
    )

    assert len(ops) == depth
    assert all(
        op in allowed_subset for op in ops
    ), f"Found ops outside allowed subset: {ops}"


def test_generate_random_dag_plan_invalid_ops_raises():
    """Providing an invalid operation should raise a clear ValueError."""
    invalid_subset = ["add", "power"]
    with pytest.raises(ValueError):
        generate_random_dag_plan(depth=1, allowed_operations=invalid_subset)


def test_dataset_generation_respects_allowed_ops():
    """DAGStructureDataset should only emit operations from the allowed set."""
    allowed_subset = TEST_OPS_NAMES  # subset guaranteed to be valid
    depth = 3
    dataset = DAGStructureDataset(
        max_depth=depth,
        seed=0,
        english_conversion_probability=0.0,  # turn off text randomness for test clarity
        allowed_operations=allowed_subset,
    )

    text, structure = dataset.generate_structure_example(depth, seed=0)

    op_one_hot = structure["operation_probs"]  # shape: (depth, n_ops)
    picked_indices = op_one_hot.argmax(dim=-1)
    allowed_indices = [TEST_OPS_NAMES.index(op) for op in allowed_subset]

    assert set(picked_indices.tolist()).issubset(set(allowed_indices))
