import math

import pytest

from data.dagset.streaming import plan_to_tensors
from models.dag_model import execute_stack


@pytest.mark.parametrize(
    "test_case",
    [
        # Test case format: (initial_values, operations, expected_value)
        (
            [2.0, 3.0, 1.0],
            ["identity", "add"],
            2.0,
        ),  # Identity preserves value before add
        (
            [2.0, 3.0, 1.0],
            ["add", "identity"],
            5.0,
        ),  # Identity preserves value after add
        ([12.23, 1.0, 1.0], ["identity", "identity"], 12.23),  # Chain of 2 identities
        (
            [12.23, 1.0, 1.0, 1.0],
            ["identity", "identity", "identity"],
            12.23,
        ),  # Chain of 3 identities
        (
            [-98.506, 1.0, 1.0, 1.0],
            ["identity", "identity", "identity"],
            -98.506,
        ),  # Regression test case
    ],
)
def test_identity_operations(test_case):
    """Test identity operations in various positions and chains."""
    initial_values, operations, expected = test_case

    # Convert plan to tensors using existing helper
    structure_dict = plan_to_tensors(
        initial_values=initial_values,
        operations=operations,
        max_digits=2,
        max_decimal_places=3,
    )

    # Extract individual tensors from structure dict
    signs = structure_dict["initial_sgn"][: len(initial_values)]  # trim padding
    digits = structure_dict["initial_digits"][: len(initial_values)]  # trim padding
    ops = structure_dict["operations"]  # original operations tensor

    # Add batch and sequence dimensions
    sign_tensor = signs.view(1, 1, -1)
    digit_probs = digits.unsqueeze(0).unsqueeze(0)
    op_probs = ops.unsqueeze(0).unsqueeze(0)

    # Run stack execution
    final_sgn, final_log = execute_stack(
        sign_tensor,
        digit_probs,
        op_probs,
        max_digits=2,
        max_decimal_places=3,
        base=10,
        ignore_clip=True,
    )

    # Convert result to value and verify
    result = final_sgn.item() * math.exp(final_log.item())
    assert math.isclose(result, expected, rel_tol=1e-4, abs_tol=1e-4)
