import pytest

from data.dagset.streaming import generate_random_dag_plan


def _has_divide_by_zero(initial_values, operations, tol: float = 1e-6) -> bool:
    """Return True if a divide operation would use (approximately) zero as the divisor."""
    for idx, op in enumerate(operations):
        if op == "divide" and abs(initial_values[idx + 1]) < tol:
            return True
    return False


@pytest.mark.parametrize("depth", [3, 6])
def test_divide_only_mode_never_divides_by_zero(depth: int) -> None:
    """When the only allowed op is 'divide' and all generated magnitudes default to 0, the
    guard should still ensure no /0 occurs (it turns such ops into identity and
    rewrites the corresponding initial value to 1.0).
    """
    for seed in range(500):
        initial_values, operations = generate_random_dag_plan(
            depth=depth,
            num_initial_values=depth + 1,
            seed=seed,
            max_digits=0,
            max_decimal_places=0,
            allowed_operations=["divide"],
        )
        assert not _has_divide_by_zero(initial_values, operations), (
            f"Divide-by-zero detected in divide-only mode for depth={depth}, seed={seed}: "
            f"initial_values={initial_values}, operations={operations}"
        )
