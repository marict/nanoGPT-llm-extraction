import math
import random

import pytest
import torch

from data.dagset.streaming import OP_NAMES, generate_random_dag_plan, pad_plan


@pytest.mark.parametrize(
    "depth, p, n, tol", [(5, 0.3, 10000, 0.05), (3, 0.0, 5000, 0.02)]
)
def test_identity_frequency(depth, p, n, tol):
    """Verify that the probability of generating an identity matches *identity_cutoff_p* (within *tol*).
    The test is repeated for p=0.3 (typical use-case) and the degenerate case p=0.0.
    """
    identities = 0
    for seed in range(n):
        _ivs, ops = generate_random_dag_plan(
            depth=depth,
            num_initial_values=depth + 1,
            seed=seed,
            identity_cutoff_p=p,
        )
        if "identity" in ops:
            identities += 1
    freq = identities / n
    assert math.isclose(
        freq, p, abs_tol=tol
    ), f"Identity frequency {freq:.3f} outside tolerance for p={p}"


def test_non_identity_distribution_uniform():
    """When *identity_cutoff_p*=0 the distribution over non-identity operations should remain uniform."""
    depth = 4
    n = 20000
    counts = {op: 0 for op in OP_NAMES if op != "identity"}
    for seed in range(n):
        _ivs, ops = generate_random_dag_plan(
            depth=depth,
            num_initial_values=depth + 1,
            seed=seed,
            identity_cutoff_p=0.0,
        )
        for op in ops:
            if op != "identity":
                counts[op] += 1
    total_non_identity = sum(counts.values())
    expected = total_non_identity / len(counts)
    for op, c in counts.items():
        assert (
            abs(c - expected) / expected < 0.05
        ), f"Non-identity op '{op}' frequency deviates more than 5% from uniform expectation"


def _evaluate_plan(initial_values, operations):
    """Evaluate a DAG plan using the same stack semantics as the generator."""
    stack = initial_values.copy()
    for op in reversed(operations):
        b = stack.pop()
        a = stack.pop()
        if op == "add":
            stack.append(a + b)
        elif op == "subtract":
            stack.append(a - b)
        elif op == "multiply":
            stack.append(a * b)
        elif op == "divide":
            # Protect against division by zero in tests
            denom = b if b != 0 else 1e-12
            stack.append(a / denom)
        elif op == "identity":
            stack.append(a)
        else:
            raise ValueError(f"Unknown operation: {op}")
    # The generator may leave additional constants on the stack when an early
    # identity truncates the plan.  The meaningful result is always the *top*
    # of the stack (index 0 after generation), matching the behavior in
    # `plan_to_string_expression`.
    return stack[0]


def test_forward_equivalence_with_padding():
    """The scalar result is identical whether or not post-identity operations are present."""
    rng = random.Random(42)
    samples = 1000
    for i in range(samples):
        depth = rng.randint(2, 6)
        ivs, ops = generate_random_dag_plan(
            depth=depth,
            num_initial_values=depth + 1,
            seed=i,
            identity_cutoff_p=1.0,  # guarantee a cutoff for robustness
        )
        # Evaluate truncated plan
        res_trunc = _evaluate_plan(ivs.copy(), ops.copy())

        # Pad the plan after the first identity to full length using the helper
        padded_vals, padded_ops = pad_plan(ivs.copy(), ops.copy())
        res_padded = _evaluate_plan(padded_vals, padded_ops)

        assert math.isclose(
            res_trunc, res_padded, rel_tol=1e-6, abs_tol=1e-6
        ), f"Evaluation mismatch (trunc={res_trunc}, padded={res_padded})"
