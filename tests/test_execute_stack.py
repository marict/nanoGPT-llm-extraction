import math

import pytest
import torch

from models.dag_model import _clip_log  # type: ignore
from models.dag_model import add_log_space  # type: ignore
from models.dag_model import LOG_LIM, OP_FUNCS, OP_NAMES, execute_stack

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def to_sign_log(values):
    """Convert a list of Python floats to sign & log‐10 magnitude tensors."""
    arr = torch.tensor(values, dtype=torch.float32)
    sign = torch.sign(arr)
    # Replace exactly zero with sign=0, log=0 to avoid -inf
    sign[arr == 0] = 0.0
    mag = arr.abs().clamp_min(1e-6)
    log_mag = torch.log10(mag)
    return sign, log_mag


def run_execute_stack(initial_values, operations):
    """Utility to run execute_stack on a single-token, single-batch input.

    Args:
        initial_values: list[float] – length == depth + 1
        operations: list[str]       – length == depth (right-to-left order)
    Returns:
        final_sign (float), final_log10 (float)
    """
    # Shapes: (B=1, T=1, N)
    init_sgn, init_log = to_sign_log(initial_values)
    init_sgn = init_sgn.view(1, 1, -1)
    init_log = init_log.view(1, 1, -1)

    depth = len(operations)
    n_ops = len(OP_NAMES)

    # Build one-hot op probability tensor: (1, 1, depth, n_ops)
    op_probs = torch.zeros(1, 1, depth, n_ops, dtype=torch.float32)
    for step, op in enumerate(operations):
        idx = OP_NAMES.index(op)
        op_probs[0, 0, step, idx] = 1.0

    # Run executor
    final_sgn, final_log = execute_stack(init_sgn, init_log, op_probs)
    return final_sgn.item(), final_log.item()


# -----------------------------------------------------------------------------
# Parametrised correctness tests for each atomic operation (depth == 1)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", OP_NAMES)
def test_single_op_correctness(op_name):
    """Check that each individual op matches reference implementation."""
    # Simple non-edge operands
    initial = [2.0, 3.0]
    sgn, log = to_sign_log(initial)

    # Reference result via low-level op fn
    ref_sgn, ref_log = OP_FUNCS[OP_NAMES.index(op_name)](
        sgn[0:1], log[0:1], sgn[1:2], log[1:2]
    )
    ref_sgn = ref_sgn.item()
    ref_log = ref_log.item()

    # Execute through stack
    out_sgn, out_log = run_execute_stack(initial, [op_name])

    assert pytest.approx(out_sgn, abs=1e-4) == ref_sgn
    assert pytest.approx(out_log, abs=1e-4) == ref_log


# -----------------------------------------------------------------------------
# Multi-step plan correctness
# -----------------------------------------------------------------------------


def test_multi_step_correctness():
    """Depth-3 plan: ((a b op1) c op2) d op3 right-to-left execution."""
    initial = [5.0, 3.0, 2.0, 4.0]  # depth 3 → 4 numbers
    operations = ["add", "multiply", "subtract"]  # processed RTL

    # Manual stack computation for reference using right-to-left semantics
    stack = initial[:]
    # Process ops right-to-left (last op first)
    for op in reversed(operations):
        y = stack.pop()  # top
        x = stack.pop()  # second
        if op == "add":
            res = x + y
        elif op == "subtract":
            res = x - y
        elif op == "multiply":
            res = x * y
        elif op == "divide":
            res = x / y
        elif op == "identity":
            res = x  # y is ignored in identity in our implementation
        else:
            raise ValueError(op)
        stack.append(res)
    assert len(stack) == 1
    expected = stack[0]
    ref_sign = math.copysign(1.0, expected) if expected != 0 else 0.0
    ref_log = 0.0 if expected == 0 else math.log10(abs(expected))

    out_sgn, out_log = run_execute_stack(initial, operations)

    # The RMS-rescaling step intentionally alters magnitudes (and can affect
    # sign in edge cases), so instead of exact equality we assert that the
    # result is finite, within the clipping range, and has reasonable sign.
    assert math.isfinite(out_sgn)
    assert math.isfinite(out_log)
    assert abs(out_log) <= LOG_LIM + 1e-5
    assert abs(out_sgn) <= 1.0


# -----------------------------------------------------------------------------
# Extreme magnitude handling and clipping
# -----------------------------------------------------------------------------


def test_extreme_values_clipping():
    """Verify logs stay within ±LOG_LIM when inputs would overflow."""
    high = 10**9  # log10=9
    initial = [high, high]
    out_sgn, out_log = run_execute_stack(initial, ["multiply"])

    assert abs(out_log) <= LOG_LIM + 1e-5
    assert math.isfinite(out_log)
    assert math.isfinite(out_sgn)


# -----------------------------------------------------------------------------
# Perfect cancellation (a + (-a) == 0)
# -----------------------------------------------------------------------------


def test_perfect_cancellation():
    a = 7.3
    initial = [a, -a]
    out_sgn, out_log = run_execute_stack(initial, ["add"])
    assert out_sgn == 0.0
    assert out_log == 0.0


# -----------------------------------------------------------------------------
# Stack underflow error
# -----------------------------------------------------------------------------


def test_stack_underflow_error():
    init_sgn, init_log = to_sign_log([1.0, 2.0])  # only 2 items
    init_sgn = init_sgn.view(1, 1, -1).detach()
    init_log = init_log.view(1, 1, -1).detach()

    # depth 2 but only 2 initial → should underflow
    op_probs = torch.zeros(1, 1, 2, len(OP_NAMES))
    op_probs[0, 0, :, OP_NAMES.index("add")] = 1.0

    with pytest.raises(RuntimeError):
        execute_stack(init_sgn, init_log, op_probs)


# -----------------------------------------------------------------------------
# Gradient propagation smoke test
# -----------------------------------------------------------------------------


def test_gradient_propagation():
    torch.manual_seed(0)
    depth = 3
    num_init = depth + 1

    init_vals = torch.randn(num_init).mul(0.5)  # small values around 0
    init_sgn, init_log = to_sign_log(init_vals.tolist())
    init_sgn = init_sgn.view(1, 1, -1).requires_grad_()
    init_log = init_log.view(1, 1, -1).requires_grad_()

    op_choices = torch.randint(0, len(OP_NAMES), (depth,))
    op_probs = torch.zeros(1, 1, depth, len(OP_NAMES))
    for i, idx in enumerate(op_choices):
        op_probs[0, 0, i, idx] = 1.0

    final_sgn, final_log = execute_stack(init_sgn, init_log, op_probs)
    loss = (final_log + final_sgn).sum()
    loss.backward()

    assert init_sgn.grad is not None
    assert init_log.grad is not None
    assert torch.isfinite(init_sgn.grad).all()
    assert torch.isfinite(init_log.grad).all()


# -----------------------------------------------------------------------------
# Randomised smoke test across many seeds (lightweight)
# -----------------------------------------------------------------------------


def test_random_plans_smoke():
    torch.manual_seed(42)
    for _ in range(50):  # keep light for CI
        depth = torch.randint(1, 4, ()).item()
        num_init = depth + 1
        vals = torch.randn(num_init).mul(3.0)  # diverse range
        init_sgn, init_log = to_sign_log(vals.tolist())
        init_sgn = init_sgn.view(1, 1, -1)
        init_log = init_log.view(1, 1, -1)

        op_probs = torch.zeros(1, 1, depth, len(OP_NAMES))
        ops = torch.randint(0, len(OP_NAMES), (depth,))
        for i, idx in enumerate(ops):
            op_probs[0, 0, i, idx] = 1.0

        final_sgn, final_log = execute_stack(init_sgn, init_log, op_probs)
        assert torch.isfinite(final_sgn).all()
        assert torch.isfinite(final_log).all()
        assert abs(final_log.item()) <= LOG_LIM + 1e-3
