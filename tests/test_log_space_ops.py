import math
import random

import torch

from dag_model import (add_log_space, divide_log_space, identity_log_space,
                       multiply_log_space, subtract_log_space)


def _encode(val: float):
    """Return (sign, log) representation used by the DAG ops."""
    if val == 0.0:
        # Represent zero with sign 0 and log 0 (value ignored when sign==0)
        return torch.tensor([0.0]), torch.tensor([0.0])
    sign = 1.0 if val > 0 else -1.0
    log = torch.tensor([math.log(abs(val))])
    return torch.tensor([sign]), log


def _decode(sign: torch.Tensor, log: torch.Tensor):
    return sign * torch.exp(log)


_OPS = [
    (add_log_space, lambda a, b: a + b, "add"),
    (subtract_log_space, lambda a, b: a - b, "sub"),
    (multiply_log_space, lambda a, b: a * b, "mul"),
    (divide_log_space, lambda a, b: a / b if b != 0 else float("inf"), "div"),
]


@torch.no_grad()
def test_log_space_ops_correctness():
    """Confirm all log-space ops reproduce real arithmetic (without clipping)."""
    rng = random.Random(0)
    test_vals = [rng.uniform(-50, 50) for _ in range(200)]

    # Ensure we have no near-zero denominators for division
    test_vals = [v for v in test_vals if abs(v) > 1e-3]

    for op_fn, ref_fn, name in _OPS:
        for a in test_vals[:50]:
            for b in test_vals[50:100]:
                # Skip invalid division by zero
                if name == "div" and b == 0:
                    continue

                sx, lx = _encode(a)
                sy, ly = _encode(b)

                # Test with ignore_clip=True for clean arithmetic verification
                s_out, l_out = op_fn(sx, lx, sy, ly, ignore_clip=True)
                val_out = _decode(s_out, l_out).item()

                # Ground truth without clipping
                ref_val = ref_fn(a, b)
                # Handle potential inf/NaN from invalid operations
                if not math.isfinite(ref_val):
                    continue

                # Direct comparison without clipping
                assert math.isclose(
                    val_out, ref_val, rel_tol=1e-5, abs_tol=1e-5
                ), f"Mismatch for {name}: {a} ? {b} → {val_out} vs {ref_val}"


@torch.no_grad()
def test_log_space_ops_with_clipping():
    """Test that clipping behavior is consistent (default behavior)."""
    # Test a case that would exceed clipping bounds
    large_val = 1e10
    sx, lx = _encode(large_val)
    sy, ly = _encode(large_val)

    # With clipping (default)
    s_clipped, l_clipped = multiply_log_space(sx, lx, sy, ly, ignore_clip=False)
    val_clipped = _decode(s_clipped, l_clipped).item()

    # Without clipping
    s_unclipped, l_unclipped = multiply_log_space(sx, lx, sy, ly, ignore_clip=True)
    val_unclipped = _decode(s_unclipped, l_unclipped).item()

    # Clipped result should be smaller than unclipped for large values
    assert val_clipped < val_unclipped, "Clipping should reduce large values"


def test_identity_log_space():
    """Identity op should return input unchanged."""
    for val in [-20.5, -3.2, 0.5, 10.0]:
        s, l = _encode(val)
        s_out, l_out = identity_log_space(s, l, ignore_clip=True)
        assert torch.allclose(s, s_out)
        assert torch.allclose(l, l_out)
