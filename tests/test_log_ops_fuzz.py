import math

import pytest
import torch

from dag_model import (LOG_LIM, _clip_log, add_log_space, divide_log_space,
                       multiply_log_space, subtract_log_space)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "op",
    [
        add_log_space,
        subtract_log_space,
        multiply_log_space,
        divide_log_space,
    ],
)
def test_log_op_fuzz(op, dtype):
    """Fuzz the log-space arithmetic ops to ensure finite outputs."""
    torch.manual_seed(0)
    for _ in range(1000):
        # Skip dtype if not supported on CPU for fp ops
        if dtype == torch.bfloat16 and not torch.tensor(0.0).bfloat16().is_cpu:
            pytest.skip("bfloat16 not supported on this CPU")

        # Random signs in {-1, 1}
        sx = (torch.randint(0, 2, (128,), dtype=torch.int32) * 2 - 1).to(dtype)
        sy = (torch.randint(0, 2, (128,), dtype=torch.int32) * 2 - 1).to(dtype)
        # Random log magnitudes in [0, LOG_LIM]
        lx = (torch.rand(128) * LOG_LIM).to(dtype)
        ly = (torch.rand(128) * LOG_LIM).to(dtype)

        try:
            s_out, l_out = op(sx, lx, sy, ly)
        except RuntimeError as e:
            # Some math functions may not support bfloat16 on CPU; skip gracefully
            if dtype == torch.bfloat16:
                pytest.skip(f"Op {op.__name__} not supported in bfloat16: {e}")
            raise

        s_out_f = s_out.to(torch.float32)
        l_out_f = l_out.to(torch.float32)

        assert torch.isfinite(
            s_out_f
        ).all(), f"NaN/inf sign output in {op.__name__}({dtype})"
        assert torch.isfinite(
            l_out_f
        ).all(), f"NaN/inf log output in {op.__name__}({dtype})"
        assert (
            l_out_f.abs() <= LOG_LIM + 1e-5
        ).all(), f"Log out of bounds in {op.__name__}({dtype})"
