import torch

from models.dag_model import LOG_LIM, OP_NAMES, execute_stack  # type: ignore

# -----------------------------------------------------------------------------
# Helper utilities (duplicated locally to avoid cross-test dependencies)
# -----------------------------------------------------------------------------


def to_sign_log(values):
    """Convert list[float] to (sign, log10(abs)) tensors suitable for execute_stack."""
    arr = torch.tensor(values, dtype=torch.float32)
    sign = torch.sign(arr)
    sign[arr == 0] = 0.0  # ensure 0 has sign 0 not nan
    mag = arr.abs().clamp_min(1e-6)
    log_mag = torch.log10(mag)
    return sign, log_mag


def build_op_probs(depth, op_name):
    """Return op probability tensor (1,1,depth,n_ops) with given op one-hot."""
    n_ops = len(OP_NAMES)
    idx = OP_NAMES.index(op_name)
    op_probs = torch.zeros(1, 1, depth, n_ops, dtype=torch.float32)
    op_probs[:, :, :, idx] = 1.0
    return op_probs


# -----------------------------------------------------------------------------
# Edge-case regression tests
# -----------------------------------------------------------------------------


def test_deep_magnitude_clipping():
    """After removing _rms_rescale, verify logs remain clipped for deep multiply chains."""
    depth = 12  # reasonably deep but fast
    large_val = 1e3  # log10 = 3

    # Need depth+1 initial values for stack execution
    initial_values = [large_val] * (depth + 1)

    init_sgn, init_log = to_sign_log(initial_values)
    init_sgn = init_sgn.view(1, 1, -1)
    init_log = init_log.view(1, 1, -1)

    op_probs = build_op_probs(depth, "multiply")

    final_sgn, final_log = execute_stack(init_sgn, init_log, op_probs)

    # Ensure outputs are finite and clipped within ±LOG_LIM.
    assert torch.isfinite(final_sgn).all()
    assert torch.isfinite(final_log).all()
    assert abs(final_log.item()) <= LOG_LIM + 1e-5
