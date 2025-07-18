import torch

from models.dag_model import LOG_LIM, OP_NAMES, execute_stack  # type: ignore

# -----------------------------------------------------------------------------
# Helper utilities for new execute_stack signature
# -----------------------------------------------------------------------------


def float_to_digit_onehot(value: float, max_digits: int, max_decimal_places: int):

    limit = 10**max_digits - 10 ** (-max_decimal_places)
    abs_val = min(abs(value), limit)
    s = f"{abs_val:.{max_decimal_places}f}"
    int_part, frac_part = s.split(".")
    int_part = int_part.zfill(max_digits)[-max_digits:]
    frac_part = (frac_part + "0" * max_decimal_places)[:max_decimal_places]
    digits = int_part + frac_part
    one_hot = torch.zeros(len(digits), 10, dtype=torch.float32)
    for i, ch in enumerate(digits):
        one_hot[i, int(ch)] = 1.0
    return one_hot


def values_to_sign_and_digits(values, max_digits=4, max_decimal_places=6):

    B, T, N = 1, 1, len(values)
    sign_tensor = torch.tensor(
        [1.0 if v > 0 else -1.0 if v < 0 else 0.0 for v in values], dtype=torch.float32
    ).view(B, T, N)

    D = max_digits + max_decimal_places
    digit_probs = torch.zeros(B, T, N, D, 10, dtype=torch.float32)
    for idx, v in enumerate(values):
        digit_probs[0, 0, idx] = float_to_digit_onehot(
            v, max_digits, max_decimal_places
        )
    return sign_tensor, digit_probs


# -----------------------------------------------------------------------------
# Helper utilities (duplicated locally to avoid cross-test dependencies)
# -----------------------------------------------------------------------------


# Replace helper to build op probs remains same
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

    sign_tensor, digit_probs = values_to_sign_and_digits(initial_values)
    op_probs = build_op_probs(depth, "multiply")

    final_sgn, final_log = execute_stack(
        sign_tensor,
        digit_probs,
        op_probs,
        max_digits=4,
        max_decimal_places=6,
    )

    # Ensure outputs are finite and clipped within ±LOG_LIM.
    assert torch.isfinite(final_sgn).all()
    assert torch.isfinite(final_log).all()
    assert abs(final_log.item()) <= LOG_LIM + 1e-5
