from typing import Tuple

import torch

from models.dag_model import (LOG_LIM, DAGPlanPredictor, GPTConfig,
                              add_log_space, safe_clamp)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _build_plan_predictor(depth: int = 2, n_embd: int = 32) -> DAGPlanPredictor:
    """Return a tiny `DAGPlanPredictor` in eval-mode for unit tests."""

    cfg = GPTConfig(
        vocab_size=50,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=n_embd,
        dropout=0.0,
        bias=False,
        dag_depth=depth,
    )
    predictor = DAGPlanPredictor(cfg)
    predictor.eval()
    return predictor


# -----------------------------------------------------------------------------
# 1. Causal-mask correctness
# -----------------------------------------------------------------------------


def test_causal_mask_invariance():
    """Changing future hidden states must NOT affect earlier predictions."""

    predictor = _build_plan_predictor(depth=2)

    B, T, H = 1, 4, predictor.n_embd
    hidden_base = torch.randn(B, T, H)

    # Clone and perturb ONLY the *last* token
    hidden_modified = hidden_base.clone()
    hidden_modified[..., -1, :] += torch.randn_like(hidden_modified[..., -1, :]) * 10.0

    with torch.no_grad():
        sign_ref, digits_ref, ops_ref = predictor(hidden_base)
        sign_mod, digits_mod, ops_mod = predictor(hidden_modified)

    assert torch.allclose(
        sign_ref[..., :-1, :], sign_mod[..., :-1, :], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        digits_ref[..., :-1, :, :], digits_mod[..., :-1, :, :], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        ops_ref[..., :-1, :, :], ops_mod[..., :-1, :, :], atol=1e-5, rtol=1e-5
    ), "Causal mask failed – future information leaked into past predictions."


# -----------------------------------------------------------------------------
# 2. Numerical stability of `safe_clamp` and softmax pathway
# -----------------------------------------------------------------------------


def test_operation_probs_finite_and_normalised_after_large_logits():
    """Feeding huge hidden activations should *not* create NaNs/Inf in op probs."""

    predictor = _build_plan_predictor(depth=3)

    B, T, H = 2, 6, predictor.n_embd
    hidden = torch.randn(B, T, H) * 1e4  # Enormous magnitude ⇒ very large logits

    with torch.no_grad():
        _sgn, _log, op_probs = predictor(hidden)

    # All probabilities must be finite and each row should sum to ~1
    assert torch.isfinite(
        op_probs
    ).all(), "Non-finite values encountered in operation probs"

    row_sums = op_probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# -----------------------------------------------------------------------------
# 3. Log-space arithmetic clipping
# -----------------------------------------------------------------------------


def _random_sign(size: Tuple[int, ...]) -> torch.Tensor:
    return torch.randint(0, 2, size).float() * 2 - 1  # ±1


def test_clip_log_bounds():
    """`add_log_space` output must respect ±LOG_LIM bound regardless of inputs."""

    torch.manual_seed(0)
    size = (4, 5)
    # Logs far outside the representable range
    l1 = torch.full(size, 50.0)
    l2 = torch.full(size, 60.0)
    s1 = _random_sign(size)
    s2 = _random_sign(size)

    _, l_out = add_log_space(s1, l1, s2, l2)

    assert torch.isfinite(l_out).all(), "Non-finite value in clipped log output"
    assert (
        l_out.abs() <= LOG_LIM + 1e-6
    ).all(), "Clipping failed – magnitude exceeds LOG_LIM"


# -----------------------------------------------------------------------------
# 4. `safe_clamp` stand-alone behaviour
# -----------------------------------------------------------------------------


def test_safe_clamp_respects_dtype_thresholds():
    """`safe_clamp` must not exceed its dtype-specific maximums."""

    for dtype, expected_max in [
        (torch.float16, 8.0),
        (torch.bfloat16, 20.0),
        (torch.float32, 40.0),
    ]:
        tensor = torch.tensor([-1000.0, 0.0, 1000.0], dtype=dtype)
        clamped = safe_clamp(tensor)
        assert clamped.max().item() <= expected_max + 1e-3
        assert clamped.min().item() >= -expected_max - 1e-3
