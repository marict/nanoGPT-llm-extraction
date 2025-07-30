from __future__ import annotations

"""Utility helpers shared by DAG predictor training and evaluation.
Moving these functions into their own module drastically reduces the size of
`train_predictor.py` while keeping behaviour unchanged.
"""

import random as _eval_random
from typing import Dict, List

import torch
import torch.nn.functional as F
from tiktoken import get_encoding

from models.dag_model import OP_NAMES, execute_stack
from tensor_utils import digits_to_magnitude

__all__ = [
    "tokenize_texts",
    "compute_dag_structure_loss",
]


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #


def tokenize_texts(texts: List[str], sequence_length: int, device: str) -> torch.Tensor:
    """Tokenise a list of mathematical expressions to fixed-length ID tensors.

    This helper is intentionally minimal and mirrors the original logic found in
    ``train_predictor.py`` – truncating or left-padding with ``0`` to achieve a
    shape of ``(batch, sequence_length)``.
    """
    enc = get_encoding("gpt2")

    batch_size = len(texts)
    tokens = torch.zeros((batch_size, sequence_length), dtype=torch.long, device=device)
    for i, text in enumerate(texts):
        ids = enc.encode_ordinary(text)[:sequence_length]
        if len(ids) < sequence_length:
            ids += [0] * (sequence_length - len(ids))
        tokens[i] = torch.tensor(ids, dtype=torch.long)
    return tokens


# --------------------------------------------------------------------------- #
# Loss computation helpers
# --------------------------------------------------------------------------- #


def _compute_sign_loss(
    pred_sign_logits: torch.Tensor,
    target_sgn: torch.Tensor,
) -> torch.Tensor:
    """Compute binary cross-entropy loss for sign prediction using logits.

    Args:
        pred_sign_logits: Raw sign logits (B,T,N)
        target_sgn: Target signs (B,T,N) in {-1,1}
    """
    # Convert target signs {-1,1} to binary targets {0,1}
    binary_targets = ((target_sgn > 0).float()).view(-1)
    pred_sign_logits_flat = pred_sign_logits.view(-1).to(torch.float32)

    # Use binary cross entropy directly (clean, no backwards compatibility)
    sign_loss = F.binary_cross_entropy_with_logits(
        pred_sign_logits_flat, binary_targets, reduction="mean"
    )
    return sign_loss


def _compute_digit_loss(
    pred_digit_logits: torch.Tensor,
    target_digits: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss for digit prediction."""

    # Unpack shape information -------------------------------------------------
    _, _, _, D, base = pred_digit_logits.shape

    # Sanity checks ------------------------------------------------------------
    if target_digits.shape[-2] != D:
        raise ValueError(
            "Shape mismatch between model-predicted digits and target digits: "
            f"predicted D={D} , target D={target_digits.shape[-2]}. "
            "Ensure that `max_digits` and `max_decimal_places` are set to the "
            "*same* values for both the dataset and the model (these values are "
            "propagated via the training config)."
        )

    if target_digits.shape[-1] != base:
        raise ValueError(
            "Shape mismatch between model-predicted base and target base: "
            f"predicted base={base} , target base={target_digits.shape[-1]}. "
            "Ensure that `base` is set to the same value for both the dataset and the model."
        )

    # Loss computation ---------------------------------------------------------
    logits_flat = pred_digit_logits.reshape(-1, base).to(
        torch.float32
    )  # (B*T*N*D,base)

    target_flat = target_digits.reshape(-1, base)
    row_sums = target_flat.sum(dim=-1)

    # Validation: every row must be a valid one-hot distribution
    if (row_sums == 0).any():
        offending = torch.nonzero(row_sums == 0).flatten()[:5].tolist()
        raise ValueError(
            "Encountered target digit rows with all zeros (invalid one-hot). "
            f"Example flat indices: {offending}. This indicates a bug in the "
            "dataset generation pipeline."
        )

    # Allow a tiny numerical tolerance when checking that rows sum to 1.0
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
        bad = torch.nonzero(
            ~torch.isclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
        )
        first_bad = bad.flatten()[:5].tolist()
        raise ValueError(
            "Target digit rows are expected to be one-hot (sum to 1). "
            f"Found rows that sum to values != 1. Example flat indices: {first_bad}."
        )

    target_idx = target_flat.argmax(dim=-1)

    # Standard cross-entropy over raw logits
    digit_loss = F.cross_entropy(logits_flat, target_idx, reduction="mean")

    return digit_loss


def _compute_op_loss(
    pred_op_logits: torch.Tensor,
    target_ops: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss for operation prediction using logits.

    Args:
        pred_op_logits: Operation logits (B,T,D,n_ops)
        target_ops: Target operation one-hot vectors (B,T,D,n_ops)
    """
    _, _, _, n_ops = pred_op_logits.shape
    target_idx = target_ops.view(-1, n_ops).argmax(dim=-1)
    pred_op_logits_flat = pred_op_logits.view(-1, n_ops).to(torch.float32)
    op_loss = F.cross_entropy(pred_op_logits_flat, target_idx, reduction="mean")
    return op_loss


# --------------------------------------------------------------------------- #
# Robust Huber helper
# --------------------------------------------------------------------------- #
def _robust_huber(err: torch.Tensor, beta: torch.Tensor | float) -> torch.Tensor:
    """Huber loss with adaptive beta scaling."""
    abs_e = err.abs()
    loss = torch.where(abs_e < beta, 0.5 * (abs_e**2) / beta, abs_e - 0.5 * beta)
    return loss.mean()


def _compute_value_loss(
    pred_digit_logits: torch.Tensor,
    target_initial_values: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """Magnitude-only value loss (log-space adaptive Huber + small absolute term)."""
    # Magnitude prediction from digit logits
    pred_digit_probs = F.softmax(pred_digit_logits.to(torch.float32), dim=-1)
    pred_mag = digits_to_magnitude(
        pred_digit_probs, cfg.max_digits, cfg.max_decimal_places, cfg.base
    )  # (B,T,N)

    tgt_mag = target_initial_values.abs().to(torch.float32)

    eps = 1e-6
    pred_ln = torch.log(pred_mag + eps)
    tgt_ln = torch.log(tgt_mag + eps)
    ln_err = pred_ln - tgt_ln
    beta_ln = torch.quantile(ln_err.detach().abs().flatten(), 0.9).clamp_min(1e-6)
    ln_loss = _robust_huber(ln_err, beta_ln)

    # Small absolute-space component to keep zeros stable
    abs_err = pred_mag - tgt_mag
    beta_abs = torch.quantile(abs_err.detach().abs().flatten(), 0.9).clamp_min(1e-6)
    abs_loss = _robust_huber(abs_err, beta_abs)

    # Scale to keep early-training value_loss in ~1 range
    return 0.001 * (ln_loss + abs_loss)


def _compute_exec_loss(
    pred_sgn: torch.Tensor,
    pred_digit_logits: torch.Tensor,
    pred_ops: torch.Tensor,
    target_final_exec: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """Execution loss on magnitudes only (log-space adaptive Huber + overflow penalty)."""
    pred_digit_probs = F.softmax(pred_digit_logits.to(torch.float32), dim=-1)

    # Stack execution returns sign & ln(|value|) – signs are included but
    # the loss ignores them (magnitude-only). Using real predicted signs
    # ensures gradients still flow through the sign branch.
    _, pred_ln = execute_stack(
        pred_sgn,
        pred_digit_probs,
        pred_ops,
        max_digits=cfg.max_digits,
        max_decimal_places=cfg.max_decimal_places,
        base=cfg.base,
        ignore_clip=True,
    )

    # Extremely tight bounds to prevent any extreme values
    # Much more aggressive clamping: tanh(x/2) * 5
    pred_ln_soft = torch.tanh(pred_ln / 2.0) * 5.0
    pred_mag = torch.exp(pred_ln_soft).reshape(-1)
    tgt_mag = target_final_exec.abs().reshape(-1).to(torch.float32)

    # Simple MSE loss in log space with tight error bounds ---------------
    eps = 1e-6
    pred_ln_flat = torch.log(pred_mag + eps)
    tgt_ln_flat = torch.log(tgt_mag + eps)

    # Clamp the log errors to prevent extreme values
    ln_err = torch.clamp(pred_ln_flat - tgt_ln_flat, -5.0, 5.0)

    # Simple MSE loss instead of Huber (more predictable)
    ln_loss = (ln_err**2).mean()

    # Simple absolute error with tight bounds ---------------------------
    abs_err = torch.clamp(pred_mag - tgt_mag, -100.0, 100.0)
    abs_loss = 0.001 * (abs_err**2).mean()  # Very small coefficient

    # No overflow penalty - let the tight bounds handle it

    # Simple combination with aggressive scaling
    raw_loss = ln_loss + abs_loss

    # Very conservative final scaling to ensure we stay well below 2.0
    return raw_loss / 20.0


def _compute_statistics_loss(
    pred_statistics: dict,  # dict with 'initial', 'intermediate', 'final' containing (B, T, num_stats) tensors
    target_statistics: dict,  # dict with 'initial', 'intermediate', 'final' containing (B, T, num_stats) tensors
) -> torch.Tensor:
    """Compute MSE loss for auxiliary statistical predictions (per-token)."""
    total_loss = 0.0
    # Compute loss for each statistics component
    for key in ["initial", "intermediate", "final"]:
        if key in pred_statistics and key in target_statistics:
            pred = pred_statistics[key].to(torch.float32)  # (B, T, num_stats)
            target = target_statistics[key].to(torch.float32)  # (B, T, num_stats)

            # Check for NaN/inf in inputs before computing loss
            if torch.isnan(pred).any() or torch.isnan(target).any():
                print(f"Warning: NaN detected in statistics {key} - skipping")
                continue
            if torch.isinf(pred).any() or torch.isinf(target).any():
                print(f"Warning: Inf detected in statistics {key} - skipping")
                continue

            # Debug: Print statistics ranges when loss is very high
            pred_min, pred_max = pred.min().item(), pred.max().item()
            target_min, target_max = target.min().item(), target.max().item()

            # MSE loss for this component (averaged over batch, sequence, and features)
            component_loss = F.mse_loss(pred, target)

            # Log details when component loss is extremely high
            if component_loss > 1e10:
                print(f"DEBUG: High {key} stats loss: {component_loss:.2e}")
                print(f"  Pred range: [{pred_min:.3e}, {pred_max:.3e}]")
                print(f"  Target range: [{target_min:.3e}, {target_max:.3e}]")
                print(f"  Mean abs diff: {(pred - target).abs().mean().item():.3e}")

            # Additional safety check
            if torch.isnan(component_loss) or torch.isinf(component_loss):
                print(f"Warning: NaN/Inf in statistics loss component {key} - skipping")
                continue

            total_loss += component_loss

    # Scale to target range: worst case (all predictions wrong) should give total stats loss = 2.0
    # We have 3 loss components (initial, intermediate, final), each contributing max 1.0 in worst case
    # So worst case total = 3.0, and we want 2.0, hence divide by 3.0/2.0 = 1.5
    scaled_loss = total_loss / 1.5

    # Final safety check - return 0 if loss is NaN/inf
    if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
        print(f"Warning: Final statistics loss is NaN/Inf, returning 0.0")
        # Try to get device from first available statistics tensor, fallback to CPU
        try:
            device = pred_statistics[next(iter(pred_statistics))].device
        except (StopIteration, KeyError):
            device = "cpu"
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    return scaled_loss


# --------------------------------------------------------------------------- #
# Main loss function
# --------------------------------------------------------------------------- #
def compute_dag_structure_loss(
    pred_sign_logits: torch.Tensor,  # (B,T,N) raw logits (not tanh-activated)
    pred_digit_logits: torch.Tensor,  # (B,T,N,D,base) raw logits (not probabilities)
    pred_op_logits: torch.Tensor,  # (B,T,depth,n_ops) raw logits (not probabilities)
    pred_statistics: dict,  # dict with 'initial', 'intermediate', 'final' containing (B,T,num_stats) tensors
    target_tensors: list[
        list[dict[str, torch.Tensor]]
    ],  # (B, T) nested list: target tensors for all positions (including zero DAGs)
    valid_mask: torch.Tensor,  # (B,T) boolean mask indicating which positions are valid
    cfg,  # DAGTrainConfig
    uncertainty_params: torch.Tensor,  # (6,) learnable log-variance parameters for uncertainty weighting
    loss_flags: dict = None,  # Optional override for cfg.loss_flags
) -> dict[str, torch.Tensor]:
    """Compute DAG-structure prediction loss across all valid tokens with masking.

    This function works with the streaming.py system that generates per-token targets
    and uses masking for invalid tokens.

    Args:
        pred_sign_logits: (B,T,N) predicted sign logits for all positions
        pred_digit_logits: (B,T,N,D,base) predicted digit logits for all positions
        pred_op_logits: (B,T,depth,n_ops) predicted operation logits for all positions
        pred_statistics: Dict with prediction statistics for all positions
        target_tensors: (B,T) nested list of target dicts (includes zero DAGs for invalid positions)
        valid_mask: (B,T) boolean mask where True indicates valid positions
        cfg: Training configuration
        uncertainty_params: Learnable uncertainty weighting parameters
        loss_flags: Optional loss component enablement flags

    Returns:
        Dict containing all loss components and total weighted loss
    """

    B, T = pred_sign_logits.shape[:2]
    device = pred_sign_logits.device

    # Check that we have valid mask with correct shape
    if valid_mask.shape != (B, T):
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} must match (B,T) = ({B},{T})"
        )

    # Count total valid positions across all sequences
    num_valid_total = valid_mask.sum().item()

    if num_valid_total == 0:
        # No valid positions - return zero losses
        zero_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        uncertainty_weights = torch.exp(-uncertainty_params).detach()

        return {
            "total_loss": zero_loss,
            "sign_loss": zero_loss,
            "digit_loss": zero_loss,
            "op_loss": zero_loss,
            "value_loss": zero_loss,
            "exec_loss": zero_loss,
            "stats_loss": zero_loss,
            "uncertainty_weights": uncertainty_weights,
        }

    # Extract predictions and targets for valid positions only
    valid_positions = torch.nonzero(valid_mask, as_tuple=False)  # (num_valid, 2)
    batch_indices = valid_positions[:, 0]  # (num_valid,)
    token_indices = valid_positions[:, 1]  # (num_valid,)

    # Extract predictions at valid positions
    valid_sign_logits = pred_sign_logits[batch_indices, token_indices]  # (num_valid, N)
    valid_digit_logits = pred_digit_logits[
        batch_indices, token_indices
    ]  # (num_valid, N, D, base)
    valid_op_logits = pred_op_logits[
        batch_indices, token_indices
    ]  # (num_valid, depth, n_ops)
    valid_statistics = {
        key: pred_statistics[key][
            batch_indices, token_indices
        ]  # (num_valid, num_stats)
        for key in pred_statistics.keys()
    }

    # Prepare target tensors - extract valid targets from nested structure
    if len(target_tensors) != B:
        raise ValueError(
            f"Number of batch sequences ({len(target_tensors)}) must match batch size ({B})"
        )

    # Check that each sequence has T target tensors
    for i, seq_targets in enumerate(target_tensors):
        if len(seq_targets) != T:
            raise ValueError(
                f"Sequence {i} has {len(seq_targets)} target tensors, expected {T}"
            )

    # Extract targets for valid positions only
    valid_targets = []
    for batch_idx, token_idx in zip(batch_indices, token_indices):
        target_dict = target_tensors[batch_idx.item()][token_idx.item()]
        valid_targets.append(target_dict)

    # Stack targets from all valid positions
    target_sgn = torch.stack(
        [t["target_initial_sgn"] for t in valid_targets]
    )  # (num_valid, N)
    target_digits = torch.stack(
        [t["target_initial_digits"] for t in valid_targets]
    )  # (num_valid, N, D, base)
    target_ops = torch.stack(
        [t["target_operation_probs"] for t in valid_targets]
    )  # (num_valid, depth, n_ops)
    target_initial_values = torch.stack(
        [t["target_initial_values"] for t in valid_targets]
    )  # (num_valid, N)
    target_final_exec = torch.tensor(
        [t["target_final_exec"] for t in valid_targets], device=device
    )  # (num_valid,)

    # Handle statistics targets
    target_statistics = {}
    stat_keys = ["initial", "intermediate", "final"]
    for key in stat_keys:
        if f"target_{key}_stats" in valid_targets[0]:
            target_statistics[key] = torch.stack(
                [t[f"target_{key}_stats"] for t in valid_targets]
            )

    # Add singleton sequence dimension for compatibility with existing loss functions
    valid_sign_logits = valid_sign_logits.unsqueeze(1)  # (num_valid, 1, N)
    valid_digit_logits = valid_digit_logits.unsqueeze(1)  # (num_valid, 1, N, D, base)
    valid_op_logits = valid_op_logits.unsqueeze(1)  # (num_valid, 1, depth, n_ops)
    valid_statistics = {
        key: value.unsqueeze(1)
        for key, value in valid_statistics.items()  # (num_valid, 1, num_stats)
    }

    target_sgn = target_sgn.unsqueeze(1)  # (num_valid, 1, N)
    target_digits = target_digits.unsqueeze(1)  # (num_valid, 1, N, D, base)
    target_ops = target_ops.unsqueeze(1)  # (num_valid, 1, depth, n_ops)
    target_initial_values = target_initial_values.unsqueeze(1)  # (num_valid, 1, N)
    target_final_exec = target_final_exec.unsqueeze(1)  # (num_valid, 1)
    target_statistics = {
        key: value.unsqueeze(1)
        for key, value in target_statistics.items()  # (num_valid, 1, num_stats)
    }

    # Compute individual loss components using valid predictions and targets
    sign_loss = _compute_sign_loss(valid_sign_logits, target_sgn)
    digit_loss = _compute_digit_loss(valid_digit_logits, target_digits)
    op_loss = _compute_op_loss(valid_op_logits, target_ops)

    # For value and execution losses, we need tanh-activated signs and probabilities
    valid_sgn = torch.tanh(valid_sign_logits)
    value_loss = _compute_value_loss(valid_digit_logits, target_initial_values, cfg)

    # For execution loss, we need probabilities, so convert logits
    valid_ops = torch.nn.functional.softmax(valid_op_logits, dim=-1)
    exec_loss = _compute_exec_loss(
        valid_sgn,
        valid_digit_logits,
        valid_ops,
        target_final_exec,
        cfg,
    )

    # Compute statistics loss
    stats_loss = _compute_statistics_loss(valid_statistics, target_statistics)

    # Use loss flags from config
    if loss_flags is None:
        loss_flags = (
            cfg.loss_flags
            if hasattr(cfg, "loss_flags")
            else {
                "sign": True,
                "digit": True,
                "op": True,
                "value": True,
                "exec": True,
                "stats": True,
            }
        )

    # Define loss names and values in order matching uncertainty_params
    loss_names = ["sign", "digit", "op", "value", "exec", "stats"]
    all_losses = [sign_loss, digit_loss, op_loss, value_loss, exec_loss, stats_loss]

    # Filter enabled losses and corresponding uncertainty parameters
    enabled_losses = []
    enabled_uncertainty_params = []

    for i, (name, loss_value) in enumerate(zip(loss_names, all_losses)):
        if loss_flags.get(name, True):  # Default to enabled if flag not specified
            enabled_losses.append(loss_value)
            enabled_uncertainty_params.append(uncertainty_params[i])

    # Combine enabled losses using learned uncertainty weighting
    if enabled_losses:
        # Stack enabled losses and uncertainty parameters
        losses_tensor = torch.stack(enabled_losses)
        enabled_uncertainty_tensor = torch.stack(enabled_uncertainty_params)

        # Apply learned uncertainty weighting: exp(-s_i) * L_i + s_i
        weighted_losses = (
            torch.exp(-enabled_uncertainty_tensor) * losses_tensor
            + enabled_uncertainty_tensor
        )
        total_loss = weighted_losses.sum()
    else:
        # If no losses are enabled, return zero loss
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    # For logging, compute the weighting factors for all losses (detached to avoid interfering with gradients)
    uncertainty_weights = torch.exp(-uncertainty_params).detach()

    return {
        "total_loss": total_loss,
        "sign_loss": sign_loss,
        "digit_loss": digit_loss,
        "op_loss": op_loss,
        "value_loss": value_loss,
        "exec_loss": exec_loss,
        "stats_loss": stats_loss,
        "uncertainty_weights": uncertainty_weights,
        "num_valid_tokens": torch.tensor(num_valid_total, device=device),
    }
