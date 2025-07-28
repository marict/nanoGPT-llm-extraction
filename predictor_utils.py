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

from data.dagset.streaming import DAGExample
from models.dag_model import OP_NAMES, execute_stack
from predictor_config import DAGTrainConfig
from tensor_utils import digits_to_magnitude

__all__ = [
    "tokenize_texts",
    "compute_dag_structure_loss",
    "evaluate_dag_model",
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
    target_sgn: torch.Tensor,  # (B,N) - single target per batch
    target_digits: torch.Tensor,  # (B,N,D,base) - single target per batch
    target_ops: torch.Tensor,  # (B,depth,n_ops) - single target per batch
    target_initial_values: torch.Tensor,  # (B,N) - single target per batch
    target_final_exec: torch.Tensor,  # (B,) - single target per batch
    target_statistics: dict,  # dict with 'initial', 'intermediate', 'final' containing (B,num_stats) tensors
    final_token_pos: torch.Tensor,  # (B,) - final token position for each batch element
    cfg: DAGTrainConfig,
    uncertainty_params: torch.Tensor,  # (6,) learnable log-variance parameters for uncertainty weighting
    loss_flags: (
        Dict[str, bool] | None
    ) = None,  # Optional override for cfg.loss_flags (deprecated)
) -> Dict[str, torch.Tensor]:
    """Compute robust DAG-structure prediction loss with position masking.

    Args:
        loss_flags: Optional dictionary to selectively disable losses. Keys are:
            - "sign": Enable/disable sign loss
            - "digit": Enable/disable digit loss
            - "op": Enable/disable operation loss
            - "value": Enable/disable value loss
            - "exec": Enable/disable execution loss
            - "stats": Enable/disable statistics loss
            Defaults to None (all losses enabled). Missing keys default to enabled.
            Disabled losses are still computed for logging but excluded from total_loss.
            The uncertainty weighting is applied only to enabled losses.
    """

    B, T = pred_sign_logits.shape[:2]

    # Check that all final token positions are within sequence bounds
    if (final_token_pos >= T).any():
        invalid_positions = final_token_pos >= T
        invalid_indices = torch.nonzero(invalid_positions).flatten()
        max_invalid_pos = final_token_pos[invalid_positions].max().item()
        raise ValueError(
            f"Final token positions exceed sequence length. "
            f"Found {invalid_indices.numel()} examples with final_token_pos >= {T}. "
            f"Maximum invalid position: {max_invalid_pos}. "
            f"This indicates training examples longer than block_size ({T}). "
            f"Consider increasing block_size or filtering longer examples during data generation."
        )

    # Verify all positions are valid after validation (should always pass)
    if (final_token_pos < 0).any() or (final_token_pos >= T).any():
        invalid_low = (final_token_pos < 0).any()
        invalid_high = (final_token_pos >= T).any()
        raise RuntimeError(
            f"BUG: Invalid final_token_pos found after validation. "
            f"Negative positions: {invalid_low}, positions >= {T}: {invalid_high}. "
            f"This indicates a bug in the validation logic above."
        )

    # Extract predictions at final token positions using advanced indexing
    batch_indices = torch.arange(B, device=pred_sign_logits.device)
    selected_sign_logits = pred_sign_logits[batch_indices, final_token_pos]  # (B,N)
    selected_digit_logits = pred_digit_logits[
        batch_indices, final_token_pos
    ]  # (B,N,D,base)
    selected_op_logits = pred_op_logits[
        batch_indices, final_token_pos
    ]  # (B,depth,n_ops)
    selected_statistics = {
        key: pred_statistics[key][batch_indices, final_token_pos]  # (B,num_stats)
        for key in pred_statistics.keys()
    }

    # Add singleton sequence dimension for compatibility with helper functions
    selected_sign_logits = selected_sign_logits.unsqueeze(1)  # (B,1,N)
    selected_digit_logits = selected_digit_logits.unsqueeze(1)  # (B,1,N,D,base)
    selected_op_logits = selected_op_logits.unsqueeze(1)  # (B,1,depth,n_ops)
    selected_statistics = {
        key: value.unsqueeze(1)
        for key, value in selected_statistics.items()  # (B,1,num_stats)
    }

    # Add singleton sequence dimension to targets
    target_sgn_seq = target_sgn.unsqueeze(1)  # (B,1,N)
    target_digits_seq = target_digits.unsqueeze(1)  # (B,1,N,D,base)
    target_ops_seq = target_ops.unsqueeze(1)  # (B,1,depth,n_ops)
    target_initial_values_seq = target_initial_values.unsqueeze(1)  # (B,1,N)
    target_final_exec_seq = target_final_exec.unsqueeze(1)  # (B,1)
    target_statistics_seq = {
        key: value.unsqueeze(1)
        for key, value in target_statistics.items()  # (B,1,num_stats)
    }

    # Compute individual loss components using selected predictions
    sign_loss = _compute_sign_loss(selected_sign_logits, target_sgn_seq)
    digit_loss = _compute_digit_loss(selected_digit_logits, target_digits_seq)
    op_loss = _compute_op_loss(selected_op_logits, target_ops_seq)

    # For value and execution losses, we need tanh-activated signs and probabilities
    selected_sgn = torch.tanh(selected_sign_logits)
    value_loss = _compute_value_loss(
        selected_digit_logits, target_initial_values_seq, cfg
    )

    # For execution loss, we need probabilities, so convert logits
    selected_ops = F.softmax(selected_op_logits, dim=-1)
    exec_loss = _compute_exec_loss(
        selected_sgn,
        selected_digit_logits,
        selected_ops,
        target_final_exec_seq,
        cfg,
    )

    # Compute statistics loss
    stats_loss = _compute_statistics_loss(selected_statistics, target_statistics_seq)

    # Use loss flags from config, with parameter override for backwards compatibility
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
        device = sign_loss.device
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
    }


# --------------------------------------------------------------------------- #
# Utility helpers for digit tensors
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Validation / evaluation
# --------------------------------------------------------------------------- #


def evaluate_dag_model(
    model: torch.nn.Module,
    val_loader,
    device: str,
    ctx,
    cfg,
    eval_iters: int,
    seed: int,
) -> Dict[str, float]:
    """Run evaluation on *eval_iters* batches and return aggregated metrics."""
    model.eval()
    _eval_random.seed(seed)

    total_losses = {
        k: 0.0
        for k in (
            "total_loss",
            "sign_loss",
            "digit_loss",
            "op_loss",
            "value_loss",
            "exec_loss",
            "stats_loss",
        )
    }
    total_metrics = {
        "op_accuracy": 0.0,
        "full_dag_op_match": 0.0,
        "sign_accuracy": 0.0,
        "executed_mse": 0.0,
        "initial_values_mse": 0.0,
    }

    num_batches = 0
    with torch.no_grad():
        for i, (texts, structures, examples) in enumerate(val_loader):
            if i >= eval_iters:
                break

            # Targets → device
            tgt_sgn = structures["target_initial_sgn"].to(device)
            tgt_digits = structures["target_initial_digits"].to(device)
            tgt_ops = structures["target_operation_probs"].to(device)

            # Inputs
            input_tokens = tokenize_texts(texts, cfg.block_size, device)

            # Forward
            with ctx:
                if cfg.full_backbone and hasattr(model, "dag"):
                    hidden = model.forward_hidden(input_tokens)
                    pred_sgn, pred_digit_probs, pred_ops, pred_statistics = (
                        model.dag.plan_predictor(hidden)
                    )
                else:
                    pred_sgn, pred_digit_probs, pred_ops, pred_statistics = model(
                        input_tokens
                    )

                # ------------------------------------------------------------------
                # Retrieve digit logits using consistent access pattern
                # ------------------------------------------------------------------
                # Use consistent dag_predictor property for all model types
                dag_predictor = model.dag_predictor
                if dag_predictor is None:
                    raise RuntimeError("Model has no DAG predictor (dag_depth=0)")

                # Get raw logits for all positions
                pred_sign_logits = dag_predictor.last_sign_logits
                pred_digit_logits = dag_predictor.last_digit_logits
                pred_op_logits = dag_predictor.last_operation_logits
                # Extract learnable uncertainty weighting parameters
                uncertainty_params = dag_predictor.uncertainty_params

                if pred_digit_logits is None:
                    raise RuntimeError("last_digit_logits not found for evaluation")
                if pred_sign_logits is None:
                    raise RuntimeError("last_sign_logits not found for evaluation")
                if pred_op_logits is None:
                    raise RuntimeError("last_operation_logits not found for evaluation")

                # Extract final token positions
                final_token_pos = structures["final_token_pos"].to(device)
                # Targets for loss calculation
                tgt_sgn = structures["target_initial_sgn"].to(device)
                tgt_digits = structures["target_initial_digits"].to(device)
                tgt_ops = structures["target_operation_probs"].to(device)

                # Extract remaining targets
                target_initial_values = structures["target_initial_values"].to(device)
                target_final_exec = structures["target_final_exec"].to(device)
                target_statistics = {
                    "initial": structures["target_initial_stats"].to(device),
                    "intermediate": structures["target_intermediate_stats"].to(device),
                    "final": structures["target_final_stats"].to(device),
                }

                # Compute loss using new signature with final token positions
                losses = compute_dag_structure_loss(
                    pred_sign_logits,
                    pred_digit_logits,
                    pred_op_logits,
                    pred_statistics,
                    tgt_sgn,
                    tgt_digits,
                    tgt_ops,
                    target_initial_values,
                    target_final_exec,
                    target_statistics,
                    final_token_pos,
                    cfg,
                    uncertainty_params=uncertainty_params,
                )

                # Extract predictions at final positions for metrics
                B = pred_sign_logits.shape[0]
                batch_indices = torch.arange(B, device=pred_sign_logits.device)
                final_token_pos_clamped = torch.clamp(
                    final_token_pos, 0, pred_sign_logits.shape[1] - 1
                )

                pred_ops_final = pred_op_logits[
                    batch_indices, final_token_pos_clamped
                ]  # (B, depth, n_ops)
                pred_sgn_final = pred_sign_logits[
                    batch_indices, final_token_pos_clamped
                ]  # (B, N)

                # Convert to probabilities and tanh-activated signs for metrics
                pred_ops = F.softmax(pred_ops_final, dim=-1)
                pred_sgn = torch.tanh(pred_sgn_final)

                # Metrics
                op_correct = pred_ops.argmax(dim=-1).eq(tgt_ops.argmax(dim=-1))
                op_acc = op_correct.float().mean()
                full_match = op_correct.all(dim=-1).float().mean()

                sign_correct = torch.sign(pred_sgn).eq(torch.sign(tgt_sgn))
                sign_acc = sign_correct.float().mean()

                # ------------------------------------------------------------------ #
                # Execute full DAGs to obtain scalar answers and compute MSE          #
                # ------------------------------------------------------------------ #

                # Target tensors --------------------------------------------------
                tgt_sign = tgt_sgn.unsqueeze(1)  # (B,1,N)
                tgt_digit_probs = tgt_digits.unsqueeze(1)  # (B,1,N,D,base)
                tgt_ops_seq = tgt_ops.unsqueeze(1)  # (B,1,depth,n_ops)

                # Prediction tensors ---------------------------------------------
                pred_sign = pred_sgn.unsqueeze(1)  # (B,1,N)
                pred_digit_probs_final = pred_digit_logits[
                    batch_indices, final_token_pos_clamped
                ]  # (B,N,D,base)
                pred_digit_probs = F.softmax(pred_digit_probs_final, dim=-1).unsqueeze(
                    1
                )  # (B,1,N,D,base)
                pred_ops_seq = pred_ops.unsqueeze(1)  # (B,1,depth,n_ops)

                # Execute stacks - use clipping for consistency with training
                tgt_final_sgn, tgt_final_log = execute_stack(
                    tgt_sign,
                    tgt_digit_probs,
                    tgt_ops_seq,
                    max_digits=cfg.max_digits,
                    max_decimal_places=cfg.max_decimal_places,
                    base=cfg.base,
                    ignore_clip=False,  # Consistent with training behavior
                )

                pred_final_sgn, pred_final_log = execute_stack(
                    pred_sign,
                    pred_digit_probs,
                    pred_ops_seq,
                    max_digits=cfg.max_digits,
                    max_decimal_places=cfg.max_decimal_places,
                    base=cfg.base,
                    ignore_clip=False,  # Consistent with training behavior
                )

                # Convert to real numbers
                tgt_final_val = tgt_final_sgn * torch.exp(tgt_final_log)
                pred_final_val = pred_final_sgn * torch.exp(pred_final_log)

                executed_mse = F.mse_loss(pred_final_val, tgt_final_val)

                # ------------------------------------------------------------------ #
                # Compute MSE between initial values of target and predicted DAGs    #
                # ------------------------------------------------------------------ #

                # Convert predicted digits to magnitudes for all samples in batch
                pred_magnitudes = digits_to_magnitude(
                    pred_digit_probs.squeeze(1),  # (B, N, D, base)
                    cfg.max_digits,
                    cfg.max_decimal_places,
                    cfg.base,
                )  # (B, N)

                # Combine sign and magnitude to get predicted initial values
                pred_initial_values = (
                    torch.sign(pred_sgn.squeeze(1)) * pred_magnitudes
                )  # (B, N)

                # Get target initial values (remove sequence dimension if present)
                tgt_initial_values = target_initial_values.squeeze(1)  # (B, N)

                # Compute MSE between initial values
                initial_values_mse = F.mse_loss(pred_initial_values, tgt_initial_values)

                # -------------------------------------------------------------- #
                # Console debug: print the last sample from the batch
                # -------------------------------------------------------------- #
                if i == 0:
                    batch_size = tgt_sgn.size(0)
                    batch_idx = batch_size - 1
                    sample_text = texts[batch_idx]
                    sample_obj: DAGExample = examples[batch_idx]
                    sample_seed = sample_obj.seed
                    did_expand = sample_obj.did_expand
                    did_simplify = sample_obj.did_simplify

                    # Sign vectors (N,) and digit logits (N,D,10)
                    pred_sign_vec = pred_sgn.squeeze(1)[batch_idx]

                    pred_digits_vec = pred_digit_logits[
                        batch_indices, final_token_pos_clamped
                    ][batch_idx].softmax(dim=-1)

                    # Convert digit distributions to magnitudes
                    pred_mag_vec = digits_to_magnitude(
                        pred_digits_vec,
                        cfg.max_digits,
                        cfg.max_decimal_places,
                        cfg.base,
                    )

                    pred_real_vals = (
                        (torch.sign(pred_sign_vec) * pred_mag_vec).cpu().tolist()
                    )

                    # Decode operations
                    tgt_ops_row = tgt_ops[batch_idx]  # (depth, n_ops)
                    pred_ops_row = pred_ops.squeeze(1)[batch_idx]
                    tgt_op_indices = tgt_ops_row.argmax(dim=-1).cpu().tolist()
                    pred_op_indices = pred_ops_row.argmax(dim=-1).cpu().tolist()
                    tgt_op_names = [OP_NAMES[idx] for idx in tgt_op_indices]
                    pred_op_names = [OP_NAMES[idx] for idx in pred_op_indices]

                    print("\n=== Validation Sample ===")
                    print(f"Sample RNG seed: {sample_seed}")
                    print(f"Depth: {sample_obj.depth}")
                    print(f"Max digits: {sample_obj.max_digits}")
                    print(f"Base: {sample_obj.base}")
                    print(f"Max decimal places: {sample_obj.max_decimal_places}")
                    print(f"Printing style: {sample_obj.printing_style}")
                    print(
                        f"English conversion probability: {sample_obj.english_conversion_probability}"
                    )
                    print(
                        f"Integer no decimal probability: {sample_obj.integer_no_decimal_probability}"
                    )
                    print(f"Text:\n-------\n{sample_text}\n-------\n")
                    print(f"Sympy expression: {sample_obj.expr}")
                    print(f"Did expand: {did_expand}")
                    print(f"Did simplify: {did_simplify}")
                    print(f"Sympy execution value: {sample_obj.final_value_sympy}")
                    print(
                        f"Target stack execution value: {sample_obj.structure_dict['target_final_exec']}"
                    )
                    print(
                        f"Predicted stack execution value: {pred_final_val[batch_idx].item()}"
                    )
                    print("Predicted sign logits:")
                    print(pred_sign_vec.cpu().tolist())
                    # Print number of tokens in the sample text to check context length
                    enc = get_encoding("gpt2")
                    token_count = len(enc.encode_ordinary(sample_text))
                    print(f"Token count: {token_count}")
                    # If we have the raw DAGExample, use its original floats for nicer printing
                    true_vals = (
                        structures["target_initial_values"][batch_idx].cpu().tolist()
                    )
                    print(
                        f"Target initial values (rounded to {cfg.max_decimal_places} dp):"
                    )
                    print([round(v, cfg.max_decimal_places) for v in true_vals])
                    print(
                        f"Predicted initial values (rounded to {cfg.max_decimal_places} dp):"
                    )
                    print([round(v, cfg.max_decimal_places) for v in pred_real_vals])
                    print("Operations (ground truth):")
                    print(tgt_op_names)
                    print("Operations (predicted):")
                    print(pred_op_names)
                    print(f"Allowed operations: {sample_obj.allowed_operations}")
                    print("==========================\n")

            # Aggregate losses (skip uncertainty_weights which is for logging only)
            for k, v in losses.items():
                if k != "uncertainty_weights":
                    total_losses[k] += v.item()
            total_metrics["op_accuracy"] += op_acc.item()
            total_metrics["full_dag_op_match"] += full_match.item()
            total_metrics["sign_accuracy"] += sign_acc.item()
            total_metrics["executed_mse"] += executed_mse.item()
            total_metrics["initial_values_mse"] += initial_values_mse.item()
            num_batches += 1

    if num_batches:
        for d in (total_losses, total_metrics):
            for k in d:
                d[k] /= num_batches

    model.train()
    return {**total_losses, **total_metrics}
