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

__all__ = [
    "tokenize_texts",
    "compute_dag_structure_loss",
    "evaluate_dag_model",
    "digits_to_magnitude",
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
    device_type: str,
) -> torch.Tensor:
    """Compute cross-entropy loss for sign prediction using logits.

    Args:
        pred_sign_logits: Raw sign logits (B,T,N)
        target_sgn: Target signs (B,T,N) in {-1,1}
        device_type: Device type for autocast
    """
    with torch.amp.autocast(device_type=device_type, enabled=False):
        # Convert target signs {-1,1} to class indices {0,1}
        sign_target_idx = ((target_sgn > 0).long()).view(-1)
        pred_sign_logits_flat = pred_sign_logits.view(-1, 1).to(torch.float32)

        # Create 2-class logits: [negative_logit, positive_logit]
        # For binary classification: logit for class 1 vs class 0
        binary_logits = torch.cat(
            [-pred_sign_logits_flat, pred_sign_logits_flat], dim=1
        )
        sign_loss = F.cross_entropy(binary_logits, sign_target_idx, reduction="mean")
    return sign_loss


def _compute_digit_loss(
    pred_digit_logits: torch.Tensor,
    target_digits: torch.Tensor,
    device_type: str,
) -> torch.Tensor:
    """Compute cross-entropy loss for digit prediction.

    Key changes compared to the previous implementation:
    1. Positions whose target vector is all zeros are *not* ignored. They are
       interpreted as the digit `0` class, providing the model with an explicit
       training signal to predict zeros when no digit information is present.
    2. Accepts arbitrary *args / **kwargs so that callers that (mistakenly) pass
       additional arguments such as `iter_num` or `cfg` will not crash.
    """

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
    with torch.amp.autocast(device_type=device_type, enabled=False):
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
    device_type: str,
) -> torch.Tensor:
    """Compute cross-entropy loss for operation prediction using logits.

    Args:
        pred_op_logits: Operation logits (B,T,D,n_ops)
        target_ops: Target operation one-hot vectors (B,T,D,n_ops)
        device_type: Device type for autocast
    """
    b, t, d, n_ops = pred_op_logits.shape
    with torch.amp.autocast(device_type=device_type, enabled=False):
        target_idx = target_ops.view(-1, n_ops).argmax(dim=-1)
        pred_op_logits_flat = pred_op_logits.view(-1, n_ops).to(torch.float32)
        op_loss = F.cross_entropy(pred_op_logits_flat, target_idx, reduction="mean")
    return op_loss


def _compute_value_loss(
    pred_sgn: torch.Tensor,
    pred_digit_logits: torch.Tensor,
    target_initial_values: torch.Tensor,
    cfg,
    device_type: str,
) -> torch.Tensor:
    """Compute robust loss between predicted and target initial values using direct regression."""
    with torch.amp.autocast(device_type=device_type, enabled=False):
        # Convert logits to probabilities for magnitude computation
        pred_digit_probs = F.softmax(pred_digit_logits.to(torch.float32), dim=-1)

        # Compute predicted magnitudes using centralized function
        pred_magnitude = digits_to_magnitude(
            pred_digit_probs, cfg.max_digits, cfg.max_decimal_places, cfg.base
        )  # (B,T,N)

        # Reconstruct full signed predicted values
        # Use continuous signs directly (maintains differentiability)
        pred_values = pred_sgn.to(torch.float32) * pred_magnitude

        # Direct regression loss on signed values
        # Use smooth L1 for robustness to outliers
        direct_loss = F.smooth_l1_loss(
            pred_values, target_initial_values.to(torch.float32)
        )

        # Optional: Add relative error component for values with different scales
        # This helps when we have both small (0.01) and large (1000) values
        epsilon = 1e-8
        relative_error = (pred_values - target_initial_values.to(torch.float32)) / (
            target_initial_values.abs() + epsilon
        )
        relative_loss = F.smooth_l1_loss(
            relative_error, torch.zeros_like(relative_error)
        )

        # Combine direct and relative losses
        return direct_loss + 0.1 * relative_loss


def _compute_exec_loss(
    pred_sgn: torch.Tensor,
    pred_digit_logits: torch.Tensor,
    pred_ops: torch.Tensor,
    target_final_exec: torch.Tensor,
    cfg,
    device_type: str,
) -> torch.Tensor:
    """Execution loss using direct regression on signed execution results."""
    with torch.amp.autocast(device_type=device_type, enabled=False):
        # Convert logits to probabilities for execution
        pred_digit_probs = F.softmax(pred_digit_logits.to(torch.float32), dim=-1)

        # Execute: returns sign tensor and ln(|value|) tensor
        pred_final_sgn, pred_final_ln = execute_stack(
            pred_sgn,
            pred_digit_probs,
            pred_ops,
            max_digits=cfg.max_digits,
            max_decimal_places=cfg.max_decimal_places,
            base=cfg.base,
            ignore_clip=True,  # raw for loss
        )

        # Convert from log space back to actual values
        pred_final_magnitude = torch.exp(
            pred_final_ln.clamp(max=50.0)
        )  # Clamp to prevent overflow

        # Use continuous signs directly (maintains differentiability)
        pred_final_values = pred_final_sgn.to(torch.float32) * pred_final_magnitude

        # Flatten for loss computation
        pred_final_values = pred_final_values.reshape(-1).to(torch.float32)
        target_flat = target_final_exec.reshape(-1).to(torch.float32)

        # Direct regression loss on signed execution results
        direct_loss = F.smooth_l1_loss(pred_final_values, target_flat)

        # Add relative error component for scale robustness
        epsilon = 1e-8
        relative_error = (pred_final_values - target_flat) / (
            target_flat.abs() + epsilon
        )
        relative_loss = F.smooth_l1_loss(
            relative_error, torch.zeros_like(relative_error)
        )

        # Add overflow penalty to prevent extreme predictions
        overflow_threshold = 1e10  # Much simpler threshold in actual value space
        overflow_penalty = (pred_final_values.abs() > overflow_threshold).float().mean()

        return direct_loss + 0.1 * relative_loss + 0.05 * overflow_penalty


# --------------------------------------------------------------------------- #
# Main loss function
# --------------------------------------------------------------------------- #
def compute_dag_structure_loss(
    pred_sign_logits: torch.Tensor,  # (B,T,N) raw logits (not tanh-activated)
    pred_digit_logits: torch.Tensor,  # (B,T,N,D,base) raw logits (not probabilities)
    pred_op_logits: torch.Tensor,  # (B,T,depth,n_ops) raw logits (not probabilities)
    target_sgn: torch.Tensor,  # (B,T,N)
    target_digits: torch.Tensor,  # (B,T,N,D,base) one-hot
    target_ops: torch.Tensor,
    target_initial_values: torch.Tensor,  # (B,T,N) - target initial values as floats
    target_final_exec: torch.Tensor,  # (B,T) - target final execution values as floats
    cfg: DAGTrainConfig,
) -> Dict[str, torch.Tensor]:
    """Compute robust DAG-structure prediction loss.

    The formulation includes cross-entropy for sign, cross-entropy for digits, cross-entropy for operations,
    and MSE for initial values and final execution values.

    All target tensors are required parameters.
    """
    # Determine device type once for proper autocast context switching
    device_type = (
        pred_sign_logits.device.type
        if isinstance(pred_sign_logits, torch.Tensor)
        else "cuda"
    )

    # Compute individual loss components (now pass logits directly for better numerical stability)
    sign_loss = _compute_sign_loss(pred_sign_logits, target_sgn, device_type)
    digit_loss = _compute_digit_loss(pred_digit_logits, target_digits, device_type)
    op_loss = _compute_op_loss(pred_op_logits, target_ops, device_type)

    # For value and execution losses, we need tanh-activated signs and probabilities
    pred_sgn = torch.tanh(pred_sign_logits)
    value_loss = _compute_value_loss(
        pred_sgn, pred_digit_logits, target_initial_values, cfg, device_type
    )

    # For execution loss, we need probabilities, so convert logits
    pred_ops = F.softmax(pred_op_logits, dim=-1)
    exec_loss = _compute_exec_loss(
        pred_sgn,
        pred_digit_logits,
        pred_ops,
        target_final_exec,
        cfg,
        device_type,
    )

    # Combine all losses with their respective weights
    total_loss = (
        cfg.sign_loss_weight * sign_loss
        + cfg.digit_loss_weight * digit_loss
        + cfg.op_loss_weight * op_loss
        + cfg.value_loss_weight * value_loss
    )

    # Only allow exec loss if all values are finite
    # since somtimes it will blow up.
    if torch.isfinite(exec_loss).all():
        total_loss += cfg.exec_loss_weight * exec_loss

    return {
        "total_loss": total_loss,
        "sign_loss": sign_loss,
        "digit_loss": digit_loss,
        "op_loss": op_loss,
        "value_loss": value_loss,
        "exec_loss": exec_loss,
    }


def compute_gradient_cosines(
    weighted_losses: Dict[str, torch.Tensor],
    total_loss: torch.Tensor,
    model_parameters: torch.nn.ParameterList,
) -> Dict[str, float]:
    """Compute cosine similarities between individual loss gradients and total gradient.

    Returns gradient cosines as ⟨g_i, g_total⟩ / ‖g_total‖² for each loss component.

    Args:
        weighted_losses: Dictionary of loss components (already weighted by cfg weights)
        total_loss: The total combined loss tensor
        model_parameters: List of model parameters to compute gradients with respect to

    Returns:
        Dictionary with keys like "grad_cosine_sign_loss" and float cosine values
    """
    # Early validation: check if total_loss is finite (single tensor check)
    if not torch.isfinite(total_loss):
        print(f"Warning: total_loss is not finite: {total_loss.item()}")
        return {f"grad_cosine_{name}": 0.0 for name in weighted_losses.keys()}

    # Compute total gradient
    try:
        total_grads = torch.autograd.grad(
            total_loss,
            model_parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
    except RuntimeError as e:
        print(f"Warning: Could not compute total gradient: {e}")
        return {f"grad_cosine_{name}": 0.0 for name in weighted_losses.keys()}

    # Handle case where some parameters don't contribute to total loss
    total_grads = [
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(total_grads, model_parameters)
    ]

    # Flatten and concatenate total gradient (defer NaN check until after computation)
    total_grad_flat = torch.cat([g.flatten() for g in total_grads])
    total_grad_norm_sq = torch.dot(total_grad_flat, total_grad_flat)

    # Combined check: finite and non-zero
    if not torch.isfinite(total_grad_norm_sq) or total_grad_norm_sq < 1e-12:
        if not torch.isfinite(total_grad_norm_sq):
            print(
                f"Warning: total_grad_norm_sq is not finite: {total_grad_norm_sq.item()}"
            )
        return {f"grad_cosine_{name}": 0.0 for name in weighted_losses.keys()}

    gradient_cosines = {}

    for loss_name, loss_value in weighted_losses.items():
        # Quick checks first: finite and non-zero
        if not torch.isfinite(loss_value) or loss_value.item() == 0.0:
            if not torch.isfinite(loss_value):
                print(f"Warning: {loss_name} is not finite: {loss_value.item()}")
            gradient_cosines[f"grad_cosine_{loss_name}"] = 0.0
            continue

        try:
            # Compute gradient for this loss component
            loss_grads = torch.autograd.grad(
                loss_value,
                model_parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

            # Handle case where some parameters don't contribute to this loss
            loss_grads = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(loss_grads, model_parameters)
            ]

            # Flatten and concatenate loss gradient
            loss_grad_flat = torch.cat([g.flatten() for g in loss_grads])

            # Compute cosine similarity: ⟨g_i, g_total⟩ / ‖g_total‖²
            dot_product = torch.dot(loss_grad_flat, total_grad_flat)
            cosine = dot_product / total_grad_norm_sq

            # Single final check for NaN in result
            if not torch.isfinite(cosine):
                print(f"Warning: cosine for {loss_name} is not finite")
                gradient_cosines[f"grad_cosine_{loss_name}"] = 0.0
            else:
                gradient_cosines[f"grad_cosine_{loss_name}"] = cosine.item()

        except RuntimeError as e:
            # Handle case where gradient computation fails
            print(f"Warning: Could not compute gradient for {loss_name}: {e}")
            gradient_cosines[f"grad_cosine_{loss_name}"] = 0.0

    return gradient_cosines


# --------------------------------------------------------------------------- #
# Utility helpers for digit tensors
# --------------------------------------------------------------------------- #


def digits_to_magnitude(
    digits: torch.Tensor,
    max_digits: int,
    max_decimal_places: int,
    base: int = 10,
) -> torch.Tensor:
    """Convert a digit tensor to absolute magnitude.

    Args:
        digits: (..., D, base) tensor where the last dimension contains digit
            probabilities (or one-hot values) for each decimal place.
        max_digits: integer digits (D1).
        max_decimal_places: fractional digits (D2).
        base: number base for digit representation.

    Returns:
        magnitude: tensor with shape digits.shape[:-2]
    """
    device, dtype = digits.device, digits.dtype
    digits_vals = (digits * torch.arange(base, device=device, dtype=dtype)).sum(
        -1
    )  # (..., D)

    int_weights = base ** torch.arange(
        max_digits - 1, -1, -1, device=device, dtype=dtype
    )
    frac_weights = base ** torch.arange(
        -1, -max_decimal_places - 1, -1, device=device, dtype=dtype
    )
    weights = torch.cat((int_weights, frac_weights))  # (D,)
    magnitude = (digits_vals * weights).sum(-1)
    return magnitude


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
        )
    }
    total_metrics = {
        "op_accuracy": 0.0,
        "full_dag_op_match": 0.0,
        "sign_accuracy": 0.0,
        # Mean-squared-error between executed DAG outputs (target vs prediction)
        "final_mse": 0.0,
    }

    num_batches = 0
    with torch.no_grad():
        for i, (texts, structures, examples) in enumerate(val_loader):
            if i >= eval_iters:
                break

            # Targets → device
            tgt_sgn = structures["initial_sgn"].to(device)
            tgt_digits = structures["initial_digits"].to(device)
            tgt_ops = structures["operation_probs"].to(device)

            # Inputs
            input_tokens = tokenize_texts(texts, cfg.block_size, device)

            # Forward
            with ctx:
                if cfg.full_backbone and hasattr(model, "dag"):
                    hidden = model.forward_hidden(input_tokens)
                    pred_sgn, _, pred_ops = model.dag.plan_predictor(hidden)
                else:
                    pred_sgn, _, pred_ops = model(input_tokens)

                pred_sgn = pred_sgn.mean(dim=1)

                # ------------------------------------------------------------------
                # Retrieve digit logits depending on model type
                # ------------------------------------------------------------------
                if hasattr(model, "dag"):
                    # GPT backbone with DAG augmentation
                    last_digit_logits = (
                        model.dag.plan_predictor.last_digit_logits
                        if hasattr(model.dag.plan_predictor, "last_digit_logits")
                        else None
                    )
                    last_operation_logits = (
                        model.dag.plan_predictor.last_operation_logits
                        if hasattr(model.dag.plan_predictor, "last_operation_logits")
                        else None
                    )
                    last_sign_logits = (
                        model.dag.plan_predictor.last_sign_logits
                        if hasattr(model.dag.plan_predictor, "last_sign_logits")
                        else None
                    )
                else:
                    # Stand-alone predictor model
                    last_digit_logits = (
                        model.dag_predictor.last_digit_logits
                        if hasattr(model.dag_predictor, "last_digit_logits")
                        else None
                    )
                    last_operation_logits = (
                        model.dag_predictor.last_operation_logits
                        if hasattr(model.dag_predictor, "last_operation_logits")
                        else None
                    )
                    last_sign_logits = (
                        model.dag_predictor.last_sign_logits
                        if hasattr(model.dag_predictor, "last_sign_logits")
                        else None
                    )

                if last_digit_logits is None:
                    raise RuntimeError("last_digit_logits not found for evaluation")
                if last_sign_logits is None:
                    raise RuntimeError("last_sign_logits not found for evaluation")
                if last_operation_logits is None:
                    raise RuntimeError("last_operation_logits not found for evaluation")

                last_digit_logits = last_digit_logits.mean(dim=1)  # (B,N,D,10)
                last_operation_logits = last_operation_logits.mean(
                    dim=1
                )  # (B,depth,n_ops)
                last_sign_logits = last_sign_logits.mean(dim=1)  # (B,N)
                pred_ops = pred_ops.mean(dim=1)

                # Convert operation logits to probabilities for metrics and execution
                if last_operation_logits is not None:
                    pred_ops = F.softmax(last_operation_logits, dim=-1)
                else:
                    # Fallback to using probabilities directly (should not happen with updated model)
                    pred_ops = pred_ops

                # Convert sign logits to tanh-activated signs for metrics and execution
                pred_sgn = torch.tanh(last_sign_logits)

                nodes, depth = tgt_sgn.size(1), tgt_ops.size(1)
                if pred_sgn.size(1) != nodes or pred_ops.size(1) != depth:
                    raise ValueError(
                        "Prediction shape mismatch: "
                        f"nodes {pred_sgn.size(1)} vs {nodes}, depth {pred_ops.size(1)} vs {depth}"
                    )

                # Sequence dimension for loss function compatibility
                last_sign_logits = last_sign_logits.unsqueeze(1)
                pred_sgn = pred_sgn.unsqueeze(1)
                last_digit_logits = last_digit_logits.unsqueeze(1)
                last_operation_logits = last_operation_logits.unsqueeze(1)

                # Extract target initial values and final execution values from structures
                target_initial_values = (
                    structures["target_initial_values"].to(device).unsqueeze(1)
                )  # Add sequence dim
                target_final_exec = (
                    structures["target_final_exec"].to(device).unsqueeze(1)
                )  # Add sequence dim

                # Compute loss using all logits for numerical stability
                losses = compute_dag_structure_loss(
                    last_sign_logits,
                    last_digit_logits,
                    last_operation_logits,
                    tgt_sgn.unsqueeze(1),
                    tgt_digits.unsqueeze(1),
                    tgt_ops.unsqueeze(1),
                    target_initial_values,
                    target_final_exec,
                    cfg,
                )

                # Metrics
                op_correct = (
                    pred_ops.squeeze(1).argmax(dim=-1).eq(tgt_ops.argmax(dim=-1))
                )
                op_acc = op_correct.float().mean()
                full_match = op_correct.all(dim=-1).float().mean()

                sign_correct = torch.sign(pred_sgn.squeeze(1)).eq(torch.sign(tgt_sgn))
                sign_acc = sign_correct.float().mean()

                # ------------------------------------------------------------------ #
                # Execute full DAGs to obtain scalar answers and compute MSE          #
                # ------------------------------------------------------------------ #

                # Target tensors --------------------------------------------------
                tgt_sign = tgt_sgn.unsqueeze(1)  # (B,1,N)
                tgt_digit_probs = tgt_digits.unsqueeze(1)  # (B,1,N,D,10)
                tgt_ops_seq = tgt_ops.unsqueeze(1)  # (B,1,depth,n_ops)

                # Prediction tensors ---------------------------------------------
                pred_sign = pred_sgn.squeeze(1).unsqueeze(1)  # (B,1,N)
                pred_digit_probs = last_digit_logits.softmax(dim=-1)  # (B,1,N,D,10)
                pred_ops_seq = pred_ops.unsqueeze(
                    1
                )  # (B,1,depth,n_ops) - converted from logits above

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

                final_mse = F.mse_loss(pred_final_val, tgt_final_val)

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

                    pred_digits_vec = last_digit_logits.squeeze(1)[batch_idx].softmax(
                        dim=-1
                    )

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
                        f"Target stack execution value: {sample_obj.final_value_exec}"
                    )
                    print(
                        f"Predicted stack execution value: {pred_final_val[batch_idx].item()}"
                    )
                    # Print number of tokens in the sample text to check context length
                    enc = get_encoding("gpt2")
                    token_count = len(enc.encode_ordinary(sample_text))
                    print(f"Token count: {token_count}")
                    # If we have the raw DAGExample, use its original floats for nicer printing
                    true_vals = sample_obj.initial_values
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

            # Aggregate
            for k, v in losses.items():
                total_losses[k] += v.item()
            total_metrics["op_accuracy"] += op_acc.item()
            total_metrics["full_dag_op_match"] += full_match.item()
            total_metrics["sign_accuracy"] += sign_acc.item()
            total_metrics["final_mse"] += final_mse.item()
            num_batches += 1

    if num_batches:
        for d in (total_losses, total_metrics):
            for k in d:
                d[k] /= num_batches

    model.train()
    return {**total_losses, **total_metrics}
