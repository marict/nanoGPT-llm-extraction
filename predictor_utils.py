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
    pred_sgn: torch.Tensor,
    target_sgn: torch.Tensor,
    device_type: str,
) -> torch.Tensor:
    """Compute binary cross-entropy loss for sign prediction."""
    with torch.amp.autocast(device_type=device_type, enabled=False):
        sign_target = (target_sgn > 0).float().to(torch.float32)
        sign_pred = ((pred_sgn + 1.0) * 0.5).to(torch.float32)
        sign_loss = F.binary_cross_entropy(
            sign_pred, sign_target, reduction="none"
        ).mean()
    return sign_loss


def _compute_digit_loss(
    pred_digit_probs: torch.Tensor,  # Now expects probabilities, not logits
    target_digits: torch.Tensor,
    device_type: str,
) -> torch.Tensor:
    """Compute cross-entropy loss for digit prediction."""
    B, T, N, D, _ = pred_digit_probs.shape

    # Sanity check: model and dataset must agree on digit slots
    if target_digits.shape[-2] != D:
        raise ValueError(
            "Shape mismatch between model-predicted digits and target digits: "
            f"predicted D={D} , target D={target_digits.shape[-2]}. "
            "Ensure that `max_digits` and `max_decimal_places` are set to the "
            "*same* values for both the dataset and the model (these values are "
            "propagated via the training config)."
        )

    with torch.amp.autocast(device_type=device_type, enabled=False):
        # Check if predictions and targets are identical (perfect match)
        # This special case should return zero loss
        if torch.allclose(pred_digit_probs, target_digits, rtol=1e-5, atol=1e-5):
            return torch.tensor(0.0, device=pred_digit_probs.device)

        # Use reshape to handle potential non-contiguous tensors (view can fail)
        pred_flat = pred_digit_probs.reshape(-1, 10).to(torch.float32)  # (B*T*N*D, 10)

        # Expect probabilities - clamp for numerical stability and take log
        log_probs = torch.log(pred_flat.clamp(min=1e-8))

        target_flat = target_digits.reshape(-1, 10)
        target_idx = target_flat.argmax(dim=-1)

        # Mask out positions where the target has no valid digit information (all zeros)
        valid_mask = target_flat.sum(dim=-1) > 0  # (B*T*N*D)
        if valid_mask.any():
            digit_loss = F.nll_loss(log_probs[valid_mask], target_idx[valid_mask])
        else:
            digit_loss = torch.tensor(0.0, device=pred_digit_probs.device)

    return digit_loss


def _compute_op_loss(
    pred_ops: torch.Tensor,
    target_ops: torch.Tensor,
    device_type: str,
) -> torch.Tensor:
    """Compute negative log-likelihood loss for operation prediction."""
    b, t, d, n_ops = pred_ops.shape
    with torch.amp.autocast(device_type=device_type, enabled=False):
        pred_ops_flat = pred_ops.view(-1, n_ops).to(torch.float32)
        target_idx = target_ops.view(-1, n_ops).argmax(dim=-1)
        op_loss = F.nll_loss(torch.log(pred_ops_flat + 1e-8), target_idx).mean()
    return op_loss


def _compute_value_loss(
    pred_sgn: torch.Tensor,
    pred_digit_probs: torch.Tensor,  # Now expects probabilities, not logits
    target_initial_values: torch.Tensor,
    cfg,
    device_type: str,
) -> torch.Tensor:
    """Compute robust loss between predicted and target initial values in log space."""
    with torch.amp.autocast(device_type=device_type, enabled=False):
        # Compute predicted log magnitudes directly from digit distributions
        # This avoids the intermediate step of reconstructing actual values
        device, dtype = pred_digit_probs.device, pred_digit_probs.dtype
        digits_vals = (
            pred_digit_probs * torch.arange(10, device=device, dtype=dtype)
        ).sum(
            -1
        )  # (..., D)

        # Build decimal place weights for log magnitude computation
        int_weights = 10 ** torch.arange(
            cfg.max_digits - 1, -1, -1, device=device, dtype=dtype
        )
        frac_weights = 10 ** torch.arange(
            -1, -cfg.max_decimal_places - 1, -1, device=device, dtype=dtype
        )
        weights = torch.cat((int_weights, frac_weights))  # (D,)
        pred_magnitude = (digits_vals * weights).sum(-1)  # (B,T,N)

        # Convert to log space (clamp for numerical stability)
        pred_log_magnitude = torch.log10(pred_magnitude.clamp(min=1e-8)).to(
            torch.float32
        )

        # Target log magnitudes
        target_log_magnitude = torch.log10(target_initial_values.abs() + 1e-8).to(
            torch.float32
        )

        # Core magnitude loss in log space (direct smooth_l1_loss)
        magnitude_loss = F.smooth_l1_loss(
            pred_log_magnitude, target_log_magnitude, beta=1.0
        )

        # Smooth sign penalty using continuous sign values
        # pred_sgn is already in [-1,1] from tanh, target signs are ±1
        target_sign_smooth = torch.sign(target_initial_values)  # ±1
        sign_mask = target_initial_values.abs() > 1e-8  # ignore target zeros

        # Smooth sign loss: penalize when pred_sgn and target_sign have different signs
        # Use smooth penalty instead of hard mismatch counting
        sign_diff = (
            pred_sgn.to(torch.float32) - target_sign_smooth.to(torch.float32)
        ) * sign_mask.float()
        sign_penalty = 0.1 * (sign_diff**2).mean()  # smooth quadratic penalty

        return magnitude_loss + sign_penalty


def _compute_exec_loss(
    pred_sgn: torch.Tensor,
    pred_digit_probs: torch.Tensor,  # Now expects probabilities, not logits
    pred_ops: torch.Tensor,
    target_final_exec: torch.Tensor,
    cfg,
    device_type: str,
) -> torch.Tensor:
    """Execution loss using log10 magnitudes directly (avoids 10** blow-up)."""
    with torch.amp.autocast(device_type=device_type, enabled=False):

        # Execute: returns sign tensor and log10(|value|) tensor
        pred_final_sgn, pred_final_log10 = execute_stack(
            pred_sgn,
            pred_digit_probs,
            pred_ops,
            max_digits=cfg.max_digits,
            max_decimal_places=cfg.max_decimal_places,
            ignore_clip=True,  # raw for loss
        )

        # Flatten and ensure float32 for smooth_l1_loss compatibility
        pred_final_log10 = pred_final_log10.reshape(-1).to(torch.float32)
        pred_final_sgn = pred_final_sgn.reshape(-1).to(torch.float32)
        target_flat = target_final_exec.reshape(-1).to(torch.float32)

        # Target log10 magnitude - ensure float32
        tgt_log10 = torch.log10(target_flat.abs() + 1e-8).to(torch.float32)

        # Magnitude loss components (inlined from safe_big_loss):
        # A) Huber loss in log space (weight 1.0)
        log_loss = F.smooth_l1_loss(pred_final_log10, tgt_log10, beta=1.0)

        # B) Relative multiplicative error approximation (reduced weight and clamped)
        log_diff = (
            (pred_final_log10 - tgt_log10).abs().clamp(max=6.0)
        )  # Reduced from 12.0
        # Clamp the exponential to prevent massive values
        rel_term = (10**log_diff - 1.0).clamp(
            max=100.0
        )  # Prevent massive relative errors
        rel_loss = 0.01 * rel_term.mean()  # Reduced weight from 0.2 to 0.01

        # C) Overflow penalty (weight 0.05)
        overflow_pen = 0.05 * (pred_final_log10.abs() > 12.0).float().mean()

        mag_loss = log_loss + rel_loss + overflow_pen

        # Sign penalty for final execution results (fixed weight 0.1)
        # This is essential: predicting -10 instead of +10 should be worse than predicting 8 instead of +10
        tgt_sign = torch.sign(target_flat)
        sign_mask = target_flat.abs() > 1e-8  # ignore target zeros
        sign_mismatch = (pred_final_sgn != tgt_sign) & sign_mask
        sign_pen = 0.1 * sign_mismatch.float().mean()

        return mag_loss + sign_pen


# --------------------------------------------------------------------------- #
# Main loss function
# --------------------------------------------------------------------------- #
def compute_dag_structure_loss(
    pred_sgn: torch.Tensor,  # (B,T,N)
    pred_digit_logits: torch.Tensor,  # (B,T,N,D,10) raw logits (not probabilities)
    pred_ops: torch.Tensor,  # (B,T,depth,n_ops)
    target_sgn: torch.Tensor,  # (B,T,N)
    target_digits: torch.Tensor,  # (B,T,N,D,10) one-hot
    target_ops: torch.Tensor,
    target_initial_values: torch.Tensor,  # (B,T,N) - target initial values as floats
    target_final_exec: torch.Tensor,  # (B,T) - target final execution values as floats
    cfg: DAGTrainConfig,
) -> Dict[str, torch.Tensor]:
    """Compute robust DAG-structure prediction loss.

    The formulation includes BCE for sign, cross-entropy for digits, NLL for operations,
    and MSE for initial values and final execution values.

    All target tensors are required parameters.
    """
    # Determine device type once for proper autocast context switching
    device_type = pred_sgn.device.type if isinstance(pred_sgn, torch.Tensor) else "cuda"

    with torch.amp.autocast(device_type=device_type, enabled=False):
        pred_digit_probs = F.softmax(pred_digit_logits.to(torch.float32), dim=-1)

    # Compute individual loss components (now pass probabilities, not logits)
    sign_loss = _compute_sign_loss(pred_sgn, target_sgn, device_type)
    digit_loss = _compute_digit_loss(pred_digit_probs, target_digits, device_type)
    op_loss = _compute_op_loss(pred_ops, target_ops, device_type)
    value_loss = _compute_value_loss(
        pred_sgn, pred_digit_probs, target_initial_values, cfg, device_type
    )
    exec_loss = _compute_exec_loss(
        pred_sgn, pred_digit_probs, pred_ops, target_final_exec, cfg, device_type
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


# --------------------------------------------------------------------------- #
# Utility helpers for digit tensors
# --------------------------------------------------------------------------- #


def digits_to_magnitude(
    digits: torch.Tensor,
    max_digits: int,
    max_decimal_places: int,
) -> torch.Tensor:
    """Convert a digit tensor to absolute magnitude.

    Args:
        digits: (..., D, 10) tensor where the last dimension contains digit
            probabilities (or one-hot values) for each decimal place.
        max_digits: integer digits (D1).
        max_decimal_places: fractional digits (D2).

    Returns:
        magnitude: tensor with shape digits.shape[:-2]
    """
    device, dtype = digits.device, digits.dtype
    digits_vals = (digits * torch.arange(10, device=device, dtype=dtype)).sum(
        -1
    )  # (..., D)

    int_weights = 10 ** torch.arange(max_digits - 1, -1, -1, device=device, dtype=dtype)
    frac_weights = 10 ** torch.arange(
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
                else:
                    # Stand-alone predictor model
                    last_digit_logits = (
                        model.dag_predictor.last_digit_logits
                        if hasattr(model.dag_predictor, "last_digit_logits")
                        else None
                    )

                if last_digit_logits is None:
                    raise RuntimeError("last_digit_logits not found for evaluation")

                last_digit_logits = last_digit_logits.mean(dim=1)  # (B,N,D,10)
                pred_ops = pred_ops.mean(dim=1)

                nodes, depth = tgt_sgn.size(1), tgt_ops.size(1)
                if pred_sgn.size(1) != nodes or pred_ops.size(1) != depth:
                    raise ValueError(
                        "Prediction shape mismatch: "
                        f"nodes {pred_sgn.size(1)} vs {nodes}, depth {pred_ops.size(1)} vs {depth}"
                    )

                # Sequence dimension for loss function compatibility
                pred_sgn = pred_sgn.unsqueeze(1)
                last_digit_logits = last_digit_logits.unsqueeze(1)
                pred_ops = pred_ops.unsqueeze(1)

                # Extract target initial values and final execution values from structures
                target_initial_values = (
                    structures["target_initial_values"].to(device).unsqueeze(1)
                )  # Add sequence dim
                target_final_exec = (
                    structures["target_final_exec"].to(device).unsqueeze(1)
                )  # Add sequence dim

                losses = compute_dag_structure_loss(
                    pred_sgn,
                    last_digit_logits,
                    pred_ops,
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
                # pred_ops already (B,1,depth,n_ops)

                # Execute stacks - use clipping for consistency with training
                tgt_final_sgn, tgt_final_log = execute_stack(
                    tgt_sign,
                    tgt_digit_probs,
                    tgt_ops_seq,
                    max_digits=cfg.max_digits,
                    max_decimal_places=cfg.max_decimal_places,
                    ignore_clip=False,  # Consistent with training behavior
                )

                pred_final_sgn, pred_final_log = execute_stack(
                    pred_sign,
                    pred_digit_probs,
                    pred_ops,
                    max_digits=cfg.max_digits,
                    max_decimal_places=cfg.max_decimal_places,
                    ignore_clip=False,  # Consistent with training behavior
                )

                # Convert to real numbers
                tgt_final_val = tgt_final_sgn * torch.pow(
                    torch.tensor(10.0, device=device, dtype=tgt_final_log.dtype),
                    tgt_final_log,
                )
                pred_final_val = pred_final_sgn * torch.pow(
                    torch.tensor(10.0, device=device, dtype=pred_final_log.dtype),
                    pred_final_log,
                )

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
