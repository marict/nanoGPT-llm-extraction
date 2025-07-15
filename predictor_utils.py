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

from models.dag_model import OP_NAMES

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
# Loss
# --------------------------------------------------------------------------- #


# Updated function to handle digit distributions instead of log magnitudes
def compute_dag_structure_loss(
    pred_sgn: torch.Tensor,  # (B,T,N)
    pred_digits: torch.Tensor,  # (B,T,N,D,10) logits or probs
    pred_ops: torch.Tensor,  # (B,T,depth,n_ops)
    target_sgn: torch.Tensor,  # (B,T,N)
    target_digits: torch.Tensor,  # (B,T,N,D,10) one-hot
    target_ops: torch.Tensor,
    cfg,
) -> Dict[str, torch.Tensor]:
    """Compute robust DAG-structure prediction loss.

    The formulation is identical to the previous implementation: BCE for sign,
    log-cosh for magnitude, and NLL for the operation category.
    """
    # Determine device type once for proper autocast context switching
    device_type = pred_sgn.device.type if isinstance(pred_sgn, torch.Tensor) else "cuda"

    # Sign (BCE on ±1 → {0,1})
    # Disable autocast to ensure computations are carried out in full precision
    # regardless of any surrounding mixed-precision context.
    with torch.amp.autocast(device_type=device_type, enabled=False):
        sign_target = (target_sgn > 0).float().to(torch.float32)
        sign_pred = ((pred_sgn + 1.0) * 0.5).to(torch.float32)
        sign_loss = F.binary_cross_entropy(
            sign_pred, sign_target, reduction="none"
        ).mean()

    # Digit prediction (cross entropy over 10-way classification per digit slot)
    # ``pred_digits`` can be either raw logits **or** probabilities depending on
    # the caller (training code passes logits, some tests pass one-hot probs).
    # We convert to log-probabilities in a way that supports both cases.
    B, T, N, D, _ = pred_digits.shape
    with torch.amp.autocast(device_type=device_type, enabled=False):
        # Use reshape to handle potential non-contiguous tensors (view can fail)
        pred_flat = pred_digits.reshape(-1, 10).to(torch.float32)  # (B*T*N*D, 10)
        if pred_flat.min() < 0 or pred_flat.max() > 1:
            # Likely raw logits – apply log_softmax
            log_probs = F.log_softmax(pred_flat, dim=-1)
        else:
            # Already probabilities (or one-hot). Clamp for numerical stability
            log_probs = torch.log(pred_flat.clamp(min=1e-8))

        target_flat = target_digits.reshape(-1, 10)
        target_idx = target_flat.argmax(dim=-1)

        # Mask out positions where the target has no valid digit information (all zeros)
        valid_mask = target_flat.sum(dim=-1) > 0  # (B*T*N*D)
        if valid_mask.any():
            digit_loss = F.nll_loss(log_probs[valid_mask], target_idx[valid_mask])
        else:
            digit_loss = torch.tensor(0.0, device=pred_sgn.device)

    # Operation (NLL over one-hot targets)
    # Convert probabilities to float32 before taking log to prevent log(0) -> -inf in FP16 when probs are tiny.
    b, t, d, n_ops = pred_ops.shape
    with torch.amp.autocast(device_type=device_type, enabled=False):
        pred_ops_flat = pred_ops.view(-1, n_ops).to(torch.float32)
        target_idx = target_ops.view(-1, n_ops).argmax(dim=-1)
        op_loss = F.nll_loss(torch.log(pred_ops_flat + 1e-8), target_idx).mean()

    total_loss = (
        cfg.sign_loss_weight * sign_loss
        + cfg.digit_loss_weight * digit_loss
        + cfg.op_loss_weight * op_loss
    )

    return {
        "total_loss": total_loss,
        "sign_loss": sign_loss,
        "digit_loss": digit_loss,
        "op_loss": op_loss,
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
        k: 0.0 for k in ("total_loss", "sign_loss", "digit_loss", "op_loss")
    }
    total_metrics = {
        "op_accuracy": 0.0,
        "full_dag_op_match": 0.0,
        "sign_accuracy": 0.0,
        "log_magnitude_mape": 0.0,
    }

    num_batches = 0
    with torch.no_grad():
        for i, (texts, structures, seeds) in enumerate(val_loader):
            if i >= eval_iters:
                break

            # Targets → device
            tgt_sgn = structures["initial_sgn"].to(device)
            tgt_digits = structures["initial_digits"].to(device)
            tgt_ops = structures["operation_probs"].to(device)

            # Inputs
            input_tokens = tokenize_texts(texts, cfg.sequence_length, device)

            # Forward
            with ctx:
                if cfg.full_backbone and hasattr(model, "dag"):
                    hidden = model.forward_hidden(input_tokens)
                    pred_sgn, _, pred_ops = model.dag.plan_predictor(hidden)
                else:
                    pred_sgn, _, pred_ops = model(input_tokens)

                pred_sgn = pred_sgn.mean(dim=1)
                digit_logits = (
                    model.dag.plan_predictor.digit_logits
                    if hasattr(model.dag.plan_predictor, "digit_logits")
                    else None
                )
                if digit_logits is None:
                    raise RuntimeError("digit_logits not found for evaluation")
                digit_logits = digit_logits.mean(dim=1)  # (B,N,D,10)
                pred_ops = pred_ops.mean(dim=1)

                nodes, depth = tgt_sgn.size(1), tgt_ops.size(1)
                if pred_sgn.size(1) != nodes or pred_ops.size(1) != depth:
                    raise ValueError(
                        "Prediction shape mismatch: "
                        f"nodes {pred_sgn.size(1)} vs {nodes}, depth {pred_ops.size(1)} vs {depth}"
                    )

                # Sequence dimension for loss function compatibility
                pred_sgn = pred_sgn.unsqueeze(1)
                digit_logits = digit_logits.unsqueeze(1)
                pred_ops = pred_ops.unsqueeze(1)

                losses = compute_dag_structure_loss(
                    pred_sgn,
                    digit_logits,
                    pred_ops,
                    tgt_sgn.unsqueeze(1),
                    tgt_digits.unsqueeze(1),
                    tgt_ops.unsqueeze(1),
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

                pred_mag = digits_to_magnitude(
                    digit_logits.squeeze(1).softmax(dim=-1),
                    cfg.max_digits,
                    cfg.max_decimal_places,
                )
                tgt_mag = digits_to_magnitude(
                    tgt_digits.squeeze(1),
                    cfg.max_digits,
                    cfg.max_decimal_places,
                )
                log_mape = ((pred_mag - tgt_mag).abs() / tgt_mag.clamp_min(1e-8)).mean()

                # -------------------------------------------------------------- #
                # Console debug: print one random sample from the first batch
                # -------------------------------------------------------------- #
                if i == 0:
                    batch_size = tgt_sgn.size(0)
                    sample_idx = _eval_random.randrange(batch_size)
                    sample_text = texts[sample_idx]
                    sample_seed = seeds[sample_idx]

                    # Sign vectors (N,) and digit logits (N,D,10)
                    pred_sign_vec = pred_sgn.squeeze(1)[sample_idx]
                    tgt_sign_vec = tgt_sgn[sample_idx]

                    pred_digits_vec = digit_logits.squeeze(1)[sample_idx].softmax(
                        dim=-1
                    )
                    tgt_digits_vec = tgt_digits[sample_idx]

                    # Convert digit distributions to magnitudes
                    pred_mag_vec = digits_to_magnitude(
                        pred_digits_vec,
                        cfg.max_digits,
                        cfg.max_decimal_places,
                    )
                    tgt_mag_vec = digits_to_magnitude(
                        tgt_digits_vec,
                        cfg.max_digits,
                        cfg.max_decimal_places,
                    )

                    pred_real_vals = (
                        (torch.sign(pred_sign_vec) * pred_mag_vec).cpu().tolist()
                    )
                    tgt_real_vals = (
                        (torch.sign(tgt_sign_vec) * tgt_mag_vec).cpu().tolist()
                    )

                    # Decode operations
                    tgt_ops_row = tgt_ops[sample_idx]  # (depth, n_ops)
                    pred_ops_row = pred_ops.squeeze(1)[sample_idx]
                    tgt_op_indices = tgt_ops_row.argmax(dim=-1).cpu().tolist()
                    pred_op_indices = pred_ops_row.argmax(dim=-1).cpu().tolist()
                    tgt_op_names = [OP_NAMES[idx] for idx in tgt_op_indices]
                    pred_op_names = [OP_NAMES[idx] for idx in pred_op_indices]

                    print("\n=== Validation Sample ===")
                    print(f"Sample RNG seed: {sample_seed}")
                    print(f"Text: {sample_text}")
                    print("Target initial values (rounded to 4 dp):")
                    print([round(v, 4) for v in tgt_real_vals])
                    print("Predicted initial values (rounded to 4 dp):")
                    print([round(v, 4) for v in pred_real_vals])
                    print("Operations (ground truth):")
                    print(tgt_op_names)
                    print("Operations (predicted):")
                    print(pred_op_names)
                    print("==========================\n")

            # Aggregate
            for k, v in losses.items():
                total_losses[k] += v.item()
            total_metrics["op_accuracy"] += op_acc.item()
            total_metrics["full_dag_op_match"] += full_match.item()
            total_metrics["sign_accuracy"] += sign_acc.item()
            total_metrics["log_magnitude_mape"] += log_mape.item()
            num_batches += 1

    if num_batches:
        for d in (total_losses, total_metrics):
            for k in d:
                d[k] /= num_batches

    model.train()
    return {**total_losses, **total_metrics}
