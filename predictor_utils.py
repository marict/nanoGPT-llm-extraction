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

from models.dag_model import LOG_LIM, OP_NAMES

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
# Loss
# --------------------------------------------------------------------------- #


def compute_dag_structure_loss(
    pred_sgn: torch.Tensor,
    pred_log: torch.Tensor,
    pred_ops: torch.Tensor,
    target_sgn: torch.Tensor,
    target_log: torch.Tensor,
    target_ops: torch.Tensor,
    cfg,
) -> Dict[str, torch.Tensor]:
    """Compute robust DAG-structure prediction loss.

    The formulation is identical to the previous implementation: BCE for sign,
    log-cosh for magnitude, and NLL for the operation category.
    """
    # Sign (BCE on ±1 → {0,1})
    with torch.cuda.amp.autocast(enabled=False):
        sign_target = (target_sgn > 0).float().to(torch.float32)
        sign_pred = ((pred_sgn + 1.0) * 0.5).to(torch.float32)
        sign_loss = F.binary_cross_entropy(
            sign_pred, sign_target, reduction="none"
        ).mean()

    # Magnitude (log-cosh on centred log-space values)
    diff = (pred_log - target_log) / LOG_LIM
    log_loss = torch.log(torch.cosh(diff + 1e-12)).mean()

    # Operation (NLL over one-hot targets)
    b, t, d, n_ops = pred_ops.shape
    pred_ops_flat = pred_ops.view(-1, n_ops)
    target_idx = target_ops.view(-1, n_ops).argmax(dim=-1)
    op_loss = F.nll_loss(torch.log(pred_ops_flat + 1e-8), target_idx).mean()

    total_loss = (
        cfg.sign_loss_weight * sign_loss
        + cfg.log_loss_weight * log_loss
        + cfg.op_loss_weight * op_loss
    )

    return {
        "total_loss": total_loss,
        "sign_loss": sign_loss,
        "log_loss": log_loss,
        "op_loss": op_loss,
    }


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

    total_losses = {k: 0.0 for k in ("total_loss", "sign_loss", "log_loss", "op_loss")}
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
            tgt_log = structures["initial_log"].to(device)
            tgt_ops = structures["operation_probs"].to(device)

            # Inputs
            input_tokens = tokenize_texts(texts, cfg.sequence_length, device)

            # Forward
            with ctx:
                if cfg.full_backbone and hasattr(model, "dag"):
                    hidden = model.forward_hidden(input_tokens)
                    pred_sgn, pred_log, pred_ops = model.dag.plan_predictor(hidden)
                else:
                    pred_sgn, pred_log, pred_ops = model(input_tokens)

                pred_sgn = pred_sgn.mean(dim=1)
                pred_log = pred_log.mean(dim=1)
                pred_ops = pred_ops.mean(dim=1)

                nodes, depth = tgt_sgn.size(1), tgt_ops.size(1)
                if pred_sgn.size(1) != nodes or pred_ops.size(1) != depth:
                    raise ValueError(
                        "Prediction shape mismatch: "
                        f"nodes {pred_sgn.size(1)} vs {nodes}, depth {pred_ops.size(1)} vs {depth}"
                    )

                # Sequence dimension for loss function compatibility
                pred_sgn = pred_sgn.unsqueeze(1)
                pred_log = pred_log.unsqueeze(1)
                pred_ops = pred_ops.unsqueeze(1)

                losses = compute_dag_structure_loss(
                    pred_sgn,
                    pred_log,
                    pred_ops,
                    tgt_sgn.unsqueeze(1),
                    tgt_log.unsqueeze(1),
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

                pred_mag = pred_log.squeeze(1).exp()
                tgt_mag = tgt_log.squeeze(1).exp()
                log_mape = ((pred_mag - tgt_mag).abs() / tgt_mag.clamp_min(1e-8)).mean()

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
