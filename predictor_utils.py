from __future__ import annotations

"""Utility helpers shared by DAG predictor training and evaluation.
Moving these functions into their own module drastically reduces the size of
`train_predictor.py` while keeping behaviour unchanged.
"""

from typing import List

import torch
import torch.nn.functional as F
from tiktoken import get_encoding


def _create_zero_losses(device: torch.device) -> dict[str, torch.Tensor]:
    """Create dictionary of zero losses/accuracies for edge cases."""
    zero = torch.tensor(0.0, device=device)
    return {
        "total_loss": zero,
        "digit_loss": zero,
        "digit_accuracy": zero,
        "V_sign_loss": zero,
        "O_loss": zero,
        "G_loss": zero,
        "exec_loss": zero,
        "sign_accuracy": zero,
        "op_accuracy": zero,
        "gate_accuracy": zero,
    }


def _compute_vsign_loss(
    pred_V_sign_valid: torch.Tensor,  # (num_valid, total_nodes) - Predicted signs (tanh outputs)
    target_V_sign: torch.Tensor,  # (num_valid, total_nodes) - Target signs
    num_initial_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute V_sign loss and sign accuracy.

    Args:
        pred_V_sign_valid: (num_valid, total_nodes) predicted signs (tanh activated)
        target_V_sign: (num_valid, total_nodes) target signs
        num_initial_nodes: Number of initial nodes (signs only matter for these)
        device: Device for tensor operations

    Returns:
        Tuple of (V_sign_loss, sign_accuracy)
    """
    # Sign loss - only on initial nodes (model already outputs tanh-activated values)
    V_sign_loss = F.mse_loss(
        pred_V_sign_valid[:, :num_initial_nodes],
        target_V_sign[:, :num_initial_nodes],
    )

    # Sign accuracy - V_sign predictions (tanh outputs) vs {-1, 1} targets
    # Convert tanh outputs to discrete values using 0.0 threshold (matches target generation)
    pred_V_sign_discrete = torch.where(
        pred_V_sign_valid[:, :num_initial_nodes] > 0.0,
        torch.tensor(1.0, device=device),
        torch.tensor(-1.0, device=device),
    )
    sign_correct = (
        pred_V_sign_discrete == target_V_sign[:, :num_initial_nodes]
    ).float()
    sign_accuracy = sign_correct.mean()  # Average across tokens and batch

    return V_sign_loss, sign_accuracy


def _compute_o_loss(
    pred_O_valid: torch.Tensor,  # (num_valid, dag_depth, total_nodes) - Predicted operand selectors
    target_O: torch.Tensor,  # (num_valid, dag_depth, total_nodes) - Target operand selectors
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute operand selector loss and operand accuracy.

    Args:
        pred_O_valid: (num_valid, dag_depth, total_nodes) predicted operand selectors
        target_O: (num_valid, dag_depth, total_nodes) target operand selectors

    Returns:
        Tuple of (O_loss, op_accuracy)
    """
    # Tensors should already be in correct shape (num_valid, dag_depth, total_nodes) from compute_dag_loss

    # Operand selector loss (L2)
    O_loss = F.mse_loss(pred_O_valid, target_O)

    # Operand accuracy - O predictions vs target operand selectors
    # Convert O predictions to nearest integers (targets can be any integer due to accumulation)
    pred_O_discrete = torch.round(pred_O_valid)
    op_correct = (pred_O_discrete == target_O).float()
    op_accuracy = op_correct.mean()  # Average across tokens and batch

    return O_loss, op_accuracy


def _compute_g_loss(
    pred_G_valid: torch.Tensor,  # (num_valid, dag_depth)
    target_G: torch.Tensor,  # (num_valid, dag_depth)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute domain gate loss and gate accuracy.

    Args:
        pred_G_valid: (num_valid, dag_depth) predicted gate values
        target_G: (num_valid, dag_depth) target gate values

    Returns:
        Tuple of (G_loss, gate_accuracy)
    """

    G_loss = F.mse_loss(pred_G_valid, target_G)
    pred_G_discrete = (pred_G_valid > 0.5).float()
    gate_correct = (pred_G_discrete == target_G).float()
    gate_accuracy = gate_correct.mean()

    return G_loss, gate_accuracy


def _compute_digit_loss(
    pred_digit_logits: torch.Tensor,  # (num_valid, N, D, base) - Raw logits for each digit
    target_digits: torch.Tensor,  # (num_valid, N, D, base) - One-hot target digits
    device_type: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cross-entropy loss and accuracy for digit prediction.

    This loss function is inherently stable because each digit prediction is bounded [0, 9].
    Maximum loss per digit is cross_entropy ≤ ln(10) ≈ 2.3, preventing NaN explosion.

    Args:
        pred_digit_logits: (num_valid, N, D, base) Raw logits for each digit position
        target_digits: (num_valid, N, D, base) One-hot target digits
        device_type: Device type for autocast

    Returns:
        Tuple of (digit_loss, digit_accuracy) where:
        - digit_loss: Scalar tensor with averaged cross-entropy loss
        - digit_accuracy: Scalar tensor with percentage of digits predicted correctly
    """
    _, _, D, base = pred_digit_logits.shape

    # Sanity checks (matching original implementation)
    if target_digits.shape[-2] != D:
        raise ValueError(
            "Shape mismatch between model-predicted digits and target digits: "
            f"predicted D={D}, target D={target_digits.shape[-2]}. "
            "Ensure that `max_digits` and `max_decimal_places` are set to the "
            "same values for both the dataset and the model."
        )

    if target_digits.shape[-1] != base:
        raise ValueError(
            "Shape mismatch between model-predicted base and target base: "
            f"predicted base={base}, target base={target_digits.shape[-1]}. "
            "Ensure that `base` is set to the same value for both the dataset and the model."
        )

    with torch.amp.autocast(device_type=device_type, enabled=False):
        # Reshape for efficient cross-entropy computation
        logits_flat = pred_digit_logits.reshape(-1, base).to(
            torch.float32
        )  # (num_valid*N*D, base)
        target_flat = target_digits.reshape(-1, base)
        row_sums = target_flat.sum(dim=-1)

        # Only compute loss on rows that have valid one-hot distributions
        # Unused initial nodes will have zero rows and should be skipped
        valid_rows = row_sums > 0
        if not valid_rows.any():
            # If no valid rows at all, return zero loss and accuracy
            return torch.tensor(0.0, device=logits_flat.device), torch.tensor(
                0.0, device=logits_flat.device
            )

        # Filter to only valid rows
        logits_valid = logits_flat[valid_rows]  # (num_valid_rows, base)
        target_flat_valid = target_flat[valid_rows]  # (num_valid_rows, base)

        # Allow tiny numerical tolerance when checking that valid rows sum to 1.0
        row_sums_valid = target_flat_valid.sum(dim=-1)
        if not torch.allclose(
            row_sums_valid, torch.ones_like(row_sums_valid), atol=1e-6
        ):
            bad = torch.nonzero(
                ~torch.isclose(
                    row_sums_valid, torch.ones_like(row_sums_valid), atol=1e-6
                )
            )
            first_bad = bad.flatten()[:5].tolist()
            raise ValueError(
                "Target digit rows are expected to be one-hot (sum to 1). "
                f"Found rows that sum to values != 1. Example flat indices: {first_bad}."
            )

        target_idx = target_flat_valid.argmax(dim=-1)  # (num_valid_rows,)

        # Standard cross-entropy over raw logits - bounded loss prevents NaN
        digit_loss = F.cross_entropy(logits_valid, target_idx, reduction="mean")

        # Compute digit accuracy - percentage of digits predicted correctly
        pred_idx = torch.argmax(logits_valid, dim=-1)  # (num_valid_rows,)
        correct_digits = (pred_idx == target_idx).float()
        digit_accuracy = correct_digits.mean()  # Average over all valid digit positions

    return digit_loss, digit_accuracy


def _compute_exec_loss(
    dag_executor,
    pred_digit_logits_valid: torch.Tensor,
    pred_V_sign_valid: torch.Tensor,
    pred_O_valid: torch.Tensor,
    pred_G_valid: torch.Tensor,
    target_final_exec: torch.Tensor,
    batch_indices: torch.Tensor,
    token_indices: torch.Tensor,
    cfg,
    device: torch.device,
) -> torch.Tensor:
    """
    Execute predicted DAG and compute robust execution loss using asinh transformation for stability.
    """
    try:
        # Add batch and time dimensions back for executor: (num_valid,) -> (num_valid, 1)
        pred_digit_logits_exec = pred_digit_logits_valid.unsqueeze(
            1
        )  # (num_valid, 1, num_initial_nodes, D, base)
        pred_V_sign_exec = pred_V_sign_valid.unsqueeze(1)  # (num_valid, 1, total_nodes)
        pred_O_exec = pred_O_valid.unsqueeze(
            1
        )  # (num_valid, 1, dag_depth, total_nodes)
        pred_G_exec = pred_G_valid.unsqueeze(1)  # (num_valid, 1, dag_depth)

        # Execute predicted DAG - executor handles digit conversion internally
        pred_final_exec = dag_executor(
            pred_digit_logits_exec, pred_V_sign_exec, pred_O_exec, pred_G_exec
        )  # (num_valid, 1)
        pred_final_exec = pred_final_exec.squeeze(1)  # (num_valid,)

        # Check for NaN/Inf in execution results and log problematic cases
        _log_problematic_dag_executions(
            pred_final_exec,
            batch_indices,
            token_indices,
            target_final_exec,
        )

        # Handle NaN/Inf predictions with penalty
        if torch.any(~torch.isfinite(pred_final_exec)):
            return torch.tensor(100.0, device=device, requires_grad=True)

        # Use asinh transformation - differentiable and handles all real values
        # asinh(x) ≈ log(|x|) for large |x|, but smooth and defined everywhere
        asinh_pred = torch.asinh(pred_final_exec)
        asinh_target = torch.asinh(target_final_exec)

        # Use Huber loss in asinh space for smooth handling of extreme errors
        # Huber loss: quadratic for small errors, linear for large errors
        # Increase delta to stay in quadratic regime longer for better gradients
        delta = 10.0  # Larger delta for stronger gradients on large errors
        exec_loss = F.huber_loss(asinh_pred, asinh_target, delta=delta)

        return exec_loss
    except Exception:
        print(
            f"Execution failed for batch {batch_indices[0].item()}, token {token_indices[0].item()}"
        )
        raise


def _log_problematic_dag_executions(
    pred_final_exec: torch.Tensor,
    batch_indices: torch.Tensor,
    token_indices: torch.Tensor,
    target_final_exec: torch.Tensor,
) -> None:
    """Log brief information about DAG executions that produced NaN/Inf values."""
    problematic_mask = ~torch.isfinite(pred_final_exec)
    if not problematic_mask.any():
        return

    # Log summary of problematic executions
    num_problematic = problematic_mask.sum().item()
    print(f"WARNING: {num_problematic} DAG executions produced NaN/Inf values")

    # Log first few problematic cases
    for i in torch.where(problematic_mask)[0][:3]:  # Limit to first 3
        batch_idx = batch_indices[i].item()
        token_idx = token_indices[i].item()
        result_val = pred_final_exec[i].item()
        target_val = target_final_exec[i].item()
        print(
            f"  batch {batch_idx}, token {token_idx}: {result_val} (target: {target_val})"
        )


def compute_dag_loss(
    pred_digit_logits: torch.Tensor,
    pred_V_sign: torch.Tensor,
    pred_O: torch.Tensor,
    pred_G: torch.Tensor,
    target_tensors: dict[str, torch.Tensor],
    dag_executor=None,
    cfg=None,
) -> dict[str, torch.Tensor]:
    """Compute DAG losses and accuracy metrics.

    Returns dictionary with losses (digit, V_mag, V_sign, O, G, exec) and
    accuracies (sign, op, gate) for all valid positions.
    """

    # Add in batch dimension if needed
    if pred_V_sign.dim() == 1:
        pred_V_sign = pred_V_sign.unsqueeze(0)  # (1, total_nodes)
    if pred_O.dim() == 2:
        pred_O = pred_O.unsqueeze(0)  # (1, dag_depth, total_nodes)
    if pred_G.dim() == 1:
        pred_G = pred_G.unsqueeze(0)  # (1, dag_depth)

    device = pred_digit_logits.device
    if "valid_mask" not in target_tensors:
        raise ValueError("valid_mask not found in target_tensors")
    valid_mask = target_tensors["valid_mask"]
    b_valid_idx, t_valid_idx = torch.where(valid_mask)  # shapes: [N_valid], [N_valid]

    pred_digit_logits_valid = pred_digit_logits[b_valid_idx, t_valid_idx]
    pred_V_sign_valid = pred_V_sign[b_valid_idx, t_valid_idx]
    pred_O_valid = pred_O[b_valid_idx, t_valid_idx]
    pred_G_valid = pred_G[b_valid_idx, t_valid_idx]

    target_digits = target_tensors["target_digits"][b_valid_idx, t_valid_idx]
    target_V_sign = target_tensors["target_V_sign"][b_valid_idx, t_valid_idx]
    target_O = target_tensors["target_O"][b_valid_idx, t_valid_idx]
    target_G = target_tensors["target_G"][b_valid_idx, t_valid_idx]
    target_final_exec = target_tensors["target_final_exec"][b_valid_idx, t_valid_idx]

    # Determine node counts from tensor shapes
    total_nodes = pred_V_sign_valid.shape[-1]
    dag_depth = (total_nodes - 1) // 2
    num_initial_nodes = dag_depth + 1

    # Digit loss and accuracy - stable bounded loss for magnitudes
    digit_loss, digit_accuracy = _compute_digit_loss(
        pred_digit_logits_valid, target_digits
    )

    # Compute V_sign loss and sign accuracy
    V_sign_loss, sign_accuracy = _compute_vsign_loss(
        pred_V_sign_valid, target_V_sign, num_initial_nodes, device
    )

    # Compute operand selector loss and operand accuracy
    O_loss, op_accuracy = _compute_o_loss(pred_O_valid, target_O)

    # Compute domain gate loss and gate accuracy
    G_loss, gate_accuracy = _compute_g_loss(pred_G_valid, target_G)

    # Execution loss (if DAG executor is provided)
    # NOTE: This might be broken now because of the new indexing
    exec_loss = torch.tensor(0.0, device=device)
    if dag_executor is not None and cfg.enable_exec_loss:
        exec_loss = _compute_exec_loss(
            dag_executor,
            pred_digit_logits_valid,
            pred_V_sign_valid,
            pred_O_valid,
            pred_G_valid,
            target_final_exec,
            b_valid_idx,
            t_valid_idx,
            cfg,
            device,
        )

    # Build total loss from enabled components
    total_loss = torch.tensor(0.0, device=device)

    # Use cfg flags directly
    enable_digit_loss = cfg.enable_digit_loss
    enable_vsign_loss = cfg.enable_vsign_loss
    enable_o_loss = cfg.enable_o_loss
    enable_g_loss = cfg.enable_g_loss
    enable_exec_loss = cfg.enable_exec_loss
    exec_loss_weight = cfg.exec_loss_weight
    g_loss_weight = getattr(
        cfg, "g_loss_weight", 1.0
    )  # Default to 1.0 if not specified

    if enable_digit_loss:
        total_loss = total_loss + digit_loss
    if enable_vsign_loss:
        total_loss = total_loss + V_sign_loss
    if enable_o_loss:
        total_loss = total_loss + O_loss
    if enable_g_loss:
        total_loss = total_loss + (G_loss * g_loss_weight)
    if enable_exec_loss:
        total_loss = total_loss + (exec_loss * exec_loss_weight)

    return {
        "total_loss": total_loss,
        "digit_loss": digit_loss,
        "digit_accuracy": digit_accuracy,
        "V_sign_loss": V_sign_loss,
        "sign_accuracy": sign_accuracy,
        "O_loss": O_loss,
        "op_accuracy": op_accuracy,
        "G_loss": G_loss * g_loss_weight,
        "gate_accuracy": gate_accuracy,
        "exec_loss": exec_loss * exec_loss_weight,
    }


__all__ = [
    "tokenize_texts",
    "compute_dag_loss",
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
