from __future__ import annotations

"""Utility helpers shared by DAG predictor training and evaluation.
Moving these functions into their own module drastically reduces the size of
`train_predictor.py` while keeping behaviour unchanged.
"""

from typing import List

import torch
import torch.nn.functional as F
from tiktoken import get_encoding


def _compute_value_loss(
    pred_values: torch.Tensor, target_values: torch.Tensor
) -> torch.Tensor:
    """
    Compute robust loss for magnitude values that can span wide ranges.

    Uses a combination of relative error and absolute error to handle both small and large values.
    This prevents exploding gradients from very large magnitude differences.
    """
    # Avoid division by zero
    epsilon = 1e-8

    # Compute relative error for non-zero targets
    target_abs = torch.abs(target_values) + epsilon
    relative_error = torch.abs(pred_values - target_values) / target_abs

    # Compute absolute error (for very small values where relative error is unreliable)
    absolute_error = torch.abs(pred_values - target_values)

    # Use Huber-like loss: relative error for large values, absolute for small
    # Choose between relative and absolute based on target magnitude
    use_relative = target_abs > 1.0
    loss_per_element = torch.where(use_relative, relative_error, absolute_error)

    # Apply smooth L1 (Huber) loss to make it more robust to outliers
    huber_delta = 1.0
    loss_per_element = torch.where(
        loss_per_element < huber_delta,
        0.5 * loss_per_element.pow(2),
        huber_delta * (loss_per_element - 0.5 * huber_delta),
    )

    return loss_per_element.mean()


def _compute_digit_loss(
    pred_digit_logits: torch.Tensor,  # (num_valid, N, D, base) - Raw logits for each digit
    target_digits: torch.Tensor,  # (num_valid, N, D, base) - One-hot target digits
    device_type: str = "cuda",
) -> torch.Tensor:
    """Compute cross-entropy loss for digit prediction.

    This loss function is inherently stable because each digit prediction is bounded [0, 9].
    Maximum loss per digit is cross_entropy ≤ ln(10) ≈ 2.3, preventing NaN explosion.

    Args:
        pred_digit_logits: (num_valid, N, D, base) Raw logits for each digit position
        target_digits: (num_valid, N, D, base) One-hot target digits
        device_type: Device type for autocast

    Returns:
        Scalar tensor with averaged digit loss
    """
    num_valid, N, D, base = pred_digit_logits.shape

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

        # Strict validation: every row must be a valid one-hot distribution (matching original)
        if (row_sums == 0).any():
            offending = torch.nonzero(row_sums == 0).flatten()[:5].tolist()
            raise ValueError(
                "Encountered target digit rows with all zeros (invalid one-hot). "
                f"Example flat indices: {offending}. This indicates a bug in the "
                "dataset generation pipeline."
            )

        # Allow tiny numerical tolerance when checking that rows sum to 1.0
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
            bad = torch.nonzero(
                ~torch.isclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
            )
            first_bad = bad.flatten()[:5].tolist()
            raise ValueError(
                "Target digit rows are expected to be one-hot (sum to 1). "
                f"Found rows that sum to values != 1. Example flat indices: {first_bad}."
            )

        target_idx = target_flat.argmax(dim=-1)  # (num_valid*N*D,)

        # Standard cross-entropy over raw logits - bounded loss prevents NaN
        digit_loss = F.cross_entropy(logits_flat, target_idx, reduction="mean")

    return digit_loss


def _compute_exec_loss(
    pred_exec: torch.Tensor, target_exec: torch.Tensor
) -> torch.Tensor:
    """
    Compute robust execution loss that handles the wide range of possible execution results.

    Uses log-space loss for very large/small values and regular loss for moderate values.
    This prevents training instability from extreme execution results.
    """
    # Handle edge cases
    if torch.any(torch.isnan(pred_exec)) or torch.any(torch.isinf(pred_exec)):
        # Heavily penalize NaN/Inf predictions
        return torch.tensor(100.0, device=pred_exec.device, requires_grad=True)

    # Separate small, moderate, and large values for different loss treatments
    target_abs = torch.abs(target_exec)

    # Define thresholds
    small_threshold = 1e-6
    large_threshold = 1e6

    # For very small values (near zero), use absolute difference
    small_mask = target_abs < small_threshold

    # For very large values, use log-space loss to prevent explosion
    large_mask = target_abs > large_threshold

    # For moderate values, use relative error
    moderate_mask = ~(small_mask | large_mask)

    total_loss = torch.tensor(0.0, device=pred_exec.device)

    if small_mask.any():
        small_loss = F.mse_loss(pred_exec[small_mask], target_exec[small_mask])
        total_loss = total_loss + small_loss

    if moderate_mask.any():
        # Relative error for moderate values
        epsilon = 1e-8
        target_moderate = target_exec[moderate_mask]
        pred_moderate = pred_exec[moderate_mask]
        relative_error = torch.abs(pred_moderate - target_moderate) / (
            torch.abs(target_moderate) + epsilon
        )
        moderate_loss = relative_error.mean()
        total_loss = total_loss + moderate_loss

    if large_mask.any():
        # Log-space loss for large values (with sign preservation)
        target_large = target_exec[large_mask]
        pred_large = pred_exec[large_mask]

        # Preserve signs
        target_sign = torch.sign(target_large)
        pred_sign = torch.sign(pred_large)

        # Work in log space for magnitudes
        target_log = torch.log(torch.abs(target_large) + 1e-12)
        pred_log = torch.log(torch.abs(pred_large) + 1e-12)

        # Combined loss: log magnitude difference + sign difference
        magnitude_loss = F.mse_loss(pred_log, target_log)
        sign_loss = F.mse_loss(pred_sign, target_sign)

        large_loss = magnitude_loss + sign_loss
        total_loss = total_loss + large_loss

    return total_loss


def compute_dag_loss(
    pred_digit_logits: torch.Tensor,  # (B, T, num_initial_nodes, D, base) predicted digit logits
    pred_V_sign: torch.Tensor,  # (B, T, total_nodes) predicted signs
    pred_O: torch.Tensor,  # (B, T, dag_depth, total_nodes) predicted operand selection
    pred_G: torch.Tensor,  # (B, T, dag_depth) predicted domain gates
    target_tensors: list[
        list[dict[str, torch.Tensor]]
    ],  # (B, T) nested list: new target tensors
    valid_mask: torch.Tensor,  # (B, T) boolean mask for valid positions
    dag_executor=None,  # Optional DAGExecutor for execution loss
) -> dict[str, torch.Tensor]:
    """Compute loss for DAG tensor format with digit prediction, V_sign, O, G targets.

    Args:
        pred_digit_logits: (B, T, num_initial_nodes, D, base) predicted digit logits for initial nodes
        pred_V_sign: (B, T, total_nodes) predicted signs
        pred_O: (B, T, dag_depth, total_nodes) predicted operand selectors
        pred_G: (B, T, dag_depth) predicted domain gates
        target_tensors: (B, T) nested list of target dictionaries with keys:
            - "target_digits": (num_initial_nodes, D, base) target digit one-hots
            - "target_V_sign": (total_nodes,) target signs
            - "target_O": (dag_depth, total_nodes) target operand selectors
            - "target_G": (dag_depth,) target domain gates
        valid_mask: (B, T) boolean mask for valid positions

    Returns:
        Dictionary with loss components
    """
    device = pred_digit_logits.device

    # Get valid positions
    valid_positions = valid_mask.nonzero(as_tuple=False)  # (num_valid, 2)

    if valid_positions.numel() == 0:
        # No valid positions - return zero losses
        return {
            "total_loss": torch.tensor(0.0, device=device),
            "digit_loss": torch.tensor(0.0, device=device),
            "V_sign_loss": torch.tensor(0.0, device=device),
            "O_loss": torch.tensor(0.0, device=device),
            "G_loss": torch.tensor(0.0, device=device),
            "exec_loss": torch.tensor(0.0, device=device),
        }

    batch_indices, token_indices = valid_positions[:, 0], valid_positions[:, 1]

    # Extract predictions for valid positions
    pred_digit_logits_valid = pred_digit_logits[
        batch_indices, token_indices
    ]  # (num_valid, num_initial_nodes, D, base)
    pred_V_sign_valid = pred_V_sign[
        batch_indices, token_indices
    ]  # (num_valid, total_nodes)
    pred_O_valid = pred_O[
        batch_indices, token_indices
    ]  # (num_valid, dag_depth, total_nodes)
    pred_G_valid = pred_G[batch_indices, token_indices]  # (num_valid, dag_depth)

    # Extract targets for valid positions
    valid_targets = []
    for batch_idx, token_idx in zip(batch_indices, token_indices):
        target_dict = target_tensors[batch_idx.item()][token_idx.item()]
        valid_targets.append(target_dict)

    # Stack targets from all valid positions
    target_digits = torch.stack([t["target_digits"] for t in valid_targets]).to(
        device
    )  # (num_valid, num_initial_nodes, D, base)
    target_V_sign = torch.stack([t["target_V_sign"] for t in valid_targets]).to(
        device
    )  # (num_valid, total_nodes)
    target_O = torch.stack([t["target_O"] for t in valid_targets]).to(
        device
    )  # (num_valid, dag_depth, total_nodes)
    target_G = torch.stack([t["target_G"] for t in valid_targets]).to(
        device
    )  # (num_valid, dag_depth)
    target_final_exec = torch.tensor(
        [t["target_final_exec"] for t in valid_targets],
        device=device,
        dtype=torch.float32,
    )  # (num_valid,)

    # Compute losses

    # Determine node counts from tensor shapes
    total_nodes = pred_V_sign_valid.shape[-1]
    dag_depth = (total_nodes - 1) // 2
    num_initial_nodes = dag_depth + 1

    # Digit loss - stable bounded loss for magnitudes
    digit_loss = _compute_digit_loss(pred_digit_logits_valid, target_digits)

    # V_mag loss - convert digits to magnitudes and compute L2 loss
    from models.dag_model import DAGExecutor

    # Convert predicted digit logits to V_mag values using shared method
    # Add batch and time dimensions for DAGExecutor interface: (num_valid, N, D, base) -> (num_valid, 1, N, D, base)
    pred_digit_logits_expanded = pred_digit_logits_valid.unsqueeze(1)
    pred_V_mag_from_digits = DAGExecutor.digits_to_vmag(
        pred_digit_logits_expanded,
        max_digits=pred_digit_logits_valid.shape[-2]
        // 2,  # Assuming D = max_digits + max_decimal_places
        max_decimal_places=pred_digit_logits_valid.shape[-2] // 2,
        base=pred_digit_logits_valid.shape[-1],
    ).squeeze(
        1
    )  # (num_valid, num_initial_nodes)

    # Convert target digits to V_mag values for comparison
    target_V_mag_from_digits = []
    for i in range(target_digits.shape[0]):
        target_digits_expanded = target_digits[i : i + 1].unsqueeze(
            1
        )  # (1, 1, N, D, base)
        target_V_mag = DAGExecutor.digits_to_vmag(
            target_digits_expanded,
            max_digits=target_digits.shape[-2] // 2,
            max_decimal_places=target_digits.shape[-2] // 2,
            base=target_digits.shape[-1],
        ).squeeze()  # (num_initial_nodes,)
        target_V_mag_from_digits.append(target_V_mag)
    target_V_mag_from_digits = torch.stack(
        target_V_mag_from_digits
    )  # (num_valid, num_initial_nodes)

    # Compute V_mag loss using robust value loss
    V_mag_loss = _compute_value_loss(pred_V_mag_from_digits, target_V_mag_from_digits)

    # Sign loss - only on initial nodes (L2 on tanh-activated predictions)
    V_sign_loss = F.mse_loss(
        torch.tanh(pred_V_sign_valid[:, :num_initial_nodes]),
        target_V_sign[:, :num_initial_nodes],
    )

    # Operand selector loss (L2)
    O_loss = F.mse_loss(pred_O_valid, target_O)

    # Domain gate loss (L2 on sigmoid-activated predictions)
    G_loss = F.mse_loss(torch.sigmoid(pred_G_valid), target_G)

    # Execution loss (if DAG executor is provided)
    exec_loss = torch.tensor(0.0, device=device)
    if dag_executor is not None:
        try:
            # Add batch and time dimensions back for executor: (num_valid,) -> (num_valid, 1)
            pred_digit_logits_exec = pred_digit_logits_valid.unsqueeze(
                1
            )  # (num_valid, 1, num_initial_nodes, D, base)
            pred_V_sign_exec = pred_V_sign_valid.unsqueeze(
                1
            )  # (num_valid, 1, total_nodes)
            pred_O_exec = pred_O_valid.unsqueeze(
                1
            )  # (num_valid, 1, dag_depth, total_nodes)
            pred_G_exec = pred_G_valid.unsqueeze(1)  # (num_valid, 1, dag_depth)

            # Execute predicted DAG - executor handles digit conversion internally
            pred_final_exec = dag_executor(
                pred_digit_logits_exec, pred_V_sign_exec, pred_O_exec, pred_G_exec
            )  # (num_valid, 1)
            pred_final_exec = pred_final_exec.squeeze(1)  # (num_valid,)

            # Compute robust execution loss
            exec_loss = _compute_exec_loss(pred_final_exec, target_final_exec)
        except Exception:
            print(
                f"Execution failed for batch {batch_indices[0].item()}, token {token_indices[0].item()}"
            )
            raise

    exec_loss_weight = 0.01
    total_loss = (
        digit_loss
        + V_mag_loss
        + V_sign_loss
        + O_loss
        + G_loss
        + (exec_loss * exec_loss_weight)
    )

    return {
        "total_loss": total_loss,
        "digit_loss": digit_loss,
        "V_mag_loss": V_mag_loss,
        "V_sign_loss": V_sign_loss,
        "O_loss": O_loss,
        "G_loss": G_loss,
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
