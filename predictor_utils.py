from __future__ import annotations

"""Utility helpers shared by DAG predictor training and evaluation.
Moving these functions into their own module drastically reduces the size of
`train_predictor.py` while keeping behaviour unchanged.
"""

from typing import List

import torch
import torch.nn.functional as F
from tiktoken import get_encoding


def compute_dag_loss(
    pred_V_mag: torch.Tensor,  # (B, T, total_nodes) predicted magnitudes
    pred_V_sign: torch.Tensor,  # (B, T, total_nodes) predicted signs
    pred_O: torch.Tensor,  # (B, T, dag_depth, total_nodes) predicted operand selection
    pred_G: torch.Tensor,  # (B, T, dag_depth) predicted domain gates
    target_tensors: list[
        list[dict[str, torch.Tensor]]
    ],  # (B, T) nested list: new target tensors
    valid_mask: torch.Tensor,  # (B, T) boolean mask for valid positions
    dag_executor=None,  # Optional DAGExecutor for execution loss
) -> dict[str, torch.Tensor]:
    """Compute loss for DAG tensor format with V_mag, V_sign, O, G targets.

    Args:
        pred_V_mag: (B, T, total_nodes) predicted magnitudes
        pred_V_sign: (B, T, total_nodes) predicted signs
        pred_O: (B, T, dag_depth, total_nodes) predicted operand selectors
        pred_G: (B, T, dag_depth) predicted domain gates
        target_tensors: (B, T) nested list of target dictionaries with keys:
            - "target_V_mag": (total_nodes,) target magnitudes
            - "target_V_sign": (total_nodes,) target signs
            - "target_O": (dag_depth, total_nodes) target operand selectors
            - "target_G": (dag_depth,) target domain gates
        valid_mask: (B, T) boolean mask for valid positions
        cfg: Training configuration

    Returns:
        Dictionary with loss components
    """
    device = pred_V_mag.device

    # Get valid positions
    valid_positions = valid_mask.nonzero(as_tuple=False)  # (num_valid, 2)

    if valid_positions.numel() == 0:
        # No valid positions - return zero losses
        return {
            "total_loss": torch.tensor(0.0, device=device),
            "V_mag_loss": torch.tensor(0.0, device=device),
            "V_sign_loss": torch.tensor(0.0, device=device),
            "O_loss": torch.tensor(0.0, device=device),
            "G_loss": torch.tensor(0.0, device=device),
            "exec_loss": torch.tensor(0.0, device=device),
        }

    batch_indices, token_indices = valid_positions[:, 0], valid_positions[:, 1]

    # Extract predictions for valid positions
    pred_V_mag_valid = pred_V_mag[
        batch_indices, token_indices
    ]  # (num_valid, total_nodes)
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
    target_V_mag = torch.stack([t["target_V_mag"] for t in valid_targets]).to(
        device
    )  # (num_valid, total_nodes)
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

    # Magnitude loss (L2)
    V_mag_loss = F.mse_loss(pred_V_mag_valid, target_V_mag)

    # Sign loss (L2 on tanh-activated predictions)
    V_sign_loss = F.mse_loss(torch.tanh(pred_V_sign_valid), target_V_sign)

    # Operand selector loss (L2)
    O_loss = F.mse_loss(pred_O_valid, target_O)

    # Domain gate loss (L2 on sigmoid-activated predictions)
    G_loss = F.mse_loss(torch.sigmoid(pred_G_valid), target_G)

    # Execution loss (if DAG executor is provided)
    exec_loss = torch.tensor(0.0, device=device)
    if dag_executor is not None:
        try:
            # Execute the predicted DAG tensors
            # Need to add batch and time dimensions back for executor: (num_valid,) -> (num_valid, 1)
            pred_V_mag_exec = pred_V_mag_valid.unsqueeze(
                1
            )  # (num_valid, 1, total_nodes)
            pred_V_sign_exec = pred_V_sign_valid.unsqueeze(
                1
            )  # (num_valid, 1, total_nodes)
            pred_O_exec = pred_O_valid.unsqueeze(
                1
            )  # (num_valid, 1, dag_depth, total_nodes)
            pred_G_exec = pred_G_valid.unsqueeze(1)  # (num_valid, 1, dag_depth)

            # Execute predicted DAG
            pred_final_exec = dag_executor(
                pred_V_mag_exec, pred_V_sign_exec, pred_O_exec, pred_G_exec
            )  # (num_valid, 1)
            pred_final_exec = pred_final_exec.squeeze(1)  # (num_valid,)

            # Compute execution loss
            exec_loss = F.mse_loss(pred_final_exec, target_final_exec)
        except Exception:
            print(
                f"Execution failed for batch {batch_indices[0].item()}, token {token_indices[0].item()}"
            )
            raise

    # Total loss (simple sum for now)
    total_loss = V_mag_loss + V_sign_loss + O_loss + G_loss + exec_loss

    return {
        "total_loss": total_loss,
        "V_mag_loss": V_mag_loss,
        "V_sign_loss": V_sign_loss,
        "O_loss": O_loss,
        "G_loss": G_loss,
        "exec_loss": exec_loss,
    }


__all__ = [
    "tokenize_texts",
    "compute_dag_structure_loss",
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

    # Use binary cross entropy directly
    sign_loss = F.binary_cross_entropy_with_logits(
        pred_sign_logits_flat, binary_targets, reduction="mean"
    )
    return sign_loss
