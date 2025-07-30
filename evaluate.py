"""Model evaluation utilities for DAG predictor models.

This module contains evaluation functions that were moved from predictor_utils.py
for better code organization and separation of concerns.
"""

import random as _eval_random
from typing import Dict, List

import torch
import torch.nn.functional as F

from predictor_utils import compute_dag_structure_loss, tokenize_texts
from tensor_utils import digits_to_magnitude


def evaluate_dag_model(
    model: torch.nn.Module,
    val_loader,
    device: str,
    ctx,
    cfg,
    eval_iters: int,
    seed: int,
) -> Dict[str, float]:
    """Run evaluation on *eval_iters* batches and return aggregated metrics.

    This function works with the new per-token streaming.py data format.
    """
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
        "executed_mse": 0.0,
        "initial_values_mse": 0.0,
        "valid_tokens": 0.0,
        "total_tokens": 0.0,
        "valid_token_rate": 0.0,
    }

    num_batches = 0
    master_process = device == "cuda" or device.startswith(
        "cuda"
    )  # Simple master process detection

    with torch.no_grad():
        for i, (texts, target_tensors, valid_mask) in enumerate(val_loader):
            if i >= eval_iters:
                break

            try:
                # Move target tensors to device
                for seq_targets in target_tensors:
                    for target_dict in seq_targets:
                        for key, tensor in target_dict.items():
                            if isinstance(tensor, torch.Tensor):
                                target_dict[key] = tensor.to(device)

                # Move valid mask to device
                valid_mask = valid_mask.to(device)

                # Inputs
                input_tokens = tokenize_texts(texts, cfg.block_size, device)

                # Model forward pass
                with ctx:
                    _ = model(input_tokens)

                # Get DAG predictor outputs
                dag_predictor = model.dag_predictor
                if dag_predictor is None:
                    raise RuntimeError("Model has no DAG predictor (dag_depth=0)")

                # Get raw logits for all positions
                pred_sign_logits = dag_predictor.last_sign_logits
                pred_digit_logits = dag_predictor.last_digit_logits
                pred_op_logits = dag_predictor.last_operation_logits
                pred_statistics = dag_predictor.last_statistics
                uncertainty_params = dag_predictor.uncertainty_params

                if pred_digit_logits is None:
                    raise RuntimeError("last_digit_logits not found for evaluation")
                if pred_sign_logits is None:
                    raise RuntimeError("last_sign_logits not found for evaluation")
                if pred_op_logits is None:
                    raise RuntimeError("last_operation_logits not found for evaluation")

                # Compute loss using new per-token loss function
                losses = compute_dag_structure_loss(
                    pred_sign_logits,
                    pred_digit_logits,
                    pred_op_logits,
                    pred_statistics,
                    target_tensors,
                    valid_mask,
                    cfg,
                    uncertainty_params,
                )

                # Accumulate losses
                for key in total_losses.keys():
                    total_losses[key] += losses[key].item()

                # Compute per-token metrics
                valid_tokens_count = valid_mask.sum().item()
                total_tokens_count = valid_mask.numel()

                total_metrics["valid_token_rate"] += (
                    valid_tokens_count / total_tokens_count
                )

                # Compute additional evaluation metrics
                # For simplicity, we'll compute MSE using valid positions only
                if valid_tokens_count > 0:
                    # Get valid positions
                    valid_positions = torch.nonzero(
                        valid_mask, as_tuple=False
                    )  # (num_valid, 2)
                    batch_indices = valid_positions[:, 0]  # (num_valid,)
                    token_indices = valid_positions[:, 1]  # (num_valid,)

                    # Extract valid targets
                    valid_targets = []
                    for batch_idx, token_idx in zip(batch_indices, token_indices):
                        target_dict = target_tensors[batch_idx.item()][token_idx.item()]
                        valid_targets.append(target_dict)

                    # Extract valid predictions
                    valid_sign_logits = pred_sign_logits[batch_indices, token_indices]
                    valid_digit_logits = pred_digit_logits[batch_indices, token_indices]
                    valid_statistics = pred_statistics["final"][
                        batch_indices, token_indices
                    ]

                    # Convert digit logits to magnitudes for initial values MSE
                    pred_initial_values = digits_to_magnitude(
                        valid_digit_logits.softmax(dim=-1),
                        cfg.max_digits,
                        cfg.max_decimal_places,
                        cfg.base,
                    )  # (num_valid, N)

                    # Get target initial values
                    target_initial_values = torch.stack(
                        [t["target_initial_values"] for t in valid_targets]
                    )

                    # Compute initial values MSE
                    initial_values_mse = F.mse_loss(
                        pred_initial_values, target_initial_values
                    )
                    total_metrics["initial_values_mse"] += initial_values_mse.item()

                    # Compute execution MSE using final statistics
                    target_final_exec = torch.tensor(
                        [t["target_final_exec"] for t in valid_targets], device=device
                    )
                    pred_final_exec = valid_statistics[
                        :, 0
                    ]  # First statistic is final exec value
                    executed_mse = F.mse_loss(pred_final_exec, target_final_exec)
                    total_metrics["executed_mse"] += executed_mse.item()

                num_batches += 1

                # Optional: Print sample validation details for debugging
                if master_process and (i % 10 == 0 or i < 3):
                    sample_text = texts[0] if texts else "N/A"
                    sample_valid_rate = (
                        valid_mask[0].float().mean().item()
                        if len(valid_mask) > 0
                        else 0.0
                    )
                    print(
                        f'[val] batch {i}: "{sample_text[:50]}...", valid_rate={sample_valid_rate:.1%}'
                    )

            except Exception as e:
                # Fail fast on ALL exceptions during evaluation - no skipping
                if master_process:
                    print(f"Error in evaluation batch {i+1}/{eval_iters}: {e}")
                raise

            if num_batches >= eval_iters:
                break

    # Average metrics over successful batches
    if num_batches > 0:
        for d in (total_losses, total_metrics):
            for k in d:
                d[k] /= num_batches

    model.train()
    return {**total_losses, **total_metrics}
