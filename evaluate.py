"""Model evaluation utilities for DAG predictor models.

This module contains evaluation functions that were moved from predictor_utils.py
for better code organization and separation of concerns.
"""

import random as _eval_random
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F

from predictor_utils import compute_dag_loss, tokenize_texts
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
            "V_mag_loss",
            "V_sign_loss",
            "O_loss",
            "G_loss",
            "exec_loss",
        )
    }
    total_metrics = {
        "valid_tokens": 0.0,
        "total_tokens": 0.0,
        "expression_valid_rate": 0.0,
    }

    num_batches = 0
    # For evaluation printing, we want to show details in most cases
    # Only skip if we're explicitly in a distributed non-master process
    try:
        master_process = not dist.is_initialized() or dist.get_rank() == 0
    except:
        master_process = True  # Default to showing details

    with torch.no_grad():
        for i, (texts, target_tensors, valid_mask) in enumerate(val_loader):
            if i >= eval_iters:
                break

            try:

                for seq_targets in target_tensors:
                    for target_dict in seq_targets:
                        for key, tensor in target_dict.items():
                            if isinstance(tensor, torch.Tensor):
                                target_dict[key] = tensor.to(device)

                valid_mask = valid_mask.to(device)

                # Inputs
                input_tokens = tokenize_texts(texts, cfg.block_size, device)

                # Model forward pass with DAG model
                with ctx:

                    hidden_states = model.forward_hidden(input_tokens)

                    dag_predictor = model.dag_predictor
                    if dag_predictor is None:
                        raise RuntimeError("Model has no DAG predictor (dag_depth=0)")

                    pred_V_mag, pred_V_sign, pred_O, pred_G = dag_predictor(
                        hidden_states
                    )

                # Compute loss using DAG tensor loss function
                dag_executor = getattr(model, "dag_executor", None)
                losses = compute_dag_loss(
                    pred_V_mag,
                    pred_V_sign,
                    pred_O,
                    pred_G,
                    target_tensors,
                    valid_mask,
                    cfg,
                    dag_executor=dag_executor,
                )

                # Accumulate losses
                for key in total_losses.keys():
                    total_losses[key] += losses[key].item()

                # Compute per-token metrics
                valid_tokens_count = valid_mask.sum().item()

                # Calculate expression-level valid rate (excluding padding)
                batch_size, seq_len = valid_mask.shape
                expression_valid_tokens = 0
                expression_total_tokens = 0

                for b in range(batch_size):
                    # Find the last non-padding token (expression length)
                    sequence_mask = valid_mask[b]  # (seq_len,)

                    # Find expression length by looking for the transition to padding
                    # Padding is always False values at the end
                    expression_length = seq_len
                    for i in range(seq_len - 1, -1, -1):
                        if (
                            i == 0
                            or sequence_mask[i]
                            or (i < seq_len - 1 and sequence_mask[i + 1])
                        ):
                            expression_length = i + 1
                            break

                    # Count valid tokens within the expression (before padding)
                    expression_tokens = sequence_mask[:expression_length]
                    expression_valid_tokens += expression_tokens.sum().item()
                    expression_total_tokens += expression_length

                if expression_total_tokens > 0:
                    total_metrics["expression_valid_rate"] += (
                        expression_valid_tokens / expression_total_tokens
                    )

                num_batches += 1

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

                    if False and valid_tokens_count > 0 and len(texts) > 0:
                        # Reconstruct seed for this sample
                        sample_seed = (
                            seed + i * cfg.batch_size + 10000
                        )  # Validation offset

                        print(
                            f"\n=== DETAILED VALIDATION SAMPLE (seed={sample_seed}) ==="
                        )
                        print(f"Full expression: '{texts[0]}'")

                        # Get the last valid token position for this sample (complete expression)
                        sample_mask = valid_mask[0]  # (T,)
                        last_valid_pos = -1
                        for pos in range(len(sample_mask)):
                            if sample_mask[pos]:
                                last_valid_pos = pos

                        if last_valid_pos >= 0:
                            # Get target and prediction for the final/complete expression
                            target_dict = target_tensors[0][last_valid_pos]

                            # Target values
                            target_digits = target_dict[
                                "target_initial_digits"
                            ]  # (N, D, base)
                            target_ops = target_dict[
                                "target_operation_probs"
                            ]  # (depth, n_ops)
                            target_final_exec = target_dict["target_final_exec"]

                            # Convert target digits to initial values
                            target_initial_values = []
                            for n in range(target_signs.shape[0]):
                                sign = target_signs[n].item()
                                digit_probs = target_digits[n]  # (D, base)
                                magnitude = digits_to_magnitude(
                                    digit_probs,
                                    cfg.max_digits,
                                    cfg.max_decimal_places,
                                    cfg.base,
                                )
                                value = sign * magnitude
                                target_initial_values.append(value)

                            # Target operations
                            target_op_names = []
                            for d in range(target_ops.shape[0]):
                                op_probs = target_ops[d]  # (n_ops,)
                                op_idx = torch.argmax(op_probs).item()
                                op_names = ["add", "multiply", "identity"]  # STATE_OPS
                                target_op_names.append(op_names[op_idx])

                            # Predicted values (from the forward pass)
                            pred_signs_logit = pred_sign_logits[
                                0, last_valid_pos
                            ]  # (N,)
                            pred_digits_logit = pred_digit_logits[
                                0, last_valid_pos
                            ]  # (N, D, base)
                            pred_ops_logit = pred_op_logits[
                                0, last_valid_pos
                            ]  # (depth, n_ops)

                            # Convert predictions
                            pred_signs = torch.tanh(pred_signs_logit)
                            pred_digit_probs = torch.softmax(pred_digits_logit, dim=-1)
                            pred_op_probs = torch.softmax(pred_ops_logit, dim=-1)

                            # Convert predicted digits to initial values
                            pred_initial_values = []
                            for n in range(pred_signs.shape[0]):
                                sign = pred_signs[n].item()
                                digit_probs = pred_digit_probs[n]  # (D, base)
                                magnitude = digits_to_magnitude(
                                    digit_probs,
                                    cfg.max_digits,
                                    cfg.max_decimal_places,
                                    cfg.base,
                                )
                                value = sign * magnitude
                                pred_initial_values.append(value)

                            # Predicted operations
                            pred_op_names = []
                            for d in range(pred_op_probs.shape[0]):
                                op_probs = pred_op_probs[d]  # (n_ops,)
                                op_idx = torch.argmax(op_probs).item()
                                op_names = ["add", "multiply", "identity"]  # STATE_OPS
                                pred_op_names.append(op_names[op_idx])

                            # Predicted final execution (from statistics)
                            pred_final_exec = pred_statistics["final"][
                                0, last_valid_pos, 0
                            ].item()  # First statistic is final exec

                            print(
                                f"Target initial values: {[f'{v:.3f}' for v in target_initial_values]}"
                            )
                            print(
                                f"Pred   initial values: {[f'{v:.3f}' for v in pred_initial_values]}"
                            )
                            print(f"Target operations:     {target_op_names}")
                            print(f"Pred   operations:     {pred_op_names}")
                            print(f"Target final exec:     {target_final_exec:.6f}")
                            print(f"Pred   final exec:     {pred_final_exec:.6f}")
                            print(
                                f"Exec error:            {abs(target_final_exec - pred_final_exec):.6f}"
                            )

                        print("=" * 60 + "\n")

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
